import os, sys
import os.path as osp
sys.path.append('..')
import numpy as np
import torch
import argparse
import lib

# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True,
                    help='Path to experiment run dir, e.g. runs/DEEP100K_nsw_...')
parser.add_argument('--step', type=int, required=True,
                    help='Checkpoint step to load (agent.{step}.pth)')
parser.add_argument('--data_dir', type=str, default='./data/DEEP100K')
parser.add_argument('--graph_type', type=str, default='nsw', choices=['nsw', 'nsg'])
parser.add_argument('--M', type=int, default=12)
parser.add_argument('--R', type=int, default=24)
parser.add_argument('--nn', type=int, default=200)
parser.add_argument('--efC', type=int, default=300)
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--ef_min', type=int, default=12)
parser.add_argument('--ef_max', type=int, default=120)
parser.add_argument('--ef_step', type=int, default=4)
parser.add_argument('--n_jobs', type=int, default=8)
parser.add_argument('--max_trajectory', type=int, default=300)
parser.add_argument('--ngt', type=int, default=100)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

DATA_DIR = args.data_dir

# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────
graph_params = {
    'vertices_path': osp.join(DATA_DIR, 'deep_base.fvecs'),
    'train_queries_path': osp.join(DATA_DIR, 'deep_learn_1m.fvecs'),
    'test_queries_path': osp.join(DATA_DIR, 'deep_query.fvecs'),
    'train_gt_path': osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
    'test_gt_path': osp.join(DATA_DIR, 'test_gt.ivecs'),
    'train_queries_size': 1,   # minimal, not used for eval
    'val_queries_size': 1,
    'ground_truth_n_neighbors': args.ngt,
    'graph_type': args.graph_type,
}

if args.graph_type == 'nsg':
    graph_params['edges_path'] = osp.join(
        DATA_DIR, 'deep_R{R}_{nn}nn.nsg'.format(R=args.R, nn=args.nn))
elif args.graph_type == 'nsw':
    graph_params['edges_path'] = osp.join(
        DATA_DIR, 'deep_nsw_M{M}_efC{efC}.ivecs'.format(M=args.M, efC=args.efC))
    graph_params['initial_vertex_id'] = 0

graph = lib.Graph(**graph_params)

# ─────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────
agent_path = osp.join(args.run_dir, 'agent.{}.pth'.format(args.step))
conf_path  = osp.join(args.run_dir, 'edge_confidence.{}.pth'.format(args.step))

print('Loading agent from:', agent_path)
agent = torch.load(agent_path, map_location=args.device, weights_only=False)
agent.to(args.device)
agent.eval()

hnsw = lib.ParallelHNSW(graph, ef=args.ef_min, k=args.k, n_jobs=args.n_jobs)
hnsw.max_trajectory = args.max_trajectory

if osp.exists(conf_path):
    print('Loading edge_confidence from:', conf_path)
    hnsw.edge_confidence = torch.load(conf_path, weights_only=False)

# ─────────────────────────────────────────────
# Sweep ef and report Recall / Distance computations
# ─────────────────────────────────────────────
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(osp.join(args.run_dir, 'eval_sweep'))

state = agent.prepare_state(graph, device=args.device)

print('\n{:<8} {:<12} {:<20}'.format('ef', 'Recall@%d' % args.k, 'Avg Distances'))
print('-' * 42)

for heap_size in range(args.ef_min, args.ef_max + 1, args.ef_step):
    algo = lib.BaseAlgorithm(
        agent=agent, hnsw=hnsw,
        reward=lambda actions, **kw: [0] * len(actions),
        writer=writer, device=args.device,
    )
    algo.hnsw.ef = heap_size
    algo.step = args.step

    metrics = algo.get_session_batch(
        graph.test_queries, graph.test_gt,
        greedy=True, summarize=True, write_logs=False,
        prefix='eval', is_evaluate=True,
    )['summary']

    recall = metrics['eval/recall@%d' % args.k]
    dcs    = metrics['eval/distance_computations']
    print('{:<8} {:<12.4f} {:<20.1f}'.format(heap_size, recall, dcs))

    writer.add_scalar('eval/recall@%d' % args.k, recall, global_step=heap_size)
    writer.add_scalar('eval/distance_computations', dcs, global_step=heap_size)

writer.close()
print('\nDone. TensorBoard logs written to:', osp.join(args.run_dir, 'eval_sweep'))
