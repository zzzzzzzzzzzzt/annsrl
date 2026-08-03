import os, sys
import os.path as osp
import argparse
# %load_ext line_profiler
# %load_ext autoreload
# %autoreload 2
# %env CUDA_VISIBLE_DEVICES=1
sys.path.append('..')
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch
import time
import lib

print("Numpy: {}, Torch: {}".format(np.__version__, torch.__version__))

##################
# CLI arguments  #
##################

parser = argparse.ArgumentParser(description='PPO training on SIFT/DEEP HNSW graph')
# Which agent class to train from scratch (or resume-shape when restore_step is
# set): 'simple' -> lib.SimpleNeuralAgent (plain from-scratch MLP over raw
# coords), 'mlp_link' -> lib.MLPLinkAgent (structurally mirrors
# mlpretrain_hard.py's MLPLinkNet, and is the only agent_type pretrained_path
# below can load into).
parser.add_argument('--agent_type', type=str, default='mlp_link',
                    choices=['simple', 'mlp_link'],
                    help="'simple'=SimpleNeuralAgent, 'mlp_link'=MLPLinkAgent")
# Path to a checkpoint produced by mlpretrain_hard.py. Leave unset to train from
# scratch; the canonical models/mlplink_{dataset}_{scorer}_best.pth default is
# used only when --agent_type mlp_link and that file exists (see below).
parser.add_argument('--pretrained_path', type=str, default=None,
                    help='checkpoint from mlpretrain_hard.py to warm-start the '
                         'agent (only valid with --agent_type mlp_link)')
# ProximityDCSReward hyperparameters.
parser.add_argument('--alpha', type=float, default=1.0,
                    help='ProximityDCSReward alpha coefficient')
parser.add_argument('--beta', type=float, default=1.0,
                    help='ProximityDCSReward beta coefficient')
args = parser.parse_args()
print(args)

DATA_DIR = './data/SIFT100K'

# if not os.path.exists(DATA_DIR):
#     assert not DATA_DIR.endswith(os.sep), 'please do not put "/" at the end of DATA_DIR'
#     !mkdir -p {DATA_DIR}
#     !wget https://www.dropbox.com/sh/vfnqvuk2jex4oqc/AADglx1ukHNaT3cNOVgAxMvfa?dl=1 -O {DATA_DIR}/deep_100k.zip
#     !cd {DATA_DIR} && unzip deep_100k.zip && rm deep_100k.zip

seed = random.randint(0, 2**32-1)
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
print("Random seed: %d" % seed)

################
# Graph params #
################

graph_type = 'nsw'        # 'nsw' or 'nsg'
M = 12                    # degree parameter for NSW (Max degree is 2*M)
ef = 10                   # search algorithm parameter, sets the operating point
R = 24                    #  degree parameter for NSG (corresponds to max degree)
k = 1                     # Number of answers per query. Need for Recall@k

assert k <= ef

nn = 200                  # Number of NN in initial KNNG that is used for NSG construction 
efC = 300                 # efConstruction used for NSW graph
ngt = 100                 # Number of ground truth answers per query

train_queries_size = 200000  # Number of training queries
val_queries_size = 20000     # Number of queries for validation

#################
# Reward params #
#################

max_dcs = 1000            # reward hyperparameter

################
# Agent params #
################

hidden_size = 2048        # number of hidden units (used by SimpleNeuralAgent; unused by MLPLinkAgent, kept for exp_name/back-compat)

# Which agent class to train from scratch (or resume-shape when restore_step
# is set): 'simple' -> lib.SimpleNeuralAgent (plain from-scratch MLP over raw
# coords), 'mlp_link' -> lib.MLPLinkAgent (structurally mirrors mlpretrain_hard.py's
# MLPLinkNet, and is the only agent_type pretrained_path below can load into).
agent_type = args.agent_type
assert agent_type in ('simple', 'mlp_link')

# MLPLinkAgent hyperparams -- must match the checkpoint's mlpretrain_hard.py run
# whenever pretrained_path is set (see override below), otherwise these are the
# from-scratch defaults (mirrors mlpretrain_hard.py's argparse defaults).
node_hidden = 96         # per-node encoder width; 0 = feed raw coords (no encoder)
node_layers = 2           # number of layers in the per-node encoder
pair_hidden = 128         # hidden width of the pairwise scoring MLP
scorer = 'dot'           # 'pair' (MLP([z_i,z_j])) or 'dot' (two-tower cosine)
norm = 'none'             # 'none', 'bn' or 'ln' -- normalization in the node encoder
dropout = 0.3

# Path to a checkpoint produced by mlpretrain_hard.py, e.g.:
#   {'state_dict': ..., 'node_hidden': ..., 'node_layers': ..., 'pair_hidden': ...,
#    'scorer': ..., 'norm': ..., 'mean': ..., 'std': ...}
# Set to None (or leave --pretrained_path unset) to train the agent from scratch, so the
# same script can be run twice -- with/without this set -- to compare pretrained vs.
# from-scratch PPO training on identical hyperparameters otherwise. Only applies
# to agent_type == 'mlp_link' -- SimpleNeuralAgent has no pretraining path.
#
# Default: the canonical best model mlpretrain_hard.py saves under models/. The
# filename scheme (mlplink_{dataset}_{scorer}_best.pth) is kept in sync with
# that script's save_checkpoint. An explicit --pretrained_path overrides
# this; the default is only used when agent_type == 'mlp_link' and the file
# actually exists (so 'simple' runs and fresh checkouts still train from scratch).
default_pretrained_path = osp.join(
    'models', 'mlplink_{}_{}_best.pth'.format(osp.split(DATA_DIR)[-1], scorer))
pretrained_path = args.pretrained_path
if pretrained_path is None and agent_type == 'mlp_link' and osp.exists(default_pretrained_path):
    pretrained_path = default_pretrained_path
    print('Using default pretrained checkpoint:', pretrained_path)
pretrained_path = pretrained_path or None
assert pretrained_path is None or agent_type == 'mlp_link', \
    "pretrained_path requires agent_type == 'mlp_link'"

####################
# Algorithm params #
####################

samples_in_batch = 4096   # PPO mini-batch size per gradient step
ppo_epochs = 3            # number of gradient passes over each session batch
lr = 2e-4                 # Adam learning rate
clip_eps = 0.1            # PPO clipping epsilon
edge_patience = 400       # How many iterations are needed without the change of edge probability
                          # to denote the prediction as confident and make it deterministic
                          # Very important for training procedure efficiency

entropy_reg = 0.01        # coefficient in front of the entropy regularizer term
batch_size = 100000       # number of sessions per batch
update_edges_every = 10   # call hnsw.update_edges() every N steps

n_jobs = 8                # Number of threads for C++ sampling
max_steps = 1000          # Max number of training iterations

# Recover settings
restore_step = None       # the iteration step from which you want to recover the model 

import lib
import os.path as osp

graph_params = { 
    'vertices_path': osp.join(DATA_DIR, 'sift_base.fvecs'),

    'train_queries_path': osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
    'test_queries_path': osp.join(DATA_DIR, 'sift_query.fvecs'),
    
    'train_gt_path': osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
    'test_gt_path': osp.join(DATA_DIR, 'test_gt.ivecs'),
#     ^-- comment these 2 lines to re-compute ground truth ids (if you don't have pre-computed ground truths)
    
    'train_queries_size': train_queries_size + val_queries_size,  # valid set is a train subset
    'val_queries_size': val_queries_size,
    'ground_truth_n_neighbors': ngt,  # for each query, finds this many nearest neighbors via brute force
    'graph_type': graph_type
}

if graph_type == 'nsg':
    graph_params['edges_path'] = osp.join(DATA_DIR, 'sift_R{R}_{nn}nn.nsg'.format(R=R, nn=nn))
elif graph_type == 'nsw':
    graph_params['edges_path'] = osp.join(DATA_DIR, 'sift_nsw_M{M}_efC{efC}.ivecs'.format(M=M, efC=efC))
    graph_params['initial_vertex_id'] = 0  # by default, starts search from this vertex
else:
    raise ValueError("Wrong graph type: ['nsg', 'nsw']")
    
graph = lib.Graph(**graph_params)

# Tag the run dir by agent_type / pretrain status so different agent choices
# and "with" vs "without" pretraining land in separate ./runs/ subfolders and
# can be compared side by side (e.g. in tensorboard) without overwriting one
# another. SimpleNeuralAgent has no pretraining, so its tag is just its name.
if agent_type == 'simple':
    pretrain_tag = 'simple'
else:
    pretrain_tag = 'mlp_link_pretrained' if pretrained_path else 'mlp_link_scratch'

if graph_type == 'nsw':
    exp_name = '{data_name}_{graph_type}_k{k}_M{M}_ef{ef}_max-dcs{max_dcs}_hid-size{hidden_size}_entropy{entropy_reg}_patience{edge_patience}_{pretrain_tag}_seed_{seed}'.format(
        data_name=osp.split(DATA_DIR)[-1], k=k, M=M, ef=ef, hidden_size=hidden_size,
        max_dcs=max_dcs, entropy_reg=entropy_reg, graph_type=graph_type, edge_patience=edge_patience,
        pretrain_tag=pretrain_tag, seed=seed
    )
elif graph_type == 'nsg':
    exp_name = '{data_name}_{graph_type}_k{k}_R{R}_ef{ef}_max-dcs{max_dcs}_hid-size{hidden_size}_entropy{entropy_reg}_patience{edge_patience}_{pretrain_tag}_seed_{seed}'.format(
        data_name=osp.split(DATA_DIR)[-1], k=k, R=R, ef=ef, hidden_size=hidden_size,
        max_dcs=max_dcs, entropy_reg=entropy_reg, graph_type=graph_type, edge_patience=edge_patience,
        pretrain_tag=pretrain_tag, seed=seed,
    )

print('exp name:', exp_name)
# !rm {'./runs/' + exp_name} -rf # KEEP COMMENTED!
assert restore_step is not None or not os.path.exists('./runs/' + exp_name)

hnsw = lib.ParallelHNSW(graph, ef=ef, k=k, edge_patience=edge_patience, n_jobs=n_jobs)

if restore_step is not None:
    agent = torch.load("runs/{}/agent.{}.pth".format(exp_name, restore_step), weights_only=False)
    baseline = torch.load("runs/{}/baseline.{}.pth".format(exp_name, restore_step), weights_only=False)
    hnsw.edge_confidence = torch.load("runs/{}/edge_confidence.{}.pth".format(exp_name, restore_step), weights_only=False)
elif pretrained_path:
    # Load a checkpoint produced by mlpretrain_hard.py: reconstruct MLPLinkAgent
    # with the SAME architecture hyperparams the checkpoint was trained with
    # (falling back to this script's own defaults for any key the checkpoint
    # doesn't carry), then load its weights.
    print('Loading pretrained agent from:', pretrained_path)
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    if 'mean' not in ckpt or 'std' not in ckpt:
        print('[WARN] checkpoint has no mean/std -- feeding un-standardized '
              'coordinates to a model pretrained on standardized ones')
    agent = lib.MLPLinkAgent(
        graph.vertices.shape[1],
        node_hidden=ckpt.get('node_hidden', node_hidden),
        node_layers=ckpt.get('node_layers', node_layers),
        pair_hidden=ckpt.get('pair_hidden', pair_hidden),
        scorer=ckpt.get('scorer', scorer),
        norm=ckpt.get('norm', norm),
        dropout=dropout,
        # mlpretrain_hard.py standardizes vertex coordinates (zero-mean/unit-std)
        # before training; the pretrained weights expect that same input
        # distribution. Standardization is applied INSIDE the agent's own
        # encode() (see MLPLinkAgent.encode) rather than on graph.vertices
        # itself, since graph.vertices also feeds HNSW's real L2 distance
        # search and the reward -- those must stay in raw coordinate space.
        feat_mean=ckpt.get('mean'), feat_std=ckpt.get('std'),
    )
    # The pretrained MLPLinkNet has no feat_mean/feat_std buffers, so its
    # state_dict lacks those keys. This agent DOES register them (populated
    # above from ckpt['mean']/['std'] via the constructor), so a strict load
    # would fail on exactly those two missing keys. Load non-strict, then assert
    # nothing UNexpected was in the checkpoint and nothing besides the two
    # standardization buffers was missing -- any other mismatch is a real bug.
    missing, unexpected = agent.load_state_dict(ckpt['state_dict'], strict=False)
    allowed_missing = {'feat_mean', 'feat_std'}
    leftover_missing = set(missing) - allowed_missing
    assert not unexpected, f"unexpected keys in checkpoint: {sorted(unexpected)}"
    assert not leftover_missing, f"unexpected missing keys: {sorted(leftover_missing)}"
    baseline = lib.SessionBaseline(graph.train_queries.size(0) + graph.vertices.size(0))
elif agent_type == 'simple':
    agent = lib.SimpleNeuralAgent(graph.vertices.shape[1], hidden_size=hidden_size)
    baseline = lib.SessionBaseline(graph.train_queries.size(0) + graph.vertices.size(0))
else:
    agent = lib.MLPLinkAgent(
        graph.vertices.shape[1],
        node_hidden=node_hidden,
        node_layers=node_layers,
        pair_hidden=pair_hidden,
        scorer=scorer,
        norm=norm,
        dropout=dropout,
    )
    baseline = lib.SessionBaseline(graph.train_queries.size(0) + graph.vertices.size(0))

reward = lib.ProximityDCSReward(graph.vertices, k=k, max_dcs=max_dcs,
                                alpha=args.alpha, beta=args.beta)
trainer = lib.OptimizedPPO(agent, hnsw, reward, baseline,
                  lr=lr,
                  clip_eps=clip_eps,
                  ppo_epochs=ppo_epochs,
                  samples_in_batch=samples_in_batch,
                  entropy_reg=entropy_reg,
                  target_kl=0.015,                   # 加入早停保障
                  writer=SummaryWriter('./runs/' + exp_name))

if restore_step is not None:
    trainer.step = restore_step

from pandas import DataFrame
from IPython.display import clear_output
import matplotlib.pyplot as plt
# %matplotlib inline
moving_average = lambda x, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(**kw).mean().values
reward_history = []
best_val_step = 0
best_val_reward = 0
# best_val_step should only be picked once edges have had a chance to become
# confident (edge_patience), but if max_steps is set below edge_patience (e.g.
# a quick test run) that gate would never open and best_val_step would stay
# stuck at its initial 0 -- with no agent.0.pth ever saved, crashing the final
# "load best checkpoint" block. Cap the gate at max_steps so it's always
# reachable, while still respecting edge_patience for normal (longer) runs.
best_val_min_step = min(edge_patience, max_steps)

# generate batches of [queries, ground truth, train_query_ids (for baseline)]
# 1. 获取图节点数量和 query 数量    
num_train_queries = graph.train_queries.size(0)    
num_vertices = graph.vertices.size(0)    
     
train_gt_top1 = graph.train_gt[:, :1]    
vertices_gt_top1 = torch.arange(num_vertices).unsqueeze(1).to(graph.train_gt.device)    
      
train_query_ids = torch.arange(num_train_queries)
vertices_query_ids = torch.arange(num_train_queries, num_train_queries + num_vertices)    
      
mixed_queries = torch.cat([graph.train_queries, graph.vertices], dim=0)    
mixed_gt = torch.cat([train_gt_top1, vertices_gt_top1], dim=0)
mixed_query_ids = torch.cat([train_query_ids, vertices_query_ids], dim=0)    
    
train_batcher = lib.utils.iterate_minibatches(mixed_queries, mixed_gt,     
                                            mixed_query_ids, batch_size=batch_size)    

# generate batches of [queries, ground truth]           
val_iterator = lib.utils.iterate_minibatches(graph.val_queries, graph.val_gt, 
                                             batch_size=graph.val_queries.size(0))

dev_iterator = lib.utils.iterate_minibatches(graph.test_queries, graph.test_gt, 
                                             batch_size=graph.test_queries.size(0))

for batch_queries, batch_gt, batch_query_ids in train_batcher:
    start = time.time()
    torch.cuda.empty_cache()
    mean_reward = trainer.train_step(batch_queries, batch_gt, query_index=batch_query_ids)
    reward_history.append(mean_reward)

    # if trainer.step % update_edges_every == 0:
    #     promoted = hnsw.update_edges()
    #     trainer.writer.add_scalar('train/promoted_edges', promoted, global_step=trainer.step)
        
    if trainer.step % 10 == 0:
        val_reward = trainer.evaluate(*next(val_iterator), prefix='val')
        if val_reward > best_val_reward and \
           trainer.step % 50 == 0 and \
           trainer.step >= edge_patience:
            best_val_reward = val_reward
            best_val_step = trainer.step
        
    if trainer.step % 50 == 0:
        _ = trainer.evaluate(*next(dev_iterator))
        print(end="Saving...")
        torch.save(agent, "runs/{}/agent.{}.pth".format(exp_name, trainer.step))
        torch.save(baseline, "runs/{}/baseline.{}.pth".format(exp_name, trainer.step))
        torch.save(hnsw.edge_confidence, "runs/{}/edge_confidence.{}.pth".format(exp_name, trainer.step))
        print('Done!')
    
    if trainer.step % 1 == 0:
        clear_output(True)
        plt.title('train reward over time')
        plt.plot(moving_average(reward_history, span=50))
        plt.scatter(range(len(reward_history)), reward_history, alpha=0.1)
        plt.grid()
        plt.show()
        print("step=%i, mean_reward=%.6f, time=%.3f" % 
              (trainer.step, np.mean(reward_history[-100:]), time.time()-start))
    
    if trainer.step >= max_steps: break

#protip: run tensorboard in ./runs to get all metrics.

print("Best step on validation: %d" % best_val_step)
agent = torch.load("runs/{}/agent.{}.pth".format(exp_name, best_val_step), weights_only=False)
hnsw.edge_confidence = torch.load("runs/{}/edge_confidence.{}.pth".format(exp_name, best_val_step), weights_only=False)
trainer.step = best_val_step

from collections import defaultdict

agent.cuda()
state = agent.prepare_state(graph, device='cuda')

new_edges = defaultdict(list)

for i in range(len(hnsw.from_vertex_ids)):
    from_vertex_ids = np.array(hnsw.from_vertex_ids[i])
    to_vertex_ids = np.array(hnsw.to_vertex_ids[i])
    edge_confidence = np.array(hnsw.edge_confidence[i])

    with torch.no_grad():
        edges_logp = agent.get_edge_logp(from_vertex_ids, to_vertex_ids,
                                        state=state, device='cuda').cpu()
        edges_mask = edges_logp.argmax(-1).numpy() == 1
        edges_mask = edges_mask | (edge_confidence == hnsw.edge_patience)
        edges_mask = edges_mask & (edge_confidence != -hnsw.edge_patience)
    from_vertex_ids = from_vertex_ids[edges_mask]
    to_vertex_ids = to_vertex_ids[edges_mask]
    
    for from_vertex_id, to_vertex_id in zip(from_vertex_ids, to_vertex_ids):
        new_edges[from_vertex_id].append(to_vertex_id)
    for i in range(len(graph.edges)):
        if len(new_edges[i]) == 0:
            new_edges[i] = []

#Save constructed graph
new_edges=dict(sorted(new_edges.items())) # to preserve edges in the correct order in the file
lib.write_edges("runs/{}/graph.{}.ivecs".format(exp_name, trainer.step), new_edges)

hnsw.max_trajectory = 300  # Set larger number of hops allowed to the search algorithm 
                           # to deal with large heap_sizes

for heap_size in range(12, 301, 4):
    algo_hnsw = lib.BaseAlgorithm(
        agent=agent, hnsw=hnsw,
        reward=lambda actions, **kw: [0] * len(actions),
        writer=trainer.writer, device='cuda',
    )
    algo_hnsw.hnsw.ef = heap_size
    algo_hnsw.step = trainer.step  # for tensorboard

    metrics = algo_hnsw.get_session_batch(graph.test_queries, graph.test_gt, greedy=True,
                             summarize=True, write_logs=False, prefix='dev', is_evaluate=True)['summary']
    sys.stderr.flush()
    print("Ef %i | Recall@%d %.4f | Distances: %.1f" % 
          (heap_size, k, metrics['dev/recall@%d' % k], metrics['dev/distance_computations']),
          flush=True,
         )