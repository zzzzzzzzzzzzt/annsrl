import os, sys
import os.path as osp
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

DATA_DIR = './data/DEEP100K'

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

hidden_size = 32         # NodeFormer hidden / output dimension
mlp_hidden_size = 128     # edge MLP hidden dimension
num_layers = 2            # number of NodeFormer message-passing layers
num_heads = 6             # number of attention heads
nb_random_features = 30   # random features for kernelized softmax
use_bn = True             # layer normalization
use_residual = True       # residual connections

####################
# Algorithm params #
####################

samples_in_batch = 1024   # PPO mini-batch size per gradient step
ppo_epochs = 4            # number of gradient passes over each session batch
lr = 3e-4                 # Adam learning rate
clip_eps = 0.2            # PPO clipping epsilon
edge_patience = 400       # How many iterations are needed without the change of edge probability
                          # to denote the prediction as confident and make it deterministic
                          # Very important for training procedure efficiency

entropy_reg = 0.01        # coefficient in front of the entropy regularizer term
batch_size = 10000       # number of sessions per batch
update_edges_every = 10   # call hnsw.update_edges() every N steps

n_jobs = 8                # Number of threads for C++ sampling
max_steps = 1000          # Max number of training iterations

# Recover settings
restore_step = None       # the iteration step from which you want to recover the model 

import lib
import os.path as osp

graph_params = { 
    'vertices_path': osp.join(DATA_DIR, 'deep_base.fvecs'),

    'train_queries_path': osp.join(DATA_DIR, 'deep_learn_1m.fvecs'),
    'test_queries_path': osp.join(DATA_DIR, 'deep_query.fvecs'),
    
    'train_gt_path': osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
    'test_gt_path': osp.join(DATA_DIR, 'test_gt.ivecs'),
#     ^-- comment these 2 lines to re-compute ground truth ids (if you don't have pre-computed ground truths)
    
    'train_queries_size': train_queries_size + val_queries_size,  # valid set is a train subset
    'val_queries_size': val_queries_size,
    'ground_truth_n_neighbors': ngt,  # for each query, finds this many nearest neighbors via brute force
    'graph_type': graph_type
}

if graph_type == 'nsg':
    graph_params['edges_path'] = osp.join(DATA_DIR, 'deep_R{R}_{nn}nn.nsg'.format(R=R, nn=nn))
elif graph_type == 'nsw':
    graph_params['edges_path'] = osp.join(DATA_DIR, 'deep_nsw_M{M}_efC{efC}.ivecs'.format(M=M, efC=efC))
    graph_params['initial_vertex_id'] = 0  # by default, starts search from this vertex
else:
    raise ValueError("Wrong graph type: ['nsg', 'nsw']")
    
graph = lib.Graph(**graph_params)

if graph_type == 'nsw':
    exp_name = '{data_name}_{graph_type}_k{k}_M{M}_ef{ef}_max-dcs{max_dcs}_hid-size{hidden_size}_entropy{entropy_reg}_patience{edge_patience}_seed_{seed}'.format(
        data_name=osp.split(DATA_DIR)[-1], k=k, M=M, ef=ef, hidden_size=hidden_size, 
        max_dcs=max_dcs, entropy_reg=entropy_reg, graph_type=graph_type, edge_patience=edge_patience, seed=seed
    )
elif graph_type == 'nsg':
    exp_name = '{data_name}_{graph_type}_k{k}_R{R}_ef{ef}_max-dcs{max_dcs}_hid-size{hidden_size}_entropy{entropy_reg}_patience{edge_patience}_seed_{seed}'.format(
        data_name=osp.split(DATA_DIR)[-1], k=k, R=R, ef=ef, hidden_size=hidden_size, 
        max_dcs=max_dcs, entropy_reg=entropy_reg, graph_type=graph_type, edge_patience=edge_patience, seed=seed,
    )
    
print('exp name:', exp_name)
# !rm {'./runs/' + exp_name} -rf # KEEP COMMENTED!
assert restore_step is not None or not os.path.exists('./runs/' + exp_name)

hnsw = lib.ParallelHNSW(graph, ef=ef, k=k, edge_patience=edge_patience, n_jobs=n_jobs)

if restore_step is not None:
    agent = torch.load("runs/{}/agent.{}.pth".format(exp_name, restore_step), weights_only=False)
    baseline = torch.load("runs/{}/baseline.{}.pth".format(exp_name, restore_step), weights_only=False)
    hnsw.edge_confidence = torch.load("runs/{}/edge_confidence.{}.pth".format(exp_name, restore_step), weights_only=False)
else:
    agent = lib.NodeFormerAgent(
        graph.vertices.shape[1],
        hidden_size,
        mlp_hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        nb_random_features=nb_random_features,
        use_bn=use_bn,
        use_residual=use_residual,
    )
    baseline = lib.SessionBaseline(graph.train_queries.size(0))
    
reward = lib.MaxDCSReward(k=k, max_dcs=max_dcs)
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

# generate batches of [queries, ground truth, train_query_ids (for baseline)]
train_query_ids = torch.arange(graph.train_queries.size(0))
train_batcher = lib.utils.iterate_minibatches(graph.train_queries, graph.train_gt, train_query_ids, 
                                              batch_size=batch_size)

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
        print("step=%i, mean_reward=%.3f, time=%.3f" % 
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

for heap_size in range(12, 121, 4):
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