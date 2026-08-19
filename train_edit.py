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

ppo_epochs = 3            # number of gradient passes over each step's node batch
lr = 2e-4                 # Adam learning rate
clip_eps = 0.1            # PPO clipping epsilon
entropy_reg = 0.01        # coefficient in front of the entropy regularizer term

# Graph-editing MDP (framework.md). Each iteration t edits `nodes_per_step` nodes,
# swapping `n_swap` of each node's edges for 2-hop candidates, then measures the
# resulting cost change on a fixed probe set of queries.
n_swap = 2                # edges replaced per node per iteration (degree is preserved)
nodes_per_step = 4096     # nodes edited per iteration
nodes_in_batch = 512      # nodes per forward chunk (memory knob)
probe_size = 4096         # queries used to measure c(s_t); the reward's sample size
probe_refresh = True      # reuse step t's post-swap search as step t+1's pre-swap one
probe_resample_every = 0  # redraw the probe set every N steps (0 = keep it fixed).
                          # Must be 0 when probe_refresh is True, since a cached
                          # pre-swap cost is only comparable on the same queries.

n_jobs = 8                # Number of threads for C++ sampling
n_hop_virtual = 24 * 24   # cap on 2-hop candidates considered per node
max_steps = 1000          # Max number of training iterations (T in framework.md)

assert not (probe_refresh and probe_resample_every), \
    'probe_refresh reuses the previous step cost, so the probe set must stay fixed'

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

edit_tag = 'nswap{}_nodes{}'.format(n_swap, nodes_per_step)

if graph_type == 'nsw':
    exp_name = '{data_name}_{graph_type}_k{k}_M{M}_ef{ef}_max-dcs{max_dcs}_hid-size{hidden_size}_entropy{entropy_reg}_{edit_tag}_{pretrain_tag}_seed_{seed}'.format(
        data_name=osp.split(DATA_DIR)[-1], k=k, M=M, ef=ef, hidden_size=hidden_size,
        max_dcs=max_dcs, entropy_reg=entropy_reg, graph_type=graph_type, edit_tag=edit_tag,
        pretrain_tag=pretrain_tag, seed=seed
    )
elif graph_type == 'nsg':
    exp_name = '{data_name}_{graph_type}_k{k}_R{R}_ef{ef}_max-dcs{max_dcs}_hid-size{hidden_size}_entropy{entropy_reg}_{edit_tag}_{pretrain_tag}_seed_{seed}'.format(
        data_name=osp.split(DATA_DIR)[-1], k=k, R=R, ef=ef, hidden_size=hidden_size,
        max_dcs=max_dcs, entropy_reg=entropy_reg, graph_type=graph_type, edit_tag=edit_tag,
        pretrain_tag=pretrain_tag, seed=seed,
    )

print('exp name:', exp_name)
# !rm {'./runs/' + exp_name} -rf # KEEP COMMENTED!
assert restore_step is not None or not os.path.exists('./runs/' + exp_name)

hnsw = lib.GraphEditHNSW(graph, ef=ef, k=k, n_jobs=n_jobs, n_hop_virtual=n_hop_virtual)

if restore_step is not None:
    agent = torch.load("runs/{}/agent.{}.pth".format(exp_name, restore_step), weights_only=False)
    baseline = torch.load("runs/{}/baseline.{}.pth".format(exp_name, restore_step), weights_only=False)
    # The graph topology IS the state, so it must be restored alongside the agent.
    hnsw.dynamic_edges = torch.load("runs/{}/dynamic_edges.{}.pth".format(exp_name, restore_step), weights_only=False)
    hnsw.adj = hnsw.build_adjacency()
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
    # One baseline slot per graph node: in the graph-edit MDP a "session" is one
    # node's edge swap, so the EMA is indexed by node id rather than by query id.
    baseline = lib.SessionBaseline(graph.vertices.size(0))
elif agent_type == 'simple':
    agent = lib.SimpleNeuralAgent(graph.vertices.shape[1], hidden_size=hidden_size)
    # One baseline slot per graph node: in the graph-edit MDP a "session" is one
    # node's edge swap, so the EMA is indexed by node id rather than by query id.
    baseline = lib.SessionBaseline(graph.vertices.size(0))
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
    # One baseline slot per graph node: in the graph-edit MDP a "session" is one
    # node's edge swap, so the EMA is indexed by node id rather than by query id.
    baseline = lib.SessionBaseline(graph.vertices.size(0))

reward = lib.ProximityDCSReward(graph.vertices, k=k, max_dcs=max_dcs,
                                alpha=args.alpha, beta=args.beta)
trainer = lib.GraphEditPPO(agent, hnsw, reward, baseline,
                  lr=lr,
                  clip_eps=clip_eps,
                  ppo_epochs=ppo_epochs,
                  n_swap=n_swap,
                  nodes_per_step=nodes_per_step,
                  nodes_in_batch=nodes_in_batch,
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
best_val_reward = -float('inf')   # r_t is a cost *delta*, so val reward may be negative
                                  # and 0 is not a safe lower bound to compare against
# The probe set: a fixed sample of training queries whose deterministic search cost
# defines c(s_t). Unlike the old per-query MDP there is no "training batch" of
# sessions to iterate -- the graph itself is the thing being trained, and queries
# only serve to measure it. best_val_step is only ever set on a checkpoint step
# (see is_checkpoint_step below), so the final "load best" block always finds a file.
num_train_queries = graph.train_queries.size(0)


def draw_probe_set(size):
    idx = torch.randperm(num_train_queries)[:size]
    return graph.train_queries[idx], graph.train_gt[idx]


probe_queries, probe_gt = draw_probe_set(probe_size)

# generate batches of [queries, ground truth]           
val_iterator = lib.utils.iterate_minibatches(graph.val_queries, graph.val_gt, 
                                             batch_size=graph.val_queries.size(0))

dev_iterator = lib.utils.iterate_minibatches(graph.test_queries, graph.test_gt, 
                                             batch_size=graph.test_queries.size(0))

# framework.md step 2: for t = 0 .. T. Each train_step covers steps 3-7
# (features H -> rule-picking -> environment step -> reward -> RL update).
for t in range(max_steps):
    start = time.time()
    torch.cuda.empty_cache()

    if probe_resample_every and trainer.step % probe_resample_every == 0:
        probe_queries, probe_gt = draw_probe_set(probe_size)

    mean_reward = trainer.train_step(probe_queries, probe_gt, probe_refresh=probe_refresh)
    if mean_reward is not None:
        reward_history.append(mean_reward)

    # Always checkpoint the final step, so short runs (max_steps < 50) still leave
    # a loadable checkpoint for the export block below.
    is_checkpoint_step = (trainer.step % 50 == 0) or (t == max_steps - 1)

    if trainer.step % 10 == 0 or is_checkpoint_step:
        val_counters = trainer.evaluate(*next(val_iterator), prefix='val')
        val_reward = val_counters['val/mean_reward']
        if val_reward > best_val_reward and is_checkpoint_step:
            best_val_reward = val_reward
            best_val_step = trainer.step

    if is_checkpoint_step:
        _ = trainer.evaluate(*next(dev_iterator))
        print(end="Saving...")
        torch.save(agent, "runs/{}/agent.{}.pth".format(exp_name, trainer.step))
        torch.save(baseline, "runs/{}/baseline.{}.pth".format(exp_name, trainer.step))
        # Save the topology, not edge_confidence: in this MDP the graph is the state.
        torch.save(hnsw.dynamic_edges, "runs/{}/dynamic_edges.{}.pth".format(exp_name, trainer.step))
        print('Done!')
    
    if reward_history:
        clear_output(True)
        plt.title('train reward over time')
        plt.plot(moving_average(reward_history, span=50))
        plt.scatter(range(len(reward_history)), reward_history, alpha=0.1)
        plt.grid()
        plt.show()
        print("step=%i, mean_reward=%.6f, time=%.3f" %
              (trainer.step, np.mean(reward_history[-100:]), time.time()-start))
    else:
        # train_step returns None when no sampled node had a usable 2-hop candidate
        # or no probe walk touched any of them, so there was nothing to credit.
        print("step=%i, no credited nodes yet, time=%.3f" % (trainer.step, time.time()-start))

#protip: run tensorboard in ./runs to get all metrics.

print("Best step on validation: %d" % best_val_step)
agent = torch.load("runs/{}/agent.{}.pth".format(exp_name, best_val_step), weights_only=False)
# The learned artifact is the topology itself, so restore the graph that scored best
# and let it define the exported edges directly -- there is no per-edge keep/drop
# mask to recompute here (that belonged to the old per-query MDP).
hnsw.dynamic_edges = torch.load("runs/{}/dynamic_edges.{}.pth".format(exp_name, best_val_step), weights_only=False)
hnsw.adj = hnsw.build_adjacency()
trainer.step = best_val_step
trainer.agent = agent

agent.cuda()

# Save constructed graph. sorted() keeps rows in vertex-id order in the file.
new_edges = dict(sorted((v, list(nbrs)) for v, nbrs in hnsw.dynamic_edges.items()))
lib.write_edges("runs/{}/graph.{}.ivecs".format(exp_name, trainer.step), new_edges)

initial_degrees = np.array([len(graph.edges[v]) for v in sorted(graph.edges)])
final_degrees = np.array([len(new_edges[v]) for v in sorted(new_edges)])
print("degree preserved: %s (initial mean %.2f, final mean %.2f)" % (
    np.array_equal(initial_degrees, final_degrees),
    initial_degrees.mean(), final_degrees.mean()))

hnsw.max_trajectory = 300  # Set larger number of hops allowed to the search algorithm 
                           # to deal with large heap_sizes

for heap_size in range(12, 301, 4):
    # The edited graph is deterministic, so the recall/DCS operating point is a
    # property of the topology alone -- sweep ef straight through the search rather
    # than replaying sampled sessions.
    metrics = trainer.evaluate(graph.test_queries, graph.test_gt, prefix='dev',
                               write_logs=False, ef=heap_size)
    sys.stderr.flush()
    print("Ef %i | Recall@%d %.4f | Distances: %.1f" %
          (heap_size, k, metrics['dev/recall@%d' % k], metrics['dev/distance_computations']),
          flush=True,
         )