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
parser.add_argument('--scratch', action='store_true', default=False,
                    help='disable the auto-load of the default pretrained checkpoint '
                         'so the agent trains from random initialisation')
# ProximityDCSReward hyperparameters.
parser.add_argument('--alpha', type=float, default=1.0,
                    help='ProximityDCSReward alpha coefficient')
parser.add_argument('--beta', type=float, default=1.0,
                    help='ProximityDCSReward beta coefficient')
parser.add_argument('--dense', type=str, default='prox', choices=['prox', 'path'],
                    help="which dense reward term to use. 'prox' scores how close the "
                         "returned ANSWERS are (approximation ratio); 'path' is the legacy "
                         "Spearman path-monotonicity term, which is measurably "
                         "ANTI-aligned with recall (corr -0.354 across queries on NSW, "
                         "-0.588 across swap batches) -- keep it only to reproduce old runs")
# Initial topology s_0. The graph-edit MDP learns the topology itself, so starting
# from a pruned HNSW/NSG graph would mix what the agent learned with what the
# construction heuristic already knew. 'random' and 'knn' build s_0 from scratch at
# a fixed out-degree instead.
parser.add_argument('--graph_type', type=str, default='random',
                    choices=['random', 'knn', 'nsw', 'nsg'],
                    help="initial graph: 'random'=uniform random edges, "
                         "'knn'=exact kNN edges, 'nsw'/'nsg'=pre-built file")
parser.add_argument('--init_degree', type=int, default=24,
                    help="out-degree of the 'random'/'knn' initial graph")
parser.add_argument('--init_seed', type=int, default=None,
                    help="seed for the 'random' initial graph (default: run seed)")
parser.add_argument('--seed', type=int, default=None,
                    help='run seed for python/numpy/torch (default: random). Set this in '
                         'A/B comparisons: it fixes the node sample and the query batch '
                         'of every step, so the two runs differ only in the thing under '
                         'test. Without it the gate is unpaired and the between-run '
                         'variance swamps the effect -- measured: the two arms started '
                         '0.0418 apart in mean reward, larger than the advantage claimed')
parser.add_argument('--run_name', type=str, default=None,
                    help='name of the runs/ subdirectory. Default builds one from the '
                         'hyperparameters, which is unambiguous but long -- set this '
                         'when you want to find the run again by name')
parser.add_argument('--max_steps', type=int, default=1000,
                    help='training iterations T (short runs are useful for diagnostics)')
parser.add_argument('--no_plot', action='store_true',
                    help='skip the inline matplotlib plot and the clear_output() call '
                         'that goes with it. Required when running in a terminal: '
                         'clear_output wipes the per-step diagnostic prints')
parser.add_argument('--commit_stride', type=int, default=10,
                    help='steps a swap stays tentative before the graph really moves. '
                         '1 = commit every step; N > 1 measures the reward of N trial '
                         'swaps from the same graph, trains on all of them, and keeps '
                         'only the last')
parser.add_argument('--logit_scale', type=float, default=20.0,
                    help="temperature on the candidate scores. The 'dot' scorer returns "
                         'a cosine in [-1, 1], which a softmax over ~240 candidates '
                         'cannot turn into a preference: at scale 1 the top candidate '
                         'gets 1.2x uniform probability. Measured on SIFT100K: '
                         '10 -> 5.1x uniform, 20 -> 16.6x, 50 -> 76x')
parser.add_argument('--learn_logit_scale', action='store_true',
                    help='let PPO tune the temperature (kept in log space) instead of '
                         'holding it fixed at --logit_scale')
parser.add_argument('--nodes_per_step', type=int, default=4096,
                    help='how many nodes get an edge swap per step')
parser.add_argument('--max_grad_norm', type=float, default=1.0,
                    help='gradient-norm clip. 0 disables it (the norm is still logged)')
parser.add_argument('--no_noop', action='store_true',
                    help='disable the per-node no-op action. Without it every sampled '
                         'node is FORCED to swap, so from a graph that is already a '
                         'local optimum the policy can only pick the least-bad '
                         'downhill move (measured on SIFT100K: 0 of 20 random swap '
                         'batches improved recall)')
parser.add_argument('--drop_mode', type=str, default='policy',
                    choices=['policy', 'argmin', 'random'],
                    help="how the dropped neighbours are chosen. 'policy' samples from "
                         "softmax(-score) and TRAINS the choice, symmetric to the "
                         "addition side. 'argmin' is the legacy deterministic argmin, "
                         "which carries no log-prob and so is never trained. 'random' is "
                         "the uniform baseline, i.e. the control for whether training the "
                         "drop side buys anything. Measured on the random s_0 (6 runs, "
                         "2 seeds x 3 drop_entropy_reg values): 'policy' and 'argmin' are "
                         "INDISTINGUISHABLE, the difference has no stable sign across "
                         "seeds. That s_0's per-batch |mean|/SE is 0.26, so it cannot "
                         "resolve an effect this size either way -- re-test on NSW once "
                         "gamma>0 + a critic exist rather than reading anything into it.")
parser.add_argument('--drop_entropy_reg', type=float, default=None,
                    help='entropy bonus on the DROP softmax, separate from the addition '
                         "side's entropy_reg (0.01). Default None follows entropy_reg, "
                         'reproducing runs before 2026-08-06 where the two shared one '
                         'coefficient. They pull against each other: the bonus pushes the '
                         'drop distribution toward uniform, and uniform IS the '
                         "drop_mode='random' baseline that 'policy' exists to beat. Set 0 "
                         'to remove the bonus from the drop side only. Ignored unless '
                         "drop_mode='policy' (the other modes have no drop distribution).")
parser.add_argument('--adv_ref', type=str, default=None,
                    choices=['batch', 'noop'],
                    help="what an advantage is measured against. 'batch' (pre-2026-08-06) "
                         'subtracts the EMA baseline then the batch mean, which forces '
                         'mean(advantage)=0 and so makes ~half the batch positive even '
                         'when EVERY sampled edit was harmful -- measured on NSW, '
                         'mean_reward is -0.001 at every step yet PPO keeps reinforcing '
                         "the least-bad half. 'noop' (the default) references the no-op's "
                         'exactly-0 reward, known by construction rather than estimated, '
                         'and skips centering, so a positive advantage means "better than '
                         'leaving this node alone": +0.0058 +- 0.0024 recall@10 over '
                         "'batch' on NSW at --accept node, 4 seeds. Defaults to 'batch' "
                         'under gamma>0, whose critic brings its own learned reference.')
parser.add_argument('--accept', type=str, default='node',
                    choices=['off', 'step', 'node'],
                    help="roll back an edit the probe measured as harmful. 'node' (the "
                         "default) keeps only the nodes whose own credited reward was "
                         'positive. Measured on NSW over 4 seeds, mean change in '
                         'recall@10 from s_0: node +0.0072, off -0.0457, i.e. a paired '
                         "+0.0529 +- 0.0070 (7.6 sigma, 4/4 seeds). 'off' (pre-2026-08-06) "
                         'commits every sampled edit unconditionally -- commit_stride does '
                         'NOT do this, its rollback ignores the reward too -- so ~134k '
                         'harmful edits land over 300 steps and the graph is reliably made '
                         "worse. 'step' is all-or-nothing on the aggregate probe delta and "
                         'is the only mode where probe cost is monotone by construction, '
                         'but from a local optimum it rejects nearly every step (measured '
                         "accept rate ~0). 'node' carries no such guarantee (a jointly "
                         'measured delta is attributed per node, so a subset\'s true '
                         'effect is not the sum of parts) and costs one extra probe search '
                         'per step when it reverts a subset.')
parser.add_argument('--gamma', type=float, default=0.0,
                    help='discount on the per-node credited reward. 0 (default) is the '
                         'historical greedy bandit: each edit is scored by its own '
                         'immediate delta only, which cannot leave a graph that is '
                         'already a local optimum of the bounded swap -- NSW is one, so '
                         'the NSW start is guaranteed to stall at gamma=0 no matter how '
                         'good the policy is. Since r_t = c(s_t)-c(s_{t+1}) telescopes, '
                         'gamma->1 makes the objective the TOTAL improvement '
                         'c(s_0)-c(s_T). Any gamma>0 switches the baseline to the critic.')
parser.add_argument('--value_coef', type=float, default=0.5,
                    help='weight of the critic MSE in the loss (ignored at gamma=0)')
parser.add_argument('--act_bias', type=float, default=2.0,
                    help='initial bias of the no-op gate. 2.0 => ~88%% of nodes act at '
                         'step 0, so the reward signal does not vanish immediately')
parser.add_argument('--init_agent_from', type=str, default=None,
                    help='path to a runs/<name>/agent.<step>.pth saved by a PREVIOUS RL '
                         'run, used as the starting policy. Distinct from '
                         '--pretrained_path, which reads mlpretrain_hard.py\'s dict '
                         'format; this loads a whole pickled agent, so it can carry a '
                         'policy trained on one s_0 onto a different one (e.g. NSW -> '
                         'kNN). Unlike restore_step it does NOT restore the graph or the '
                         'baseline: s_0 comes from --graph_type as usual, which is the '
                         'point -- only the policy transfers.')
parser.add_argument('--freeze_agent', action='store_true',
                    help='do not update the policy; only apply its edits. Use with '
                         '--init_agent_from to test whether a learned policy transfers '
                         'to a new s_0 as-is, separating transfer from re-adaptation.')
parser.add_argument('--init_graph_from', type=str, default=None,
                    help='path to a runs/<name>/dynamic_edges.<step>.pth snapshot to load '
                         'as s_0, replacing the graph --graph_type would have built. With '
                         '--max_steps 0 this evaluates a saved topology as-is, which is how '
                         'you get an ef sweep on a mid-run graph (e.g. the peak of a run '
                         'that later degraded) without re-running it.')
parser.add_argument('--long_frac', type=float, default=0.0,
                    help='B2: keep only the longest FRAC of each node\'s 2-hop '
                         'candidate pool, so the scorer cannot choose a short edge '
                         'because none is on the menu. 0 (default) disables. The '
                         'short-edge bias is architectural -- rho(score, length) is '
                         '-0.80 at RANDOM init, worse than the -0.58 after '
                         'pretraining -- so restricting the action space is the only '
                         'lever that does not require changing the scorer\'s form.')
parser.add_argument('--cand_band', type=float, nargs=2, default=None,
                    metavar=('LO', 'HI'),
                    help='D1: keep only candidates whose distance from the node falls in '
                         'the [LO, HI] percentile band of random pair distances. C2 swept '
                         'this directly and found an INTERIOR optimum near 65 (kNN edges '
                         'sit at 7.6, the policy\'s hubs at 93.2, a uniform random target '
                         'at 46.9), so "55 75" is the evidence-backed setting. Needs '
                         '--n_rand_cand: the 2-hop pool tops out at percentile 8.95 on '
                         'kNN and cannot reach the band on its own.')
parser.add_argument('--n_rand_cand', type=int, default=0,
                    help='D1: uniformly sampled vertices appended to each node\'s '
                         'candidate row before the band filter. About (HI-LO)%% of them '
                         'survive, so 256 leaves ~50 in a 20-point band.')
parser.add_argument('--max_periph_pct', type=float, default=100.0,
                    help='D1: drop candidates whose distance-to-centroid percentile '
                         'exceeds this. The policy\'s runaway hubs are centroid-distance '
                         'percentile 98.8-99.8 outliers on every seed, and C1b showed the '
                         'harm is their IDENTITY, not in-degree (relabelling them to '
                         'random nodes gains +0.030 while holding in-degree exact). '
                         '100 (default) disables.')
parser.add_argument('--len_head', action='store_true',
                    help='D2/Stage 1: learn the length percentile p_u per node instead '
                         'of fixing it at C2\'s global optimum. p_u = p0 + span*tanh(d), '
                         'zero-initialised so the starting policy IS the p=65 rule and '
                         'anything the head does is a strict improvement over it. '
                         'Requires --n_rand_cand (the band is a per-node RANK, estimated '
                         'from the row\'s own uniform sample). Stage 0 measured the '
                         'learning-free version of this as NULL, but only for linear '
                         'maps of 3 closed-form features -- this is the '
                         'function-class-free version, judged by stage1_eval.py.')
parser.add_argument('--len_p0', type=float, default=65.0,
                    help='centre of the length band; 65 is C2\'s measured optimum')
parser.add_argument('--len_span', type=float, default=25.0,
                    help='max |p_u - p0|. Bounded so the head cannot wander into p>90, '
                         'which C1b showed is where the dead-end outliers live')
parser.add_argument('--len_sigma', type=float, default=6.0,
                    help='exploration sd of the sampled p_u, in percentile points')
parser.add_argument('--len_half', type=float, default=10.0,
                    help='half-width of the candidate band around p_u')
parser.add_argument('--len_fixed', type=float, default=None,
                    help='matched CONTROL for --len_head: use the same per-node RANK '
                         'band but centred at this constant, with nothing learned. '
                         'Comparing --len_head against D1 instead would confound the '
                         'head with the change from a global-CDF band to a per-node one.')
parser.add_argument('--actor_ctx', action='store_true',
                    help="condition the pair scorer on the source node's current "
                         'neighbourhood (a zero-initialized residual head over '
                         '[z_i, z_j, ctx_i, |z_j-ctx_i|]). WITHOUT it the score of a '
                         'pair is a fixed function of the two endpoints, identical in '
                         'every graph state, so the policy cannot represent "my '
                         'neighbourhood is already too local, add a long-range link". '
                         'Off by default until the A/B is measured; the head is zero-'
                         'init so step 0 behaviour is identical to the baseline.')
parser.add_argument('--indeg_ctx', action='store_true',
                    help='give the pair scorer the in-degree of both endpoints, as a '
                         'zero-initialized residual over [log1p(indeg_i), log1p(indeg_j)]. '
                         'WITHOUT it the policy has no view of connectivity at all, so a '
                         'candidate 1000 nodes already point at looks identical to one '
                         'nobody points at; since the attractive targets are attractive to '
                         'every node, the independent per-node decisions pile onto the same '
                         'destinations (measured on SIFT100K: max in-degree 8039 vs NSW 71). '
                         'This has to be an input, not a reward penalty -- a scalar penalty '
                         'cannot tell the policy WHICH candidate is the saturated one.')
parser.add_argument('--indeg_noop', action='store_true',
                    help="give the no-op gate the mean log in-degree of the node's current "
                         'neighbours (one scalar weight, zero-init). This is the learnable '
                         'termination rule: the gate otherwise sees only z_i, so it cannot '
                         'distinguish a node whose neighbours are fresh from one whose '
                         'neighbours are already saturated -- the state in which continuing '
                         'to edit is what makes the graph worse.')
parser.add_argument('--k', type=int, default=1,
                    help='answers per query for recall@k in the reward')
parser.add_argument('--ef', type=int, default=10,
                    help='beam width during training, i.e. the operating point. Must be '
                         '>= k. Raise it to 32+ when using --dcs_budget: below that the '
                         'beam stops the walk before the budget binds (measured at '
                         'budget=300: 49%% of queries saturate at ef=10, 100%% at ef=32)')
# Note the name: this is a HARD cap inside the search kernel, distinct from the
# ProximityDCSReward `max_dcs` (=1000) which merely normalises the r_cost term.
parser.add_argument('--dcs_budget', type=int, default=0,
                    help='hard distance-computation budget per query inside the search '
                         'kernel; 0 = unlimited (old behaviour). With a budget, DCS is '
                         'a constant of the environment rather than something an edge '
                         'swap can move, so recall is comparable across graphs without '
                         'picking a recall/DCS exchange rate -- use it with --beta 0. '
                         'Without it, ef fixes the beam width but not the cost: DCS ~ '
                         'degree*hops*(1-revisit_rate), and shortening edges raises the '
                         'revisit rate. Measured: the unconstrained policy cut DCS 8% '
                         '(354->326) with hops FLAT (14.78->14.81), i.e. all of it came '
                         'from neighbourhood overlap, and recall did not improve.')
args = parser.parse_args()
if args.max_grad_norm <= 0:
    args.max_grad_norm = None   # GraphEditPPO reads None as "report but don't clip"
print(args)

DATA_DIR = './data/SIFT100K'

# if not os.path.exists(DATA_DIR):
#     assert not DATA_DIR.endswith(os.sep), 'please do not put "/" at the end of DATA_DIR'
#     !mkdir -p {DATA_DIR}
#     !wget https://www.dropbox.com/sh/vfnqvuk2jex4oqc/AADglx1ukHNaT3cNOVgAxMvfa?dl=1 -O {DATA_DIR}/deep_100k.zip
#     !cd {DATA_DIR} && unzip deep_100k.zip && rm deep_100k.zip

seed = args.seed if args.seed is not None else random.randint(0, 2**32-1)
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
print("Random seed: %d" % seed)

################
# Graph params #
################

graph_type = args.graph_type   # 'random', 'knn', 'nsw' or 'nsg'
init_degree = args.init_degree  # out-degree of the built ('random'/'knn') initial graph
M = 12                    # degree parameter for NSW (Max degree is 2*M)
ef = args.ef              # search algorithm parameter, sets the operating point
R = 24                    #  degree parameter for NSG (corresponds to max degree)
k = args.k                # Number of answers per query. Need for Recall@k

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
dropout = 0.0             # deliberately 0 for PPO, unlike pretraining. The ratio
                          # exp(logp - old_logp) is only meaningful if both sides
                          # score the pair with the SAME network; a fresh dropout
                          # mask per re-encode would make the ratio pure noise and
                          # the clipping meaningless. (Exploration comes from the
                          # Gumbel-top-k sampling, not from dropout.)

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
if pretrained_path is None and not args.scratch and agent_type == 'mlp_link' and osp.exists(default_pretrained_path):
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
drop_entropy_reg = args.drop_entropy_reg   # None => follow entropy_reg (pre-2026-08-06)

# Graph-editing MDP (framework.md). Each iteration t edits `nodes_per_step` nodes,
# swapping `n_swap` of each node's edges for 2-hop candidates, then measures the
# resulting cost change on a fixed probe set of queries.
n_swap = 2                # edges replaced per node per iteration (degree is preserved)
nodes_per_step = args.nodes_per_step  # nodes edited per iteration
nodes_in_batch = 512      # nodes per forward chunk (memory knob)
logit_scale = args.logit_scale
learn_logit_scale = args.learn_logit_scale
probe_size = 4096         # queries used to measure c(s_t); the reward's sample size
probe_refresh = True      # reuse step t's post-swap search as step t+1's pre-swap one
probe_resample_every = 0  # redraw the probe set every N steps (0 = keep it fixed).
                          # Must be 0 when probe_refresh is True, since a cached
                          # pre-swap cost is only comparable on the same queries.
commit_stride = args.commit_stride
                          # steps a swap stays tentative. 1 = apply every swap; N > 1
                          # trains on N trial swaps from the same s_t (each rolled
                          # back after its reward is measured) and only keeps the last.

n_jobs = 8                # Number of threads for C++ sampling
n_hop_virtual = None      # cap on 2-hop candidates per node (None = max_degree**2,
                          # i.e. the whole 2-hop pool)
max_steps = args.max_steps  # Max number of training iterations (T in framework.md)

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
elif graph_type in ('random', 'knn'):
    # s_0 is built from the base points, so there is no edge file to read.
    graph_params['edges_path'] = None
    graph_params['initial_vertex_id'] = 0
    graph_params['init_degree'] = init_degree
    graph_params['init_seed'] = args.init_seed if args.init_seed is not None else seed
    if graph_type == 'knn':
        # Reuse the on-disk neighbour cache across runs -- an exact kNN over the
        # whole base set is the expensive part of building this s_0.
        knn_cache_dir = osp.join(DATA_DIR, 'knn_cache')
        os.makedirs(knn_cache_dir, exist_ok=True)
        graph_params['knn_cache_path'] = osp.join(
            knn_cache_dir, 'init_knn_k{}.npz'.format(init_degree + 1))
else:
    raise ValueError("Wrong graph type: ['random', 'knn', 'nsg', 'nsw']")

graph = lib.Graph(**graph_params)

# Tag the run dir by agent_type / pretrain status so different agent choices
# and "with" vs "without" pretraining land in separate ./runs/ subfolders and
# can be compared side by side (e.g. in tensorboard) without overwriting one
# another. SimpleNeuralAgent has no pretraining, so its tag is just its name.
if agent_type == 'simple':
    pretrain_tag = 'simple'
else:
    # init_agent_from wins the label because it wins the load order below: an RL
    # checkpoint already contains the pretrained weights, so calling such a run
    # 'pretrained' would understate where its policy came from.
    if args.init_agent_from:
        pretrain_tag = 'mlp_link_rlinit'
    else:
        pretrain_tag = 'mlp_link_pretrained' if pretrained_path else 'mlp_link_scratch'

edit_tag = 'nswap{}_nodes{}_stride{}_lscale{:g}{}{}'.format(
    n_swap, nodes_per_step, commit_stride, logit_scale,
    '_learned' if learn_logit_scale else '',
    '_nonoop' if args.no_noop else '_noop{:g}'.format(args.act_bias)) \
    + ('_budget{}'.format(args.dcs_budget) if args.dcs_budget > 0 else '') \
    + ('_ctx' if args.actor_ctx else '') \
    + ('_ideg' if args.indeg_ctx else '') \
    + ('_idnoop' if args.indeg_noop else '') \
    + ('_xfer' if args.init_agent_from else '') \
    + ('_frozen' if args.freeze_agent else '')

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
else:
    # 'random' / 'knn': degree is the only topology hyperparameter of s_0.
    exp_name = '{data_name}_{graph_type}{init_degree}_k{k}_ef{ef}_max-dcs{max_dcs}_hid-size{hidden_size}_entropy{entropy_reg}_{edit_tag}_{pretrain_tag}_seed_{seed}'.format(
        data_name=osp.split(DATA_DIR)[-1], k=k, ef=ef, hidden_size=hidden_size,
        max_dcs=max_dcs, entropy_reg=entropy_reg, graph_type=graph_type,
        init_degree=init_degree, edit_tag=edit_tag, pretrain_tag=pretrain_tag, seed=seed,
    )

if args.run_name:
    exp_name = args.run_name

print('exp name:', exp_name)
print('diagnostics:  python dump_diag.py runs/%s 0 50 100 150 200' % exp_name)
# !rm {'./runs/' + exp_name} -rf # KEEP COMMENTED!
assert restore_step is not None or not os.path.exists('./runs/' + exp_name)

hnsw = lib.GraphEditHNSW(graph, ef=ef, k=k, n_jobs=n_jobs, n_hop_virtual=n_hop_virtual,
                         max_dcs=args.dcs_budget)

if args.init_graph_from:
    # Same restore path as restore_step, but decoupled from exp_name so a snapshot from
    # any run can be evaluated under any config.
    print('Loading s_0 topology from:', args.init_graph_from)
    hnsw.dynamic_edges = torch.load(args.init_graph_from, weights_only=False)
    hnsw.adj = hnsw.build_adjacency()

if restore_step is not None:
    agent = torch.load("runs/{}/agent.{}.pth".format(exp_name, restore_step), weights_only=False)
    baseline = torch.load("runs/{}/baseline.{}.pth".format(exp_name, restore_step), weights_only=False)
    # The graph topology IS the state, so it must be restored alongside the agent.
    hnsw.dynamic_edges = torch.load("runs/{}/dynamic_edges.{}.pth".format(exp_name, restore_step), weights_only=False)
    hnsw.adj = hnsw.build_adjacency()
elif args.init_agent_from:
    # A whole agent pickled by a previous RL run's torch.save(agent, ...). Loaded
    # BEFORE the pretrained_path branch because it is strictly later in the training
    # chain: that checkpoint's weights are already inside this one.
    #
    # Deliberately NOT restoring dynamic_edges or the baseline. s_0 must come from
    # --graph_type so the policy meets a genuinely new state; carrying the old graph
    # over would just resume the old run under a new name. The baseline is a per-node
    # EMA of rewards measured on the OLD topology, so on a new s_0 it is stale by
    # construction and starting it fresh is correct.
    print('Loading trained agent from:', args.init_agent_from)
    agent = torch.load(args.init_agent_from, map_location='cpu', weights_only=False)
    if not isinstance(agent, torch.nn.Module):
        raise TypeError('--init_agent_from expects a pickled agent Module, got %s. A '
                        'mlpretrain_hard.py dict checkpoint goes to --pretrained_path '
                        'instead.' % type(agent).__name__)
    # feat_mean/feat_std were computed from the SOURCE run's graph.vertices. Both runs
    # normalize the same base vectors ('global'), so they agree here -- but assert it
    # rather than trust it: a mismatch silently feeds the encoder the wrong scale, which
    # is exactly the failure that collapsed every embedding once before.
    if getattr(agent, 'feat_mean', None) is not None:
        want_mean = graph.vertices.mean(dim=0, keepdim=True)
        want_std = graph.vertices.std(dim=0, keepdim=True)
        for nm, have, want in (('feat_mean', agent.feat_mean, want_mean),
                               ('feat_std', agent.feat_std, want_std)):
            rel = ((have.cpu() - want).norm() / (want.norm() + 1e-12)).item()
            assert rel < 1e-4, (
                '%s differs from this run\'s graph by %.3e relative -- the source run '
                'used different vertex data or normalization' % (nm, rel))
        print('[feat_stats] source-run buffers match this graph (rel < 1e-4)')
    print('[init_agent_from] actor_ctx=%s indeg_ctx=%s indeg_noop=%s; '
          's_0 and baseline are NOT restored'
          % (getattr(agent, 'actor_ctx', False),
             getattr(agent, 'indeg_ctx', False),
             getattr(agent, 'indeg_noop', False)))
    if args.actor_ctx and not getattr(agent, 'actor_ctx', False):
        raise ValueError('--actor_ctx was requested but the loaded agent has no '
                         'ctx_head. Re-run the source training with --actor_ctx, or '
                         'drop the flag here.')
    # Same contract for the in-degree heads: the flag selects an architecture, and a
    # checkpoint that never had the parameters cannot silently acquire them here.
    for flag, attr, param in (('--indeg_ctx', 'indeg_ctx', 'indeg_head'),
                              ('--indeg_noop', 'indeg_noop', 'act_indeg_w')):
        if getattr(args, attr) and not getattr(agent, attr, False):
            raise ValueError('%s was requested but the loaded agent has no %s. Re-run '
                             'the source training with %s, or drop the flag here.'
                             % (flag, param, flag))
    baseline = lib.SessionBaseline(graph.vertices.size(0))
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
        #
        # The stats are recomputed from graph.vertices instead of taken from the
        # checkpoint, because the two live in DIFFERENT coordinate scales:
        # lib.pretrain_graph accepts a `normalization` argument but never
        # applies it, so ckpt['mean']/['std'] are in raw fvecs units (|mean| ~
        # 331 on SIFT100K), while lib.Graph does apply normalization='global',
        # leaving graph.vertices at unit norm. Feeding unit-norm x through
        # (x - 331) / 32.7 yields the constant -mean/std vector plus a ~0.3%
        # per-node perturbation: all 100k embeddings collapse onto one point,
        # every candidate cosine reads 1.0 to float32 precision, and the
        # candidate softmax is EXACTLY uniform -- the policy cannot express any
        # preference and its gradient vanishes (measured: grad_norm ~4e-08).
        # Recomputing here is not an approximation: global normalization is a
        # single uniform rescale x -> x/c, and (x/c - mean/c) / (std/c) equals
        # (x - mean) / std, so this reproduces pretraining's input exactly.
        feat_mean=graph.vertices.mean(dim=0, keepdim=True),
        feat_std=graph.vertices.std(dim=0, keepdim=True),
        logit_scale=logit_scale, learn_logit_scale=learn_logit_scale,
        act_bias=args.act_bias, actor_ctx=args.actor_ctx,
        indeg_ctx=args.indeg_ctx, indeg_noop=args.indeg_noop,
    )
    ck_mean = ckpt.get('mean')
    if ck_mean is not None:
        scale_ratio = (ck_mean.norm() / (graph.vertices.mean(dim=0).norm() + 1e-12)).item()
        print('[feat_stats] recomputed from graph.vertices; checkpoint stats were '
              '{:.1f}x larger in norm (raw-coordinate scale)'.format(scale_ratio))
    # The pretrained MLPLinkNet has no feat_mean/feat_std buffers, so its
    # state_dict lacks those keys. This agent DOES register them (populated
    # above from ckpt['mean']/['std'] via the constructor), so a strict load
    # would fail on exactly those two missing keys. Load non-strict, then assert
    # nothing UNexpected was in the checkpoint and nothing besides the two
    # standardization buffers was missing -- any other mismatch is a real bug.
    missing, unexpected = agent.load_state_dict(ckpt['state_dict'], strict=False)
    # log_logit_scale is this script's own temperature knob, not something
    # mlpretrain_hard.py ever trained, so it is legitimately absent.
    # act_head is the no-op gate ("edit this node or leave it alone"), which is
    # this script's own addition to the MDP; the pretraining task had no such
    # decision, so the checkpoint cannot contain it. It is initialized to
    # zero weight / act_bias bias, i.e. a constant sigmoid(act_bias) prior.
    # value_head is the critic (gamma>0), likewise absent from a checkpoint trained on
    # a pure link-prediction task with no notion of return. Its output layer is
    # zero-init, so an unloaded critic starts at V == 0 and the first advantages are
    # exactly the raw rewards.
    # ctx_head (--actor_ctx) is the topology-conditioned correction to the pair score.
    # Pretraining scored pairs in isolation with no graph at all, so it has no such
    # parameters; also zero-init at the output, so loading a checkpoint without it
    # reproduces the baseline scorer exactly.
    # indeg_head (--indeg_ctx) and act_indeg_w (--indeg_noop) are likewise graph-state
    # heads with no pretraining counterpart, and are zero-init for the same reason.
    # len_head (--len_head) is the D2 per-node length-percentile head. Same story: a
    # graph-state head with no pretraining counterpart, zero-init at the output, so a
    # checkpoint without it starts at exactly p = len_p0 for every node.
    allowed_missing = {'feat_mean', 'feat_std', 'log_logit_scale',
                       'act_head.weight', 'act_head.bias',
                       'value_head.0.weight', 'value_head.0.bias',
                       'value_head.2.weight', 'value_head.2.bias',
                       'ctx_head.0.weight', 'ctx_head.0.bias',
                       'ctx_head.2.weight', 'ctx_head.2.bias',
                       'len_head.0.weight', 'len_head.0.bias',
                       'len_head.2.weight', 'len_head.2.bias',
                       'indeg_head', 'act_indeg_w'}
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
        feat_mean=graph.vertices.mean(dim=0, keepdim=True),
        feat_std=graph.vertices.std(dim=0, keepdim=True),
        logit_scale=logit_scale, learn_logit_scale=learn_logit_scale,
        act_bias=args.act_bias, actor_ctx=args.actor_ctx,
        indeg_ctx=args.indeg_ctx, indeg_noop=args.indeg_noop,
    )
    # One baseline slot per graph node: in the graph-edit MDP a "session" is one
    # node's edge swap, so the EMA is indexed by node id rather than by query id.
    baseline = lib.SessionBaseline(graph.vertices.size(0))

reward = lib.ProximityDCSReward(graph.vertices, k=k, max_dcs=max_dcs,
                                alpha=args.alpha, beta=args.beta, dense=args.dense)
trainer = lib.GraphEditPPO(agent, hnsw, reward, baseline,
                  lr=lr,
                  clip_eps=clip_eps,
                  # 0 epochs = the PPO update loop never runs, so the policy is frozen
                  # while its edits are still applied and measured. Every aggregation in
                  # that loop is guarded by n_updates>0, so this is a clean no-op rather
                  # than a division by zero. Preferred over lr=0, which would still
                  # forward/backward and would still let Adam step on stored momentum.
                  ppo_epochs=0 if args.freeze_agent else ppo_epochs,
                  n_swap=n_swap,
                  nodes_per_step=nodes_per_step,
                  nodes_in_batch=nodes_in_batch,
                  entropy_reg=entropy_reg,
                  drop_entropy_reg=drop_entropy_reg,
                  commit_stride=commit_stride,
                  target_kl=0.015,                   # 加入早停保障
                  max_grad_norm=args.max_grad_norm,
                  use_noop=not args.no_noop,
                  drop_mode=args.drop_mode,
                  gamma=args.gamma,
                  value_coef=args.value_coef,
                  accept=args.accept,
                  adv_ref=args.adv_ref,
                  long_frac=args.long_frac,
                  cand_band=args.cand_band,
                  n_rand_cand=args.n_rand_cand,
                  max_periph_pct=args.max_periph_pct,
                  len_head=args.len_head, len_p0=args.len_p0,
                  len_span=args.len_span, len_sigma=args.len_sigma,
                  len_half=args.len_half, len_fixed=args.len_fixed,
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

# Baseline: s_0 measured before a single edit, so every later number has something
# to be compared against. If val recall never gets back above this line, the agent
# is making the graph worse no matter what the reward curve says.
init_counters = trainer.evaluate(*next(val_iterator), prefix='val')
init_stats = hnsw.graph_stats()
print('s_0 baseline: val recall@1={:.4f} reward={:.4f} dcs={:.1f} | '
      'reachable={:.4f} deg_mean={:.1f} edge_len={:.4f}'.format(
          init_counters['val/recall@1'], init_counters['val/mean_reward'],
          init_counters['val/distance_computations'],
          init_stats['reachable_frac'], init_stats['degree_mean'],
          init_stats['edge_len_mean']))
for key, value in init_stats.items():
    trainer.writer.add_scalar('graph/' + key, value, global_step=0)

# framework.md step 2: for t = 0 .. T. Each train_step covers steps 3-7
# (features H -> rule-picking -> environment step -> reward -> RL update).
for t in range(max_steps):
    start = time.time()
    torch.cuda.empty_cache()

    if probe_resample_every and trainer.step % probe_resample_every == 0:
        probe_queries, probe_gt = draw_probe_set(probe_size)
        # The cached cost was measured on the old probes, so it is no longer a
        # comparable baseline for this step's swap.
        trainer.invalidate_probe_cache()

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

        # Structural health of s_t. Degree is invariant under the bounded swap, so
        # degeneration shows up as lost reachability / shrinking edge lengths rather
        # than as missing edges. BFS over the whole graph, hence the same cadence
        # as validation rather than every step.
        stats = hnsw.graph_stats()
        for key, value in stats.items():
            trainer.writer.add_scalar('graph/' + key, value, global_step=trainer.step)
        print('  graph: reachable={reachable_frac:.4f} deg_mean={degree_mean:.1f} '
              'edge_len={edge_len_mean:.4f} | val recall@1={recall:.4f} reward={reward:.4f}'.format(
                  recall=val_counters['val/recall@1'], reward=val_reward, **stats))

        # Optimizer health. grad_norm is the number that matters: 0 means no signal
        # reaches the weights, so a flat KL is a plumbing problem rather than a
        # too-small learning rate.
        opt_diag = trainer.last_opt_diagnostics
        if opt_diag:
            print('  optim: grad_norm={grad_norm:.3e} steps {steps_taken}/{steps_skipped} '
                  '(taken/skipped) scale={scaler_scale:.0f} '
                  'dparam_rel={param_delta_rel:.3e} pi_loss={policy_loss:+.3e} '
                  'kl={kl:+.2e} H={entropy:.4f}'.format(**opt_diag))

    if is_checkpoint_step:
        _ = trainer.evaluate(*next(dev_iterator))
        print(end="Saving...")
        torch.save(agent, "runs/{}/agent.{}.pth".format(exp_name, trainer.step))
        torch.save(baseline, "runs/{}/baseline.{}.pth".format(exp_name, trainer.step))
        # Save the topology, not edge_confidence: in this MDP the graph is the state.
        torch.save(hnsw.dynamic_edges, "runs/{}/dynamic_edges.{}.pth".format(exp_name, trainer.step))
        print('Done!')
    
    if reward_history:
        if not args.no_plot:
            # NB: clear_output wipes everything printed above, including the graph/val
            # diagnostics, so --no_plot is the right choice outside a notebook.
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

if max_steps == 0:
    # Nothing trained, so no checkpoint exists to restore -- the graph currently loaded
    # (from --init_graph_from, or s_0 from --graph_type) is exactly what should be swept.
    print('[eval] max_steps=0: sweeping the loaded topology without restoring a checkpoint')
else:
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

# The sweep's whole purpose is to trace the recall-DCS frontier, so the training
# budget has to come off here: with it on, every ef >= budget/degree reports the same
# clamped DCS and the frontier collapses to a single point. Training used the cap to
# make DCS a constant; evaluation needs it variable again.
if args.dcs_budget > 0:
    print('[eval] lifting the training DCS budget (%d) for the ef sweep'
          % args.dcs_budget)
    hnsw.max_dcs = 0

for heap_size in range(12, 160, 4):
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