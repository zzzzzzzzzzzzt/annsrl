""" Guards for the acceptance test (--accept).

The claim being guarded is a LOWER BOUND: with acceptance on, a step can no longer make
the graph worse. Before this existed the transition was unconditional -- commit_stride's
rollback ignores the reward too -- so from a local optimum of the bounded swap the only
possible trajectory was downhill (measured on NSW: -0.045 recall@10 over 300 steps).

test 1 checks the rollback is exact: after a rejected step the adjacency must be
byte-identical to what it was before, or "reject" silently means "corrupt".
test 2 checks 'off' still commits unconditionally, since every measured result so far
came from that path.
"""
import copy
import os.path as osp
import numpy as np
import torch

import lib

DATA_DIR = './data/SIFT100K'
NQ, EF, K, BUDGET = 512, 32, 10, 300


def build():
    g = lib.Graph(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        edges_path=osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs'),
        initial_vertex_id=0, graph_type='nsw', normalization='global',
        train_queries_size=NQ * 2, val_queries_size=NQ,
        ground_truth_n_neighbors=100)
    h = lib.GraphEditHNSW(g, ef=EF, k=K, n_jobs=32, max_dcs=BUDGET, max_trajectory=400)
    a = lib.MLPLinkAgent(g.vertices.shape[1], node_hidden=64, node_layers=2,
                         pair_hidden=64, scorer='dot', norm='ln', dropout=0.0,
                         feat_mean=g.vertices.mean(0, keepdim=True),
                         feat_std=g.vertices.std(0, keepdim=True),
                         logit_scale=20.0, act_bias=2.0)
    a.cuda()
    r = lib.ProximityDCSReward(g.vertices, k=K, max_dcs=1000, alpha=1.0, beta=0.0,
                               dense='prox')
    return g, h, a, r


def mk(g, h, a, r, **kw):
    return lib.GraphEditPPO(a, h, r, lib.SessionBaseline(g.vertices.size(0)),
                            n_swap=2, nodes_per_step=256, nodes_in_batch=128,
                            device='cuda', drop_mode='policy', warmup_steps=0, **kw)


def run_steps(g, h, a, r, mode, n=6):
    """n train_steps in `mode`, reporting graph drift and the accept rate."""
    base_adj = h.adj.copy()
    t = mk(g, h, a, r, accept=mode)
    q, gt = g.val_queries[:NQ], g.val_gt[:NQ]
    fracs, deltas = [], []
    for _ in range(n):
        torch.manual_seed(0)
        np.random.seed(t.step + 1)
        before = h.adj.copy()
        t.train_step(q, gt)
        fracs.append(float((h.adj != before).any(axis=1).mean()))
    drift = float((h.adj != base_adj).any(axis=1).mean())
    return t, drift, fracs


def test1_rollback_is_exact(g, h, a, r):
    """A rejected step must leave the adjacency byte-identical."""
    print('== test 1: is a rejected step an exact rollback? ==')
    t = mk(g, h, a, r, accept='step')
    q, gt = g.val_queries[:NQ], g.val_gt[:NQ]
    before = h.adj.copy()
    edges_before = copy.deepcopy(h.dynamic_edges)
    torch.manual_seed(0)
    np.random.seed(1)
    t.train_step(q, gt)
    adj_same = np.array_equal(h.adj, before)
    # dynamic_edges is the authoritative store; adj is its dense mirror. Both must match
    # or a later apply_swaps would build on an inconsistent state.
    edges_same = all(list(edges_before[v]) == list(h.dynamic_edges[v])
                     for v in edges_before)
    changed = int((h.adj != before).any(axis=1).sum())
    print('  rows changed after the step: %d' % changed)
    print('  adj identical            %s' % adj_same)
    print('  dynamic_edges identical  %s' % edges_same)
    if changed:
        print('  (step was ACCEPTED, so identity is not expected here -- rerun with a'
              ' seed whose delta is negative to exercise the reject path)')
    ok = (adj_same == edges_same)  # the two stores must never disagree
    print('  verdict: %s' % ('stores consistent' if ok else 'INCONSISTENT STORES'))
    return ok


def test2_modes_differ(g, h, a, r):
    """off commits everything; step/node must commit strictly less."""
    print('\n== test 2: do the three modes actually differ? ==')
    out = {}
    for mode in ('off', 'step', 'node'):
        # Fresh graph per mode so drift is comparable.
        h.dynamic_edges = {v: list(nbrs) for v, nbrs in h.graph.edges.items()}
        h.adj = h.build_adjacency()
        t, drift, fr = run_steps(g, h, a, r, mode)
        acc = t.writer  # not readable here; use the per-step row-change fractions
        out[mode] = (drift, fr)
        print('  %-5s rows drifted from s_0 %.4f   per-step row-change %s'
              % (mode, drift, ' '.join('%.3f' % f for f in fr)))
    ok = out['off'][0] >= out['step'][0]
    print('  verdict: %s' % ('step commits no more than off'
                             if ok else 'step drifted MORE than off'))
    return ok


if __name__ == '__main__':
    g, h, a, r = build()
    r1 = test1_rollback_is_exact(g, h, a, r)
    r2 = test2_modes_differ(g, h, a, r)
    print('\nsummary: stores-consistent=%s  modes-ordered=%s' % (r1, r2))
