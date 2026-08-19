"""Does (policy + hub ablation) beat random rewiring at the SAME TOTAL edit budget?

ablate_hub_long.py showed hub destruction gains +0.029 +- 0.008 over a length-matched
control on all three B2 seeds. But its controls are matched to the ABLATION step only
(~40k edges). The pipeline as a whole spends the policy's edits too (232k-315k), and
the question that decides whether any of this is a result is whether the whole pipeline
beats one line of numpy given the same number of edge moves from s_0.

So: rebuild the arm, ablate its top-10 hubs, count every slot that differs from s_0,
and put a random control at exactly that count. Two extra reference rows:

  s_0 + ablation-only   -- kNN has no hubs to destroy, so this should be a no-op; it
                           is here to confirm the gain is not just "moving 40k edges".
  arm, no ablation      -- the previously reported number, re-scored here so the two
                           are on one harness.
"""
import argparse
import numpy as np
import torch
import lib
import lib_control
from probe_visits import NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees, recall_at

SEED = 1234
TOPN = 10


def make(graph, edges=None):
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                          max_dcs=DCS_BUDGET)
    if edges is not None:
        h.dynamic_edges = edges
        h.adj = h.build_adjacency()
    return h


def score_adj(graph, adj, pad):
    h = make(graph)
    h.adj = adj
    res = h.search_deterministic(graph.val_queries[:NQ])
    rec, se = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
    return rec, se, int(in_degrees(h).max())


def ablate(adj, pad, ind, rng, topn=TOPN):
    """Redirect every in-edge of the top-N in-degree vertices to random targets."""
    tgt = np.argsort(ind)[::-1][:topn].astype(adj.dtype)
    idx = np.argwhere(np.isin(adj, tgt) & (adj != pad))
    out = adj.copy()
    out[idx[:, 0], idx[:, 1]] = rng.integers(0, adj.shape[0], size=len(idx)).astype(adj.dtype)
    return out, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--graph_type', default='knn')
    args = ap.parse_args()

    graph = build(args.graph_type)
    h0 = make(graph)
    base_adj, pad = h0.adj.copy(), h0.service_labels['pad']
    r0 = score_adj(graph, base_adj.copy(), pad)
    print('== policy + hub ablation vs count-matched random, NQ=%d dcs=%d ==' % (NQ, DCS_BUDGET))
    print('%-24s %8s %8s %9s %9s %11s' %
          ('arm', 'recall', '+-SE', 'ind_max', 'n_moved', 'd(vs ctrl)'))
    print('%-24s %8.4f %8.4f %9d %9s %11s'
          % ('s_0 (%s)' % args.graph_type, r0[0], r0[1], r0[2], '-', '-'))

    # s_0 + ablation: kNN's max in-degree is small, so this moves few edges and should
    # do nothing. If it gained, the effect would be "random edges", not "hub removal".
    rng = np.random.default_rng(SEED)
    a0, k0 = ablate(base_adj, pad, in_degrees(h0), rng)
    r = score_adj(graph, a0, pad)
    print('%-24s %8.4f %8.4f %9d %9d %11s'
          % ('  s_0 + ablate', r[0], r[1], r[2], k0, '-'))

    deltas = []
    for p in args.paths:
        edges = torch.load(p, weights_only=False)
        hp = make(graph, edges)
        arm_adj, ind = hp.adj.copy(), in_degrees(hp)
        name = p.split('/')[-2]

        ra = score_adj(graph, arm_adj.copy(), pad)
        moved_arm = lib_control.changed_slots(base_adj, arm_adj, pad)
        ce, ck = lib_control.rewire(base_adj, pad, n_edges=moved_arm, seed=2)
        rc = score_adj(graph, make(graph, ce).adj, pad)
        print('%-24s %8.4f %8.4f %9d %9d %+11.4f'
              % (name, ra[0], ra[1], ra[2], moved_arm, ra[0] - rc[0]))

        abl_adj, _ = ablate(arm_adj, pad, ind, np.random.default_rng(SEED))
        rb = score_adj(graph, abl_adj.copy(), pad)
        moved_tot = lib_control.changed_slots(base_adj, abl_adj, pad)
        ce2, _ = lib_control.rewire(base_adj, pad, n_edges=moved_tot, seed=2)
        rc2 = score_adj(graph, make(graph, ce2).adj, pad)
        d = rb[0] - rc2[0]
        deltas.append(d)
        print('%-24s %8.4f %8.4f %9d %9d %+11.4f'
              % ('  + ablate top-%d' % TOPN, rb[0], rb[1], rb[2], moved_tot, d))
        print('%-24s %8.4f %8.4f %9s %9d %11s'
              % ('    ctrl @ that count', rc2[0], rc2[1], '-', moved_tot, '-'))

    v = np.asarray(deltas, dtype=np.float64)
    if len(v) > 1:
        print('\nablated arm vs count-matched random: %+.4f +- %.4f  (%d/%d positive)'
              % (v.mean(), v.std(ddof=1) / np.sqrt(len(v)), int((v > 0).sum()), len(v)))


if __name__ == '__main__':
    main()
