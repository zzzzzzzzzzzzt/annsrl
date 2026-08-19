"""How many of rewiring's long edges survive the policy?

B0 showed the a2 runs remove edges at length-percentile 13.4 while adding at 1.6,
from a start whose own edges average 2.8. That is preferential removal of the long
links. This turns it into one number: of the 120k edges random rewiring introduced,
what fraction is still present at step 500?

The control matters as much as the number: the policy removes ~80k edges per
20k-node sample regardless, so a low survival rate proves nothing unless the
ORIGINAL kNN edges survive at a higher rate. Equal rates = indiscriminate
removal. Lower rate for long edges = targeting.
"""
import os.path as osp
import numpy as np
import torch
import lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build

STEP = 500
REWIRED = 'runs/rewired_s0/dynamic_edges.0.pth'
A2 = [('a2_s42', 42), ('a2_s123', 123), ('a2_s456', 456)]


def adj_of(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    return hnsw.adj.copy(), hnsw.service_labels['pad']


def rowsets(adj, pad, n):
    return [set(adj[i][adj[i] != pad].tolist()) for i in range(n)]


def main():
    print('== survival of rewiring\'s long edges under the policy ==')
    graph = build('knn')
    base, pad = adj_of(graph, None)
    rew, _ = adj_of(graph, REWIRED)
    n = base.shape[0]

    bs = rowsets(base, pad, n)
    rs = rowsets(rew, pad, n)
    # Introduced = in rewired, not in kNN (the long random links).
    # Retained   = in both (the surviving original kNN edges) -- the control set.
    intro = [rs[i] - bs[i] for i in range(n)]
    orig = [rs[i] & bs[i] for i in range(n)]
    n_intro = sum(len(s) for s in intro)
    n_orig = sum(len(s) for s in orig)
    print('rewired s_0: %d introduced (long) edges, %d surviving original kNN edges'
          % (n_intro, n_orig))

    print('\n%-10s %14s %14s %12s' %
          ('run', 'long survived', 'kNN survived', 'ratio'))
    for name, seed in A2:
        p = 'runs/a2_rewired_s%d/dynamic_edges.%d.pth' % (seed, STEP)
        if not osp.exists(p):
            print('%-10s MISSING' % name)
            continue
        fin, _ = adj_of(graph, p)
        fs = rowsets(fin, pad, n)
        li = sum(len(intro[i] & fs[i]) for i in range(n))
        oi = sum(len(orig[i] & fs[i]) for i in range(n))
        lr, orr = li / max(1, n_intro), oi / max(1, n_orig)
        print('%-10s %13.1f%% %13.1f%% %12.2fx'
              % (name, 100 * lr, 100 * orr, lr / orr if orr else float('nan')))

    print('\nratio < 1 => long edges are removed preferentially, i.e. the policy')
    print('targets exactly what makes the rewired graph good. ratio ~1 => removal')
    print('is indiscriminate and the length story is wrong.')


if __name__ == '__main__':
    main()
