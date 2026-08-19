"""A2: trained from kNN+5% rewiring. Did it go up from there, or down?

The policy adds a stable +0.017 over kNN s_0 (probe_a3_reversal, ex-s789, both
operating points). The question A2 answers is whether that +0.017 is available
from ANY start, or only from the bad one: if s_0 already has the long-range edges
the policy's gain came from, is there anything left for it to add?

  ends above 0.524 -> the policy contributes on top of the free gain; the two
                      are complementary and should be stacked
  ends at ~0.524    -> the policy only recovers what rewiring already supplied
  ends below        -> training actively damages a good graph, which is the
                      [[no-fixed-point-unbounded-edge-growth]] behaviour again
"""
import os.path as osp
import numpy as np
import torch
import lib
from probe_visits import NQ, NJ, K, build, visit_counts, recall_at

POINTS = [(300, 32), (3000, 128)]
SEEDS = [42, 123, 456]
STEPS = [100, 200, 300, 400, 500]
S0 = 'runs/rewired_s0/dynamic_edges.0.pth'


def score(graph, edges_path, budget, ef):
    hnsw = lib.GraphEditHNSW(graph, ef=ef, k=K, n_jobs=NJ,
                             max_trajectory=budget + 100, max_dcs=budget)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)
    return rec, float((vis > 0).mean())


def main():
    print('== A2: training from kNN + 5% random rewiring ==')
    print('NQ=%d k=%d' % (NQ, K))
    graph = build('knn')

    b = {}
    for budget, ef in POINTS:
        b[(budget, ef)] = score(graph, S0, budget, ef)
        print('rewired s_0 @ budget=%d ef=%d: recall %.4f coverage %.4f'
              % (budget, ef, b[(budget, ef)][0], b[(budget, ef)][1]))
    kb = score(graph, None, 300, 32)
    print('kNN s_0     @ budget=300 ef=32:  recall %.4f coverage %.4f' % kb)

    print('\n--- trajectory at budget=300 ef=32 (s_0 = %.4f) ---' % b[(300, 32)][0])
    print('%-8s %s' % ('seed', ' '.join('%9s' % ('step%d' % s) for s in STEPS)))
    finals = []
    for seed in SEEDS:
        row = []
        for st in STEPS:
            p = 'runs/a2_rewired_s%d/dynamic_edges.%d.pth' % (seed, st)
            row.append(score(graph, p, 300, 32)[0] if osp.exists(p) else float('nan'))
        finals.append(row[-1])
        print('%-8d %s' % (seed, ' '.join('%9.4f' % v for v in row)))

    print('\n--- final (step 500) at both operating points ---')
    print('%-8s %12s %12s %14s %14s'
          % ('seed', 'r@300/32', 'r@3000/128', 'd vs s_0 tight', 'd vs s_0 loose'))
    dt, dl = [], []
    for seed in SEEDS:
        p = 'runs/a2_rewired_s%d/dynamic_edges.500.pth' % seed
        if not osp.exists(p):
            continue
        rt = score(graph, p, 300, 32)[0]
        rl = score(graph, p, 3000, 128)[0]
        a, c = rt - b[(300, 32)][0], rl - b[(3000, 128)][0]
        dt.append(a); dl.append(c)
        print('%-8d %12.4f %12.4f %+14.4f %+14.4f' % (seed, rt, rl, a, c))

    if dt:
        n = len(dt)
        for nm, v in (('tight 300/32', np.array(dt)), ('loose 3000/128', np.array(dl))):
            print('mean d %-16s %+.4f +- %.4f (%d/%d positive)'
                  % (nm, v.mean(), v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
                     int((v > 0).sum()), n))
    print('\nReference: policy from kNN s_0 gained +0.017 (ex-s789, both points).')
    print('An equal gain here means the two are complementary. A null or negative')
    print('means the policy was only ever recovering the long-range edges that')
    print('rewiring already put in.')


if __name__ == '__main__':
    main()
