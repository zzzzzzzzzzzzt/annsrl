"""Seed-robustness for the random-rewiring baseline.

probe_random_rewire found kNN + 5% uniform rewiring = 0.5750 against 0.308-0.322
for the learned arms. The effect is ~75 SE so seed noise cannot plausibly erase
it, but the claim redirects the whole plan, so it gets checked rather than
assumed. Also reports the learned arms' own spread for a like-for-like read.
"""
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

DOSES = [0.02, 0.05, 0.08]
SEEDS = [1, 2, 3]


def measure(hnsw, graph):
    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)
    return rec, se, float((vis > 0).mean())


def main():
    print('== random-rewiring baseline: seed robustness ==')
    print('NQ=%d ef=%d k=%d budget=%d' % (NQ, EF, K, DCS_BUDGET))
    graph = build('knn')
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    pad = hnsw.service_labels['pad']
    base = hnsw.adj.copy()
    valid = np.argwhere(base != pad)
    n = hnsw.num_vertices

    print('\n%8s %8s %8s %9s' % ('dose', 'seed', 'recall', 'coverage'))
    summary = {}
    for d in DOSES:
        recs = []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            adj = base.copy()
            k = int(round(d * len(valid)))
            pick = valid[rng.choice(len(valid), size=k, replace=False)]
            adj[pick[:, 0], pick[:, 1]] = rng.integers(0, n, size=k).astype(adj.dtype)
            hnsw.adj = adj
            rec, se, cov = measure(hnsw, graph)
            recs.append(rec)
            print('%8.3f %8d %8.4f %9.4f' % (d, s, rec, cov))
        summary[d] = (float(np.mean(recs)), float(np.std(recs, ddof=1)))
    hnsw.adj = base

    print('\n%8s %10s %10s' % ('dose', 'mean', 'sd(seeds)'))
    for d, (m, sd) in summary.items():
        print('%8.3f %10.4f %10.4f' % (d, m, sd))

    learned = [0.3218, 0.3222, 0.3079, 0.4997, 0.3200, 0.3115, 0.3090, 0.3189]
    print('\nlearned arms (8 runs): mean %.4f  sd %.4f  max %.4f'
          % (np.mean(learned), np.std(learned, ddof=1), max(learned)))
    print('kNN s_0 0.2985   NSW s_0 0.6756')


if __name__ == '__main__':
    main()
