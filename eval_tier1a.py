"""Re-evaluate the Tier 1a arms under ONE harness.

The per-run numbers printed inside training are not comparable across runs: each
run evaluates on its own sampled query batch, so a lucky batch shows up as a
recall that the graph itself does not support (measured: ctrl_s789 reported
0.8096 at FEWER distances than peers whose graphs are structurally identical).
Here every saved graph is scored on the same val_queries[:NQ] with the same
ef/k/budget, so a difference between arms is a difference between graphs.

Also reports `coverage` -- the fraction of vertices the query set actually
reaches under the DCS budget. Training only logs unbounded BFS `reachable`
(~0.994 everywhere, uninformative); coverage is the quantity that ordered
recall perfectly across the Tier 0 graphs, so it is the one to watch here.
"""
import os.path as osp
import numpy as np
import torch
import lib
from probe_visits import (DATA_DIR, NQ, NJ, EF, K, DCS_BUDGET, build,
                          in_degrees, visit_counts, recall_at, norm_entropy, gini)

SEEDS = [42, 123, 456, 789]
STEP = 500
ARMS = [('ctrl', 'runs/tier1a_ctrl_s%d/dynamic_edges.%d.pth'),
        ('ideg', 'runs/tier1a_ideg_s%d/dynamic_edges.%d.pth')]


def score(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)
    ind = in_degrees(hnsw)
    vis, total_hops = visit_counts(res, hnsw.num_vertices,
                                   hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)
    seen = vis > 0
    return dict(recall=rec, se=se,
                dcs=float(res['total_distance_computations'].mean()),
                coverage=float(seen.mean()),
                pops_per_seen=float(vis[seen].mean()) if seen.any() else 0.0,
                ind_max=int(ind.max()), ind_gini=gini(ind),
                edge_len=None)


def main():
    print('== Tier 1a re-evaluation, one harness ==')
    print('NQ=%d ef=%d k=%d dcs_budget=%d step=%d' % (NQ, EF, K, DCS_BUDGET, STEP))
    print()
    graph = build('knn')          # s_0 is the same for every arm/seed
    base = score(graph, None)
    print('%-14s %8s %8s %9s %9s %8s %8s' %
          ('run', 'recall', '+-SE', 'coverage', 'pops/seen', 'ind_max', 'ind_gini'))
    print('%-14s %8.4f %8.4f %9.4f %9.2f %8d %8.4f' %
          ('kNN s_0', base['recall'], base['se'], base['coverage'],
           base['pops_per_seen'], base['ind_max'], base['ind_gini']))

    out = {}
    for arm, tmpl in ARMS:
        for seed in SEEDS:
            path = tmpl % (seed, STEP)
            if not osp.exists(path):
                print('%-14s MISSING %s' % ('%s_s%d' % (arm, seed), path))
                continue
            r = score(graph, path)
            out[(arm, seed)] = r
            print('%-14s %8.4f %8.4f %9.4f %9.2f %8d %8.4f' %
                  ('%s_s%d' % (arm, seed), r['recall'], r['se'], r['coverage'],
                   r['pops_per_seen'], r['ind_max'], r['ind_gini']))

    print()
    print('--- paired ideg - ctrl (same seed => same node sample & query batch) ---')
    print('%-8s %10s %10s %10s %11s' %
          ('seed', 'd_recall', 'd_coverage', 'd_ind_max', 'd_ind_gini'))
    d_rec, d_cov, d_max, d_gini = [], [], [], []
    for seed in SEEDS:
        a, b = out.get(('ideg', seed)), out.get(('ctrl', seed))
        if a is None or b is None:
            continue
        dr = a['recall'] - b['recall']
        dc = a['coverage'] - b['coverage']
        dm = a['ind_max'] - b['ind_max']
        dg = a['ind_gini'] - b['ind_gini']
        d_rec.append(dr); d_cov.append(dc); d_max.append(dm); d_gini.append(dg)
        print('%-8d %+10.4f %+10.4f %+10d %+11.4f' % (seed, dr, dc, dm, dg))

    if len(d_rec) > 1:
        n = len(d_rec)
        for name, v in (('recall', d_rec), ('coverage', d_cov),
                        ('ind_max', d_max), ('ind_gini', d_gini)):
            v = np.asarray(v, dtype=np.float64)
            se = v.std(ddof=1) / np.sqrt(n)
            wins = int((v > 0).sum())
            print('paired d_%-9s = %+.5f +- %.5f  (%d/%d seeds positive)'
                  % (name, v.mean(), se, wins, n))
        print()
        print('The brake is working on its own terms only if d_ind_max is clearly')
        print('negative. Whether that BUYS anything is d_recall, and Tier 0 says')
        print('d_coverage is the variable to watch alongside it.')


if __name__ == '__main__':
    main()
