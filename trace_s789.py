"""Did ctrl_s789's hub CAUSE its coverage, or merely accompany it?

ctrl_s789 ends with ind_max=5225 (others 174-211), the highest recall (0.4997
vs ~0.32) and the highest coverage (0.4362 vs ~0.32). Two readings:

  (a) the hub is a routing shortcut -- one very-high-degree vertex reached early
      hands the walk access to many regions, raising coverage per unit budget
      (consistent with its low pops/seen of 4.29 vs ~5.7), or
  (b) the hub is incidental and something else about that seed's trajectory
      produced the coverage.

Reading (a) predicts ind_max and coverage rise TOGETHER over training. Reading
(b) predicts coverage rises before, or independently of, the hub. Tracing both
against a seed that stayed in the ordinary regime (s42) separates them.
"""
import os.path as osp
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at, gini)

STEPS = [50, 100, 200, 300, 400, 500]
RUNS = [('ctrl_s789', 'runs/tier1a_ctrl_s789/dynamic_edges.%d.pth'),
        ('ctrl_s42',  'runs/tier1a_ctrl_s42/dynamic_edges.%d.pth'),
        ('ideg_s789', 'runs/tier1a_ideg_s789/dynamic_edges.%d.pth')]


def score(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)
    ind = in_degrees(hnsw)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, _ = recall_at(res['best_vertex_ids'], gt, K)
    seen = vis > 0
    # Is the hub actually ON the search path? A shortcut that is never popped
    # cannot be raising coverage, which decides between the two readings above.
    hub = int(ind.argmax())
    return dict(recall=rec, coverage=float(seen.mean()),
                pops_per_seen=float(vis[seen].mean()) if seen.any() else 0.0,
                ind_max=int(ind.max()), ind_gini=gini(ind),
                hub_visits=int(vis[hub]),
                hub_visit_rank=int((vis > vis[hub]).sum()))


def main():
    graph = build('knn')
    print('== trajectory: does the hub grow WITH coverage? ==')
    print('NQ=%d ef=%d k=%d budget=%d' % (NQ, EF, K, DCS_BUDGET))
    b = score(graph, None)
    print('\n%-12s %6s %8s %9s %10s %8s %10s %9s' %
          ('run', 'step', 'recall', 'coverage', 'pops/seen', 'ind_max',
           'hub_visits', 'hub_rank'))
    print('%-12s %6s %8.4f %9.4f %10.2f %8d %10d %9d' %
          ('kNN s_0', '0', b['recall'], b['coverage'], b['pops_per_seen'],
           b['ind_max'], b['hub_visits'], b['hub_visit_rank']))
    for name, tmpl in RUNS:
        for st in STEPS:
            p = tmpl % st
            if not osp.exists(p):
                continue
            r = score(graph, p)
            print('%-12s %6d %8.4f %9.4f %10.2f %8d %10d %9d' %
                  (name, st, r['recall'], r['coverage'], r['pops_per_seen'],
                   r['ind_max'], r['hub_visits'], r['hub_visit_rank']))
    print()
    print('hub_visits = times the highest-in-degree vertex was popped (of %d queries).' % NQ)
    print('hub_rank   = how many vertices were popped MORE often than it.')
    print('If the hub is a routing shortcut it should be popped by a large share')
    print('of queries, i.e. hub_rank near 0. If hub_rank is large the hub is')
    print('mostly bypassed and cannot be what raised coverage.')


if __name__ == '__main__':
    main()
