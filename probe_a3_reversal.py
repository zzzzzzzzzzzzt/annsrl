"""Does the policy's small gain REVERSE at a generous budget?

A3 showed, for one run (ctrl_s42): at budget=300/ef=32 it is +0.023 over kNN s_0,
but at budget=3000/ef=128 it is -0.014, i.e. below its own start. If that holds
across all 8 learned graphs it is a qualitative change in the finding -- the
policy is not "helping a little", it is trading generous-budget recall for
tight-budget recall, and the tight budget is the only place its gain exists.

Evaluates every Tier 1a arm at both operating points against the same s_0.
"""
import numpy as np
import torch
import lib
from probe_visits import NQ, NJ, K, build, visit_counts, recall_at

POINTS = [(300, 32), (3000, 128)]
STEP = 500
RUNS = [('ctrl_s42', 'ctrl', 42), ('ctrl_s123', 'ctrl', 123),
        ('ctrl_s456', 'ctrl', 456), ('ctrl_s789', 'ctrl', 789),
        ('ideg_s42', 'ideg', 42), ('ideg_s123', 'ideg', 123),
        ('ideg_s456', 'ideg', 456), ('ideg_s789', 'ideg', 789)]


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
    return rec, se, float((vis > 0).mean())


def main():
    print('== does the policy gain reverse at generous budget? ==')
    print('NQ=%d k=%d step=%d' % (NQ, K, STEP))
    graph = build('knn')

    base = {}
    for budget, ef in POINTS:
        base[(budget, ef)] = score(graph, None, budget, ef)
        print('kNN s_0 @ budget=%d ef=%d: recall %.4f coverage %.4f'
              % (budget, ef, base[(budget, ef)][0], base[(budget, ef)][2]))

    print('\n%-12s %12s %12s %14s %14s' %
          ('run', 'r@300/32', 'r@3000/128', 'd vs s_0 tight', 'd vs s_0 loose'))
    d_tight, d_loose = [], []
    for name, arm, seed in RUNS:
        p = 'runs/tier1a_%s_s%d/dynamic_edges.%d.pth' % (arm, seed, STEP)
        rt = score(graph, p, 300, 32)[0]
        rl = score(graph, p, 3000, 128)[0]
        dt = rt - base[(300, 32)][0]
        dl = rl - base[(3000, 128)][0]
        d_tight.append(dt)
        d_loose.append(dl)
        print('%-12s %12.4f %12.4f %+14.4f %+14.4f' % (name, rt, rl, dt, dl))

    dt, dl = np.array(d_tight), np.array(d_loose)
    n = len(dt)
    print('\n%-22s %+.4f +- %.4f  (%d/%d positive)'
          % ('mean d @300/32', dt.mean(), dt.std(ddof=1) / np.sqrt(n), int((dt > 0).sum()), n))
    print('%-22s %+.4f +- %.4f  (%d/%d positive)'
          % ('mean d @3000/128', dl.mean(), dl.std(ddof=1) / np.sqrt(n), int((dl > 0).sum()), n))
    print('\nIf tight is positive and loose is negative, the policy is not adding')
    print('graph quality -- it is specialising to the tight operating point, and')
    print('the gain does not survive a change of budget.')


if __name__ == '__main__':
    main()
