"""A3: does the ranking survive a generous budget?

Every result so far is at dcs_budget=300 / ef=32, where cost is the binding
constraint and long-range edges are worth an unusual amount -- exactly the regime
that favours random rewiring. If the policy wins once the budget stops binding,
that is the regime it should be claimed in. If it loses everywhere, the
random-rewiring result is not an artefact of one operating point.

Arms: kNN s_0, the best and a typical learned graph, kNN+5% random rewiring, and
NSW s_0 as the ceiling. Swept over budget x ef.
"""
import itertools
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, K, build, visit_counts, recall_at)

BUDGETS = [300, 1000, 3000]
EFS = [32, 128]
REWIRE_DOSE = 0.05
REWIRE_SEED = 2          # the middle seed of the three, not the luckiest
STEP = 500

ARMS = [
    ('kNN s_0',        'knn', None),
    ('policy typical', 'knn', 'runs/tier1a_ctrl_s42/dynamic_edges.%d.pth' % STEP),
    ('policy best',    'knn', 'runs/tier1a_ctrl_s789/dynamic_edges.%d.pth' % STEP),
    ('kNN+5% rewire',  'knn', 'REWIRE'),
    ('NSW s_0',        'nsw', None),
]


def main():
    print('== A3: budget x ef sweep ==')
    print('NQ=%d k=%d rewire dose=%.2f (seed %d)' % (NQ, K, REWIRE_DOSE, REWIRE_SEED))
    graphs = {'knn': build('knn'), 'nsw': build('nsw')}

    print('\n%-16s %8s %6s %9s %9s %10s %9s' %
          ('arm', 'budget', 'ef', 'recall', '+-SE', 'coverage', 'DCS/q'))
    results = {}
    for budget, ef in itertools.product(BUDGETS, EFS):
        for label, gtype, path in ARMS:
            graph = graphs[gtype]
            # max_trajectory must exceed the hops the budget allows, or the walk is
            # truncated by the buffer instead of the budget and the higher budgets
            # would be silently capped.
            hnsw = lib.GraphEditHNSW(graph, ef=ef, k=K, n_jobs=NJ,
                                     max_trajectory=budget + 100, max_dcs=budget)
            if path == 'REWIRE':
                pad, n = hnsw.service_labels['pad'], hnsw.num_vertices
                base = hnsw.adj.copy()
                valid = np.argwhere(base != pad)
                rng = np.random.default_rng(REWIRE_SEED)
                kk = int(round(REWIRE_DOSE * len(valid)))
                pick = valid[rng.choice(len(valid), size=kk, replace=False)]
                base[pick[:, 0], pick[:, 1]] = rng.integers(0, n, size=kk).astype(base.dtype)
                hnsw.adj = base
            elif path is not None:
                hnsw.dynamic_edges = torch.load(path, weights_only=False)
                hnsw.adj = hnsw.build_adjacency()

            queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
            res = hnsw.search_deterministic(queries)
            vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
            rec, se = recall_at(res['best_vertex_ids'], gt, K)
            cov = float((vis > 0).mean())
            dcs = float(res['total_distance_computations'].mean())
            results[(label, budget, ef)] = rec
            print('%-16s %8d %6d %9.4f %9.4f %10.4f %9.1f'
                  % (label, budget, ef, rec, se, cov, dcs))
        print()

    print('--- policy best/typical minus kNN+5%% rewire (positive = policy wins) ---')
    print('%8s %6s %14s %14s' % ('budget', 'ef', 'best-rewire', 'typical-rewire'))
    for budget, ef in itertools.product(BUDGETS, EFS):
        r = results[('kNN+5% rewire', budget, ef)]
        print('%8d %6d %+14.4f %+14.4f'
              % (budget, ef,
                 results[('policy best', budget, ef)] - r,
                 results[('policy typical', budget, ef)] - r))
    print('\nA positive column at some operating point is the policy\'s regime.')
    print('All-negative means the budget=300 result was not an artefact.')


if __name__ == '__main__':
    main()
