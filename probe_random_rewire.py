"""Control that should have existed before any of the RL runs.

ablate_hub found that randomising 18327 arbitrary edges of ctrl_s789's graph
RAISED recall 0.4997 -> 0.5221. Randomising a kNN edge replaces a short local
link with a long one, which is the small-world recipe NSW gets its navigability
from -- so the obvious question is how much of the learned policy's gain is just
"some long edges got added", reachable by a rule with no learning in it at all.

Dose sweep of uniform random rewiring on:
  - kNN s_0        (what the policy starts from)
  - ctrl_s789      (the best learned graph, 0.4997)
against the learned arms as reference. If kNN s_0 + random rewiring reaches
~0.50 at some dose, the policy's advantage over its own start is not evidence
that it learned anything useful about WHICH edges to add.
"""
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

SEED = 1234
# Fractions of the ~2.4M directed edges to randomise.
DOSES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
LEARNED = 'runs/tier1a_ctrl_s789/dynamic_edges.500.pth'


def load(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    return hnsw


def measure(hnsw, graph):
    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)
    seen = vis > 0
    ind = in_degrees(hnsw)
    return dict(recall=rec, se=se, coverage=float(seen.mean()),
                pops=float(vis[seen].mean()) if seen.any() else 0.0,
                ind_max=int(ind.max()))


def sweep(name, graph, edges_path):
    hnsw = load(graph, edges_path)
    pad = hnsw.service_labels['pad']
    n = hnsw.num_vertices
    base = hnsw.adj.copy()
    valid = np.argwhere(base != pad)
    rng = np.random.default_rng(SEED)
    print('\n--- %s (%d directed edges) ---' % (name, len(valid)))
    print('%8s %8s %8s %8s %9s %8s %8s' %
          ('dose', 'nedges', 'recall', '+-SE', 'coverage', 'pops', 'ind_max'))
    for d in DOSES:
        adj = base.copy()
        k = int(round(d * len(valid)))
        if k:
            pick = valid[rng.choice(len(valid), size=k, replace=False)]
            adj[pick[:, 0], pick[:, 1]] = rng.integers(0, n, size=k).astype(adj.dtype)
        hnsw.adj = adj
        r = measure(hnsw, graph)
        print('%8.3f %8d %8.4f %8.4f %9.4f %8.2f %8d' %
              (d, k, r['recall'], r['se'], r['coverage'], r['pops'], r['ind_max']))


def main():
    print('== how much of the gain is just "add some long edges"? ==')
    print('NQ=%d ef=%d k=%d budget=%d' % (NQ, EF, K, DCS_BUDGET))
    graph = build('knn')
    sweep('kNN s_0 + random rewiring', graph, None)
    sweep('ctrl_s789 step500 + random rewiring', graph, LEARNED)
    print('\nReference: NSW s_0 = 0.6756, kNN s_0 = 0.2985,')
    print('learned arms = 0.308-0.322 except ctrl_s789 = 0.4997.')
    print('Any dose of a LEARNING-FREE rule that matches the learned arms is a')
    print('baseline those arms have to beat before their gain means anything.')


if __name__ == '__main__':
    main()
