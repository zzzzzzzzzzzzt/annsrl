"""Is ctrl_s789's hub CAUSAL for its coverage, or just correlated?

s789 ends at recall 0.4997 / coverage 0.4362 with ind_max=5225, while every
other arm sits at ~0.32 / ~0.32 with ind_max ~200. The trajectory shows the hub
growing, becoming heavily popped (visit rank 191 -> 15), and coverage rising
together. That is consistent with the hub acting as a routing shortcut, but
correlation over three checkpoints is not causation.

Test: redirect every edge that points AT the hub to a uniformly random other
vertex. Out-degree is preserved exactly for every node, so this removes "points
at the hub" WITHOUT removing "has 24 edges" -- the confound that plain edge
deletion would introduce. Controls:

  - rewire the hub          -> if coverage collapses to ~0.33, the hub is causal
  - rewire an equal number of RANDOM edges -> how much damage does rewiring that
    many edges do on its own? Without this the hub result is unfalsifiable.
"""
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

PATH = 'runs/tier1a_ctrl_s789/dynamic_edges.500.pth'
SEED = 1234


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
    ind = in_degrees(hnsw)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)
    seen = vis > 0
    return dict(recall=rec, se=se, coverage=float(seen.mean()),
                pops=float(vis[seen].mean()) if seen.any() else 0.0,
                ind_max=int(ind.max()))


def rewire(adj, pad, targets, rng, n_vert):
    """Point every in-edge of `targets` somewhere else, keeping out-degree fixed."""
    adj = adj.copy()
    mask = np.isin(adj, np.asarray(list(targets), dtype=adj.dtype)) & (adj != pad)
    idx = np.argwhere(mask)
    repl = rng.integers(0, n_vert, size=len(idx)).astype(adj.dtype)
    adj[idx[:, 0], idx[:, 1]] = repl
    return adj, len(idx)


def rewire_random(adj, pad, n_edges, rng, n_vert):
    """Same number of edges moved, chosen uniformly -- the dose-matched control."""
    adj = adj.copy()
    valid = np.argwhere(adj != pad)
    pick = valid[rng.choice(len(valid), size=min(n_edges, len(valid)), replace=False)]
    adj[pick[:, 0], pick[:, 1]] = rng.integers(0, n_vert, size=len(pick)).astype(adj.dtype)
    return adj, len(pick)


def main():
    rng = np.random.default_rng(SEED)
    graph = build('knn')
    hnsw = load(graph, PATH)
    pad = hnsw.service_labels['pad']
    n = hnsw.num_vertices
    ind = in_degrees(hnsw)
    order = np.argsort(ind)[::-1]

    print('== is the hub causal? (ctrl_s789 step 500) ==')
    print('top-10 in-degrees:', ind[order[:10]].tolist())
    base_adj = hnsw.adj.copy()

    print('\n%-26s %8s %8s %9s %8s %8s %8s' %
          ('variant', 'recall', '+-SE', 'coverage', 'pops', 'ind_max', 'moved'))
    r = measure(hnsw, graph)
    print('%-26s %8.4f %8.4f %9.4f %8.2f %8d %8s' %
          ('s789 intact', r['recall'], r['se'], r['coverage'], r['pops'],
           r['ind_max'], '-'))

    for label, tgt in (('rewire top-1 hub', order[:1]), ('rewire top-10 hubs', order[:10])):
        adj2, moved = rewire(base_adj, pad, set(tgt.tolist()), rng, n)
        hnsw.adj = adj2
        r = measure(hnsw, graph)
        print('%-26s %8.4f %8.4f %9.4f %8.2f %8d %8d' %
              (label, r['recall'], r['se'], r['coverage'], r['pops'],
               r['ind_max'], moved))
        # Dose-matched control: same number of edges randomised, no hub targeting.
        adj3, moved3 = rewire_random(base_adj, pad, moved, rng, n)
        hnsw.adj = adj3
        r3 = measure(hnsw, graph)
        print('%-26s %8.4f %8.4f %9.4f %8.2f %8d %8d' %
              ('  ctrl: %d random edges' % moved3, r3['recall'], r3['se'],
               r3['coverage'], r3['pops'], r3['ind_max'], moved3))

    hnsw.adj = base_adj
    print('\nIf targeting the hub costs much more coverage than moving the same')
    print('number of random edges, the hub is carrying the search. If the two are')
    print('comparable, s789 gained for some other reason and the hub is incidental.')


if __name__ == '__main__':
    main()
