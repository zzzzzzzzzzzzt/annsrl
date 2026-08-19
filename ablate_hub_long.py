"""Are the hubs causal on the LONG-EDGE graphs that B2 produced?

Hubs were shown non-causal once already (ctrl_s789: ind_max 5225 -> 550 left recall
unchanged), but that was measured on a SHORT-edge graph. B2's frac=0.01 runs reach
ind_max 5456-11616 while adding edges at length percentile ~66, which is a different
structure, so the old null does not transfer for free.

The old ablation redirected hub in-edges to UNIFORMLY RANDOM targets. That test is
weak here: on a graph whose own added edges already sit near percentile 66, a random
target is roughly an in-distribution edge, so the intervention barely perturbs
anything and a null would mean nothing.

Two controls are therefore run at every dose, both moving exactly as many edges:

  hub     -- redirect in-edges OF THE TOP-N HUBS to random targets
  random  -- redirect the same COUNT of randomly chosen edges to random targets
  length  -- redirect the same count of edges chosen to MATCH THE HUB EDGES' LENGTH
             distribution, to random targets

`random` answers "does moving this many edges hurt at all". `length` answers "is it
the hubs specifically, or just that hub in-edges happen to be long?" -- the confound
the old script could not see because on a short-edge graph there was no length
variation to confound with.

Out-degree is preserved exactly everywhere (slots are overwritten, never deleted), so
no variant can win or lose through degree.
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

SEED = 1234
DOSES = [1, 10, 100]


def load(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    return hnsw


def measure(hnsw, graph):
    res = hnsw.search_deterministic(graph.val_queries[:NQ])
    ind = in_degrees(hnsw)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
    seen = vis > 0
    return dict(recall=rec, se=se, coverage=float(seen.mean()),
                pops=float(vis[seen].mean()) if seen.any() else 0.0,
                ind_max=int(ind.max()))


def edge_lengths(V, idx, adj):
    """L2 length of each edge given as (row, col) index pairs into adj."""
    d = V[idx[:, 0]].astype(np.float64) - V[adj[idx[:, 0], idx[:, 1]]].astype(np.float64)
    return np.sqrt((d * d).sum(1))


def apply_rewire(adj, idx, rng, n_vert):
    """Point the given edge slots at uniformly random vertices, keeping out-degree."""
    adj = adj.copy()
    adj[idx[:, 0], idx[:, 1]] = rng.integers(0, n_vert, size=len(idx)).astype(adj.dtype)
    return adj


def pick_length_matched(V, adj, pad, hub_idx, rng, exclude_targets):
    """Edges matching the hub in-edges' LENGTH distribution, but not pointing at hubs.

    Stratified by decile of the hub edges' own length distribution, so the returned
    set has the same length profile rather than merely the same mean. Without this
    the 'length' control could be matched on average yet differ in shape.
    """
    valid = np.argwhere(adj != pad)
    tgt = adj[valid[:, 0], valid[:, 1]]
    ok = ~np.isin(tgt, exclude_targets)
    valid = valid[ok]
    pool_len = edge_lengths(V, valid, adj)
    hub_len = edge_lengths(V, hub_idx, adj)

    edges = np.percentile(hub_len, np.linspace(0, 100, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    chosen = []
    for i in range(10):
        want = int(((hub_len >= edges[i]) & (hub_len < edges[i + 1])).sum())
        if not want:
            continue
        cand = np.flatnonzero((pool_len >= edges[i]) & (pool_len < edges[i + 1]))
        if not len(cand):
            continue
        chosen.append(rng.choice(cand, size=min(want, len(cand)), replace=False))
    return valid[np.concatenate(chosen)] if chosen else valid[:0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='dynamic_edges snapshot to ablate')
    ap.add_argument('--doses', type=int, nargs='*', default=DOSES,
                    help='how many top hubs to neutralise')
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    hnsw = load(graph, args.path)
    pad, n = hnsw.service_labels['pad'], hnsw.num_vertices
    base_adj = hnsw.adj.copy()
    ind = in_degrees(hnsw)
    order = np.argsort(ind)[::-1]

    print('== hub ablation on %s ==' % args.path)
    print('top-10 in-degrees:', ind[order[:10]].tolist())
    print('\n%-30s %8s %8s %9s %8s %9s %8s' %
          ('variant', 'recall', '+-SE', 'coverage', 'pops', 'ind_max', 'moved'))
    r = measure(hnsw, graph)
    print('%-30s %8.4f %8.4f %9.4f %8.2f %9d %8s' %
          ('intact', r['recall'], r['se'], r['coverage'], r['pops'], r['ind_max'], '-'))

    for topn in args.doses:
        tgt = order[:topn]
        mask = np.isin(base_adj, tgt.astype(base_adj.dtype)) & (base_adj != pad)
        hub_idx = np.argwhere(mask)
        if not len(hub_idx):
            continue
        hub_len = edge_lengths(V, hub_idx, base_adj)

        variants = [('hub top-%d' % topn, hub_idx)]

        valid = np.argwhere(base_adj != pad)
        variants.append(('  ctrl random n=%d' % len(hub_idx),
                         valid[rng.choice(len(valid), size=len(hub_idx), replace=False)]))
        variants.append(('  ctrl length-matched', pick_length_matched(
            V, base_adj, pad, hub_idx, rng, tgt.astype(base_adj.dtype))))

        print('  [hub edges: n=%d  mean_len %.4f]' % (len(hub_idx), hub_len.mean()))
        for label, idx in variants:
            if not len(idx):
                continue
            hnsw.adj = apply_rewire(base_adj, idx, rng, n)
            rr = measure(hnsw, graph)
            print('%-30s %8.4f %8.4f %9.4f %8.2f %9d %8d' %
                  (label, rr['recall'], rr['se'], rr['coverage'], rr['pops'],
                   rr['ind_max'], len(idx)))

    hnsw.adj = base_adj
    print('\nHubs are causal only if "hub top-N" loses materially MORE recall than')
    print('BOTH controls. Losing the same as length-matched means the hub in-edges')
    print('mattered because of their length, not because of where they point.')


if __name__ == '__main__':
    main()
