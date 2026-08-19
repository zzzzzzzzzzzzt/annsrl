"""Write kNN + 5% random rewiring as a dynamic_edges snapshot for --init_graph_from.

A2 asks whether the policy can add anything ON TOP of the free gain: if it trains
from the rewired graph (0.52) and ends up above it, the policy contributes
something random rewiring does not. If it flattens or decays from there, it does
not -- and the earlier "+0.04 over kNN s_0" was only ever recovering a fraction of
what one line of numpy supplies.

Rows must stay tail-compact: search_hnsw.cc stops at the first -1, so an interior
pad would silently truncate a neighbour list (see build_adjacency's docstring).
Writing plain lists per vertex keeps that invariant by construction.
"""
import numpy as np
import torch
import lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build

DOSE = 0.05
SEED = 2                 # middle of the three measured seeds, not the luckiest
OUT = 'runs/rewired_s0/dynamic_edges.0.pth'


def main():
    import os
    os.makedirs('runs/rewired_s0', exist_ok=True)
    graph = build('knn')
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    pad, n = hnsw.service_labels['pad'], hnsw.num_vertices
    adj = hnsw.adj.copy()
    valid = np.argwhere(adj != pad)
    rng = np.random.default_rng(SEED)
    k = int(round(DOSE * len(valid)))
    pick = valid[rng.choice(len(valid), size=k, replace=False)]
    adj[pick[:, 0], pick[:, 1]] = rng.integers(0, n, size=k).astype(adj.dtype)

    # Deduplicate and drop self-loops per row, then store compactly. Random targets
    # can collide with an existing neighbour or the node itself; leaving those in
    # would give some nodes a lower effective degree than others for reasons
    # unrelated to the rewiring.
    edges = {}
    for i in range(n):
        row = adj[i]
        row = row[row != pad]
        seen, out = set(), []
        for t in row.tolist():
            if t != i and t not in seen:
                seen.add(t)
                out.append(int(t))
        edges[i] = out

    torch.save(edges, OUT)
    deg = np.array([len(v) for v in edges.values()])
    print('wrote %s' % OUT)
    print('rewired %d of %d directed edges (%.1f%%)' % (k, len(valid), 100.0 * k / len(valid)))
    print('out-degree: mean %.2f min %d max %d' % (deg.mean(), deg.min(), deg.max()))


if __name__ == '__main__':
    main()
