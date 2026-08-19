"""How much of the hub-ablation gain is the particular random draw?

ablate_hub_long.py and eval_hubablate.py report 0.5369 and 0.5012 for the SAME
operation on the same graph, differing only in which uniform targets numpy handed
back. If that spread is typical, every single-draw ablation number in this project
is quoted with an error bar it does not have.
"""
import numpy as np, torch, lib
from probe_visits import NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees, recall_at

graph = build('knn')
h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
h.dynamic_edges = torch.load('runs/b2_frac001_s42/dynamic_edges.500.pth', weights_only=False)
h.adj = h.build_adjacency()
adj, pad, n = h.adj.copy(), h.service_labels['pad'], h.num_vertices
ind = in_degrees(h)
tgt = np.argsort(ind)[::-1][:10].astype(adj.dtype)
idx = np.argwhere(np.isin(adj, tgt) & (adj != pad))
print('hub in-edges:', len(idx))

vals = []
for s in range(6):
    rng = np.random.default_rng(1000 + s)
    a = adj.copy()
    a[idx[:, 0], idx[:, 1]] = rng.integers(0, n, size=len(idx)).astype(adj.dtype)
    h.adj = a
    r = h.search_deterministic(graph.val_queries[:NQ])
    rec, _ = recall_at(r['best_vertex_ids'], graph.val_gt[:NQ], K)
    vals.append(rec); print('  draw %d: %.4f' % (s, rec))
v = np.array(vals)
print('mean %.4f  sd %.4f  se %.4f  range %.4f' % (v.mean(), v.std(ddof=1), v.std(ddof=1)/np.sqrt(len(v)), v.ptp()))
