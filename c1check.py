"""Sanity-check the rank-preserving reshuffle before spending an hour of searches:
the per-edge length must be preserved, not merely the mean, and the target must change.
"""
import numpy as np, torch, lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build, in_degrees
from c1_reshuffle import rank_neighbourhoods, apply_reshuffle, edge_lengths

import sys
BAND = int(sys.argv[1])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = build('knn')
V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
h.dynamic_edges = torch.load('runs/b2_frac001_s42/dynamic_edges.500.pth', weights_only=False)
h.adj = h.build_adjacency()
adj, pad = h.adj, h.service_labels['pad']
valid = np.argwhere(adj != pad)
L = edge_lengths(V, valid, adj)
N = 3000
idx = valid[np.argpartition(L, len(L)-N)[len(L)-N:]]
old_len = edge_lengths(V, idx, adj)
cand = rank_neighbourhoods(V, idx[:,0], adj[idx[:,0], idx[:,1]], BAND, dev)
a = apply_reshuffle(adj, idx, cand, np.random.default_rng(0))
new_len = edge_lengths(V, idx, a)
same = (a[idx[:,0], idx[:,1]] == adj[idx[:,0], idx[:,1]]).mean()
rel = np.abs(new_len - old_len) / old_len
print('device %s  n=%d' % (dev, N))
print('old len mean %.4f  new len mean %.4f' % (old_len.mean(), new_len.mean()))
print('per-edge |rel change|: mean %.5f  p50 %.5f  p95 %.5f  max %.5f'
      % (rel.mean(), *np.percentile(rel, [50, 95]), rel.max()))
print('fraction of targets unchanged: %.4f' % (same))
print('self-loops created:', int((a[idx[:,0], idx[:,1]] == idx[:,0]).sum()))
