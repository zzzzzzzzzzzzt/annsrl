"""What length band does the winning 'length-matched' ablation actually target?

That control lifted recall more than uniform selection did (+0.040 vs +0.009), so the
band it selects is the actionable part. Quoted as a percentile of random vertex-pair
distance, which is the only length scale comparable across graphs.
"""
import numpy as np, torch, lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build, in_degrees

graph = build('knn')
V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
rng = np.random.default_rng(7); n = V.shape[0]
a, b = rng.integers(0, n, 200000), rng.integers(0, n, 200000)
k = a != b
d = V[a[k]].astype(np.float64) - V[b[k]].astype(np.float64)
ref = np.sort(np.sqrt((d * d).sum(1)))
pct = lambda x: 100.0 * np.searchsorted(ref, x) / len(ref)

h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
h.dynamic_edges = torch.load('runs/b2_frac001_s42/dynamic_edges.500.pth', weights_only=False)
h.adj = h.build_adjacency()
adj, pad = h.adj, h.service_labels['pad']
ind = in_degrees(h)
tgt = np.argsort(ind)[::-1][:10].astype(adj.dtype)

def lens(idx):
    dd = V[idx[:, 0]].astype(np.float64) - V[adj[idx[:, 0], idx[:, 1]]].astype(np.float64)
    return np.sqrt((dd * dd).sum(1))

hub = np.argwhere(np.isin(adj, tgt) & (adj != pad))
allv = np.argwhere(adj != pad)
for name, idx in (('hub in-edges', hub), ('all edges', allv)):
    q = pct(lens(idx))
    print('%-14s n=%-8d pct mean %5.1f  p10 %5.1f  p50 %5.1f  p90 %5.1f'
          % (name, len(idx), q.mean(), *np.percentile(q, [10, 50, 90])))
# A uniform random target is NOT at percentile 100 -- it is a draw from the very
# distribution the percentile is defined against, so it sits at percentile ~50.
u = rng.integers(0, len(V), size=len(hub))
q = pct(np.sqrt(((V[hub[:, 0]].astype(np.float64) - V[u].astype(np.float64)) ** 2).sum(1)))
print('%-14s n=%-8d pct mean %5.1f  p10 %5.1f  p50 %5.1f  p90 %5.1f'
      % ('uniform tgt', len(u), q.mean(), *np.percentile(q, [10, 50, 90])))
print('mean random-pair distance %.4f' % ref.mean())
