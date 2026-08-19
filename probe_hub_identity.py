"""What is wrong with the 10 specific nodes the policy converges on?

C1b: permuting WHICH source uses WHICH hub is null (+0.0003), and relabelling the hubs to
random nodes while keeping the sharing pattern and in-degree EXACTLY gains +0.030. So it is
neither the assignment nor the concentration -- it is the identity of those particular
nodes. The obvious candidate is that a proximity oracle converges on the densest region of
the dataset, and a target in a dense region is a useless highway exit: everything reachable
from it was already reachable.
"""
import numpy as np, torch, lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build, in_degrees

graph = build('knn')
V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
V = V.astype(np.float64)
mu = V.mean(0)
d_mu = np.sqrt(((V - mu) ** 2).sum(1))

rng = np.random.default_rng(0)
samp = rng.choice(len(V), 20000, replace=False)
# local density: mean distance to the 100 nearest of a 20k sample (lower = denser)
def density(ids):
    out = []
    for s in range(0, len(ids), 256):
        chunk = V[ids[s:s + 256]]
        d = np.sqrt(((chunk[:, None, :] - V[samp][None, :, :]) ** 2).sum(-1))
        out.append(np.partition(d, 100, axis=1)[:, :100].mean(1))
    return np.concatenate(out)

for p in ('runs/b2_frac001_s42', 'runs/b2_frac001_s123', 'runs/b2_frac001_s456'):
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
    h.dynamic_edges = torch.load(p + '/dynamic_edges.500.pth', weights_only=False)
    h.adj = h.build_adjacency()
    hubs = np.argsort(in_degrees(h))[::-1][:10]
    pct_c = 100.0 * (d_mu < d_mu[hubs].mean()).mean()
    dh, dr = density(hubs), density(rng.choice(len(V), 2000, replace=False))
    print('%-22s dist-to-centroid: hubs %.4f vs all %.4f (hub mean at pct %.1f)'
          % (p.split('/')[-1], d_mu[hubs].mean(), d_mu.mean(), pct_c))
    print('%-22s knn100 radius:    hubs %.4f vs random %.4f  (lower = denser)'
          % ('', dh.mean(), dr.mean()))
