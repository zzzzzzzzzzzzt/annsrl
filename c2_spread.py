"""Stage 0 fallout: the gain was SPREAD, not adaptivity. So how much spread?

The shuffled control gave +0.0107 +- 0.0025 (11/12 cells) with the node correspondence
destroyed, i.e. simply widening the distribution of the added edges' length percentile is
worth about as much as anything Stage 0's features bought. C2's rule puts every new edge at
p=65 +- 0.5 (the jitter); this sweeps the width directly:

    p_u ~ clip(Normal(65, sigma), 2, 98)

sigma=0.5 reproduces C2's rule. This is a change to the LEARNING-FREE bar, so it has to be
measured before any learned arm is compared against it.
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import NQ, NJ, EF, K, DCS_BUDGET, build, visit_counts, recall_at

P0, DRAW_SEED0 = 65.0, 1000
SIGMAS = [0.5, 5.0, 10.0, 15.0, 22.0, 30.0]

ap = argparse.ArgumentParser()
ap.add_argument('--dose', type=float, default=0.05)
ap.add_argument('--draws', type=int, default=3)
ap.add_argument('--slot_seed', type=int, default=11)
ap.add_argument('--sigmas', type=float, nargs='*', default=SIGMAS)
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = build('knn')
V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
adj, pad, n = h.adj.copy(), h.service_labels['pad'], h.num_vertices

valid = np.argwhere(adj != pad)
k = int(round(args.dose * len(valid)))
slots = valid[np.random.default_rng(args.slot_seed).choice(len(valid), size=k, replace=False)]
srcs = slots[:, 0]

ranks = {}
for sg in args.sigmas:
    for d in range(args.draws):
        p = np.clip(P0 + sg * np.random.default_rng(DRAW_SEED0 + d + int(sg * 97)).normal(size=k),
                    2.0, 98.0)
        ranks[(sg, d)] = np.clip(np.rint(p / 100.0 * (n - 1)).astype(np.int64), 1, n - 1)

tgt = {key: np.empty(k, dtype=np.int64) for key in ranks}
Vt = torch.as_tensor(V, dtype=torch.float32, device=dev)
for s in range(0, k, 256):
    e = min(s + 256, k)
    su = torch.as_tensor(srcs[s:e].astype(np.int64), device=dev)
    order = torch.cdist(Vt[su], Vt).argsort(dim=1)
    for key, r in ranks.items():
        tgt[key][s:e] = order.gather(
            1, torch.as_tensor(r[s:e], device=dev)[:, None])[:, 0].cpu().numpy()
    del order

print('== C2 rule with a WIDE p distribution (dose %.0f%%, %d draws, slot %d) =='
      % (100 * args.dose, args.draws, args.slot_seed))
print('%-10s %8s %8s %9s' % ('sigma', 'recall', 'SE', 'coverage'))
for sg in args.sigmas:
    rec, cov = [], []
    for d in range(args.draws):
        a = adj.copy()
        a[slots[:, 0], slots[:, 1]] = tgt[(sg, d)].astype(adj.dtype)
        h.adj = a
        res = h.search_deterministic(graph.val_queries[:NQ])
        r, _ = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
        vis, _ = visit_counts(res, h.num_vertices, pad)
        rec.append(r); cov.append(float((vis > 0).mean()))
    v = np.asarray(rec)
    print('%-10.1f %8.4f %8.4f %9.4f'
          % (sg, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), float(np.mean(cov))))
print('\nsigma=0.5 is C2\'s delta-at-65 rule. If a wider sigma wins, the learning-free bar moves.')
