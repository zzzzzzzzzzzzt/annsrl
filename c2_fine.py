"""Is p=65 really the optimum, or was the C2 grid too coarse to tell?

The Stage 1b heads at narrow span behave as a 1-parameter search over a GLOBAL p (their
p_u has sd 0.9-3.1, i.e. no per-node structure at all) and they land at 61.3 / 63.9 / 65.1
while gaining +0.0066 +- 0.0028 over p=65. C2 sampled p at 55/60/65/70/75 -- five points,
five percentile apart -- so it could not have resolved an optimum at 62-64.

This matters because p=65 IS the bar every learned arm is judged against. If the true
optimum is a percentile point or two away, the bar is understated and every "the policy
nearly caught up" statement is off by that much.

Fine grid, proper power: 5 draws x 3 slot seeds, paired within slot seed.
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import NQ, NJ, EF, K, DCS_BUDGET, build, visit_counts, recall_at

DRAW_SEED0, JITTER = 1000, 500

ap = argparse.ArgumentParser()
ap.add_argument('--dose', type=float, default=0.05)
ap.add_argument('--draws', type=int, default=5)
ap.add_argument('--slot_seeds', type=int, nargs='*', default=[11, 23, 37])
ap.add_argument('--pcts', type=float, nargs='*',
                default=[58, 61, 63, 65, 67, 70])
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = build('knn')
V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
adj, pad, n = h.adj.copy(), h.service_labels['pad'], h.num_vertices

print('== C2 fine grid around 65 (dose %.0f%%, %d draws x %d slot seeds) =='
      % (100 * args.dose, args.draws, len(args.slot_seeds)))
per = {p: [] for p in args.pcts}
for ss in args.slot_seeds:
    valid = np.argwhere(adj != pad)
    k = int(round(args.dose * len(valid)))
    slots = valid[np.random.default_rng(ss).choice(len(valid), size=k, replace=False)]
    srcs = slots[:, 0]
    # jitter drawn once per draw over ALL sources -- re-seeding inside the chunk loop
    # makes it periodic with period `chunk` (the bug found in stage0_adaptive_p.py)
    jit = [np.random.default_rng(DRAW_SEED0 + d).integers(-JITTER, JITTER + 1, size=k)
           for d in range(args.draws)]
    ranks = {(p, d): np.clip(int(round(p / 100.0 * (n - 1))) + jit[d], 1, n - 1)
             for p in args.pcts for d in range(args.draws)}
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
    for p in args.pcts:
        rec = []
        for d in range(args.draws):
            a = adj.copy()
            a[slots[:, 0], slots[:, 1]] = tgt[(p, d)].astype(adj.dtype)
            h.adj = a
            res = h.search_deterministic(graph.val_queries[:NQ])
            rec.append(recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)[0])
        per[p].append(float(np.mean(rec)))
    print('  slot %d done: %s' % (ss, ' '.join('%g:%.4f' % (p, per[p][-1]) for p in args.pcts)))

print('\n%-8s %9s %9s %12s' % ('p', 'mean', 'SE', 'vs p=65'))
base = np.asarray(per[65.0]) if 65.0 in per else None
for p in args.pcts:
    v = np.asarray(per[p])
    tail = ''
    if base is not None and p != 65.0:
        d = v - base
        tail = '%+.4f (%d/%d)' % (d.mean(), int((d > 0).sum()), len(d))
    print('%-8g %9.4f %9.4f %12s' % (p, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), tail))
print('\nIf some p beats 65 consistently, the learning-free bar moves and every')
print('"the learned arm nearly caught up" statement shifts by that much.')
