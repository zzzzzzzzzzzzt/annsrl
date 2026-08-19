"""Stage 0: does p need to be per-node at all? A one-parameter, learning-free test.

C2 found a single global length percentile p=65 (realised as a rank in each SOURCE's own
distance ordering) and got 0.5566. The open question is whether p should VARY per node.
Answering that with a learned head first would confound two things: "p should not vary"
and "the head could not learn it". So this measures the headroom with no learning at all:

    p_u = 65 + a * standardize(feature_u),   clipped to [2, 98]

and sweeps the single slope `a` per candidate feature. a=0 is exactly C2's global rule, so
every curve passes through the same baseline point and the comparison is paired.

  best a == 0 for every feature  -> adaptivity has no headroom along anything cheap;
                                    a learned head is unlikely to find any either
  best a != 0                    -> a LOWER BOUND on what adaptivity is worth, and it
                                    names the feature the Stage 1 head should be fed

Features (all closed-form, all computed from the same 20-NN pass):
  knn_radius   distance to the 10th nearest neighbour -- local scale / density
  centroid     distance to the dataset centroid -- peripherality. C1b showed the policy's
               harmful hubs were centroid-percentile 98.8-99.8 outliers, so this is the
               one feature already known to separate good targets from dead ends.
  lid          local intrinsic dimension, Levina-Bickel MLE over the 20 nearest. The
               textbook predictor of ANN hardness, and the variable H1 actually named.

Targets are realised by PER-NODE RANK, matching C2's definition. NOTE this differs from
D1's band filter, which uses the GLOBAL pair-distance CDF -- the two coincide only if every
node's distance distribution equals the global one.
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build,
                          visit_counts, recall_at)

P0 = 65.0                       # C2's global optimum; a=0 reproduces it exactly
SLOPES = [-15.0, -7.0, 0.0, 7.0, 15.0]
KNN_K = 10                      # neighbour index defining the local radius
LID_K = 20                      # neighbours entering the Levina-Bickel estimate
DRAW_SEED0 = 1000
SLOT_SEED = 11
JITTER = 500


def measure(h, graph, adj):
    h.adj = adj
    res = h.search_deterministic(graph.val_queries[:NQ])
    rec, _ = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
    vis, _ = visit_counts(res, h.num_vertices, h.service_labels['pad'])
    return rec, float((vis > 0).mean())


def node_features(V, device, chunk=256):
    """knn_radius and LID for every vertex, from one pass of 20-NN distances."""
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    n = Vt.shape[0]
    rad = np.empty(n, dtype=np.float64)
    lid = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(Vt[s:e], Vt)
        # +1 because the nearest is the point itself at distance 0
        nn = d.topk(LID_K + 1, dim=1, largest=False).values[:, 1:].double()
        rad[s:e] = nn[:, KNN_K - 1].cpu().numpy()
        # Levina-Bickel MLE: LID = -[ mean_i log(d_i / d_k) ]^-1  over i < k
        ratio = (nn[:, :LID_K - 1] / nn[:, LID_K - 1:LID_K]).clamp_min(1e-12).log()
        lid[s:e] = (-1.0 / ratio.mean(1).clamp_max(-1e-12)).cpu().numpy()
        del d, nn, ratio
    return rad, lid


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dose', type=float, default=0.05)
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--slot_seed', type=int, default=SLOT_SEED)
    ap.add_argument('--slopes', type=float, nargs='*', default=SLOPES)
    ap.add_argument('--shuffle', action='store_true',
                    help='permute each feature across nodes. This preserves the multiset '
                         'of p_u values EXACTLY -- same marginal distribution, same '
                         'variance, same skew -- and destroys only which node gets which '
                         'p. It is the control that separates "adaptivity" from "a wider '
                         'spread of edge lengths helps regardless of who gets what".')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
    adj, pad, n = h.adj.copy(), h.service_labels['pad'], h.num_vertices

    print('== Stage 0: is a per-node p worth anything? (dose %.0f%%, %d draws, slot seed %d) =='
          % (100 * args.dose, args.draws, args.slot_seed))
    rad, lid = node_features(V, device)
    cent = np.sqrt(((V.astype(np.float64) - V.astype(np.float64).mean(0)) ** 2).sum(1))
    feats = {'knn_radius': zscore(rad), 'centroid': zscore(cent), 'lid': zscore(lid)}
    if args.shuffle:
        perm_rng = np.random.default_rng(31337)
        feats = {k: perm_rng.permutation(v) for k, v in feats.items()}
        print('[SHUFFLE CONTROL] features permuted across nodes; p_u marginals unchanged')
    print('feature spread (z-scored, so this is the raw scale before standardising):')
    for name, raw in (('knn_radius', rad), ('centroid', cent), ('lid', lid)):
        print('  %-12s mean %8.3f  sd %8.3f  p5 %8.3f  p95 %8.3f'
              % (name, raw.mean(), raw.std(), *np.percentile(raw, [5, 95])))

    valid = np.argwhere(adj != pad)
    k = int(round(args.dose * len(valid)))
    slots = valid[np.random.default_rng(args.slot_seed).choice(len(valid), size=k, replace=False)]
    srcs = slots[:, 0]

    # One argsort pass over the sources serves every (feature, slope, draw) cell.
    cfgs = [(name, a) for name in feats for a in args.slopes if a != 0.0]
    cfgs = [('baseline', 0.0)] + cfgs
    tgt = {(c, d): np.empty(k, dtype=np.int64) for c in cfgs for d in range(args.draws)}
    # One jitter vector per draw, covering ALL sources. Re-seeding inside the chunk loop
    # (as this script and c2_lengthsweep.py both used to) makes the offsets periodic with
    # period `chunk`, so 120k sources share only 256 distinct offsets and the band is
    # sampled far more coarsely than intended.
    jitters = [np.random.default_rng(DRAW_SEED0 + d).integers(-JITTER, JITTER + 1, size=k)
               for d in range(args.draws)]
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    for s in range(0, k, 256):
        e = min(s + 256, k)
        su = torch.as_tensor(srcs[s:e].astype(np.int64), device=device)
        order = torch.cdist(Vt[su], Vt).argsort(dim=1)
        for (name, a) in cfgs:
            f = 0.0 if a == 0.0 else feats[name][srcs[s:e]]
            p = np.clip(P0 + a * f, 2.0, 98.0)
            base = np.rint(p / 100.0 * (n - 1)).astype(np.int64)
            for d in range(args.draws):
                r = np.clip(base + jitters[d][s:e], 1, n - 1)
                tgt[((name, a), d)][s:e] = order.gather(
                    1, torch.as_tensor(r, device=device)[:, None])[:, 0].cpu().numpy()
        del order

    print('\n%-14s %7s %8s %8s %9s %11s'
          % ('feature', 'slope', 'recall', 'SE', 'coverage', 'vs a=0'))
    base_mean = None
    for (name, a) in cfgs:
        rec, cov = [], []
        for d in range(args.draws):
            aa = adj.copy()
            aa[slots[:, 0], slots[:, 1]] = tgt[((name, a), d)].astype(adj.dtype)
            r, c = measure(h, graph, aa)
            rec.append(r); cov.append(c)
        v = np.asarray(rec)
        if base_mean is None:
            base_mean = v.mean()
        print('%-14s %7.0f %8.4f %8.4f %9.4f %11s'
              % (name, a, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), float(np.mean(cov)),
                 '-' if a == 0.0 else '%+.4f' % (v.mean() - base_mean)))

    print('\na=0 is C2\'s global p=65 rule. A slope that beats it is a LOWER BOUND on what')
    print('a per-node p is worth, and names the feature the Stage 1 head should be fed.')
    print('All slopes ~0 => adaptivity has no cheap headroom; do not build the head.')


if __name__ == '__main__':
    main()
