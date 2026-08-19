"""Is the C2 rule still the C2 rule when its rank is ESTIMATED from a sample?

The bar (kNN + 5% of slots rewired to length percentile 65 = 0.5520 +- 0.0103) was
measured with a FULL argsort of all N distances per source -- O(N) per edge, O(N^2)
overall, which is not a rule anyone could run on a large dataset. If that is the only
way to get it, the bar is an oracle rather than a baseline and every comparison against
it is mis-stated.

It should not be. To place an edge at percentile p from u, draw M uniform vertices, sort
their distances, take the one at rank p*M/100: O(M) per edge, independent of N, and an
unbiased estimator of the population rank. `augment_candidates` already does exactly this
during training (M = n_rand_cand), so the trained policy is O(M) too -- but the two have
never been measured against each other, and the whole comparison assumes they agree.

Quantile-estimation error at rank p from M samples is ~sqrt(p(1-p)/M) in percentile units:
3.0 points at M=256, 1.5 at M=1024. The band the policy uses is +-10 wide, so M=256 should
be comfortable -- but "should" is what this measures.
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import NQ, NJ, EF, K, DCS_BUDGET, build, visit_counts, recall_at

P0, DRAW_SEED0 = 65.0, 1000


def dists(V, srcs, tgts):
    d = V[srcs].astype(np.float64) - V[tgts].astype(np.float64)
    return np.sqrt((d * d).sum(-1))


def sampled_targets(V, srcs, p, M, rng, chunk=2048):
    """Target at the estimated p-th percentile, using M uniform probes per source."""
    n, k = V.shape[0], len(srcs)
    out = np.empty(k, dtype=np.int64)
    idx = int(round(P0 / 100.0 * (M - 1))) if np.isscalar(p) else None
    for s in range(0, k, chunk):
        e = min(s + chunk, k)
        cand = rng.integers(0, n, size=(e - s, M))
        d = np.sqrt((((V[cand].astype(np.float64)
                       - V[srcs[s:e]].astype(np.float64)[:, None, :]) ** 2).sum(-1)))
        order = np.argsort(d, axis=1)
        pp = p if np.isscalar(p) else p[s:e]
        r = np.clip(np.rint(np.asarray(pp) / 100.0 * (M - 1)).astype(np.int64), 0, M - 1)
        out[s:e] = cand[np.arange(e - s), order[np.arange(e - s), r]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dose', type=float, default=0.05)
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--slot_seeds', type=int, nargs='*', default=[11, 23, 37])
    ap.add_argument('--Ms', type=int, nargs='*', default=[64, 256, 1024])
    args = ap.parse_args()

    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
    adj, pad = h.adj.copy(), h.service_labels['pad']

    print('== C2 rule at p=65: sampled rank (O(M)) vs the full argsort (O(N)) ==')
    print('%-16s %8s %8s %9s' % ('rank source', 'recall', 'SE', 'coverage'))
    agg = {}
    for M in args.Ms:
        per_seed = []
        for ss in args.slot_seeds:
            valid = np.argwhere(adj != pad)
            k = int(round(args.dose * len(valid)))
            slots = valid[np.random.default_rng(ss).choice(len(valid), size=k, replace=False)]
            rec = []
            for d in range(args.draws):
                rng = np.random.default_rng(DRAW_SEED0 + d + 13 * M)
                tgt = sampled_targets(V, slots[:, 0], P0, M, rng)
                a = adj.copy()
                a[slots[:, 0], slots[:, 1]] = tgt.astype(adj.dtype)
                h.adj = a
                res = h.search_deterministic(graph.val_queries[:NQ])
                r, _ = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
                rec.append(r)
            per_seed.append(float(np.mean(rec)))
        v = np.asarray(per_seed)
        agg[M] = v
        print('%-16s %8.4f %8.4f %9s'
              % ('sampled M=%d' % M, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), '-'))

    print('\nreference: full argsort over 3 slot seeds = 0.5520 +- 0.0103')
    print('(logs/spread_s{11,23,37}.log, sigma=0.5 row)')
    full = np.array([0.5314, 0.5626, 0.5619])
    for M, v in agg.items():
        d = v - full
        print('  M=%-5d vs full: %+.4f +- %.4f  (%d/%d)'
              % (M, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)))
    print('\nIf these agree, the bar is an O(M) rule and the scalability objection does not')
    print('bite. If sampling loses materially, the bar has to be restated as an oracle.')


if __name__ == '__main__':
    main()
