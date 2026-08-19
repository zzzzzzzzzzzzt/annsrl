"""Hub ablation, averaged over random draws.

`probe_ablate_rng.py` measured the variance the first pass ignored: repeating the
IDENTICAL top-10 ablation with six different uniform target draws gives recalls
0.4761-0.5461, sd 0.0277. Every cell in the first ablation table was a single draw,
so an apparent +0.029 hub-vs-control gap is inside one draw's noise and means nothing.

Here each variant is scored over D draws and reported as mean +- SE over draws. Draw
seeds are shared across variants, so hub / random / length-matched all see the same
sequence of target randomness and the comparison is paired.

Verdict rule unchanged: hubs are causal only if the hub variant separates from BOTH
controls by more than the draw noise. The length-matched control is the one that
matters -- it answers "hubs specifically, or just that hub in-edges are long?"
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

DRAWS = 5
DRAW_SEED0 = 1000


def load(graph, path):
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                          max_dcs=DCS_BUDGET)
    if path is not None:
        h.dynamic_edges = torch.load(path, weights_only=False)
        h.adj = h.build_adjacency()
    return h


def measure(h, graph, adj):
    h.adj = adj
    res = h.search_deterministic(graph.val_queries[:NQ])
    rec, _ = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
    vis, _ = visit_counts(res, h.num_vertices, h.service_labels['pad'])
    return rec, float((vis > 0).mean())


def edge_lengths(V, idx, adj):
    d = V[idx[:, 0]].astype(np.float64) - V[adj[idx[:, 0], idx[:, 1]]].astype(np.float64)
    return np.sqrt((d * d).sum(1))


def pick_length_matched(V, adj, pad, hub_idx, rng, exclude):
    """Same count and same length-decile profile as the hub in-edges."""
    valid = np.argwhere(adj != pad)
    valid = valid[~np.isin(adj[valid[:, 0], valid[:, 1]], exclude)]
    pool_len = edge_lengths(V, valid, adj)
    hub_len = edge_lengths(V, hub_idx, adj)
    bins = np.percentile(hub_len, np.linspace(0, 100, 11))
    bins[0], bins[-1] = -np.inf, np.inf
    chosen = []
    for i in range(10):
        want = int(((hub_len >= bins[i]) & (hub_len < bins[i + 1])).sum())
        cand = np.flatnonzero((pool_len >= bins[i]) & (pool_len < bins[i + 1]))
        if want and len(cand):
            chosen.append(rng.choice(cand, size=min(want, len(cand)), replace=False))
    return valid[np.concatenate(chosen)] if chosen else valid[:0]


def run_draws(h, graph, adj, idx, draws):
    """Score `idx` redirected to uniform targets, once per draw seed."""
    n = adj.shape[0]
    rec, cov = [], []
    for s in range(draws):
        rng = np.random.default_rng(DRAW_SEED0 + s)
        a = adj.copy()
        a[idx[:, 0], idx[:, 1]] = rng.integers(0, n, size=len(idx)).astype(adj.dtype)
        r, c = measure(h, graph, a)
        rec.append(r); cov.append(c)
    v = np.asarray(rec)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), float(np.mean(cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--topn', type=int, default=10)
    ap.add_argument('--draws', type=int, default=DRAWS)
    args = ap.parse_args()

    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)

    print('== hub ablation, top-%d, mean over %d target draws ==' % (args.topn, args.draws))
    print('%-26s %8s %8s %9s %8s' % ('variant', 'recall', 'SE(draw)', 'coverage', 'n_edges'))
    gaps_len, gaps_rnd = [], []

    for p in args.paths:
        h = load(graph, p)
        adj, pad = h.adj.copy(), h.service_labels['pad']
        ind = in_degrees(h)
        tgt = np.argsort(ind)[::-1][:args.topn].astype(adj.dtype)
        hub_idx = np.argwhere(np.isin(adj, tgt) & (adj != pad))

        r, c = measure(h, graph, adj.copy())
        print('%-26s %8.4f %8s %9.4f %8s' % (p.split('/')[-2] + ' intact', r, '-', c, '-'))

        sel = np.random.default_rng(7)          # which edges, fixed across variants
        valid = np.argwhere(adj != pad)
        variants = [
            ('  hub top-%d' % args.topn, hub_idx),
            ('  ctrl random', valid[sel.choice(len(valid), size=len(hub_idx), replace=False)]),
            ('  ctrl length-matched', pick_length_matched(V, adj, pad, hub_idx, sel, tgt)),
        ]
        res = {}
        for label, idx in variants:
            m, se, cov = run_draws(h, graph, adj, idx, args.draws)
            res[label.strip()] = m
            print('%-26s %8.4f %8.4f %9.4f %8d' % (label, m, se, cov, len(idx)))
        hub = res['hub top-%d' % args.topn]
        gaps_rnd.append(hub - res['ctrl random'])
        gaps_len.append(hub - res['ctrl length-matched'])

    for name, g in (('vs random', gaps_rnd), ('vs length-matched', gaps_len)):
        v = np.asarray(g)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
        print('hub %-18s %+.4f +- %.4f  (%d/%d positive)'
              % (name, v.mean(), se, int((v > 0).sum()), len(v)))


if __name__ == '__main__':
    main()
