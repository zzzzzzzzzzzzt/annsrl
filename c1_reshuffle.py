"""C1: does the policy's choice of WHICH node carry anything beyond distance?

The hub ablation showed that replacing the policy's long-edge targets with uniform
random ones GAINS +0.040 ([[policy-long-edge-targets-worse-than-random]]). But a uniform
target sits at length percentile ~100 while the policy's sit at ~93, so that test
confounds "target identity" with "even longer". This removes the confound.

For each selected edge (u -> v), v is replaced by v', drawn uniformly from the nodes
whose RANK in u's distance ordering is within +-`band` of v's rank. Distance from the
source is preserved to within 0.5% of the vertex set, so the edge-length distribution is
unchanged by construction and the ONLY thing destroyed is which particular node was
chosen.

  recall unchanged  -> the policy's target selection carries no information beyond
                       distance; the learned part of the pipeline is worth zero.
  recall drops      -> it knows something, and a length-stratified action space (D1) is
                       worth building.

Four variants, all scored over `--draws` paired random draws because a single draw has
sd 0.028 ([[ablation-needs-draw-averaging]]):

  intact
  C1 reshuffle-long   rank-preserving reshuffle of the N longest edges   <- the test
  ref uniform-long    same N edges sent to uniform targets               <- confounded ref
  calib reshuffle-rnd rank-preserving reshuffle of N random edges        <- should be null
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

DRAWS = 5
DRAW_SEED0 = 1000
BAND = 100              # +-0.25% of the vertex set, in rank space
SEL_SEED = 7            # which edges get touched; fixed across variants


def load(graph, path):
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                          max_dcs=DCS_BUDGET)
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


def rank_neighbourhoods(V, srcs, tgts, band, device):
    """For each (src, tgt) pair, the candidate node ids at nearly the same distance.

    Returns an int32 array (n_pairs, 2*band+1) of node ids whose rank in src's distance
    ordering is within `band` of tgt's rank. Sampling a column uniformly therefore
    preserves the edge's length to within band/|V| of the distance CDF.

    Done on the GPU in chunks: a full argsort per source is 100k int64 per row, so the
    chunk size trades memory against the number of kernel launches.
    """
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    n = Vt.shape[0]
    width = 2 * band + 1
    out = np.empty((len(srcs), width), dtype=np.int32)

    chunk = 256
    span = torch.arange(width, device=device)
    for s in range(0, len(srcs), chunk):
        e = min(s + chunk, len(srcs))
        su = torch.as_tensor(srcs[s:e].astype(np.int64), device=device)
        tu = torch.as_tensor(tgts[s:e].astype(np.int64), device=device)
        d = torch.cdist(Vt[su], Vt)                       # (c, n)
        order = d.argsort(dim=1)                          # rank -> node id
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(n, device=device).expand(order.shape))
        r = rank.gather(1, tu[:, None])                   # (c, 1) rank of the real target
        # `start` is the FIRST rank of the window, so the span added to it must run
        # 0..width-1, not -band..+band -- the latter shifts the window a whole band low
        # and silently biases every reshuffled edge SHORT. rank 0 is the source itself,
        # so the window is clamped into [1, n-1].
        start = (r - band).clamp_(1, n - width)
        out[s:e] = order.gather(1, start + span[None, :]).to(torch.int32).cpu().numpy()
        del d, order, rank
    return out


def apply_reshuffle(adj, idx, cand, rng):
    """Replace each selected edge's target with a uniformly drawn same-rank candidate."""
    a = adj.copy()
    pick = rng.integers(0, cand.shape[1], size=len(idx))
    a[idx[:, 0], idx[:, 1]] = cand[np.arange(len(idx)), pick].astype(adj.dtype)
    return a


def apply_uniform(adj, idx, rng):
    a = adj.copy()
    a[idx[:, 0], idx[:, 1]] = rng.integers(0, adj.shape[0], size=len(idx)).astype(adj.dtype)
    return a


def run_draws(h, graph, fn, draws):
    rec, cov = [], []
    for s in range(draws):
        r, c = measure(h, graph, fn(np.random.default_rng(DRAW_SEED0 + s)))
        rec.append(r); cov.append(c)
    v = np.asarray(rec)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), float(np.mean(cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--band', type=int, default=BAND)
    ap.add_argument('--draws', type=int, default=DRAWS)
    ap.add_argument('--exclude_hub', type=int, default=0,
                    help='drop the in-edges of the top-N in-degree vertices from the long '
                         'selection. C1b found the reshuffle HELPS on hub edges (+0.025) '
                         'while C1 found it HURTS on the longest ones (-0.014), so the '
                         'learned signal has to live in the long NON-hub edges; this '
                         'isolates them.')
    ap.add_argument('--n_edges', type=int, default=0,
                    help='how many longest edges to touch; 0 = match the top-10 hub in-edge count')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)

    print('== C1 length-preserving target reshuffle (band +-%d ranks, %d draws, %s) =='
          % (args.band, args.draws, device))
    print('%-28s %8s %8s %9s %9s' % ('variant', 'recall', 'SE(draw)', 'coverage', 'mean_len'))
    c1_gaps, ref_gaps, calib_gaps = [], [], []

    for p in args.paths:
        h = load(graph, p)
        adj, pad = h.adj.copy(), h.service_labels['pad']
        valid = np.argwhere(adj != pad)
        L = edge_lengths(V, valid, adj)

        n_sel = args.n_edges
        if not n_sel:
            ind = in_degrees(h)
            tgt = np.argsort(ind)[::-1][:10].astype(adj.dtype)
            n_sel = int((np.isin(adj, tgt) & (adj != pad)).sum())

        pool, Lp = valid, L
        if args.exclude_hub:
            hubs = np.argsort(in_degrees(h))[::-1][:args.exclude_hub].astype(adj.dtype)
            keep = ~np.isin(adj[valid[:, 0], valid[:, 1]], hubs)
            pool, Lp = valid[keep], L[keep]
        n_sel = min(n_sel, len(Lp))
        long_idx = pool[np.argpartition(Lp, len(Lp) - n_sel)[len(Lp) - n_sel:]]
        sel = np.random.default_rng(SEL_SEED)
        rnd_idx = valid[sel.choice(len(valid), size=n_sel, replace=False)]

        name = p.split('/')[-2]
        r, c = measure(h, graph, adj.copy())
        print('%-28s %8.4f %8s %9.4f %9.4f'
              % (name + ' intact', r, '-', c, L.mean()))

        cand_long = rank_neighbourhoods(V, long_idx[:, 0], adj[long_idx[:, 0], long_idx[:, 1]],
                                        args.band, device)
        cand_rnd = rank_neighbourhoods(V, rnd_idx[:, 0], adj[rnd_idx[:, 0], rnd_idx[:, 1]],
                                       args.band, device)

        rows = [
            ('  C1 reshuffle-long', lambda g: apply_reshuffle(adj, long_idx, cand_long, g)),
            ('  ref uniform-long', lambda g: apply_uniform(adj, long_idx, g)),
            ('  calib reshuffle-rnd', lambda g: apply_reshuffle(adj, rnd_idx, cand_rnd, g)),
        ]
        got = {}
        for label, fn in rows:
            m, se, cov = run_draws(h, graph, fn, args.draws)
            got[label.strip().split()[0]] = m
            ml = edge_lengths(V, valid, fn(np.random.default_rng(DRAW_SEED0))).mean()
            print('%-28s %8.4f %8.4f %9.4f %9.4f' % (label, m, se, cov, ml))
        c1_gaps.append(got['C1'] - r)
        ref_gaps.append(got['ref'] - r)
        calib_gaps.append(got['calib'] - r)
        print('%-28s n_selected=%d  sel_mean_len=%.4f'
              % ('', n_sel, edge_lengths(V, long_idx, adj).mean()))

    for name, g in (('C1 reshuffle-long', c1_gaps), ('ref uniform-long', ref_gaps),
                    ('calib reshuffle-rnd', calib_gaps)):
        v = np.asarray(g)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
        print('%-22s vs intact: %+.4f +- %.4f  (%d/%d positive)'
              % (name, v.mean(), se, int((v > 0).sum()), len(v)))
    print('\nC1 ~ 0 means the policy\'s target identity carries nothing beyond distance.')
    print('calib should also be ~0; if it is not, the reshuffle itself is the effect.')


if __name__ == '__main__':
    main()
