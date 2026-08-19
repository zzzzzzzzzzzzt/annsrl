"""C2: where is the interior optimum of added-edge length?

The framing this project used -- "longer is better" -- is wrong. A uniform random target
sits at length percentile 46.9, not 100 ([[hub-identity-is-the-liability]]). kNN's own
edges are at pct 7.6 and recall 0.299; the policy's hub edges reach pct 93.2 and are
actively harmful. So the optimum is interior and nobody has measured it.

This rewires a fixed random subset of kNN's slots and controls ONE variable: the length
percentile of the new target, realised per-source as a rank in that source's own distance
ordering (rank r/n from u is the percentile of the pair-distance distribution conditioned
on u, which averages to the global one). A `uniform` row is included because it is NOT the
same intervention as p=47: uniform SPREADS percentiles over [0,100] with mean 47, while
p=47 concentrates them, and the spread may matter more than the mean.

Everything is averaged over `--draws` paired draws ([[ablation-needs-draw-averaging]]).
"""
import argparse
import numpy as np
import torch
import lib
import lib_control
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

PCTS = [5, 15, 30, 47, 65, 80, 92, 99]
JITTER = 500            # +-0.5% of the vertex set, so a band is sampled and not one shell
DRAW_SEED0 = 1000
SLOT_SEED = 11


def measure(h, graph, adj):
    h.adj = adj
    res = h.search_deterministic(graph.val_queries[:NQ])
    rec, _ = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
    vis, _ = visit_counts(res, h.num_vertices, h.service_labels['pad'])
    ind = np.bincount(adj[adj != h.service_labels['pad']].ravel().astype(np.int64),
                      minlength=adj.shape[0])
    return rec, float((vis > 0).mean()), int(ind.max())


def targets_by_percentile(V, srcs, pcts, draws, jitter, device):
    """out[(p, d)] -> node ids at rank ~p% of each source's distance ordering.

    One argsort pass per chunk serves every (percentile, draw) pair, so the cost is set by
    the number of sources rather than by the size of the sweep.
    """
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    n = Vt.shape[0]
    out = {(p, d): np.empty(len(srcs), dtype=np.int64) for p in pcts for d in range(draws)}
    chunk = 256
    for s in range(0, len(srcs), chunk):
        e = min(s + chunk, len(srcs))
        su = torch.as_tensor(srcs[s:e].astype(np.int64), device=device)
        order = torch.cdist(Vt[su], Vt).argsort(dim=1)
        for p in pcts:
            base = int(round(p / 100.0 * (n - 1)))
            for d in range(draws):
                g = torch.Generator(device=device).manual_seed(DRAW_SEED0 + d + 7919 * p)
                r = (base + torch.randint(-jitter, jitter + 1, (e - s,), device=device,
                                          generator=g)).clamp_(1, n - 1)
                out[(p, d)][s:e] = order.gather(1, r[:, None])[:, 0].cpu().numpy()
        del order
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dose', type=float, default=0.05)
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--pcts', type=int, nargs='*', default=PCTS)
    ap.add_argument('--jitter', type=int, default=JITTER)
    ap.add_argument('--slot_seed', type=int, default=SLOT_SEED,
                    help='which slots get rewired. Varying this is a SEPARATE error '
                         'component from --draws (which only varies the targets), and it '
                         'is the one that was unmeasured when the dose curve looked '
                         'non-monotonic between 0.01 and 0.05.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
    adj, pad, n = h.adj.copy(), h.service_labels['pad'], h.num_vertices

    valid = np.argwhere(adj != pad)
    k = int(round(args.dose * len(valid)))
    slots = valid[np.random.default_rng(args.slot_seed).choice(len(valid), size=k, replace=False)]

    print('== C2 added-edge length sweep: kNN, dose %.0f%% (%d slots), %d draws, jitter +-%d =='
          % (100 * args.dose, k, args.draws, args.jitter))
    r0 = measure(h, graph, adj.copy())
    print('%-22s %8s %8s %9s %9s' % ('target pct', 'recall', 'SE(draw)', 'coverage', 'ind_max'))
    print('%-22s %8.4f %8s %9.4f %9d' % ('s_0 kNN (no rewire)', r0[0], '-', r0[1], r0[2]))

    tgts = targets_by_percentile(V, slots[:, 0], args.pcts, args.draws, args.jitter, device)

    best = None
    for p in args.pcts:
        rec, cov, im = [], [], []
        for d in range(args.draws):
            a = adj.copy()
            a[slots[:, 0], slots[:, 1]] = tgts[(p, d)].astype(adj.dtype)
            x = measure(h, graph, a)
            rec.append(x[0]); cov.append(x[1]); im.append(x[2])
        v = np.asarray(rec)
        se = v.std(ddof=1) / np.sqrt(len(v))
        print('%-22s %8.4f %8.4f %9.4f %9d'
              % ('p = %d' % p, v.mean(), se, float(np.mean(cov)), int(np.mean(im))))
        if best is None or v.mean() > best[1]:
            best = (p, v.mean())

    # uniform is NOT p=47: it spreads percentiles over [0,100] with mean 47.
    rec, cov, im = [], [], []
    for d in range(args.draws):
        rng = np.random.default_rng(DRAW_SEED0 + d)
        a = adj.copy()
        a[slots[:, 0], slots[:, 1]] = rng.integers(0, n, size=k).astype(adj.dtype)
        x = measure(h, graph, a)
        rec.append(x[0]); cov.append(x[1]); im.append(x[2])
    v = np.asarray(rec)
    print('%-22s %8.4f %8.4f %9.4f %9d'
          % ('uniform (spread)', v.mean(), v.std(ddof=1) / np.sqrt(len(v)),
             float(np.mean(cov)), int(np.mean(im))))

    print('\nbest concentrated percentile: p = %d at %.4f   (uniform %.4f)'
          % (best[0], best[1], v.mean()))
    print('Reference points: kNN edges pct 7.6, policy added edges pct ~66,')
    print('policy hub edges pct 93.2, best learned arm 0.502.')


if __name__ == '__main__':
    main()
