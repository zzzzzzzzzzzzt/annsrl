"""Unit-check the B2 candidate filter before spending GPU hours on it.

Three things must hold or the training runs measure nothing:
  1. The kept set really is the LONG tail -- mean length percentile of the filtered
     pool should rise sharply as long_frac shrinks.
  2. Every row keeps at least n_swap candidates, or log_softmax over an all-masked
     row is NaN and the entropy term poisons the gradient.
  3. cand_ids is untouched (only the mask changes), so padded slots still hold a
     valid index and stay safe to embed.
"""
import numpy as np
import torch
import lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build

N_NODES = 500
N_PAIRS = 100000
SEED = 7
FRACS = [1.0, 0.25, 0.10, 0.05, 0.01]


class FakeTrainer:
    """Just enough of the trainer for filter_long_candidates to run."""
    def __init__(self, hnsw, long_frac, n_swap=2):
        self.hnsw, self.long_frac, self.n_swap = hnsw, long_frac, n_swap

    filter_long_candidates = lib.algorithm.GraphEditPPO.filter_long_candidates


def dists(V, a, b):
    d = V[a].astype(np.float64) - V[b].astype(np.float64)
    return np.sqrt((d * d).sum(-1))


def main():
    rng = np.random.default_rng(SEED)
    print('== B2 filter check ==')
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    n = V.shape[0]

    pa, pb = rng.integers(0, n, size=N_PAIRS), rng.integers(0, n, size=N_PAIRS)
    keep = pa != pb
    ref = np.sort(dists(V, pa[keep], pb[keep]))
    pct = lambda d: 100.0 * np.searchsorted(ref, d) / len(ref)

    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    nodes = rng.choice(n, size=N_NODES, replace=False)
    cand_ids, cand_mask0 = hnsw.get_candidates(nodes)
    ids_before = cand_ids.copy()

    print('%-10s %10s %10s %11s %11s %9s' %
          ('long_frac', 'kept/node', 'min_kept', 'mean_pct', 'max_pct', 'ids_ok'))
    for f in FRACS:
        t = FakeTrainer(hnsw, long_frac=0.0 if f >= 1.0 else f)
        m = t.filter_long_candidates(nodes, cand_ids, cand_mask0.copy())

        src = np.repeat(nodes, m.sum(1))
        dst = cand_ids[m].astype(np.int64)
        q = pct(dists(V, src, dst))

        per_max = []
        off = 0
        for c in m.sum(1):
            if c:
                per_max.append(q[off:off + c].max())
            off += c

        print('%-10s %10.1f %10d %11.3f %11.3f %9s' %
              ('%.2f' % f, m.sum(1).mean(), m.sum(1).min(), q.mean(),
               float(np.mean(per_max)), np.array_equal(cand_ids, ids_before)))

    print('\nmean_pct must RISE as long_frac falls (the filter keeps the long tail),')
    print('min_kept must stay >= n_swap=2, and ids_ok must be True at every row.')


if __name__ == '__main__':
    main()
