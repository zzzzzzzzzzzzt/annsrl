"""Unit-check the D1 candidate pool before training on it.

Five things must hold or the runs measure nothing:
  1. The band actually binds -- mean candidate length percentile should land inside
     [lo, hi], not at the 2-hop pool's native ~2.5.
  2. The band is REACHABLE, i.e. n_rand_cand is large enough that rows are not
     constantly falling back to the closest-to-centre rule.
  3. Every row keeps >= n_swap candidates (an all-masked row makes log_softmax NaN).
  4. No candidate is the node itself or an existing neighbour, or the "addition" is a
     no-op that still consumes a swap.
  5. max_periph_pct excludes the outliers C1b blamed (centroid pct 98.8-99.8).
"""
import numpy as np
import torch
import lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build

N_NODES = 400
SEED = 7


class Fake:
    """Just enough of GraphEditPPO for augment_candidates to run."""
    def __init__(self, hnsw, **kw):
        self.hnsw, self.n_swap = hnsw, 2
        self.cand_band, self.n_rand_cand, self.max_periph_pct = None, 0, 100.0
        self._len_ref = self._periph_pct = None
        self._cand_rng_seed = 0
        self.__dict__.update(kw)

    _length_ref = lib.algorithm.GraphEditPPO._length_ref
    _peripherality = lib.algorithm.GraphEditPPO._peripherality
    augment_candidates = lib.algorithm.GraphEditPPO.augment_candidates


def main():
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    pad = hnsw.service_labels['pad']
    nodes = np.random.default_rng(SEED).choice(V.shape[0], size=N_NODES, replace=False)
    ids0, mask0 = hnsw.get_candidates(nodes)

    ref = Fake(hnsw)._length_ref()
    periph = Fake(hnsw)._peripherality()
    pct = lambda d: 100.0 * np.searchsorted(ref, d) / len(ref)

    print('== D1 candidate pool check (2-hop pool width %d) ==' % ids0.shape[1])
    print('%-34s %9s %9s %9s %9s %8s %8s'
          % ('config', 'kept/node', 'min_kept', 'mean_pct', 'in_band%', 'periph', 'clean'))

    cfgs = [
        ('baseline (no D1)', {}),
        ('band 55-75, no rand cand', dict(cand_band=(55., 75.))),
        ('band 55-75, rand 64', dict(cand_band=(55., 75.), n_rand_cand=64)),
        ('band 55-75, rand 256', dict(cand_band=(55., 75.), n_rand_cand=256)),
        ('band 55-75, rand 256, periph<=95', dict(cand_band=(55., 75.), n_rand_cand=256,
                                                  max_periph_pct=95.)),
    ]
    for name, kw in cfgs:
        f = Fake(hnsw, **kw)
        ids, mask = f.augment_candidates(nodes, ids0.copy(), mask0.copy())
        src = np.repeat(nodes, mask.sum(1))
        dst = ids[mask].astype(np.int64)
        d = np.sqrt(((V[src].astype(np.float64) - V[dst].astype(np.float64)) ** 2).sum(1))
        q = pct(d)
        lo, hi = f.cand_band if f.cand_band else (-1, 101)
        # clean = no self-loop and no candidate the node already points at
        bad = int((src == dst).sum())
        for i, node in enumerate(nodes):
            nb = hnsw.adj[node]
            bad += int(np.isin(ids[i][mask[i]], nb[nb != pad]).sum())
        print('%-34s %9.1f %9d %9.2f %9.1f %8.1f %8s'
              % (name, mask.sum(1).mean(), mask.sum(1).min(), q.mean(),
                 100.0 * ((q >= lo) & (q <= hi)).mean(), periph[dst].max(),
                 'yes' if bad == 0 else 'NO(%d)' % bad))

    print('\nmean_pct must sit inside the band, in_band%% near 100, min_kept >= 2,')
    print('clean = yes, and periph must fall below 95 in the last row.')


if __name__ == '__main__':
    main()
