"""B2 pre-check: is the short-edge attractor a SCORER bias or an ACTION-SPACE ceiling?

B0 concluded the policy prefers short edges. But `get_candidates` restricts every
addition to the 2-hop neighbourhood N2, and on a kNN graph N2 is a tight geometric
ball: the 2-hop neighbours of a node are the neighbours of its 24 nearest neighbours.
If the longest edge available in N2 already sits near pct 2, then the observed
attractor at add_pct 1.6-2 is not a preference at all -- it is the policy taking the
LONGEST edge it is permitted to take, and the bias lives in the action space, not the
weights.

This distinguishes the two by measuring the candidate pool itself:
  - pool_max_pct: the longest edge the policy COULD add
  - pool_mean_pct: what a uniform pick from N2 would give
against the measured add_pct (~1.67 for ctrl_s42).

If add_pct is close to pool_max_pct, the policy is already maximising reach and B1
(random-init scorer) is predicted null for a reason that has nothing to do with the
reward. The fix is then to widen the candidate pool, not to retrain the scorer.
"""
import numpy as np
import torch
import lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build

N_NODES = 2000         # nodes to draw candidate pools for (pools are wide)
N_PAIRS = 200000
SEED = 7
REWIRED = 'runs/rewired_s0/dynamic_edges.0.pth'


def dists(V, a, b, chunk=200000):
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), chunk):
        e = min(s + chunk, len(a))
        d = V[a[s:e]].astype(np.float64) - V[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt((d * d).sum(1))
    return out


def main():
    rng = np.random.default_rng(SEED)
    print('== B2 pre-check: does the 2-hop pool even CONTAIN long edges? ==')
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    n = V.shape[0]

    pa, pb = rng.integers(0, n, size=N_PAIRS), rng.integers(0, n, size=N_PAIRS)
    keep = pa != pb
    ref = np.sort(dists(V, pa[keep], pb[keep]))

    def pct(d):
        return 100.0 * np.searchsorted(ref, d) / len(ref)

    nodes = rng.choice(n, size=N_NODES, replace=False)

    for label, path in (('kNN s_0', None), ('rewired s_0', REWIRED)):
        hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                                 max_dcs=DCS_BUDGET)
        if path is not None:
            hnsw.dynamic_edges = torch.load(path, weights_only=False)
            hnsw.adj = hnsw.build_adjacency()

        cand_ids, cand_mask = hnsw.get_candidates(nodes)
        print('\n--- %s ---' % label)
        print('pool shape %s, valid per node: mean %.1f  min %d  max %d'
              % (cand_ids.shape, cand_mask.sum(1).mean(),
                 cand_mask.sum(1).min(), cand_mask.sum(1).max()))

        src = np.repeat(nodes, cand_mask.sum(1))
        dst = cand_ids[cand_mask].astype(np.int64)
        q = pct(dists(V, src, dst))
        print('pool edge length: mean_pct %6.3f  median_pct %6.3f  p99_pct %6.3f'
              % (q.mean(), np.median(q), np.percentile(q, 99)))

        # Per-node max: the longest edge THIS node could possibly add.
        per_node_max = np.full(len(nodes), np.nan)
        off = 0
        counts = cand_mask.sum(1)
        for i, c in enumerate(counts):
            if c:
                per_node_max[i] = q[off:off + c].max()
            off += c
        print('per-node MAX available pct: mean %6.3f  median %6.3f  p90 %6.3f'
              % (np.nanmean(per_node_max), np.nanmedian(per_node_max),
                 np.nanpercentile(per_node_max, 90)))
        print('%% of pool edges above pct 10: %5.2f%%   above pct 50: %5.2f%%'
              % (100.0 * (q > 10).mean(), 100.0 * (q > 50).mean()))

    print('\nInterpretation: ctrl_s42 measured add_pct = 1.67. If the kNN pool\'s')
    print('per-node max is near that, the policy is already picking the longest')
    print('edge available and the ACTION SPACE is the binding constraint -- B1 is')
    print('then predicted null, and widening the pool is the real intervention.')


if __name__ == '__main__':
    main()
