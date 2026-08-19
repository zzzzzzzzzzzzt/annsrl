"""B0: does the policy replace long edges with short ones?

The short-edge-bias mechanism explains every A-tier observation, but it is still
an inference. This measures it directly: for each learned graph, split its edits
into ADDED edges (present in the learned graph, absent from its own s_0) and
REMOVED edges (the reverse), and compare their lengths.

Lengths are reported as a PERCENTILE of the distance distribution between random
vertex pairs, which makes them interpretable and comparable across graphs: a
uniformly random target sits at ~50, a kNN edge near 0. Raw distances in a
128-dim normalized space carry no intuition on their own.

The sharpest test is the A2 runs, which start from kNN+5% rewiring. If the bias
is real, the edges they REMOVE should be exactly the long ones rewiring added,
and removed-percentile should far exceed added-percentile there.
"""
import os.path as osp
import numpy as np
import torch
import lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build

N_NODES = 20000        # sampled nodes for the edit-length statistics
N_PAIRS = 200000       # random pairs defining the reference distance CDF
SEED = 7
STEP = 500
REWIRED = 'runs/rewired_s0/dynamic_edges.0.pth'

CASES = [
    ('ctrl_s42  (typical)', None, 'runs/tier1a_ctrl_s42/dynamic_edges.%d.pth' % STEP),
    ('ctrl_s123 (typical)', None, 'runs/tier1a_ctrl_s123/dynamic_edges.%d.pth' % STEP),
    ('ctrl_s789 (best)',    None, 'runs/tier1a_ctrl_s789/dynamic_edges.%d.pth' % STEP),
    ('ideg_s42',            None, 'runs/tier1a_ideg_s42/dynamic_edges.%d.pth' % STEP),
    ('a2_s42  (from rewire)', REWIRED, 'runs/a2_rewired_s42/dynamic_edges.%d.pth' % STEP),
    ('a2_s123 (from rewire)', REWIRED, 'runs/a2_rewired_s123/dynamic_edges.%d.pth' % STEP),
    ('a2_s456 (from rewire)', REWIRED, 'runs/a2_rewired_s456/dynamic_edges.%d.pth' % STEP),
]


def adj_of(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    return hnsw.adj.copy(), hnsw.service_labels['pad']


def dists(V, a, b, chunk=200000):
    """L2 between V[a] and V[b], chunked to bound peak memory."""
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), chunk):
        e = min(s + chunk, len(a))
        d = V[a[s:e]].astype(np.float64) - V[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt((d * d).sum(1))
    return out


def diff_edges(base, learned, pad, nodes):
    """Per node: which targets were added, which removed. Returns edge endpoint arrays."""
    a_src, a_dst, r_src, r_dst = [], [], [], []
    for i in nodes:
        b = base[i]; b = set(b[b != pad].tolist())
        l = learned[i]; l = set(l[l != pad].tolist())
        for t in (l - b):
            a_src.append(i); a_dst.append(t)
        for t in (b - l):
            r_src.append(i); r_dst.append(t)
    return (np.array(a_src, dtype=np.int64), np.array(a_dst, dtype=np.int64),
            np.array(r_src, dtype=np.int64), np.array(r_dst, dtype=np.int64))


def main():
    rng = np.random.default_rng(SEED)
    print('== B0: are the policy\'s added edges shorter than the ones it removes? ==')
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    n = V.shape[0]
    print('vertices %s' % (V.shape,))

    # Reference CDF from random pairs, so a length can be quoted as a percentile.
    pa = rng.integers(0, n, size=N_PAIRS)
    pb = rng.integers(0, n, size=N_PAIRS)
    keep = pa != pb
    ref = np.sort(dists(V, pa[keep], pb[keep]))
    print('random-pair distance: mean %.4f  p1 %.4f  p50 %.4f  p99 %.4f'
          % (ref.mean(), *np.percentile(ref, [1, 50, 99])))

    def pct(d):
        return 100.0 * np.searchsorted(ref, d) / len(ref)

    nodes = rng.choice(n, size=min(N_NODES, n), replace=False)
    base_knn, pad = adj_of(graph, None)

    # Reference rows: what the starting graphs' own edges look like.
    print('\n--- reference edge lengths (percentile of random pairs) ---')
    for label, path, gtype in (('kNN s_0 edges', None, 'knn'),
                               ('rewired s_0 edges', REWIRED, 'knn'),
                               ('NSW s_0 edges', None, 'nsw')):
        g = graph if gtype == 'knn' else build('nsw')
        A, p = adj_of(g, path)
        Vg = g.vertices.numpy() if torch.is_tensor(g.vertices) else np.asarray(g.vertices)
        src = np.repeat(nodes, (A[nodes] != p).sum(1))
        dst = A[nodes][A[nodes] != p].astype(np.int64)
        d = dists(Vg, src, dst)
        q = pct(d) if gtype == 'knn' else 100.0 * np.searchsorted(np.sort(dists(
            Vg, pa[keep], pb[keep])), d) / keep.sum()
        print('%-20s n=%8d  mean_pct %6.2f  median_pct %6.2f  %%above p50 %5.1f%%'
              % (label, len(d), q.mean(), np.median(q), 100.0 * (q > 50).mean()))

    print('\n--- policy edits: ADDED vs REMOVED ---')
    print('%-22s %9s %9s %9s %9s %11s' %
          ('run', 'n_added', 'add_pct', 'n_removed', 'rem_pct', 'add - rem'))
    for label, base_path, learned_path in CASES:
        if not osp.exists(learned_path):
            print('%-22s MISSING' % label)
            continue
        base = base_knn if base_path is None else adj_of(graph, base_path)[0]
        learned, _ = adj_of(graph, learned_path)
        a_s, a_d, r_s, r_d = diff_edges(base, learned, pad, nodes)
        qa = pct(dists(V, a_s, a_d)) if len(a_s) else np.array([np.nan])
        qr = pct(dists(V, r_s, r_d)) if len(r_s) else np.array([np.nan])
        print('%-22s %9d %9.2f %9d %9.2f %+11.2f'
              % (label, len(a_s), qa.mean(), len(r_s), qr.mean(),
                 qa.mean() - qr.mean()))

    print('\nadd - rem < 0  => the policy is net-SHORTENING its edges, confirming')
    print('the short-edge bias. On the a2_* rows (which start from a graph whose')
    print('advantage IS long edges) a strongly negative value means it is removing')
    print('exactly the long-range links that random rewiring supplied.')


if __name__ == '__main__':
    main()
