""" Check 1: is the policy dragging NSW toward a kNN graph?

The actor cannot condition on topology. prepare_state does z = encode(x) with no
adjacency, and _pair_features returns cat([z_i, z_j]), so the score of a pair (i, j) is
a FIXED function of the two endpoints' features, identical in every graph state. Such a
policy can only express one global "which edges are good" ranking -- it cannot represent
"my neighbourhood is already too local, add a long-range link", which is the core
tradeoff of an ANN graph.

If that ranking prefers short edges, the policy rewrites NSW into something kNN-like.
Measured earlier: kNN recall@10 = 0.3213 against NSW's 0.7092, so that drift would
explain the monotone -0.045 decline directly, and the root cause would be the policy
CLASS rather than anything in the RL.

Edge length (L2 between endpoints) is the discriminator: kNN keeps only the shortest
edges by construction, NSW deliberately keeps some long ones for navigability.
"""
import os.path as osp
import numpy as np

from lib.utils import read_edges

DATA_DIR = './data/SIFT100K'
NSW_PATH = osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs')
KNN_CACHE = osp.join(DATA_DIR, 'knn_cache', 'init_knn_k25.npz')
EDITED = 'runs/nsw_g0_s1234/graph.50.ivecs'


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def edge_lengths(edges, x, cap=200000):
    """Mean L2 over a capped sample of edges, so all graphs cost the same to measure."""
    src, dst = [], []
    for v, nbrs in edges.items():
        for u in nbrs:
            src.append(v)
            dst.append(u)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if len(src) > cap:
        idx = np.random.RandomState(0).choice(len(src), cap, replace=False)
        src, dst = src[idx], dst[idx]
    d = np.linalg.norm(x[src] - x[dst], axis=1)
    return d, len(edges)


def report(name, d, nv):
    q = np.percentile(d, [50, 90, 99])
    print('  %-22s mean %8.2f  median %8.2f  p90 %8.2f  p99 %8.2f  nodes %d'
          % (name, d.mean(), q[0], q[1], q[2], nv))
    return d.mean()


if __name__ == '__main__':
    x = read_fvecs(osp.join(DATA_DIR, 'sift_base.fvecs'))
    print('base vectors', x.shape)

    print('\nedge length (L2 between endpoints), lower = more local:')
    nsw = read_edges(NSW_PATH)
    m_nsw = report('NSW s_0', *edge_lengths(read_edges(NSW_PATH), x))

    # kNN reference: the cache stores the exact kNN ids used to build a kNN start.
    z = np.load(KNN_CACHE)
    key = [k for k in z.files if z[k].ndim == 2][0]
    ids = z[key]
    # column 0 is the point itself for an exact kNN over the base set
    knn_edges = {i: [int(j) for j in ids[i, 1:25] if int(j) != i]
                 for i in range(ids.shape[0])}
    m_knn = report('pure kNN (k=24)', *edge_lengths(knn_edges, x))

    m_ed = report('NSW edited, step 50', *edge_lengths(read_edges(EDITED), x))

    print('\ninterpretation:')
    span = m_nsw - m_knn
    print('  NSW -> kNN span            %8.2f' % span)
    print('  edited moved from NSW      %+8.2f' % (m_ed - m_nsw))
    if span > 0:
        print('  i.e. %.1f%% of the way toward kNN' % (100.0 * (m_nsw - m_ed) / span))
    # The whole-graph mean above dilutes the policy's preference across the ~98% of
    # edges it never touched (50 steps x 512 nodes x 2 swaps is ~2% of 2.4M edges), so
    # its resolution for a kNN-ward pull is only ~0.3. Compare the CHANGED edges
    # directly instead: what the policy dropped against what it added, which isolates
    # the preference with no dilution at all.
    print('\nchanged edges only (isolates the policy preference):')
    ed = read_edges(EDITED)
    added_s, added_d, dropped_s, dropped_d = [], [], [], []
    for v, before in nsw.items():
        after = ed.get(v)
        if after is None:
            continue
        b, a = set(before), set(after)
        for u in a - b:
            added_s.append(v)
            added_d.append(u)
        for u in b - a:
            dropped_s.append(v)
            dropped_d.append(u)
    if not added_s:
        print('  no edge changed -- nothing to compare')
    else:
        la = np.linalg.norm(x[np.asarray(added_s)] - x[np.asarray(added_d)], axis=1)
        ld = np.linalg.norm(x[np.asarray(dropped_s)] - x[np.asarray(dropped_d)], axis=1)
        print('  added   n=%7d  mean %8.2f  median %8.2f' % (len(la), la.mean(),
                                                             np.median(la)))
        print('  dropped n=%7d  mean %8.2f  median %8.2f' % (len(ld), ld.mean(),
                                                             np.median(ld)))
        print('  added - dropped        %+8.2f  (negative = policy shortens edges,'
              ' i.e. kNN-ward)' % (la.mean() - ld.mean()))
        print('  vs the whole NSW->kNN span of %.2f' % span)

    print('\n  If the added edges are clearly shorter than the dropped ones, the actor is'
          '\n  applying a fixed feature-space preference regardless of state and the fix'
          '\n  is ARCHITECTURAL (give the actor topology input). If they are comparable,'
          '\n  the decline comes from the unconditional-commit / advantage-normalization'
          '\n  side instead and the fix is ALGORITHMIC.')
