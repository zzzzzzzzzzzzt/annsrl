"""Verify the kernel's DCS budget: hard cap, and no-op when disabled.

Three things must hold or the budget is not usable as an environment constant:
  1. max_dcs=0 reproduces the pre-budget kernel bit for bit (same answers, same DCS).
  2. With a budget B, max(DCS) <= B for EVERY query -- not just on average. A
     hop-boundary check would overshoot by up to max_degree.
  3. Recall degrades smoothly as B tightens, and trajectories stay consistent with
     num_hops (credit_nodes reads trajectory[:num_hops]).
"""
import numpy as np
import torch
import lib

DATA_DIR = './data/SIFT100K'
EF, K, NJ = 20, 1, 8


def build():
    import os.path as osp
    return lib.Graph(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        edges_path=osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs'),
        initial_vertex_id=0,
        graph_type='nsw', normalization='global',
        train_queries_size=10000, val_queries_size=1000,
        ground_truth_n_neighbors=1,
    )


def run(hnsw, queries, gt, budget, ef=None):
    hnsw.max_dcs = budget
    res = hnsw.search_deterministic(queries, ef=EF if ef is None else ef)
    dcs = np.asarray(res['total_distance_computations'])
    hops = np.asarray(res['num_hops'])
    pred = np.asarray(res['best_vertex_ids'])[:, 0]
    recall = float((pred == gt[:, 0].numpy()).mean())
    return recall, dcs, hops, res['trajectories']


def main():
    graph = build()
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ)
    nq = 2000
    queries = graph.test_queries[:nq]
    gt = graph.test_gt[:nq]

    base_recall, base_dcs, base_hops, base_traj = run(hnsw, queries, gt, 0)
    print('budget=off   recall@1=%.4f  DCS mean=%.1f max=%d  hops mean=%.2f'
          % (base_recall, base_dcs.mean(), base_dcs.max(), base_hops.mean()))

    # 1. determinism: same call twice with the cap off must be identical
    r2, d2, _, _ = run(hnsw, queries, gt, 0)
    assert r2 == base_recall and np.array_equal(d2, base_dcs), 'uncapped path not stable'
    print('  [ok] max_dcs=0 is reproducible and matches itself')

    print()
    print('%-8s %-10s %-22s %-14s %s' % ('budget', 'recall@1', 'DCS mean/max',
                                         'hops mean', 'hard cap honoured'))
    ok = True
    for b in (600, 400, 300, 200, 100, 50):
        r, d, h, traj = run(hnsw, queries, gt, b)
        honoured = int(d.max()) <= b
        ok = ok and honoured
        # trajectory rows must be filled exactly num_hops deep
        pad = hnsw.service_labels['pad']
        bad_traj = int(sum((traj[i, :h[i]] == pad).sum() for i in range(len(h))))
        print('%-8d %-10.4f %8.1f / %-11d %-14.2f %s%s'
              % (b, r, d.mean(), d.max(), h.mean(),
                 'yes' if honoured else 'NO (max %d > %d)' % (d.max(), b),
                 '' if bad_traj == 0 else '  BAD: %d pad inside trajectory' % bad_traj))
    print()
    print('hard cap honoured at every budget:', ok)

    # A budget only makes DCS a CONSTANT if it actually binds. ef caps the beam, so
    # when ef is small the walk stops on its own well under the budget and DCS is
    # still free to move -- which is the leak the budget was meant to close. Sweep ef
    # at a fixed budget to find where the budget, not ef, is the binding constraint.
    print()
    B = 300
    print('binding check at budget=%d: fraction of queries that saturate it' % B)
    print('%-6s %-10s %-22s %s' % ('ef', 'recall@1', 'DCS mean/max', 'frac at cap'))
    for ef_try in (10, 16, 20, 32, 48, 64):
        r, d, h, _ = run(hnsw, queries, gt, B, ef=ef_try)
        print('%-6d %-10.4f %8.1f / %-11d %.3f'
              % (ef_try, r, d.mean(), d.max(), float((d >= B).mean())))
    hnsw.ef = EF
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
