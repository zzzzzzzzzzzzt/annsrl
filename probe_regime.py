"""Guess 1: is there an operating point where recall on the random graph is measurable?

The fixed-cost gate FAILED with val recall@1 ~= 0.0033 on 20000 queries, i.e. ~66 hits.
The binomial noise floor there is sqrt(p(1-p)/n) = 4.1e-4, so the +-7e-4 movements the
gate was asked to judge are ~1.7 sigma -- marginal, not decisive. This asks whether a
different (budget, k) makes the signal bigger than the noise, reporting recall together
with its standard error and the ratio of the two.

Run this AFTER the build_random_edges fix: the old s_0 left 35% of nodes with in-degree
0, which caps recall no matter what the budget is.
"""
import numpy as np
import os.path as osp
import torch
import lib

DATA_DIR = './data/SIFT100K'
NQ = 20000          # same as the trainer's val set, so numbers are comparable
NJ = 32
INIT_DEGREE = 24


def build(graph_type, seed=1234):
    kw = dict(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        initial_vertex_id=0, graph_type=graph_type, normalization='global',
        train_queries_size=NQ, val_queries_size=NQ // 2,
        ground_truth_n_neighbors=100,
    )
    if graph_type == 'nsw':
        kw['edges_path'] = osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs')
    else:
        kw['edges_path'] = None
        kw['init_degree'] = INIT_DEGREE
        kw['init_seed'] = seed
    return lib.Graph(**kw)


def recall_at(pred, gt, k):
    """Fraction of the top-k ground truth found, averaged over queries, + its SE.

    gt arrives as a torch tensor; numpy == torch broadcasts to a scalar bool rather
    than an elementwise array, so it must be converted first.
    """
    gt = gt.numpy() if torch.is_tensor(gt) else np.asarray(gt)
    if k == 1:
        hit = (pred[:, 0] == gt[:, 0]).astype(np.float64)
    else:
        hit = np.array([len(set(p[:k]) & set(g[:k])) / k for p, g in zip(pred, gt)])
    return hit.mean(), hit.std(ddof=1) / np.sqrt(len(hit))


def main():
    print('== guess 1: where is recall measurable on the random graph? ==')
    print('NQ=%d, degree=%d. SE is the binomial/sample standard error of recall;' % (NQ, INIT_DEGREE))
    print('a gate needs the effect it is judging to exceed a few SE.')

    for graph_type in ('random', 'nsw'):
        graph = build(graph_type)
        queries = graph.val_queries[:NQ]
        gt = graph.val_gt[:NQ]
        print()
        print('--- s_0 = %s ---' % graph_type)
        # max_trajectory must exceed the hop count a loose budget allows, or the walk
        # is truncated by the trajectory buffer instead of by the budget.
        hnsw = lib.GraphEditHNSW(graph, ef=32, k=1, n_jobs=NJ, max_trajectory=400)
        print('reachable_frac=%.4f degree_mean=%.1f edge_len=%.4f'
              % tuple(hnsw.graph_stats()[key] for key in
                      ('reachable_frac', 'degree_mean', 'edge_len_mean')))
        print('%-9s %-6s %-6s %-18s %-8s %s'
              % ('budget', 'k', 'ef', 'recall@k +- SE', 'DCS', 'recall / SE'))
        # ef must be >= k or the kernel's ef-sized result heap cannot hold k answers
        # and best_vertex_ids comes back mis-ordered -- every order-sensitive metric
        # then reads exactly 0. Retrieving k answers inherently needs a beam that wide,
        # so k and ef cannot be swept independently.
        for budget in (300, 1000, 3000, 10000, 0):
            for k in (1, 10, 100):
                ef = max(32, k)
                hnsw.k = k
                hnsw.max_dcs = budget
                res = hnsw.search_deterministic(queries, ef=ef)
                dcs = np.asarray(res['total_distance_computations'])
                pred = np.asarray(res['best_vertex_ids'])
                r, se = recall_at(pred, gt, k)
                print('%-9s %-6d %-6d %-18s %-8.0f %s'
                      % (budget if budget else 'off', k, ef,
                         '%.4f +- %.4f' % (r, se), dcs.mean(),
                         '%.0f' % (r / se) if se else '-'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
