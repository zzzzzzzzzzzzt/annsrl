"""Guess 2: does r_dense actually align with recall?

r_dense is 99.6% of the reward once beta=0 (measured: r_dense 0.6356 vs r_target 0.0024),
so if it does not point the same way as recall then it, not gamma, is the root cause of
the gate failure. r_dense = (1 - rho)/2 where rho is the Spearman correlation of
dist(v_t, gt) against position along the walk, so high r_dense means "the walk moved
steadily toward the answer".

Three independent tests, weakest to strongest:
  1. across a quality ladder of graphs (random -> nsw): does r_dense rise with recall?
  2. across queries within one graph: do the queries with high r_dense hit?
  3. across single swap batches from one s_0: does d_r_dense agree in SIGN with
     d_recall? This is the decisive one -- it is exactly the signal credit_nodes
     turns into a per-node reward, so a near-zero or negative correlation here means
     the policy gradient points somewhere unrelated to recall.

Test 3 also scores a disjoint held-out query set every batch, which answers a prior
question: whether the probe set can support a reward at all. The random s_0 fails it --
d_recall there has mean -6.1e-07 against std 1.5e-05, so both terms' correlations are
noise and their sign agreement falls below chance. That is why 'knn' is in GRAPHS.
"""
import numpy as np
import os.path as osp
import torch
import lib

DATA_DIR = './data/SIFT100K'
NQ = 4096            # trainer's probe_size, so per-batch deltas are the same size
NJ = 32
EF, K = 32, 10       # k=10: guess 1 measured ~3x the signal/noise of k=1 at equal cost
BUDGET = 300
# s_0 candidates, weakest first. 'random' has no measurable per-batch signal (measured
# d_recall mean -6.1e-07 against std 1.5e-05), which is why 'knn' is here: a pure kNN
# graph is locally strong but globally unnavigable, so it should have both signal and
# headroom -- the thing to learn is trading a few local edges for long-range links.
GRAPHS = ('random', 'knn', 'nsw')
INIT_DEGREE = 24


def build(graph_type, seed=1234):
    kw = dict(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        initial_vertex_id=0, graph_type=graph_type, normalization='global',
        # train_queries_size is 2*NQ so val takes NQ off the tail and NQ is left in
        # train -- that leftover is the held-out set test 3 uses.
        train_queries_size=NQ * 2, val_queries_size=NQ,
        ground_truth_n_neighbors=100,
    )
    if graph_type == 'nsw':
        kw['edges_path'] = osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs')
    else:
        kw['edges_path'] = None
        kw['init_degree'] = INIT_DEGREE
        kw['init_seed'] = seed
        if graph_type == 'knn':
            # Same cache path the trainer uses, so this shares the file with real
            # runs; an exact kNN over the whole base set is the expensive part.
            kw['knn_cache_path'] = osp.join(
                DATA_DIR, 'knn_cache', 'init_knn_k%d.npz' % (INIT_DEGREE + 1))
            kw['knn_n_jobs'] = NJ
    return lib.Graph(**kw)


def make_reward(graph, dense):
    return lib.ProximityDCSReward(graph.vertices, k=K, max_dcs=1000,
                                  alpha=1.0, beta=0.0, dense=dense)


def measure(hnsw, rewards, queries, gt):
    """One search, scored by every reward in `rewards`.

    Returns (recall_per_query, {name: r_dense_per_query}, raw_result). Scoring every
    variant off the SAME search is both the fair comparison -- prox and path then differ
    only in the scoring function, not in the searches they saw -- and half the cost,
    which is what pays for test 3's held-out set.
    """
    res = hnsw.search_deterministic(queries)
    kw = dict(best_vertex_ids=res['best_vertex_ids'], ground_truth_ids=gt,
              total_distance_computations=res['total_distance_computations'],
              trajectories=res['trajectories'], num_hops=res['num_hops'],
              queries=queries)
    rt, dense = None, {}
    for name, reward in rewards.items():
        terms = reward.reward_terms_batch(**kw)
        # r_target IS recall@k per query (|answers n gts| / k) and does not depend on
        # `dense`, so whichever variant computes it, it is the same vector.
        rt = terms['r_target']
        dense[name] = terms['r_dense']
    return rt, dense, res


def pearson(x, y):
    """Correlation plus its ~SE; returns nan when either side is constant."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan'), float('nan')
    r = float(np.corrcoef(x, y)[0, 1])
    se = float(np.sqrt(max(1.0 - r * r, 0.0) / max(len(x) - 2, 1)))
    return r, se


def test1_ladder():
    """Across graphs of different quality: does r_dense rise together with recall?"""
    print('== test 1: quality ladder (weak evidence -- %d points, no controls) =='
          % len(GRAPHS))
    print('%-10s %-12s %-12s %-12s %-10s'
          % ('graph', 'recall@%d' % K, 'prox', 'path', 'edge_len'))
    for graph_type in GRAPHS:
        graph = build(graph_type)
        hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_dcs=BUDGET,
                                 max_trajectory=400)
        rewards = {d: make_reward(graph, d) for d in ('prox', 'path')}
        rt, rd, _ = measure(hnsw, rewards, graph.val_queries[:NQ], graph.val_gt[:NQ])
        print('%-10s %-12.4f %-12.4f %-12.4f %-10.4f'
              % (graph_type, rt.mean(), rd['prox'].mean(), rd['path'].mean(),
                 hnsw.graph_stats()['edge_len_mean']))


def test2_within_graph():
    """Across queries on ONE graph: do high-r_dense queries hit more often?

    Sign convention: both terms are "higher is better", so a POSITIVE correlation
    means the term agrees with recall. For 'path', r_dense = (1 - rho)/2 and rho is
    the correlation of dist(v_t, gt) with position, so a walk that closes in on the
    answer has rho < 0 and scores above 0.5.
    """
    print()
    print('== test 2: within one graph, across queries ==')
    print('%-10s %-7s %-22s %-14s %s'
          % ('graph', 'dense', 'corr(term, recall)', 'hit - miss', 'verdict'))
    for graph_type in GRAPHS:
        graph = build(graph_type)
        hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_dcs=BUDGET,
                                 max_trajectory=400)
        rewards = {d: make_reward(graph, d) for d in ('prox', 'path')}
        rt, dense_vals, _ = measure(hnsw, rewards,
                                    graph.val_queries[:NQ], graph.val_gt[:NQ])
        hit = rt > 0
        for dense, rd in dense_vals.items():
            r, se = pearson(rd, rt)
            gap = (rd[hit].mean() - rd[~hit].mean()) \
                if hit.any() and (~hit).any() else float('nan')
            print('%-10s %-7s %-22s %-14s %s'
                  % (graph_type, dense, '%+.4f +- %.4f' % (r, se), '%+.4f' % gap,
                     'aligned' if r > 3 * se else
                     ('ANTI-aligned' if r < -3 * se else 'no relation')))


def random_swap_batch(hnsw, node_ids, n_swap, rng):
    """A uniformly random bounded swap on `node_ids`, same shape the policy produces.

    Drops are uniform over each node's current neighbours, additions uniform over its
    2-hop candidates -- i.e. exactly run B of the gate. Nodes with no candidates are
    dropped from the batch.
    """
    cand_ids, cand_mask = hnsw.get_candidates(node_ids)
    keep = cand_mask.any(1)
    node_ids, cand_ids, cand_mask = node_ids[keep], cand_ids[keep], cand_mask[keep]

    pad = hnsw.service_labels['pad']
    adj = hnsw.adj[node_ids]
    drops = np.zeros((len(node_ids), n_swap), np.int64)
    adds = np.zeros((len(node_ids), n_swap), np.int64)
    for i in range(len(node_ids)):
        nbrs = adj[i][adj[i] != pad].astype(np.int64)
        pool = cand_ids[i][cand_mask[i]]
        # replace=True is fine: apply_swaps skips a repeat as a no-op (it checks
        # `drop not in current`), so a collision costs one swap, not correctness.
        drops[i] = rng.choice(nbrs, size=n_swap, replace=len(nbrs) < n_swap)
        adds[i] = rng.choice(pool, size=n_swap, replace=len(pool) < n_swap)
    return node_ids, drops, adds


def report_pair(label, d_x, d_y):
    """corr + sign agreement of two delta series, against their chance floor."""
    r, se = pearson(d_x, d_y)
    # Sign agreement has a floor: with frac>0 = p and q on the two sides,
    # independent signs already agree pq + (1-p)(1-q) of the time. Reporting
    # the baseline next to it stops 0.20 from looking like weak-but-positive.
    p, q = (d_x > 0).mean(), (d_y > 0).mean()
    chance = p * q + (1 - p) * (1 - q)
    agree = float(np.mean(np.sign(d_x) == np.sign(d_y)))
    print('    %-26s %+.4f +- %.4f | sign agree %.2f (chance %.2f) -> %s'
          % (label, r, se, agree, chance,
             'ALIGNED' if r > 3 * se else
             ('ANTI-ALIGNED' if r < -3 * se else 'NO RELATION')))
    return r, se


def test3_per_batch(n_batches=40, nodes_per_step=512, n_swap=2, seed=0):
    """The decisive test: across single swap batches, does d_r_dense track d_recall?

    This is the quantity credit_nodes actually distributes to nodes, so its sign
    agreement with recall is what decides whether the policy gradient points at
    recall at all. Each batch is applied, measured, then rolled back, so all
    batches are independent rollouts from the same s_0.

    Two query sets are searched per batch: the PROBE set (the NQ queries the trainer
    would score its reward on) and a disjoint HELD-OUT set. That extra search buys the
    one number that decides whether a gate on this s_0 can be trusted at all --
    corr(d_recall_probe, d_recall_holdout). If a batch's measured probe gain does not
    predict its gain on unseen queries, then no reward read off the probe set can
    either, however well-designed, and the alignment numbers below are measuring
    agreement with probe-set noise rather than with recall.
    """
    print()
    print('== test 3: per swap batch from one s_0 (decisive) ==')
    rng = np.random.default_rng(seed)
    for graph_type in GRAPHS:
        graph = build(graph_type)
        hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_dcs=BUDGET,
                                 max_trajectory=400)
        rewards = {d: make_reward(graph, d) for d in ('prox', 'path')}
        # val_queries is the trainer's probe set; the NQ left in train_queries after
        # the val split are disjoint from it and never scored by the reward.
        probe = (graph.val_queries[:NQ], graph.val_gt[:NQ])
        hold = (graph.train_queries[:NQ], graph.train_gt[:NQ])

        rt_p, rd_p, _ = measure(hnsw, rewards, *probe)
        rt_h, _, _ = measure(hnsw, rewards, *hold)
        base = {'recall': rt_p.mean(), 'recall_hold': rt_h.mean()}
        base.update({d: v.mean() for d, v in rd_p.items()})

        keys = ('recall', 'recall_hold', 'prox', 'path')
        deltas = {key: [] for key in keys}
        for _ in range(n_batches):
            nodes = rng.choice(hnsw.num_vertices, size=nodes_per_step, replace=False)
            nodes, drops, adds = random_swap_batch(hnsw, nodes, n_swap, rng)
            undo = hnsw.snapshot(nodes)
            hnsw.apply_swaps(nodes, drops, adds)
            rt_p, rd_p, _ = measure(hnsw, rewards, *probe)
            rt_h, _, _ = measure(hnsw, rewards, *hold)
            hnsw.restore(undo)
            deltas['recall'].append(rt_p.mean() - base['recall'])
            deltas['recall_hold'].append(rt_h.mean() - base['recall_hold'])
            for d in ('prox', 'path'):
                deltas[d].append(rd_p[d].mean() - base[d])
        deltas = {key: np.array(val) for key, val in deltas.items()}

        d_rt, d_rt_h = deltas['recall'], deltas['recall_hold']
        print('--- s_0 = %s (recall@%d probe %.4f / hold %.4f | prox %.4f path %.4f) ---'
              % (graph_type, K, base['recall'], base['recall_hold'],
                 base['prox'], base['path']))
        for name, arr in (('d_recall(probe)', d_rt), ('d_recall(hold) ', d_rt_h)):
            # |mean| / (std/sqrt(n)) is how many sigma the average batch effect stands
            # above the measurement noise of this probe size. Below ~1 there is nothing
            # for any reward to latch onto.
            snr = abs(arr.mean()) / max(arr.std() / np.sqrt(len(arr)), 1e-30)
            print('  %s mean %+.2e  std %.2e  frac>0 %.2f  |mean|/SE %.2f'
                  % (name, arr.mean(), arr.std(), (arr > 0).mean(), snr))
        print('  [is the probe set trustworthy at all?]')
        report_pair('corr(probe, hold)', d_rt, d_rt_h)
        for d in ('prox', 'path'):
            d_rd = deltas[d]
            print('  d_%-6s mean %+.2e  std %.2e  frac>0 %.2f'
                  % (d, d_rd.mean(), d_rd.std(), (d_rd > 0).mean()))
            report_pair('corr(d_%s, d_recall)' % d, d_rd, d_rt)
            # vs held-out recall is the honest test of the reward term: it cannot be
            # inflated by the term and the recall reading sharing probe-set noise.
            report_pair('corr(d_%s, d_hold)' % d, d_rd, d_rt_h)


def main():
    test1_ladder()
    test2_within_graph()
    test3_per_batch()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
