"""
Implements a family of rewards. Reward is a callable that:
- takes **session_record - see hnsw.EdgeHNSW.record_sessions or hnsw.ParallelHNSW.record_sessions
- a single number for each action -
"""
import numpy as np
import torch


class MaxDCSReward:
    def __init__(self, max_dcs=1000, k=1, scale=False):
        self.max_dcs = max_dcs
        self.scale = scale
        self.k = k

    def __call__(self, best_vertex_ids, ground_truth_id,
                 total_distance_computations, actions, **etc):
        assert len(best_vertex_ids) >= self.k
        assert len(ground_truth_id) >= self.k

        answers = set(best_vertex_ids[:self.k])
        gts = set(ground_truth_id[:self.k].tolist())
        recall = float(len(answers & gts)) / self.k
        reward = recall * max(self.max_dcs - total_distance_computations, 1)
        if self.scale:
            reward /= self.max_dcs
        return [reward] * len(actions)


class ProximityDCSReward:
    """
    Composite shaped reward: R = R_target + alpha * R_dense + beta * R_cost

    - R_target = 1(v_final == v_gt)                          (recall for k > 1)
        Preserves the hard correctness signal so the optimum stays "reach the true target".
    - R_dense, dense='prox' (DEFAULT) = mean_j dist(q, gt_j) / dist(q, found_j)  in [0, 1]
        The approximation ratio of the ANSWERS, not the shape of the walk. 1.0 means the
        returned neighbours are exactly as close as the true ones (recall is then 1 up to
        distance ties); 0.5 means they are twice as far. This is dense where r_target is
        not: when recall is 0 it still distinguishes "returned a point just outside the
        true top-k" from "returned garbage from across the dataset", so a swap that moves
        the answer closer is rewarded even before it flips a hit. Being a function of the
        answers alone, it cannot be gamed by walking further, and it is monotone in the
        same direction as recall by construction.

    - R_dense, dense='path' = (1 - rho) / 2   in [0, 1]      (path monotonicity, LEGACY)
        Rewards a search path whose distance to the ground-truth node decreases monotonically
        as the walk progresses. We record dist(v_t, v_gt) for every node v_t on the trajectory,
        then score how monotonically-decreasing that sequence is with Spearman's rank
        correlation coefficient rho (image formula, no ties):

            rho = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))

        where d_i is the difference between the rank of the i-th distance and its position index
        i along the path, and n is the path length. A strictly *decreasing* distance sequence
        (the ideal "always getting closer" walk) yields rho = -1; a strictly *increasing* one
        yields rho = +1. Mapping R_dense = (1 - rho) / 2 turns this into [0, 1] where a perfectly
        monotone approach scores 1.0 and a monotone retreat scores 0.0. This shapes the whole
        trajectory rather than only its endpoint, rewarding smooth descent toward the target.
    - R_cost   = -(DCN / DCN_max)                            (step / distance-computation penalty)
        Per-search negative penalty pushing the agent toward the SHORTEST path, not just any
        connected one. Leave beta=0 when the search runs under a hard DCS budget
        (hnsw.ParallelHNSW's max_dcs): DCN is then a constant of the environment and this
        term only reintroduces a recall/DCS exchange rate the budget already removed.

    Distances are euclidean in the feature space of `vertices` -- the same space the search
    runs in, so `vertices` must be the normalized graph.vertices, not the raw file.
    """
    def __init__(self, vertices, max_dcs=1000, k=1, alpha=1.0, beta=1.0, dense='prox'):
        """
        :param vertices: graph.vertices tensor [N, D] (the same tensor the search runs on).
        :param max_dcs: DCN_max, the distance-computation budget used to normalise R_cost.
        :param alpha: weight of the dense term R_dense.
        :param beta: weight of the efficiency penalty term R_cost.
        :param dense: which dense term to use, 'prox' (default) or 'path'.

            'path' is the original Spearman path-monotonicity term described above.
            It is ANTI-ALIGNED with recall and should only be used to reproduce old
            runs. Measured on SIFT100K (probe_dense.py, k=10, ef=32, budget=300):
            within one NSW graph across 4096 queries, corr(r_dense, recall) =
            -0.354 +- 0.015, and hit queries score 0.102 LOWER than missed ones;
            across 40 independent random swap batches, corr(d_r_dense, d_recall) =
            -0.588 +- 0.131. The cause is structural rather than a tuning issue: a
            successful ANN search does not descend monotonically, it reaches the
            answer's neighbourhood and then explores the ef-wide beam, popping
            near-equidistant and sometimes farther candidates, so rho ~ 0. A search
            that walks steadily in one direction and never converges scores higher.

            'prox' scores how close the returned answers are instead of what shape
            the walk had -- see _answer_ratio_batch. It needs the query vectors,
            which is why reward_batch/reward_terms_batch take a `queries` argument.
        """
        self.vertices = vertices
        self.max_dcs = max_dcs
        self.k = k
        self.alpha = alpha
        self.beta = beta
        if dense not in ('prox', 'path'):
            raise ValueError("dense must be 'prox' or 'path', got %r" % (dense,))
        self.dense = dense

    def _dist(self, found_id, gt_id):
        """ euclidean distance between the found node and the true nearest neighbour. """
        return torch.norm(self.vertices[int(found_id)] - self.vertices[int(gt_id)]).item()

    def _path_distances(self, path, gt_id):
        """ distance from every node on the search path to the ground-truth node. """
        gt = self.vertices[int(gt_id)]
        return [torch.norm(self.vertices[int(v)] - gt).item() for v in path]

    @staticmethod
    def _spearman_rho(values):
        """
        Spearman's rank correlation between a sequence's values and their positions.
        rho = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1)), with d_i = rank_i - i.
        Ties are broken with average ranks. Returns None when n < 2 (undefined).
        """
        n = len(values)
        if n < 2:
            return None
        values = np.asarray(values, dtype=np.float64)

        # Average ranks (0..n-1) so equal distances do not create spurious (dis)order.
        order = np.argsort(values, kind='mergesort')
        sorted_vals = values[order]
        ranks_sorted = np.arange(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                ranks_sorted[i:j + 1] = (i + j) / 2.0
            i = j + 1
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = ranks_sorted

        d = ranks - np.arange(n, dtype=np.float64)
        return 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))

    def reward_scalar(self, best_vertex_ids, ground_truth_id,
                      total_distance_computations, path=None, query=None, **etc):
        """
        Performance of ONE search, as a single float (higher is better).

        This is the quantity the per-action `__call__` broadcasts. The graph-edit
        MDP (see lib.algorithm.GraphEditPPO) needs the scalar directly, since it
        has no notion of per-action credit inside a session -- its actions are
        per-node edge swaps, not per-edge keep/drop decisions.
        """
        assert len(best_vertex_ids) >= self.k
        assert len(ground_truth_id) >= self.k

        answers = set(int(v) for v in best_vertex_ids[:self.k])
        gts = set(int(v) for v in ground_truth_id[:self.k])
        r_target = float(len(answers & gts)) / self.k

        if self.dense == 'prox':
            if query is None:
                raise ValueError("dense='prox' needs `query`, the vector this search "
                                 "ran on, or construct with dense='path'")
            # One-row reuse of the batch implementation, so the scalar and batch paths
            # cannot drift apart in their edge-case handling (missing answers, d=0).
            r_dense = float(self._answer_ratio_batch(
                np.asarray(best_vertex_ids)[None, :],
                np.asarray(ground_truth_id)[None, :],
                torch.as_tensor(query, dtype=self.vertices.dtype)[None, :])[0])
        else:
            # Monotonicity of the path's distance-to-gt: rho = -1 for a strictly decreasing
            # (always-approaching) walk, mapped to R_dense = 1.0; +1 (retreating) -> 0.0.
            rho = None
            if path is not None:
                dists = self._path_distances(path, ground_truth_id[0])
                rho = self._spearman_rho(dists)
            # Undefined monotonicity (path shorter than 2 nodes) -> neutral 0.5.
            r_dense = 0.5 if rho is None else (1.0 - rho) / 2.0

        r_cost = -(float(total_distance_computations) / self.max_dcs)

        return r_target + self.alpha * r_dense + self.beta * r_cost

    def cost_scalar(self, **rec):
        """
        Search COST c(.) of one search -- the negation of reward_scalar.

        framework.md step 6 defines the RL reward as r_t = c(s_t) - c(s_{t+1}),
        i.e. cost must DECREASE for the reward to be positive. reward_scalar is
        a performance (higher is better), so the cost is simply its negation and

            r_t = c(s_t) - c(s_{t+1}) = R(s_{t+1}) - R(s_t)

        is the *improvement* in performance produced by the edge swap. Keep this
        sign convention in mind when changing either function.
        """
        return -self.reward_scalar(**rec)

    @torch.no_grad()
    def _batch_path_rho(self, trajectories, num_hops, gt_ids, chunk=512):
        """
        Vectorized Spearman rho of dist(v_t, v_gt) along each query's path.

        :param trajectories: int array [nq, max_path], -1 padded
        :param num_hops: int array [nq], number of valid entries per row
        :param gt_ids: int array [nq], the ground-truth node of each query
        :return: float array [nq] of rho, NaN where the path is shorter than 2

        Unlike the scalar `_spearman_rho`, ties are NOT averaged here: exact ties
        between two real distances are a measure-zero event on float coords, and
        padded slots are pushed to +inf so they always rank last and are excluded
        from the d_i sum by the validity mask.
        """
        nq, max_path = trajectories.shape
        positions = torch.arange(max_path, dtype=torch.float64)
        out = np.full(nq, np.nan, dtype=np.float64)

        traj = torch.as_tensor(trajectories, dtype=torch.long)
        hops = torch.as_tensor(num_hops, dtype=torch.long)
        gts = torch.as_tensor(gt_ids, dtype=torch.long)

        for start in range(0, nq, chunk):
            end = min(start + chunk, nq)
            t = traj[start:end]
            n = hops[start:end]
            valid = torch.arange(max_path)[None, :] < n[:, None]

            # dist(v_t, v_gt) for every slot; padded slots -> +inf (rank last)
            nodes = self.vertices[t.clamp(min=0)]                  # [c, max_path, D]
            gt_vecs = self.vertices[gts[start:end]][:, None, :]     # [c, 1, D]
            dist = torch.norm(nodes - gt_vecs, dim=-1).double()
            dist = torch.where(valid, dist, torch.full_like(dist, float('inf')))

            # ranks: rank of each slot among its row's distances
            order = dist.argsort(dim=-1, stable=True)
            ranks = torch.empty_like(dist)
            ranks.scatter_(-1, order, positions.expand_as(dist).contiguous())

            d = torch.where(valid, ranks - positions[None, :], torch.zeros_like(ranks))
            nf = n.double()
            denom = nf * (nf * nf - 1.0)
            rho = 1.0 - (6.0 * (d ** 2).sum(-1)) / denom
            rho[n < 2] = float('nan')
            out[start:end] = rho.numpy()
        return out

    @torch.no_grad()
    def _answer_ratio_batch(self, best_vertex_ids, ground_truth_ids, queries, chunk=2048):
        """
        Approximation ratio of the returned answers: mean_j d(q, gt_j) / d(q, found_j).

        In [0, 1], higher is better, and 1.0 iff every answer is exactly as close as the
        true j-th neighbour. Unlike the path-monotonicity term this depends only on WHAT
        the search returned, never on how it got there, so it cannot be raised by walking
        further -- and it is dense below recall = 0, which is the regime a random s_0
        starts in (measured recall@10 = 0.003).

        Edge cases, all of which occur on real data:
        The j-th answer is paired with the j-th true neighbour after both distance lists
        are sorted ascending, which makes the term ORDER-INVARIANT: recall@k is a set
        metric, so returning the correct k points in a different order must not be
        penalised. Without the sort, answers [gt_1, gt_0] score 0.5 while recall is 1.0
        -- exactly the kind of anti-alignment this term exists to remove. In practice the
        kernel already returns answers ascending by distance, so the sort is a guard
        against relying on that rather than a correction.

        Edge cases, all of which occur on real data:
          - found_j = -1 (fewer than k answers survived the beam) -> ratio 0, the worst
            possible score. Treating a missing answer as neutral would let a graph that
            returns nothing outscore one that returns something far away.
          - d(q, found_j) = 0, i.e. the query IS a base point that was found: the ideal
            distance is 0 too, so the ratio is defined as 1.0 rather than 0/0.
          - d_gt > d_found by a hair. The ground truth is exact, so this only comes from
            float rounding or exact distance ties; clamped to 1.0 so the term stays a
            ratio and ties cannot push it above the perfect score.
        """
        k = self.k
        ans = torch.as_tensor(np.asarray(best_vertex_ids)[:, :k], dtype=torch.long)
        gts = torch.as_tensor(np.asarray(ground_truth_ids)[:, :k], dtype=torch.long)
        q_all = torch.as_tensor(queries, dtype=self.vertices.dtype)
        nq = q_all.shape[0]
        out = np.empty(nq, dtype=np.float64)

        for start in range(0, nq, chunk):
            end = min(start + chunk, nq)
            q = q_all[start:end][:, None, :]                       # [c, 1, D]
            a = ans[start:end]
            found = a >= 0
            # clamp(min=0) keeps the gather legal for pad slots; `found` masks them after.
            d_found = torch.norm(self.vertices[a.clamp(min=0)] - q, dim=-1).double()
            d_gt = torch.norm(self.vertices[gts[start:end]] - q, dim=-1).double()

            # Missing answers sort last at +inf, so a real answer is never paired with
            # the gt slot a pad slot would have taken.
            d_found = torch.where(found, d_found, torch.full_like(d_found, float('inf')))
            d_found, _ = d_found.sort(dim=-1)
            d_gt, _ = d_gt.sort(dim=-1)
            found = d_found.isfinite()

            ratio = torch.where(d_found > 0, d_gt / torch.clamp(d_found, min=1e-12),
                                torch.ones_like(d_found))
            ratio = ratio.clamp(max=1.0)
            ratio = torch.where(found, ratio, torch.zeros_like(ratio))
            out[start:end] = ratio.mean(-1).numpy()
        return out

    @torch.no_grad()
    def reward_batch(self, best_vertex_ids, ground_truth_ids,
                     total_distance_computations, trajectories=None, num_hops=None,
                     queries=None):
        """
        Vectorized `reward_scalar` over a whole probe batch -> float array [nq].

        Used by the graph-edit MDP, which evaluates the same probe batch twice per
        iteration (before and after the edge swap); the per-query Python loop in
        `reward_scalar` / `_path_distances` would dominate the step time there.

        :param best_vertex_ids: int array [nq, >=k] search answers
        :param ground_truth_ids: int array [nq, >=k] true neighbors
        :param total_distance_computations: int array [nq]
        :param trajectories: int array [nq, max_path] (-1 padded); dense='path' only
        :param num_hops: int array [nq], required when trajectories is given
        :param queries: float array/tensor [nq, D], required when dense='prox'
        """
        return self.reward_terms_batch(
            best_vertex_ids=best_vertex_ids, ground_truth_ids=ground_truth_ids,
            total_distance_computations=total_distance_computations,
            trajectories=trajectories, num_hops=num_hops, queries=queries)['total']

    @torch.no_grad()
    def reward_terms_batch(self, best_vertex_ids, ground_truth_ids,
                           total_distance_computations, trajectories=None, num_hops=None,
                           queries=None):
        """
        Same as reward_batch, but returns each term separately for diagnostics.

        R = r_target + alpha * r_dense + beta * r_cost, and the three terms pull in
        different directions once the *graph itself* is the action space: r_cost
        rises when the search does fewer distance computations, which a worse graph
        achieves by terminating early. Logging the terms separately is the cheapest
        way to see whether the agent is improving recall or just shortening walks.

        :return: dict of float arrays [nq] with keys
                 'r_target', 'r_dense', 'r_cost' (unweighted) and 'total' (weighted sum)
        """
        if self.dense == 'prox' and queries is None:
            # Silently falling back to a constant would make r_dense a no-op and the
            # reward would quietly become r_target alone -- a binary signal that is
            # zero on 99.7% of queries from a random s_0.
            raise ValueError("dense='prox' needs `queries`; pass the same query batch "
                             "the search ran on, or construct with dense='path'")
        answers = torch.as_tensor(np.asarray(best_vertex_ids)[:, :self.k], dtype=torch.long)
        gts = torch.as_tensor(np.asarray(ground_truth_ids)[:, :self.k], dtype=torch.long)
        # |answers ∩ gts| / k, computed as a k x k match count (k is small)
        matches = (answers[:, :, None] == gts[:, None, :]).any(-1).sum(-1)
        r_target = matches.double().numpy() / self.k

        if self.dense == 'prox':
            r_dense = self._answer_ratio_batch(best_vertex_ids, ground_truth_ids, queries)
        elif trajectories is None:
            r_dense = np.full(len(r_target), 0.5, dtype=np.float64)
        else:
            rho = self._batch_path_rho(trajectories, num_hops,
                                       np.asarray(ground_truth_ids)[:, 0])
            r_dense = np.where(np.isnan(rho), 0.5, (1.0 - rho) / 2.0)

        r_cost = -(np.asarray(total_distance_computations, dtype=np.float64) / self.max_dcs)
        return dict(r_target=r_target, r_dense=r_dense, r_cost=r_cost,
                    total=r_target + self.alpha * r_dense + self.beta * r_cost)

    @torch.no_grad()
    def cost_batch(self, **rec):
        """ Search cost c(.) for a whole probe batch -- negation of reward_batch. """
        return -self.reward_batch(**rec)

    def __call__(self, best_vertex_ids, ground_truth_id,
                 total_distance_computations, actions, path=None, query=None, **etc):
        # `query` comes from the session record (BaseAlgorithm.get_session_batch sets
        # rec['query']) and must be forwarded, or dense='prox' raises on the per-edge
        # TRPO/PPO path even though the caller did supply it.
        reward = self.reward_scalar(
            best_vertex_ids=best_vertex_ids, ground_truth_id=ground_truth_id,
            total_distance_computations=total_distance_computations, path=path,
            query=query)
        return [reward] * len(actions)


class RecallReward:
    def __call__(self, best_vertex_id, ground_truth_id, actions, **etc):
        recall = int(best_vertex_id == ground_truth_id[0])
        return [recall] * len(actions)


class WeightedRecallReward:
    def __init__(self, decay=0.5):
        self.decay = decay

    def __call__(self, best_vertex_id, ground_truth_id, actions, **etc):
        recall = 0.
        for i, gt in enumerate(ground_truth_id):
            if gt == best_vertex_id:
                recall = self.decay ** i
                break
        return [recall] * len(actions)
