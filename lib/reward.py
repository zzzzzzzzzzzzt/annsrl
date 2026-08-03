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
    - R_dense  = (1 - rho) / 2   in [0, 1]                   (path monotonicity)
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
        connected one.

    dist(v_t, v_gt) is the euclidean distance between a visited node and the true nearest
    neighbour in feature space (both looked up from `vertices`).
    """
    def __init__(self, vertices, max_dcs=1000, k=1, alpha=1.0, beta=1.0):
        """
        :param vertices: graph.vertices tensor [N, D] (the same tensor the search runs on).
        :param max_dcs: DCN_max, the distance-computation budget used to normalise R_cost.
        :param alpha: weight of the dense monotonicity term R_dense.
        :param beta: weight of the efficiency penalty term R_cost.
        """
        self.vertices = vertices
        self.max_dcs = max_dcs
        self.k = k
        self.alpha = alpha
        self.beta = beta

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

    def __call__(self, best_vertex_ids, ground_truth_id,
                 total_distance_computations, actions, path=None, **etc):
        assert len(best_vertex_ids) >= self.k
        assert len(ground_truth_id) >= self.k

        answers = set(best_vertex_ids[:self.k])
        gts = set(ground_truth_id[:self.k].tolist())
        r_target = float(len(answers & gts)) / self.k

        # Monotonicity of the path's distance-to-gt: rho = -1 for a strictly decreasing
        # (always-approaching) walk, mapped to R_dense = 1.0; +1 (retreating) -> 0.0.
        rho = None
        if path is not None:
            dists = self._path_distances(path, ground_truth_id[0])
            rho = self._spearman_rho(dists)
        # Undefined monotonicity (path shorter than 2 nodes) -> neutral 0.5.
        r_dense = 0.5 if rho is None else (1.0 - rho) / 2.0

        r_cost = -(float(total_distance_computations) / self.max_dcs)

        reward = r_target + self.alpha * r_dense + self.beta * r_cost
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
