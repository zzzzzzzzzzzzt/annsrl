from heapq import heappush, heappop, nlargest, nsmallest
import numpy as np
import torch

import multiprocessing
from .search_hnsw_swig import search_hnsw


class HNSW:
    def __init__(self, graph, ef=1):
        """ Main class that handles approximate nearest neighbor search using HNSW heap-based search algorithm.
            :param graph: graph on which the search algorithm is performed
            :param ef: regulates the search algorithm "greediness"
        """
        self.graph = graph
        self.ef = ef

    def get_enterpoint(self, query, **kwargs):
        vertex_id = self.get_initial_vertex_id(**kwargs)
        curdist = self.get_distance(query, self.graph.vertices[vertex_id])

        for level in range(self.graph.max_level)[::-1]:
            changed = True
            while changed:
                changed = False
                edges = list(self.graph.level_edges[vertex_id][level])
                if len(edges) == 0:
                    break

                distances = self.get_distance(query, self.graph.vertices[edges])
                for edge, dist in zip(edges, distances):
                    if dist < curdist:
                        curdist = dist
                        vertex_id = edge
                        changed = True
        return vertex_id

    def find_nearest(self, query, **kwargs):
        """
        Performs nearest neighbor lookup and returns statistics.
        :param query: vector [vertex_size] to find nearest neighbor for
        :return: nearest neighbor vertex id
        """
        if self.graph.max_level == 0:
            vertex_id = self.get_initial_vertex_id(**kwargs)
        else:
            vertex_id = self.get_enterpoint(query, **kwargs)
            self.start_session()

        visited_ids = {vertex_id}  # a set of vertices already visited by graph walker

        topResults, candidateSet = [], []
        distance = self.get_distance(query, self.graph.vertices[vertex_id])
        heappush(topResults, (-distance, vertex_id))
        heappush(candidateSet, (distance, vertex_id))
        lowerBound = distance

        while len(candidateSet) > 0:
            dist, vertex_id = heappop(candidateSet)
            if dist > lowerBound: break

            neighbor_ids = self.get_neighbors(vertex_id, visited_ids, **kwargs)
            if not len(neighbor_ids): continue

            distances = self.get_distance(query, self.graph.vertices[neighbor_ids])
            for i, (distance, neighbor_id) in enumerate(zip(distances, neighbor_ids)):
                if distance < lowerBound or len(topResults) < self.ef:
                    heappush(candidateSet, (distance, neighbor_id))
                    heappush(topResults, (-distance, neighbor_id))

                    if len(topResults) > self.ef:
                        heappop(topResults)

                    lowerBound = -nsmallest(1, topResults)[0][0]

            visited_ids.update(neighbor_ids)

        best_neighbor_id = nlargest(1, topResults)[0][1]
        return best_neighbor_id

    def start_session(self):
        """ Resets all logs """
        self._distance_computations = []  # number of times distance was evaluated at each step

    def get_initial_vertex_id(self, **kwargs):
        return self.graph.initial_vertex_id

    def get_neighbors(self, vertex_id, visited_ids, **kwargs):
        """ :return: a list of neighbor ids available from given vector_id. """
        neighbors = [edge for edge in self.graph.edges[vertex_id]
                     if edge not in visited_ids]
        return neighbors

    def get_distance(self, vector, vector_or_vectors):
        if len(vector_or_vectors.shape) == 1:
            self._distance_computations.append(1)
        else:
            self._distance_computations.append(vector_or_vectors.shape[0])
        return ((vector - vector_or_vectors) ** 2).sum(-1)


class ParallelHNSW(HNSW):
    def __init__(self, graph, k=1, ef=1, max_trajectory=80, batch_size=500000,
                 edge_patience=100, n_jobs=1, n_hop_virtual=0):
        """ Optimized EdgeHNSW for fast session sampling. Uses wrapped C++ code for HNSW search algorithm
            :param max_trajectory: maximum expected number of hops. Needed for swig.
                   The larger ef is, the larger max_trajectory you should set
            :param batch_size: number of edges in batch
            :param n_jobs: number of threads for C++ session sampling
            :param n_hop_virtual: number of 2-hop virtual edge slots per vertex
        """
        super().__init__(graph, ef)
        assert graph.graph_type != 'hnsw', 'hnsw.ParallelHNSW does not support hierarchy. Use hnsw.EdgeHNSW.'

        self.k = k
        self.max_trajectory = max_trajectory
        self.n_jobs = self._check_n_jobs(n_jobs)

        self.batch_size = batch_size
        self.edge_patience = edge_patience
        self.n_hop_virtual = n_hop_virtual
        self.effective_max_degree = graph.max_degree + n_hop_virtual

        # Service labels to denote service fields and boundary situations
        self.service_labels = {'pad': -1, 'unused': -2, 'no_actions': -3}

        # Mutable topology: {from_id: [to_id, ...]} — evolves as virtual edges get promoted
        self.dynamic_edges = {v: list(neighbors) for v, neighbors in graph.edges.items()}

        # Edge confidence dict: {(from_id, to_id): float}
        self.edge_confidence = {}
        for v, neighbors in graph.edges.items():
            for nb in neighbors:
                self.edge_confidence[(int(v), int(nb))] = 0.0

        # Cache of last virtual edge scores: {from_id: [(to_id, prob), ...]} sorted descending
        self._last_virtual_scores = {}

        # Pre-sampled virtual edge candidates (fixed at init): {from_id: [to_id, ...]}
        self._virtual_candidates = {}

        # Flat batch arrays rebuilt from dynamic_edges
        self.num_edges = 0
        self.from_vertex_ids, self.to_vertex_ids, self.degrees = [], [], []
        self._build_batches()
        self._init_virtual_candidates()

    def _build_batches(self):
        """Rebuild flat batch arrays from current dynamic_edges."""
        self.from_vertex_ids, self.to_vertex_ids, self.degrees = [], [], []
        chunk_from, chunk_to, chunk_deg = [], [], []
        total = 0
        for vertex_id, neighbors in self.dynamic_edges.items():
            degree = len(neighbors)
            if degree == 0:
                continue
            chunk_from.extend([vertex_id] * degree)
            chunk_to.extend(neighbors)
            chunk_deg.append(degree)
            if sum(chunk_deg) >= self.batch_size:
                total += sum(chunk_deg)
                self.from_vertex_ids.append(np.array(chunk_from, dtype=np.int64))
                self.to_vertex_ids.append(np.array(chunk_to, dtype=np.int64))
                self.degrees.append(np.array(chunk_deg, dtype=np.int64))
                chunk_from, chunk_to, chunk_deg = [], [], []
        if chunk_deg:
            total += sum(chunk_deg)
            self.from_vertex_ids.append(np.array(chunk_from, dtype=np.int64))
            self.to_vertex_ids.append(np.array(chunk_to, dtype=np.int64))
            self.degrees.append(np.array(chunk_deg, dtype=np.int64))
        self.num_edges = total

    def _init_virtual_candidates(self):
        """Precompute virtual edge candidates once at init by randomly sampling n_hop_virtual 2-hop neighbors."""
        self._virtual_candidates = {}
        self._virt_from_arrs = []
        self._virt_to_arrs = []
        self._virt_col_arrs = []
        self._virt_batch_slices = []  # list of {from_id: (local_start, local_end)} per batch
        if self.n_hop_virtual == 0:
            return
        rng = np.random.default_rng()
        for from_id in self.dynamic_edges:
            candidates = self._compute_hop2_virtual(from_id)
            if not candidates:
                continue
            if len(candidates) <= self.n_hop_virtual:
                sampled = candidates
            else:
                sampled = rng.choice(candidates, size=self.n_hop_virtual, replace=False).tolist()
            self._virtual_candidates[from_id] = sampled

        # Build batched arrays (chunked by vertex, like _build_batches)
        chunk_from, chunk_to, chunk_cols, chunk_slices, chunk_size = [], [], [], {}, 0
        for from_id, candidates in self._virtual_candidates.items():
            real_deg = len(self.dynamic_edges[from_id])
            local_start = len(chunk_from)
            chunk_from.extend([from_id] * len(candidates))
            chunk_to.extend(candidates)
            chunk_cols.extend(real_deg + slot for slot in range(len(candidates)))
            chunk_slices[from_id] = (local_start, local_start + len(candidates))
            chunk_size += len(candidates)
            if chunk_size >= self.batch_size:
                self._virt_from_arrs.append(np.array(chunk_from, dtype=np.int64))
                self._virt_to_arrs.append(np.array(chunk_to, dtype=np.int64))
                self._virt_col_arrs.append(np.array(chunk_cols, dtype=np.int32))
                self._virt_batch_slices.append(chunk_slices)
                chunk_from, chunk_to, chunk_cols, chunk_slices, chunk_size = [], [], [], {}, 0
        if chunk_from:
            self._virt_from_arrs.append(np.array(chunk_from, dtype=np.int64))
            self._virt_to_arrs.append(np.array(chunk_to, dtype=np.int64))
            self._virt_col_arrs.append(np.array(chunk_cols, dtype=np.int32))
            self._virt_batch_slices.append(chunk_slices)

    def _compute_hop2_virtual(self, from_id):
        """2-hop virtual edge candidates for from_id, excluding already existing edges."""
        current = set(self.dynamic_edges.get(from_id, []))
        candidates = set()
        for nb in current:
            for nb2 in self.dynamic_edges.get(nb, []):
                if nb2 != from_id and nb2 not in current:
                    candidates.add(nb2)
        return list(candidates)

    @torch.no_grad()
    def prepare_edges_with_probs(self, agent, state=None, is_evaluate=False, greedy=False, **kwargs):
        """ :param state: cached agent memory state. If not specified, calls agent.prepare_state """
        probs = np.full([state.vertices.size(0), self.effective_max_degree],
                        self.service_labels['pad'], dtype=np.float32)
        edges = np.full([state.vertices.size(0), self.effective_max_degree],
                        self.service_labels['pad'], dtype=np.int32)

        if state is None:
            state = agent.prepare_state(self.graph, **kwargs)

        upper_prob_bound = 0.99989
        lower_prob_bound = 0.00011

        # --- Score real edges (batch-wise) ---
        for i in range(len(self.from_vertex_ids)):
            flat_from = self.from_vertex_ids[i]
            flat_to = self.to_vertex_ids[i]

            edge_logp = agent.get_edge_logp(flat_from, flat_to, state=state, **kwargs).cpu()
            if greedy:
                edge_probs = edge_logp.argmax(-1).numpy().astype(np.float32)
            else:
                edge_probs = edge_logp[:, 1].exp().numpy()

            # Retrieve confidence for this batch as a numpy array
            conf_arr = np.array([self.edge_confidence.get((int(f), int(t)), 0.0)
                                 for f, t in zip(flat_from, flat_to)], dtype=np.float32)

            # Freeze confidently-decided edges
            edge_probs[conf_arr == self.edge_patience] = 1.1
            edge_probs[conf_arr == -self.edge_patience] = -0.1

            if not is_evaluate:
                conf_arr[(1. > edge_probs) & (edge_probs > upper_prob_bound)] += 1.
                conf_arr[(0. < edge_probs) & (edge_probs < lower_prob_bound)] -= 1.
                conf_arr[(conf_arr > 0) & (edge_probs < upper_prob_bound)] = 0.
                conf_arr[(conf_arr < 0) & (edge_probs > lower_prob_bound)] = 0.
                self.edge_confidence.update(
                    zip(zip(flat_from.tolist(), flat_to.tolist()), conf_arr.tolist()))

            idxs = np.cumsum(self.degrees[i][:-1])
            idxs = np.pad(idxs, (1, 0), 'constant', constant_values=0)
            vertex_ids = flat_from[idxs]

            mask = np.arange(self.graph.max_degree) < self.degrees[i][:, None]
            m_probs = np.full((len(self.degrees[i]), self.graph.max_degree),
                              self.service_labels['pad'], dtype=np.float32)
            m_probs[mask] = edge_probs
            probs[vertex_ids, :self.graph.max_degree] = m_probs

            m_edges = np.full((len(self.degrees[i]), self.graph.max_degree),
                              self.service_labels['pad'], dtype=np.int32)
            m_edges[mask] = flat_to
            edges[vertex_ids, :self.graph.max_degree] = m_edges

        # --- Score virtual (2-hop) edges, cache scores, fill virtual columns ---
        if self.n_hop_virtual > 0 and not is_evaluate and self._virt_from_arrs:
            new_virtual_scores = {}
            for from_arr, to_arr, col_arr, batch_slices in zip(
                    self._virt_from_arrs, self._virt_to_arrs, self._virt_col_arrs, self._virt_batch_slices):
                virt_logp = agent.get_edge_logp(from_arr, to_arr, state=state, **kwargs).cpu()
                virt_probs = virt_logp[:, 1].exp().numpy()
                probs[from_arr, col_arr] = virt_probs
                edges[from_arr, col_arr] = to_arr
                for from_id, (start, end) in batch_slices.items():
                    new_virtual_scores[from_id] = list(zip(
                        to_arr[start:end].tolist(), virt_probs[start:end].tolist()))
            self._last_virtual_scores = new_virtual_scores

        torch.cuda.empty_cache()
        return edges, probs

    def update_edges(self):
        """
        Evolve graph topology:
        1. Remove real edges with confidence <= -edge_patience (frozen-low).
        2. Promote top cached virtual edges (from _last_virtual_scores) to fill vacated slots.
        Returns number of newly promoted edges.
        """
        promoted = 0
        removed = 0
        for from_id, neighbors in self.dynamic_edges.items():
            # Remove low-confidence frozen edges
            to_remove = [nb for nb in neighbors
                         if self.edge_confidence.get((from_id, nb), 0.0) <= -self.edge_patience]
            for nb in to_remove:
                neighbors.remove(nb)
                self.edge_confidence.pop((from_id, nb), None)
                removed += 1

            # Fill vacated slots from cached virtual scores (sort by prob descending at update time)
            slots = self.graph.max_degree - len(neighbors)
            if slots > 0 and from_id in self._last_virtual_scores:
                existing = set(neighbors)
                for to_id, _prob in sorted(self._last_virtual_scores[from_id], key=lambda x: -x[1]):
                    if slots == 0:
                        break
                    if to_id not in existing:
                        neighbors.append(to_id)
                        self.edge_confidence[(from_id, to_id)] = 0.0
                        existing.add(to_id)
                        promoted += 1
                        slots -= 1

        if promoted > 0 or removed > 0:
            self._build_batches()
            self._init_virtual_candidates()
        return promoted

    def record_sessions(self, agent, queries, **kwargs):
        """
        finds nearest neighbors for several queries, computes reward and returns all that
        :param agent: lib.agent.BaseAgent
        :param queries: a batch of query vectors
        :return: a dict with a lot of metrics
        """
        edges, edge_probs = self.prepare_edges_with_probs(agent, **kwargs)
        num_actions = self.max_trajectory * self.effective_max_degree
        num_results = self.k + 2 + num_actions
        search_results = np.full([queries.shape[0], num_results], self.service_labels['pad'], dtype=np.int32)
        trajectories = np.full([queries.shape[0], self.max_trajectory], self.service_labels['pad'], dtype=np.int32)
        uniform_samples = np.random.rand(queries.shape[0], num_actions).astype(np.float32)

        # search_results = [:, answer, dcs, hops]
        search_hnsw(self.graph.vertices.numpy().astype(np.float32),
                    edges, edge_probs,
                    queries.numpy().astype(np.float32),
                    trajectories, uniform_samples, search_results,
                    self.k, self.graph.initial_vertex_id,
                    self.ef, self.n_jobs)

        # Collect records
        session_records = []
        best_vertex_ids = search_results[:, :self.k]
        total_distance_computations = search_results[:, self.k]
        num_hops = search_results[:, self.k + 1]
        total_actions = search_results[:, self.k + 2:]

        for i in range(queries.shape[0]):
            trajectory = trajectories[i, :num_hops[i]]
            session_actions = total_actions[i, :num_hops[i] * self.effective_max_degree]
            session_edges = edges[trajectory].reshape(-1)
            session_mask = (session_actions != self.service_labels['pad']) & \
                           (session_actions != self.service_labels['unused'])
            actions = session_actions[session_mask]
            to_vertex_ids = session_edges[session_mask]

            idxs = np.arange(1, len(trajectory)) * self.effective_max_degree
            session_num_samples = np.array(np.array_split(session_mask, idxs)).sum(-1)
            from_vertex_ids = np.repeat(trajectory, session_num_samples)

            rec = dict(
                from_vertex_ids=from_vertex_ids.tolist(),
                to_vertex_ids=to_vertex_ids.tolist(),
                actions=actions.tolist(),
                best_vertex_id=best_vertex_ids[i][0],
                best_vertex_ids=best_vertex_ids[i],
                total_distance_computations=total_distance_computations[i],
                num_hops=num_hops[i],
            )
            session_records.append(rec)
        return session_records

    @staticmethod
    def _check_n_jobs(n_jobs):
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() + 1 - n_jobs
        assert n_jobs > 0
        return n_jobs
