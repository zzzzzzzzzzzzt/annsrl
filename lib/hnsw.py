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
                 edge_patience=100, n_jobs=1, max_dcs=0):
        """ Optimized EdgeHNSW for fast session sampling. Uses wrapped C++ code for HNSW search algorithm
            :param max_trajectory: maximum expected number of hops. Needed for swig.
                   The larger ef is, the larger max_trajectory you should set
            :param batch_size: number of edges in batch
            :param n_jobs: number of threads for C++ session sampling
            :param max_dcs: hard distance-computation budget per query; <=0 = unlimited.

            With a budget the search stops mid-hop the moment it would exceed it, so
            total_distance_computations is a CONSTANT of the environment rather than
            something the graph edit can move. That is what makes recall comparable
            across graphs without choosing a recall/DCS exchange rate: the reward can
            be pure recall (beta=0) because there is no longer a cost to trade against.
            Without the cap, ef fixes the beam width but not the cost -- DCS ~
            degree * hops * (1 - revisit_rate), and shortening edges raises the
            revisit rate, which is how the unconstrained policy cut DCS 8% while
            leaving hops flat and recall untouched.
        """
        super().__init__(graph, ef)
        assert graph.graph_type != 'hnsw', 'hnsw.ParallelHNSW does not support hierarchy. Use hnsw.EdgeHNSW.'

        self.k = k
        self.max_trajectory = max_trajectory
        self.n_jobs = self._check_n_jobs(n_jobs)
        self.max_dcs = int(max_dcs)

        self.batch_size = batch_size
        self.edge_patience = edge_patience
        self.edge_confidence = []

        self.num_edges = 0
        self.num_confident = 0

        # Service labels to denote service fields and boundary situations
        self.service_labels = {'pad': -1, 'unused': -2, 'no_actions': -3}

        self.from_vertex_ids, self.to_vertex_ids, self.degrees = [], [], []
        chunk_from_vertex_ids, chunk_to_vertex_ids, chunk_degrees = [], [], []

        for vertex_id, neighbor_ids in self.graph.edges.items():
            degree = len(neighbor_ids)
            chunk_from_vertex_ids.extend([vertex_id] * degree)
            chunk_to_vertex_ids.extend(neighbor_ids)
            chunk_degrees.append(degree)
            if sum(chunk_degrees) > self.batch_size:
                self.num_edges += sum(chunk_degrees)
                self.from_vertex_ids.append(np.array(chunk_from_vertex_ids))
                self.to_vertex_ids.append(np.array(chunk_to_vertex_ids))
                self.degrees.append(np.array(chunk_degrees))
                self.edge_confidence.append(np.zeros(sum(chunk_degrees)))
                chunk_from_vertex_ids, chunk_to_vertex_ids, chunk_degrees = [], [], []

        # Remained samples
        if len(chunk_degrees) > 0:
            self.num_edges += sum(chunk_degrees)
            self.from_vertex_ids.append(np.array(chunk_from_vertex_ids))
            self.to_vertex_ids.append(np.array(chunk_to_vertex_ids))
            self.degrees.append(np.array(chunk_degrees))
            self.edge_confidence.append(np.zeros(sum(chunk_degrees)))

    @torch.no_grad()
    def prepare_edges_with_probs(self, agent, state=None, is_evaluate=False, greedy=False, **kwargs):
        """ :param state: cached agent memory state. If not specified, calls agent.prepare_state """
        probs = np.full([state.vertices.size(0), self.graph.max_degree], self.service_labels['pad'], dtype=np.float32)
        edges = np.full([state.vertices.size(0), self.graph.max_degree], self.service_labels['pad'], dtype=np.int32)

        if state is None:
            state = agent.prepare_state(self.graph, **kwargs)

        upper_prob_bound = 0.99989
        lower_prob_bound = 0.00011

        for i in range(len(self.from_vertex_ids)):
            edge_logp = agent.get_edge_logp(self.from_vertex_ids[i], self.to_vertex_ids[i],
                                            state=state, **kwargs).cpu()
            if greedy:
                edge_probs = edge_logp.argmax(-1).numpy()
            else:
                edge_probs = edge_logp[:, 1].exp().numpy()

            # Freeze edges that are consistently confident.
            # Set probs of confident edges to 1.1 or -0.1 to indicate the search algorithm do not sample them
            edge_probs[self.edge_confidence[i] == self.edge_patience] = 1.1
            edge_probs[self.edge_confidence[i] == -self.edge_patience] = -0.1

            if not is_evaluate:
                self.edge_confidence[i][(1. > edge_probs) & (edge_probs > upper_prob_bound)] += 1.
                self.edge_confidence[i][(0. < edge_probs) & (edge_probs < lower_prob_bound)] -= 1.
                self.edge_confidence[i][(self.edge_confidence[i] > 0) & (edge_probs < upper_prob_bound)] = 0.
                self.edge_confidence[i][(self.edge_confidence[i] < 0) & (edge_probs > lower_prob_bound)] = 0.

            idxs = np.cumsum(self.degrees[i][:-1])
            idxs = np.pad(idxs, (1, 0), 'constant', constant_values=0)
            vertex_ids = self.from_vertex_ids[i][idxs]

            mask = np.arange(self.graph.max_degree) < self.degrees[i][:, None]
            m_probs = np.full_like(mask, self.service_labels['pad'], dtype=np.float32)
            m_probs[mask] = edge_probs
            probs[vertex_ids] = m_probs

            m_edges = np.full_like(mask, self.service_labels['pad'], np.float32)
            m_edges[mask] = self.to_vertex_ids[i]
            edges[vertex_ids] = m_edges

        torch.cuda.empty_cache()
        return edges, probs

    def record_sessions(self, agent, queries, **kwargs):
        """
        finds nearest neighbors for several queries, computes reward and returns all that
        :param agent: lib.agent.BaseAgent
        :param queries: a batch of query vectors
        :return: a dict with a lot of metrics
        """
        edges, edge_probs = self.prepare_edges_with_probs(agent, **kwargs)
        num_actions = self.max_trajectory * self.graph.max_degree
        num_results = self.k + 2 + num_actions
        search_results = np.full([queries.shape[0], num_results], self.service_labels['pad'], dtype=np.int32)
        trajectories = np.full([queries.shape[0], self.max_trajectory], self.service_labels['pad'], dtype=np.int32)
        uniform_samples = np.random.rand(queries.shape[0], num_actions).astype(np.float32)

        # search_results = [:, answer, dcs, hps]
        search_hnsw(self.graph.vertices.numpy().astype(np.float32),
                    edges, edge_probs,
                    queries.numpy().astype(np.float32),
                    trajectories, uniform_samples, search_results,
                    self.k, self.graph.initial_vertex_id,
                    self.ef, self.n_jobs, self.max_dcs)

        # Collect records
        session_records = []
        best_vertex_ids = search_results[:, :self.k]
        total_distance_computations = search_results[:, self.k]
        num_hops = search_results[:, self.k + 1]
        total_actions = search_results[:, self.k + 2:]

        for i in range(queries.shape[0]):
            trajectory = trajectories[i, :num_hops[i]]
            session_actions = total_actions[i, :num_hops[i]*self.graph.max_degree]
            session_edges = edges[trajectory].reshape(-1)
            session_mask = (session_actions != self.service_labels['pad']) & \
                           (session_actions != self.service_labels['unused'])
            actions = session_actions[session_mask]
            to_vertex_ids = session_edges[session_mask]

            idxs = np.arange(1, len(trajectory)) * self.graph.max_degree
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
                path=trajectory.tolist(),
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


class GraphEditHNSW(ParallelHNSW):
    """ HNSW whose *topology* is the RL state (framework.md).

    Unlike ParallelHNSW -- where an episode is one query's search session and the
    action is a per-edge keep/drop sampled inside the C++ beam search -- here the
    state s_t is the whole adjacency structure and the action is a bounded edge
    swap applied to a batch of nodes. Searches are therefore run *deterministically*
    (every edge traversed) and are only used to measure the cost c(s_t).

    :param n_hop_virtual: cap on the number of 2-hop candidates considered per node.
           Defaults to max_degree ** 2, the size of the full 2-hop pool.
    """

    def __init__(self, graph, n_hop_virtual=None, **kwargs):
        super().__init__(graph, **kwargs)
        self.n_hop_virtual = n_hop_virtual or self.graph.max_degree ** 2
        # dynamic_edges is the mutable state s_t; graph.edges stays as s_0.
        self.dynamic_edges = {v: list(nbrs) for v, nbrs in self.graph.edges.items()}
        self.adj = self.build_adjacency()

    @property
    def num_vertices(self):
        return self.graph.vertices.size(0)

    def build_adjacency(self):
        """ Materializes dynamic_edges as a dense [N, max_degree] int32 array.

        Rows are -1 padded *at the tail only*: search_hnsw.cc walks neighbours with
        `while (neighbor_ids[j] != -1 && j < max_degree)`, so an interior -1 would
        silently truncate the neighbour list. Every writer must keep rows compact.
        """
        adj = np.full([self.num_vertices, self.graph.max_degree],
                      self.service_labels['pad'], dtype=np.int32)
        for vertex_id, neighbor_ids in self.dynamic_edges.items():
            degree = min(len(neighbor_ids), self.graph.max_degree)
            if degree:
                adj[vertex_id, :degree] = np.asarray(neighbor_ids[:degree], dtype=np.int32)
        return adj

    def get_candidates(self, node_ids, chunk=512):
        """ framework.md step 4: the 2-hop neighbourhood N2 of each node, excluding
        the node itself and its current 1-hop neighbours N1 (those are already edges).

        Fully vectorized -- this runs over thousands of nodes every iteration, so the
        obvious per-node Python loop would dominate the step time. Chunked over nodes
        to bound the peak size of the N1-membership test.

        :param node_ids: int array [B] of nodes to propose new edges for
        :return: (cand_ids [B, C] int64, cand_mask [B, C] bool) -- padded slots hold
                 index 0 and mask False, so cand_ids is always safe to embed.
        """
        node_ids = np.asarray(node_ids, dtype=np.int64)
        pad = self.service_labels['pad']
        max_degree = self.graph.max_degree
        width = int(min(self.n_hop_virtual, max_degree * max_degree))
        # Sorts above every real vertex id, so excluded/duplicate slots end up last.
        sentinel = self.num_vertices

        cand_ids = np.zeros([len(node_ids), width], dtype=np.int64)
        cand_mask = np.zeros([len(node_ids), width], dtype=bool)

        for start in range(0, len(node_ids), chunk):
            batch = node_ids[start:start + chunk]
            hop1 = self.adj[batch].astype(np.int64)                  # [b, D]
            hop1_valid = hop1 != pad
            hop2 = self.adj[np.where(hop1_valid, hop1, 0)].astype(np.int64)   # [b, D, D]
            hop2_valid = (hop2 != pad) & hop1_valid[:, :, None]
            pool = np.where(hop2_valid, hop2, sentinel).reshape(len(batch), -1)

            pool[pool == batch[:, None]] = sentinel                  # drop self-loops
            # Drop anything already a 1-hop neighbour. -1 for invalid hop1 slots can
            # never match a pooled id, so it is a safe filler here.
            already = np.where(hop1_valid, hop1, -1)
            pool[(pool[:, :, None] == already[:, None, :]).any(-1)] = sentinel

            # Deduplicate: sorting groups equal ids (and pushes sentinels to the end),
            # so a value is new iff it differs from its predecessor.
            pool.sort(axis=1)
            keep = np.ones_like(pool, dtype=bool)
            keep[:, 1:] = pool[:, 1:] != pool[:, :-1]
            keep &= pool != sentinel

            # Compact the kept ids to the left; argsort on ~keep is stable so their
            # relative (ascending) order is preserved.
            order = np.argsort(~keep, axis=1, kind='stable')[:, :width]
            end = start + len(batch)
            cand_ids[start:end] = np.take_along_axis(pool, order, axis=1)
            cand_mask[start:end] = np.take_along_axis(keep, order, axis=1)

        # Masked slots must hold a legal index, since they are still embedded.
        cand_ids[~cand_mask] = 0
        # Trim columns no node actually filled.
        used = int(cand_mask.any(0).sum())
        used = max(used, 1)
        return cand_ids[:, :used], cand_mask[:, :used]

    def apply_swaps(self, node_ids, drop_nb_ids, new_nb_ids):
        """ framework.md step 5: environment transition s_t -> s_{t+1}.

        For each node, drops the given existing neighbours and adds the given new
        ones. Degree is preserved (bounded swap), so the graph cannot fragment and
        the [N, max_degree] array shape stays valid.

        :param node_ids: int array [B]
        :param drop_nb_ids: int array [B, n_swap] -- existing neighbours to remove
        :param new_nb_ids: int array [B, n_swap] -- 2-hop candidates to insert
        :return: number of edges actually changed
        """
        node_ids = np.asarray(node_ids, dtype=np.int64)
        drop_nb_ids = np.asarray(drop_nb_ids, dtype=np.int64).reshape(len(node_ids), -1)
        new_nb_ids = np.asarray(new_nb_ids, dtype=np.int64).reshape(len(node_ids), -1)

        num_changed = 0
        for i, v in enumerate(node_ids):
            v = int(v)
            neighbors = self.dynamic_edges[v]
            current = set(neighbors)
            for drop, new in zip(drop_nb_ids[i], new_nb_ids[i]):
                drop, new = int(drop), int(new)
                # Skip no-ops and anything that would duplicate an edge or self-loop.
                if drop == new or new == v or new in current or drop not in current:
                    continue
                neighbors[neighbors.index(drop)] = new
                current.discard(drop)
                current.add(new)
                num_changed += 1

            degree = min(len(neighbors), self.graph.max_degree)
            self.adj[v, :degree] = np.asarray(neighbors[:degree], dtype=np.int32)
            self.adj[v, degree:] = self.service_labels['pad']
        return num_changed

    def snapshot(self, node_ids):
        """ Copies the neighbour lists of `node_ids` so a swap can be undone.

        apply_swaps only ever writes the rows of the nodes it is given, so a
        snapshot of those rows is enough to restore s_t exactly. Copying the whole
        adjacency instead would cost N * max_degree per trial step.

        :param node_ids: int array [B] -- the same nodes apply_swaps will be given
        :return: opaque state to hand back to restore()
        """
        return {int(v): list(self.dynamic_edges[int(v)]) for v in np.asarray(node_ids)}

    def restore(self, state):
        """ Undoes a trial swap: puts the snapshotted neighbour lists back.

        :param state: return value of snapshot()
        """
        pad = self.service_labels['pad']
        for vertex_id, neighbor_ids in state.items():
            self.dynamic_edges[vertex_id] = list(neighbor_ids)
            degree = min(len(neighbor_ids), self.graph.max_degree)
            self.adj[vertex_id, :degree] = np.asarray(neighbor_ids[:degree], dtype=np.int32)
            self.adj[vertex_id, degree:] = pad

    def graph_stats(self, sample=4096, seed=0):
        """ Cheap structural health check of the current topology s_t.

        The bounded swap preserves out-degree, so degeneration does not show up as
        nodes losing edges -- it shows up in *reachability*. Two failure modes worth
        watching while the agent edits the graph:

        - reachable_frac collapsing: fewer nodes are reachable from the entry point,
          so whole regions of the dataset can no longer be answered at all.
        - mean_edge_len collapsing toward the kNN radius: the agent traded away the
          long-range links that make the graph navigable, leaving a locally-dense
          but globally-disconnected graph. Search then terminates early -- which a
          DCS-based reward term will happily score as an improvement.

        :param sample: nodes sampled for the edge-length estimate (0 = all)
        :return: dict of floats
        """
        pad = self.service_labels['pad']
        valid = self.adj != pad
        degrees = valid.sum(-1)

        # BFS from the entry point over the current adjacency.
        num_vertices = self.num_vertices
        seen = np.zeros(num_vertices, dtype=bool)
        frontier = np.array([self.graph.initial_vertex_id], dtype=np.int64)
        seen[frontier] = True
        while frontier.size:
            nxt = self.adj[frontier]
            nxt = nxt[nxt != pad].astype(np.int64)
            nxt = np.unique(nxt)
            nxt = nxt[~seen[nxt]]
            seen[nxt] = True
            frontier = nxt

        stats = {
            'reachable_frac': float(seen.mean()),
            'degree_mean': float(degrees.mean()),
            'degree_min': float(degrees.min()),
        }

        # Mean length of an out-edge, over a random node sample. Rising means the
        # graph is keeping long-range links; falling toward zero means it is
        # collapsing into a purely local (kNN-like) graph.
        rng = np.random.default_rng(seed)
        num = num_vertices if not sample else min(sample, num_vertices)
        nodes = rng.choice(num_vertices, size=num, replace=False)
        rows, cols = np.nonzero(valid[nodes])
        if rows.size:
            src = self.graph.vertices[torch.as_tensor(nodes[rows], dtype=torch.long)]
            dst = self.graph.vertices[torch.as_tensor(
                self.adj[nodes][rows, cols].astype(np.int64), dtype=torch.long)]
            lengths = torch.norm(src - dst, dim=-1)
            stats['edge_len_mean'] = float(lengths.mean())
            stats['edge_len_max'] = float(lengths.max())
        return stats

    def search_deterministic(self, queries, ef=None):
        """ Runs search on the current graph s_t with *every* edge traversed.

        Setting all valid edge probs to 1.1 makes `prob > sample` always true in
        search_hnsw.cc (samples live in [0, 1)), so the walk is deterministic; the
        kernel also marks such edges `*action = -2` ("unused"), which is why
        record_sessions cannot be reused here -- it would find zero actions.

        :return: dict of best_vertex_ids [nq, k], total_distance_computations [nq],
                 num_hops [nq], trajectories [nq, max_trajectory]
        """
        num_queries = queries.shape[0]
        num_actions = self.max_trajectory * self.graph.max_degree
        num_results = self.k + 2 + num_actions

        edges = self.adj
        probs = np.where(edges != self.service_labels['pad'],
                         np.float32(1.1), np.float32(self.service_labels['pad'])).astype(np.float32)

        search_results = np.full([num_queries, num_results], self.service_labels['pad'], dtype=np.int32)
        trajectories = np.full([num_queries, self.max_trajectory], self.service_labels['pad'], dtype=np.int32)
        uniform_samples = np.zeros([num_queries, num_actions], dtype=np.float32)

        search_hnsw(self.graph.vertices.numpy().astype(np.float32),
                    edges, probs,
                    queries.numpy().astype(np.float32),
                    trajectories, uniform_samples, search_results,
                    self.k, self.graph.initial_vertex_id,
                    self.ef if ef is None else ef, self.n_jobs, self.max_dcs)

        return dict(
            best_vertex_ids=search_results[:, :self.k],
            total_distance_computations=search_results[:, self.k],
            num_hops=search_results[:, self.k + 1],
            trajectories=trajectories,
        )
