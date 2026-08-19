import os
import warnings
from .utils import knn, read_edges, read_fvecs, read_ivecs, read_nsg
import torch
import numpy as np
from torch_geometric.utils import subgraph, to_undirected


# --------------------------------------------------------------------------- #
# kNN graph: exact k-nearest-neighbors over the WHOLE graph (all N nodes) via
# faiss. Built ONCE on the raw (pre-standardization) coordinates -- the space
# the proximity graph was pruned in -- and cached to disk for reuse. Replaces
# an O(N^2) torch.cdist scan: faiss.IndexFlatL2 is exact (same result as a
# brute cdist topk) but blocked/SIMD-optimized and far lighter on memory.
#
# Returns (idx [N, k], dist [N, k]) on CPU. dist is EUCLIDEAN (sqrt of faiss'
# squared L2) so it can be compared directly against a neighbor radius. The
# self node sits at rank 0 with distance 0; callers pass k = topk + 1 to
# absorb it.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_knn_graph(coords, k, cache_path=None):
    import faiss
    n = coords.shape[0]
    k = min(n, max(1, k))
    if cache_path and os.path.exists(cache_path):
        blob = np.load(cache_path)
        idx, dist = blob['idx'], blob['dist']
        if idx.shape == (n, k):
            print(f"[KNN] loaded cache {cache_path} shape={idx.shape}")
            return torch.from_numpy(idx).long(), torch.from_numpy(dist).float()
        print(f"[KNN] cache {cache_path} shape {idx.shape} != {(n, k)}, rebuilding")

    x = np.ascontiguousarray(coords.detach().cpu().numpy(), dtype='float32')
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    d2, idx = index.search(x, k)                 # d2: squared L2, idx: [N, k]
    dist = np.sqrt(np.maximum(d2, 0.0))          # euclidean, matches radius test
    if cache_path:
        try:
            np.savez(cache_path, idx=idx.astype('int64'), dist=dist.astype('float32'))
            print(f"[KNN] built + cached {cache_path} shape={idx.shape}")
        except Exception as ex:
            print(f"[KNN] cache save skipped ({ex})")
    return torch.from_numpy(idx.astype('int64')).long(), \
           torch.from_numpy(dist.astype('float32')).float()

# --------------------------------------------------------------------------- #
# Initial-topology builders.
#
# The graph-edit MDP (framework.md / lib.algorithm.GraphEditPPO) treats the
# topology as the state it learns to improve, so s_0 does NOT have to be a good
# proximity graph -- and using a pruned HNSW/NSG graph as s_0 conflates "what the
# agent learned" with "what the construction heuristic already knew". These build
# s_0 from scratch instead, at a fixed out-degree so the bounded-swap degree
# invariant has a well-defined starting point.
# --------------------------------------------------------------------------- #
def build_random_edges(num_vertices, degree, seed=None):
    """ Uniformly random out-edges: the weakest possible s_0 (no distance info).

    Each node gets exactly `degree` distinct non-self neighbours, so the agent has
    to discover proximity structure entirely on its own.
    """
    degree = int(min(max(degree, 1), num_vertices - 1))
    rng = np.random.default_rng(seed)

    # Oversample + dedup is much faster than a per-node rng.choice; with
    # degree << num_vertices a single pass almost always fills every row.
    pool = rng.integers(0, num_vertices, size=(num_vertices, degree * 2), dtype=np.int64)
    self_ids = np.arange(num_vertices, dtype=np.int64)[:, None]
    pool[pool == self_ids] = -1                     # mark self-loops for removal
    pool.sort(axis=1)
    keep = np.ones_like(pool, dtype=bool)
    keep[:, 1:] = pool[:, 1:] != pool[:, :-1]       # drop duplicates
    keep &= pool >= 0
    # The sort above was only a device for deduplication, so the surviving ids sit in
    # ascending order: compacting them left and taking the first `degree` would keep
    # the SMALLEST ids of each row and starve the top of the id range of in-edges
    # entirely. Measured on N=100k, degree=24: mean neighbour id 25525 instead of
    # 50000, no id above 78324 ever sampled, 35% of nodes with in-degree 0, and
    # reachable_frac stuck at 0.65 no matter what the agent does. Draw `degree` of
    # the kept slots uniformly at random instead -- random keys on kept slots and
    # +inf on the rest, so the kept ids come first in random order.
    order = np.argsort(np.where(keep, rng.random(pool.shape), np.inf), axis=1)
    pool = np.take_along_axis(pool, order, axis=1)
    counts = keep.sum(1)

    edges = {}
    short = np.flatnonzero(counts < degree)
    for v in range(num_vertices):
        edges[v] = pool[v, :degree].tolist()
    # Rare rows that deduplicated below `degree` get an exact resample.
    for v in short:
        choice = rng.choice(num_vertices - 1, size=degree, replace=False)
        choice[choice >= v] += 1                    # skip self without resampling
        edges[int(v)] = choice.astype(np.int64).tolist()
    return edges


def build_knn_edges(vertices, degree, cache_path=None, n_jobs=1):
    """ Exact kNN out-edges: a strong *local* s_0 with no long-range links.

    A pure kNN graph is highly clustered and poorly navigable -- search from a
    fixed entry point tends to get stuck in the entry point's neighbourhood -- so
    it isolates exactly what the agent has to learn: which few edges to trade for
    long-range connectivity.
    """
    num_vertices = vertices.shape[0]
    degree = int(min(max(degree, 1), num_vertices - 1))

    # k = degree + 1 to absorb the self match at rank 0.
    try:
        idx, _ = build_knn_graph(vertices, degree + 1, cache_path=cache_path)
        idx = idx.numpy()
    except ImportError:
        warnings.warn('faiss unavailable, falling back to sklearn brute-force kNN')
        idx = knn(vertices, vertices, n_neighbors=degree + 1, n_jobs=n_jobs).numpy()

    # Drop self wherever it landed (ties can move it off rank 0) and left-compact,
    # so every row keeps exactly `degree` real neighbours in ascending-distance order.
    keep = idx != np.arange(num_vertices, dtype=idx.dtype)[:, None]
    order = np.argsort(~keep, axis=1, kind='stable')[:, :degree]
    idx = np.take_along_axis(idx, order, axis=1)
    return {v: idx[v].tolist() for v in range(num_vertices)}


class Graph:
    def __init__(self, vertices_path, edges_path,
                 train_queries_path, test_queries_path,
                 train_gt_path=None, test_gt_path=None,
                 vertices_size=None, train_queries_size=None, 
                 val_queries_size=None, test_queries_size=None, 
                 ground_truth_n_neighbors=1, knn_n_jobs=1,
                 initial_vertex_id=0, normalization='global', graph_type='nsw',
                 init_degree=24, init_seed=None, knn_cache_path=None):
        """
        Graph is a data class that stores all CONSTANT data about the graph: vertices, edges, etc.
        :param vertices_path: path to base datapoints
        :param edges_path: path to initial graph edges. Ignored (and may be None)
               for the 'random' and 'knn' graph types, which build s_0 themselves.
        :param train_queries_path: path to train queries
        :param test_queries_path: path to test queries

        :param vertices_size: number of base datapoints in graph
        :param train_queries_size: number of training queries
        :param val_queries_size: number of validation queries
        :param test_queries_size: number of test queries

        :param ground_truth_n_neighbors: finds this many nearest neighbors for ground truth ids
        :param knn_n_jobs: number of jobs used to precompute ground truth
        :param initial_vertex_id: starts search from this vertex
        :param normalization: normalization of base datapoints {'none', 'global', 'instance'}
        :param graph_type: one of
               'nsw'    -- read a pre-built NSW/HNSW graph from edges_path
               'nsg'    -- read a pre-built NSG graph from edges_path
               'random' -- uniformly random out-edges, built here
               'knn'    -- exact kNN out-edges, built here
        :param init_degree: out-degree for the 'random' and 'knn' graph types.
               Every node gets exactly this many edges, which is also the degree
               the graph-edit MDP's bounded swap then preserves.
        :param init_seed: seed for the 'random' graph type (None = nondeterministic)
        :param knn_cache_path: .npz cache for the 'knn' graph type's neighbour ids
        """
        self.graph_type = graph_type
        vertices = torch.tensor(read_fvecs(vertices_path, vertices_size))
        self.max_level = 0
        built_types = ('random', 'knn')
        if graph_type == 'nsw':
            self.edges = read_edges(edges_path, vertices.shape[0])
            self.initial_vertex_id = initial_vertex_id
            self.max_degree = max(map(len, self.edges.values()))
        elif graph_type == 'nsg':
            info, self.edges = read_nsg(edges_path)
            self.initial_vertex_id = info['enterpoint_node']
            self.max_degree = info['width']
        elif graph_type in built_types:
            # Deferred until after normalization below: 'instance' normalization is
            # not a monotone rescaling, so it can reorder nearest neighbours. Build
            # kNN in the same space the search will actually run in.
            self.initial_vertex_id = initial_vertex_id
            self.max_degree = int(min(max(init_degree, 1), vertices.shape[0] - 1))
        else:
            raise ValueError("Only ['nsw', 'nsg', 'random', 'knn'] graph types are supported")

        train_queries = torch.tensor(read_fvecs(train_queries_path, train_queries_size))
        test_queries = torch.tensor(read_fvecs(test_queries_path, test_queries_size))

        if normalization == 'none':
            warnings.warn("Data not normalized, individual norms:",
                          ((vertices ** 2).sum(-1) ** 0.5).cpu().numpy())
            normalize = lambda v: v
        elif normalization == 'global':
            mean_norm = ((vertices ** 2).sum(-1) ** 0.5).mean().item()
            normalize = lambda v: v / mean_norm
        elif normalization == 'instance':
            normalize = lambda v: v / (v ** 2).sum(-1, keepdim=True) ** 0.5
        else:
            raise ValueError("normalization parameter must be in ['none', 'global', 'instance']")

        self.vertices, self.train_queries, self.test_queries = \
            map(normalize, [vertices, train_queries, test_queries])

        if graph_type == 'random':
            self.edges = build_random_edges(self.vertices.shape[0], self.max_degree,
                                            seed=init_seed)
            print('[graph] random s_0: {} nodes, out-degree {}'.format(
                self.vertices.shape[0], self.max_degree))
        elif graph_type == 'knn':
            self.edges = build_knn_edges(self.vertices, self.max_degree,
                                          cache_path=knn_cache_path, n_jobs=knn_n_jobs)
            print('[graph] kNN s_0: {} nodes, out-degree {}'.format(
                self.vertices.shape[0], self.max_degree))

        if train_gt_path is None:
            self.train_gt = knn(self.vertices, self.train_queries,
                                n_neighbors=ground_truth_n_neighbors, n_jobs=knn_n_jobs)
        else:
            self.train_gt = torch.tensor(read_ivecs(train_gt_path, train_queries_size), dtype=torch.long)

        if test_gt_path is None:
            self.test_gt = knn(self.vertices, self.test_queries,
                               n_neighbors=ground_truth_n_neighbors, n_jobs=knn_n_jobs)
        else:
            self.test_gt = torch.tensor(read_ivecs(test_gt_path, test_queries_size), dtype=torch.long)

        # Split train queries on train and val
        self.val_queries = self.train_queries[-val_queries_size:]
        self.val_gt = self.train_gt[-val_queries_size:]

        self.train_queries = self.train_queries[:-val_queries_size]
        self.train_gt = self.train_gt[:-val_queries_size]

        
class pretrain_graph:
    def __init__(self, vertices_path, edges_path, graph_type='nsw',
                train_prop=.5, valid_prop=.25,
                vertices_size=None, normalization='global', undirected=False):
        """
        :param vertices_path: path to base datapoints
        :param normalization: normalization of base datapoints {'none', 'global', 'instance'}
        :param graph_type: supported graph types: {'nsw', 'nsg'}.
        :param undirected: symmetrize edges before building the train subgraph.
        """
        self.graph_type = graph_type
        self.vertices = torch.tensor(read_fvecs(vertices_path, vertices_size))
        if vertices_size == None:
            self.vertices_size = self.vertices.shape[0]
        else:
            self.vertices_size = vertices_size
        self.max_level = 0
        if graph_type == 'nsw':
            self.edgeindex = read_edges(edges_path, self.vertices.shape[0])
            self.max_degree = max(map(len, self.edgeindex.values()))
        elif graph_type == 'nsg':
            info, self.edgeindex = read_nsg(edges_path)
            self.initial_vertex_id = info['enterpoint_node']
            self.max_degree = info['width']
        else:
            raise ValueError("Only ['nsw', 'nsg'] graph types are supported")
        
        src = []
        dst = []
        for u, neighbors in self.edgeindex.items():
            for v in neighbors:
                src.append(u)
                dst.append(v)
        
        self.edges = torch.tensor([src, dst], dtype=torch.long)
        if undirected:
            self.edges = to_undirected(self.edges, num_nodes=self.vertices_size)
            self.max_degree = int(torch.bincount(self.edges[0], minlength=self.vertices_size).max().item())

        # get the splits for all runs
        self.split_idx_lst = self.get_idx_split(train_prop=train_prop, valid_prop=valid_prop)
        
        # get train subgraph
        self.train_edges, self.node_map = subgraph(self.split_idx_lst['train'], 
                                                    self.edges, relabel_nodes=True)
        pass

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            n = self.vertices_size
            train_num = int(n * train_prop)
            valid_num = int(n * valid_prop)

            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            if(train_prop == 1):
                val_indices=train_indices
                test_indices=train_indices
            else:
                val_indices = perm[train_num:]
                test_indices = perm[train_num:]

        return {'train':train_indices, 'valid':val_indices, 'test':test_indices}

    # ----------------------------------------------------------------------- #
    # Hard negatives: nodes that sit CLOSE to a source (inside the radius of
    # its farthest true out-neighbor) yet are NOT connected to it. In a pruned
    # proximity graph (NSW/HNSW) these are exactly the near points whose edges
    # the build-time pruning heuristic dropped -- the most confusable
    # non-neighbors, and the ones that force a model to learn the graph's
    # topology rather than raw proximity.
    #
    # Computed on the FULL, ORIGINAL graph: all `vertices_size` nodes, raw
    # `self.vertices` coordinates (never normalized/mutated by this class),
    # and `self.edgeindex` (the original NSW/NSG adjacency, global node ids)
    # as ground truth. This method is intentionally unaware of
    # `self.split_idx_lst` / `self.train_edges` -- no train/valid/test
    # filtering happens here, so it must be called before any caller mutates
    # `self.vertices` (e.g. standardization). Callers that only want a
    # training-node view should row-index the returned table by their own
    # node subset afterwards (`table[train_idx]`); the columns stay GLOBAL
    # node ids, they are not remapped to a compact/relabeled space.
    #
    # Returns a padded [vertices_size, max_pool] LongTensor of global node ids
    # (-1 = empty slot), plus coverage (fraction of nodes with >=1 hard
    # negative) and avg_pool (mean pool size among covered nodes) -- both
    # measured over the full node set.
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def build_hard_negatives(self, hard_neg_topk=None, knn_cache_path=None, device=None):
        n = self.vertices_size
        topk = hard_neg_topk if hard_neg_topk and hard_neg_topk > 0 else self.max_degree
        knn_idx, knn_dist = build_knn_graph(self.vertices, topk + 1, cache_path=knn_cache_path)

        coords_cpu = self.vertices.detach().cpu()
        knn_idx = knn_idx.cpu().tolist()
        knn_dist = knn_dist.cpu().tolist()

        pools = []
        max_pool = 0
        for gi in range(n):
            nbr = set(self.edgeindex.get(gi, []))
            if not nbr:
                pools.append([])                    # no neighbor radius -> no hard neg
                continue
            # neighbor radius: distance to the farthest true out-neighbor
            # (euclidean, raw coords).
            nbr_idx = torch.tensor(sorted(nbr), dtype=torch.long)
            d_max = torch.cdist(coords_cpu[gi:gi + 1], coords_cpu[nbr_idx]).max().item()
            hard = []
            for gc, dc in zip(knn_idx[gi], knn_dist[gi]):
                if gc == gi or gc in nbr or dc >= d_max:
                    continue
                hard.append(gc)
            pools.append(hard)
            max_pool = max(max_pool, len(hard))

        out_device = device if device is not None else self.vertices.device
        if max_pool == 0:
            table = torch.full((n, 1), -1, dtype=torch.long, device=out_device)
            self.hard_neg_table, self.hard_neg_coverage, self.hard_neg_avg_pool = table, 0.0, 0.0
            return self.hard_neg_table, self.hard_neg_coverage, self.hard_neg_avg_pool

        table = torch.full((n, max_pool), -1, dtype=torch.long, device=out_device)
        covered, total = 0, 0
        for gi, hard in enumerate(pools):
            if hard:
                table[gi, :len(hard)] = torch.tensor(hard, dtype=torch.long, device=out_device)
                covered += 1
                total += len(hard)
        coverage = covered / n
        avg_pool = total / max(1, covered)

        self.hard_neg_table = table
        self.hard_neg_coverage = coverage
        self.hard_neg_avg_pool = avg_pool
        return self.hard_neg_table, self.hard_neg_coverage, self.hard_neg_avg_pool