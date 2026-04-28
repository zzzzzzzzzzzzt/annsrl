import warnings
from .utils import knn, read_edges, read_fvecs, read_ivecs, read_nsg
import torch
import numpy as np
from torch_geometric.utils import subgraph

class Graph:
    def __init__(self, vertices_path, edges_path,
                 train_queries_path, test_queries_path,
                 train_gt_path=None, test_gt_path=None,
                 vertices_size=None, train_queries_size=None, 
                 val_queries_size=None, test_queries_size=None, 
                 ground_truth_n_neighbors=1, knn_n_jobs=1,
                 initial_vertex_id=0, normalization='global', graph_type='nsw'):
        """
        Graph is a data class that stores all CONSTANT data about the graph: vertices, edges, etc.
        :param vertices_path: path to base datapoints
        :param edges_path: path to initial graph edges
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
        :param graph_type: supported graph types: {'nsw', 'nsg'}.
        """
        self.graph_type = graph_type
        vertices = torch.tensor(read_fvecs(vertices_path, vertices_size))
        self.max_level = 0
        if graph_type == 'nsw':
            self.edges = read_edges(edges_path, vertices.shape[0])
            self.initial_vertex_id = initial_vertex_id
            self.max_degree = max(map(len, self.edges.values()))
        elif graph_type == 'nsg':
            info, self.edges = read_nsg(edges_path)
            self.initial_vertex_id = info['enterpoint_node']
            self.max_degree = info['width']
        else:
            raise ValueError("Only ['nsw', 'nsg'] graph types are supported")

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
                vertices_size=None, normalization='global'):
        """
        :param vertices_path: path to base datapoints
        :param normalization: normalization of base datapoints {'none', 'global', 'instance'}
        :param graph_type: supported graph types: {'nsw', 'nsg'}.
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

        # get the splits for all runs
        self.split_idx_lst = self.get_idx_split(train_prop=train_prop, valid_prop=valid_prop)
        
        # get train subgraph
        self.train_edges, self.node_map = subgraph(self.split_idx_lst['train'], 
                                                    self.edges, relabel_nodes=True)

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
            val_indices = perm[train_num:train_num + valid_num]
            test_indices = perm[train_num + valid_num:]

        return {'train':train_indices, 'valid':val_indices, 'test':test_indices}