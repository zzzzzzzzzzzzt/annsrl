import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, subgraph
from sklearn.neighbors import kneighbors_graph

from lib.logger import Logger
from lib.parse import parser_add_main_args
from lib.graph import pretrain_graph
from lib.Nodeformer import NodeFormer
import time

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = pretrain_graph(args.vertices_path, args.edges_path)

### Basic information of datasets ###
n = dataset.vertices_size
e = dataset.edges.shape[0]
d = dataset.vertices.shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d}")

### Load method ###
model=NodeFormer(d, args.hidden_channels, d, num_layers=args.num_layers, dropout=args.dropout,
            num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
            use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
            nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Adj storage for relational bias ###
adjs = []
# adj, _ = remove_self_loops(dataset.graph['edge_index'])
adj, _ = add_self_loops(dataset.edges, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1): # edge_index of high order adjacency
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)
dataset.graph['adjs'] = adjs

dataset.vertices, dataset.edges, dataset.train_edges = \
    dataset.vertices.to(device), dataset.edges.to(device), dataset.train_edges.to(device)

train_idx, valid_idx, test_idx = \
    dataset.split_idx_lst['train'].to(device), \
    dataset.split_idx_lst['valid'].to(device), \
    dataset.split_idx_lst['test'].to(device)

### Training loop ###
for run in range(args.runs):
    split_idx = dataset.split_idx_lst

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        _, link_loss_, _ = model(dataset.vertices[train_idx], dataset.train_edges, args.tau)
        loss = -torch.mean(link_loss_[-1])

        loss.backward()
        optimizer.step()

        if epoch % args.eval_step == 0:
            model.eval()
            with torch.no_grad():
                _, _, weight = model(dataset.vertices, dataset.edges, args.tau)

                train_acc = weight[-1][train_idx].mean().item()
                valid_acc = weight[-1][valid_idx].mean().item()
                test_acc = weight[-1][test_idx].mean().item()

            if valid_acc > best_val:
                best_val = valid_acc
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j

@torch.no_grad()
def evaluate(model, dataset, split_idx, args):
    model.eval()
    _, _, weight = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)

    train_acc = weight[-1][split_idx['train']].mean().item()
    valid_acc = weight[-1][split_idx['valid']].mean().item()
    test_acc = weight[-1][split_idx['test']].mean().item()

    return train_acc, valid_acc, test_acc