import argparse
import sys
import os, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, subgraph
from sklearn.neighbors import kneighbors_graph

from lib.logger import Logger
from lib.parse import parser_add_main_args
from lib.graph import pretrain_graph
from lib.Nodeformer import NodeFormer
from lib.edge_pi_viz import maybe_plot_edge_attention
from lib.z_viz import maybe_plot_z_layers
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

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j


def current_tau(args, epoch):
    if args.tau_min is None or args.epochs <= 1:
        return args.tau
    progress = epoch / (args.epochs - 1)
    return args.tau + (args.tau_min - args.tau) * progress


def edge_mass(edge_weight, edge_src, mask, num_nodes):
    scores = edge_weight[:, mask].mean(dim=0)
    src = edge_src[mask]

    mass_per_node = torch.zeros(num_nodes, device=edge_weight.device, dtype=scores.dtype)
    active_node = torch.zeros(num_nodes, device=edge_weight.device, dtype=torch.bool)

    mass_per_node.scatter_add_(0, src, scores)
    active_node[src] = True
    if not active_node.any():
        return 0.0

    return mass_per_node[active_node].mean().item()


def save_metric_plot(history, run, args):
    if not history:
        return
    out_dir = os.path.join('results', 'pretrain_metrics')
    os.makedirs(out_dir, exist_ok=True)

    epochs = [item['epoch'] for item in history]
    fig, ax_loss = plt.subplots(figsize=(8, 4.8))
    ax_mass = ax_loss.twinx()

    ax_loss.plot(epochs, [item['loss'] for item in history], label='Loss', color='tab:red')
    ax_loss.plot(epochs, [item['tau'] for item in history], label='Tau', color='tab:purple', linestyle='--')
    ax_mass.plot(epochs, [item['train_mass'] for item in history], label='Train_mass', color='tab:blue')
    ax_mass.plot(epochs, [item['valid_mass'] for item in history], label='Valid_mass', color='tab:green')
    ax_mass.plot(epochs, [item['test_mass'] for item in history], label='Test_mass', color='tab:orange')

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_mass.set_ylabel('Edge mass')
    ax_loss.grid(True, alpha=0.25)

    lines = ax_loss.get_lines() + ax_mass.get_lines()
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc='best')
    fig.tight_layout()

    path = os.path.join(out_dir, f'{args.dataset}_{args.method}_run{run:02d}_metrics.svg')
    fig.savefig(path, format='svg')
    plt.close(fig)
    print(f'[METRIC_PLOT] saved {path}')

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
dataset = pretrain_graph(args.vertices_path, args.edges_path, graph_type=args.graph_type,
                         train_prop=args.train_prop, valid_prop=args.valid_prop)

### Basic information of datasets ###
n = dataset.vertices_size
e = dataset.edges.shape[1]
d = dataset.vertices.shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d}")

### Load method ###
model=NodeFormer(d, args.hidden_channels, d, num_layers=args.num_layers, dropout=args.dropout,
            num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
            use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
            nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans,
            sample_hop=args.sample_hop, mass_alpha=args.mass_alpha).to(device)

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
# dataset.graph['adjs'] = adjs

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
    metric_history = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        tau = current_tau(args, epoch)
        _, link_loss_, _ = model(dataset.vertices[train_idx], dataset.train_edges, tau)
        loss = link_loss_[-1]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % args.eval_step == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                _, _, weight, z_stages = model(dataset.vertices, dataset.edges, tau, return_z=True)
                edge_weight = weight[-1]
                edge_src = dataset.edges[0]
                train_edge_mask = torch.isin(edge_src, train_idx)
                valid_edge_mask = torch.isin(edge_src, valid_idx)
                test_edge_mask = torch.isin(edge_src, test_idx)

                train_acc = edge_mass(edge_weight, edge_src, train_edge_mask, n)
                valid_acc = edge_mass(edge_weight, edge_src, valid_edge_mask, n)
                test_acc = edge_mass(edge_weight, edge_src, test_edge_mask, n)

                maybe_plot_edge_attention(model, dataset.edges, z_stages, tau, run, epoch, args.seed)
                maybe_plot_z_layers(z_stages, tau, run, epoch, args.seed)
            logger.add_result(run, (train_acc, valid_acc, test_acc, loss.item()))
            metric_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'tau': tau,
                'train_mass': train_acc,
                'valid_mass': valid_acc,
                'test_mass': test_acc,
            })

            if valid_acc > best_val:
                best_val = valid_acc
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

            # print(f'Epoch: {epoch:02d}, '
            #       f'Loss: {loss:.6f}, '
            #       f'Train: {100 * train_acc:.4f}%, '
            #       f'Valid: {100 * valid_acc:.4f}%, '
            #       f'Test: {100 * test_acc:.4}%')

            print(f'Epoch: {epoch:02d}, '
                f'Tau: {tau:.4f}, '
                f'Loss: {loss:.6f}, '
                f'Train_mass: {train_acc:.8f}, '
                f'Valid_mass: {valid_acc:.8f}, '
                f'Test_mass: {test_acc:.8f}')

    save_metric_plot(metric_history, run, args)
    logger.print_statistics(run)

results = logger.print_statistics()


@torch.no_grad()
def evaluate(model, dataset, split_idx, args):
    model.eval()
    _, _, weight = model(dataset.vertices, dataset.edges, args.tau)
    edge_weight = weight[-1]
    edge_src = dataset.edges[0]
    train_edge_mask = torch.isin(edge_src, split_idx["train"].to(edge_src.device))
    valid_edge_mask = torch.isin(edge_src, split_idx["valid"].to(edge_src.device))
    test_edge_mask = torch.isin(edge_src, split_idx["test"].to(edge_src.device))

    train_acc = edge_mass(edge_weight, edge_src, train_edge_mask, n)
    valid_acc = edge_mass(edge_weight, edge_src, valid_edge_mask, n)
    test_acc = edge_mass(edge_weight, edge_src, test_edge_mask, n)

    return train_acc, valid_acc, test_acc