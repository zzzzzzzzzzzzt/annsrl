import argparse
import sys
import os, random, csv
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


def build_out_neighbors(edge_index, num_nodes, device):
    neighbors = [set() for _ in range(num_nodes)]
    src, dst = edge_index.detach().cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            neighbors[u].add(v)

    neighbor_tensors = [torch.tensor(sorted(v), dtype=torch.long, device=device) for v in neighbors]
    degrees = torch.tensor([len(v) for v in neighbors], dtype=torch.long, device=device)
    return neighbor_tensors, degrees


def negative_type_counts(neg_ratio, hnsw_m, neg_type_ratios):
    total = max(0, int(round(neg_ratio * hnsw_m)))
    ratio_sum = sum(neg_type_ratios)
    if ratio_sum <= 0:
        raise ValueError("neg_type_ratios must sum to a positive value")
    ratios = [r / ratio_sum for r in neg_type_ratios]
    raw = [total * r for r in ratios]
    counts = [int(v) for v in raw]
    for i in sorted(range(3), key=lambda k: raw[k] - counts[k], reverse=True)[:total - sum(counts)]:
        counts[i] += 1
    return counts


def take_candidates(selected, candidates, target):
    for v in candidates:
        if v not in selected:
            selected.add(v)
            if len(selected) >= target:
                break


def fill_random_negatives(selected, neighbors_i, node, num_nodes, target):
    target = min(target, num_nodes - 1 - len(neighbors_i))
    attempts = 0
    while len(selected) < target and attempts < 100 + 20 * target:
        v = int(torch.randint(num_nodes, (1,)).item())
        if v != node and v not in neighbors_i and v not in selected:
            selected.add(v)
        attempts += 1
    for v in range(num_nodes):
        if len(selected) >= target:
            break
        if v != node and v not in neighbors_i and v not in selected:
            selected.add(v)


def h_hop_candidates(neighbors, node, negative_hop):
    visited, frontier = {node}, {node}
    for _ in range(negative_hop):
        frontier = {v for u in frontier for v in neighbors[u]} - visited
        if not frontier:
            break
        visited.update(frontier)
    return [v for v in visited if v != node and v not in neighbors[node]]


@torch.no_grad()
def build_negative_edges(x, pos_edges, args):
    if args.negative_hop < 1:
        raise ValueError("negative_hop must be at least 1")
    x = x.detach()
    num_nodes = x.shape[0]
    if num_nodes <= 1:
        return pos_edges.new_empty((2, 0))

    row, col = pos_edges.detach().cpu().tolist()
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in zip(row, col):
        if u != v:
            neighbors[u].add(v)

    hard_negative_k = args.hard_negative_k or args.hnsw_m
    hard_k = max(1, hard_negative_k // 2) if args.hard_negative_mode == 'topk_half' else hard_negative_k
    type_counts = negative_type_counts(args.neg_ratio, args.hnsw_m, args.neg_type_ratios)
    neg_src, neg_dst = [], []

    for i in range(num_nodes):
        selected = set()
        neighbors_i = neighbors[i]
        if len(neighbors_i) >= num_nodes - 1:
            continue

        if type_counts[0] > 0 and neighbors_i:
            neighbor_idx = torch.tensor(list(neighbors_i), device=x.device)
            max_neighbor_dist = torch.norm(x[neighbor_idx] - x[i], dim=1).max().item()
            k = min(num_nodes, hard_k + 1)
            dist, idx = torch.topk(torch.norm(x - x[i], dim=1), k=k, largest=False)
            hard_candidates = [
                v for v, d in zip(idx.cpu().tolist(), dist.cpu().tolist())
                if v != i and v not in neighbors_i and d < max_neighbor_dist
            ]
            take_candidates(selected, hard_candidates, type_counts[0])

        if type_counts[1] > 0:
            take_candidates(
                selected,
                h_hop_candidates(neighbors, i, args.negative_hop),
                len(selected) + type_counts[1],
            )

        fill_random_negatives(selected, neighbors_i, i, num_nodes, sum(type_counts))
        selected = sorted(selected)
        neg_src.extend([i] * len(selected))
        neg_dst.extend(selected)

    if not neg_src:
        return pos_edges.new_empty((2, 0))
    return torch.tensor([neg_src, neg_dst], dtype=torch.long, device=pos_edges.device)


@torch.no_grad()
def eval_edge_weight_and_kernel(model, vertices, edges, tau):
    x = vertices.unsqueeze(0)
    adjs = edges.unsqueeze(0)
    topology = model._encode_topology(x, adjs, tau)
    fused_z = model._fuse_features(x, topology)
    query_prime, key_prime = model._link_kernel(fused_z, tau)
    edge_weight = model._edge_prob(query_prime, key_prime, edges).clamp_min(1e-30)
    return edge_weight, query_prime[0, :, 0], key_prime[0, :, 0]


def topn_neighbor_ratio(query_prime, key_prime, node_idx, out_neighbors, out_degree, batch_size):
    device = query_prime.device
    key_prime_t = key_prime.t()
    node_idx = node_idx.to(device)
    total_ratio = 0.0
    total_nodes = 0

    for src in node_idx.split(batch_size):
        degrees = out_degree[src]
        valid = degrees > 0
        if not valid.any():
            continue

        src = src[valid]
        degrees = degrees[valid]
        max_k = min(int(degrees.max().item()), key_prime.shape[0] - 1)
        if max_k <= 0:
            continue

        scores = query_prime[src].matmul(key_prime_t)
        scores[torch.arange(src.numel(), device=device), src] = -float('inf')
        top_idx = scores.topk(max_k, dim=1).indices

        for row in range(src.numel()):
            k = int(degrees[row].item())
            pred = top_idx[row, :k]
            truth = out_neighbors[int(src[row].item())]
            hits = (pred[:, None] == truth[None, :]).any(dim=1).sum().item()
            total_ratio += hits / k
            total_nodes += 1

    return total_ratio / total_nodes if total_nodes > 0 else 0.0


def neighbor_prob_mass(query_prime, key_prime, edge_index, num_nodes):
    row, col = edge_index
    keep = row != col
    row, col = row[keep], col[keep]
    key_sum = key_prime.sum(dim=0)
    numerator = (query_prime[row] * key_prime[col]).sum(dim=-1)
    denominator = (query_prime[row] * (key_sum - key_prime[row])).sum(dim=-1).clamp_min(1e-30)
    edge_prob = numerator / denominator

    mass = torch.zeros(num_nodes, device=query_prime.device, dtype=query_prime.dtype)
    mass.scatter_add_(0, row, edge_prob)
    return mass


def choose_representative_node(query_prime, key_prime, edge_index, candidate_idx, out_degree, num_nodes):
    mass = neighbor_prob_mass(query_prime, key_prime, edge_index, num_nodes)
    candidates = candidate_idx[out_degree[candidate_idx] > 0]
    if candidates.numel() == 0:
        candidates = torch.nonzero(out_degree > 0, as_tuple=False).view(-1)
    if candidates.numel() == 0:
        return None, mass

    degree_vals = out_degree[candidates].float()
    mass_vals = mass[candidates]
    degree_scale = (degree_vals.max() - degree_vals.min()).clamp_min(1.0)
    mass_scale = (mass_vals.max() - mass_vals.min()).clamp_min(1e-12)
    distance = (degree_vals - degree_vals.median()).abs() / degree_scale
    distance += (mass_vals - mass_vals.median()).abs() / mass_scale
    return int(candidates[distance.argmin()].item()), mass


def save_representative_prob_plot(query_prime, key_prime, edge_index, test_idx, out_neighbors, out_degree, run, args):
    num_nodes = query_prime.shape[0]
    node_id, mass = choose_representative_node(query_prime, key_prime, edge_index, test_idx, out_degree, num_nodes)
    if node_id is None:
        print('[NODE_PROB_PLOT] skipped: no node with non-self outgoing edges')
        return

    scores = query_prime[node_id].matmul(key_prime.t())
    scores[node_id] = 0
    probs = scores / scores.sum().clamp_min(1e-30)

    degree_i = int(out_degree[node_id].item())
    topk = min(max(50, 3 * degree_i), 200, num_nodes - 1)
    all_nodes = torch.arange(num_nodes, device=query_prime.device)
    all_nodes = all_nodes[all_nodes != node_id]
    sorted_all_probs, sorted_all_order = probs[all_nodes].sort(descending=True)
    sorted_all_probs = sorted_all_probs.clamp_min(1e-30)
    sorted_all_nodes = all_nodes[sorted_all_order]
    top_nodes = sorted_all_nodes[:topk]

    selected = torch.zeros(num_nodes, device=query_prime.device, dtype=torch.bool)
    selected[top_nodes] = True
    selected[out_neighbors[node_id]] = True
    selected[node_id] = False

    display_nodes = torch.nonzero(selected, as_tuple=False).view(-1)
    display_probs = probs[display_nodes]
    display_probs, order = display_probs.sort(descending=True)
    display_nodes = display_nodes[order]

    is_neighbor = torch.zeros(num_nodes, device=query_prime.device, dtype=torch.bool)
    is_neighbor[out_neighbors[node_id]] = True
    display_is_neighbor = is_neighbor[display_nodes].detach().cpu().numpy()

    out_dir = os.path.join('results/Edge_Weight_viz', f'pretrain_metrics_oldlossfuction_topologyActivation{args.topology_activation}'
                           f'_topology_factor{args.topology_factor}_lossFunction{args.loss_function}/{args.tau}tau')
    os.makedirs(out_dir, exist_ok=True)

    curve_k = min(max(100, 5 * degree_i), 500, num_nodes - 1)
    curve_probs = sorted_all_probs[:curve_k]
    curve_nodes = sorted_all_nodes[:curve_k]
    rank_x = np.arange(1, curve_k + 1)
    curve_y = curve_probs.detach().cpu().numpy()
    curve_is_neighbor = is_neighbor[curve_nodes].detach().cpu().numpy()

    log_gaps = sorted_all_probs[:-1].log() - sorted_all_probs[1:].log()
    max_gap_rank = int(log_gaps[:max(curve_k - 1, 1)].argmax().item()) + 1
    max_gap = log_gaps[max_gap_rank - 1].item()
    if degree_i < sorted_all_probs.numel():
        degree_gap = (sorted_all_probs[degree_i - 1].log() - sorted_all_probs[degree_i].log()).item()
    else:
        degree_gap = float('nan')

    x = np.arange(display_nodes.numel())
    colors = np.where(display_is_neighbor, 'tab:orange', 'tab:blue')
    fig, axes = plt.subplots(1, 3, figsize=(22, 4.8), gridspec_kw={'width_ratios': [1.35, 1, 1]})

    axes[0].bar(x, display_probs.detach().cpu().numpy(), color=colors, width=0.85)
    axes[0].scatter([], [], marker='s', color='tab:orange', label='True out-neighbor')
    axes[0].scatter([], [], marker='s', color='tab:blue', label='Top prediction')
    axes[0].set_xlabel('Target node, sorted by probability')
    axes[0].set_ylabel(f'p({node_id} -> j)')
    axes[0].set_title('Selected target bars')
    axes[0].grid(True, axis='y', alpha=0.25)
    axes[0].legend(loc='best')

    tick_step = max(1, display_nodes.numel() // 45)
    tick_pos = x[::tick_step]
    axes[0].set_xticks(tick_pos)
    axes[0].set_xticklabels(display_nodes.detach().cpu().numpy()[::tick_step], rotation=90, fontsize=6)

    for ax, use_log in [(axes[1], False), (axes[2], True)]:
        ax.plot(rank_x, curve_y, color='tab:blue', linewidth=1.6, label='Sorted edge weight')
        ax.scatter(rank_x[curve_is_neighbor], curve_y[curve_is_neighbor], color='tab:orange', s=18,
                   label='True out-neighbor', zorder=3)
        ax.axvline(degree_i, color='tab:green', linestyle='--', linewidth=1.2, label=f'true degree={degree_i}')
        ax.axvline(max_gap_rank, color='tab:red', linestyle=':', linewidth=1.5, label=f'max drop={max_gap_rank}')
        if use_log:
            ax.set_yscale('log')
            ax.set_title('Sorted curve (log y)')
        else:
            ax.set_title('Sorted curve')
        ax.set_xlabel('Sorted rank')
        ax.set_ylabel(f'p({node_id} -> j)')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=8)

    fig.suptitle(
        f'Representative node {node_id} | degree={degree_i} | neighbor mass={mass[node_id].item():.4f} | '
        f'max gap rank={max_gap_rank}, gap={max_gap:.3f}, degree gap={degree_gap:.3f}'
    )
    fig.tight_layout()

    path = os.path.join(out_dir, f'{args.dataset}_{args.method}_run{run:02d}_representative_node{node_id}_prob.svg')
    fig.savefig(path, format='svg')
    plt.close(fig)
    print(f'[NODE_PROB_PLOT] saved {path}')


def save_metric_plot(history, run, args):
    if not history:
        return
    out_dir = os.path.join('results/Topology_factor&Activation_Experiment_pretrain_metrics', f'pretrain_metrics_oldlossfuction_topologyActivation{args.topology_activation}'
                           f'_topology_factor{args.topology_factor}_lossFunction{args.loss_function}/{args.tau}tau')
    os.makedirs(out_dir, exist_ok=True)

    epochs = [item['epoch'] for item in history]
    has_topn = 'train_topn_ratio' in history[0]
    if has_topn:
        fig, (ax_loss, ax_topn) = plt.subplots(1, 2, figsize=(12, 4.8))
    else:
        fig, ax_loss = plt.subplots(figsize=(8, 4.8))
        ax_topn = None
    ax_mass = ax_loss.twinx()

    ax_loss.plot(epochs, [item['loss'] for item in history], label='Loss', color='tab:red')
    # ax_loss.plot(epochs, [item['tau'] for item in history], label='Tau', color='tab:purple', linestyle='--')
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

    if has_topn:
        ax_topn.plot(epochs, [item['train_topn_ratio'] for item in history], label='Train_topN', color='tab:blue')
        ax_topn.plot(epochs, [item['valid_topn_ratio'] for item in history], label='Valid_topN', color='tab:green')
        ax_topn.plot(epochs, [item['test_topn_ratio'] for item in history], label='Test_topN', color='tab:orange')
        ax_topn.set_xlabel('Epoch')
        ax_topn.set_ylabel('Top-N recall')
        ax_topn.set_ylim(0, 1)
        ax_topn.grid(True, alpha=0.25)
        ax_topn.legend(loc='best')

    fig.tight_layout()

    path = os.path.join(out_dir, f'{args.dataset}_{args.method}_run{run:02d}_metrics.svg')
    fig.savefig(path, format='svg')
    plt.close(fig)
    print(f'[METRIC_PLOT] saved {path}')

def save_metric_history(history, run, args):
    if not history:
        return
    out_dir = os.path.join("results/Topology_factor&Activation_Experiment", f"pretrain_metrics_oldlossfuction_topologyActivation{args.topology_activation}"
                           f"_topology_factor{args.topology_factor}_lossFunction{args.loss_function}/{args.tau}tau")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{args.dataset}_{args.method}_run{run:02d}_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[METRIC_CSV] saved {path}")

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
            topology_factor=args.topology_factor,
            topology_activation=args.topology_activation, loss_function=args.loss_function).to(device)

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

out_neighbors, out_degree = build_out_neighbors(dataset.edges, n, device)
uses_negative_edges = args.loss_function in ['contrastive', 'contrastive_only_numerator', 'sigmoid_loss']
train_neg_edges = build_negative_edges(dataset.vertices[train_idx], dataset.train_edges, args) if uses_negative_edges else None

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

        _, loss, _ = model(dataset.vertices[train_idx], dataset.train_edges, args.tau, negative_edges=train_neg_edges)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % args.eval_step == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                _, _, edge_weight = model(dataset.vertices, dataset.edges, args.tau)
                edge_weight, query_prime, key_prime = eval_edge_weight_and_kernel(model, dataset.vertices, dataset.edges, args.tau)
                edge_src = dataset.edges[0]
                train_edge_mask = torch.isin(edge_src, train_idx)
                valid_edge_mask = torch.isin(edge_src, valid_idx)
                test_edge_mask = torch.isin(edge_src, test_idx)

                train_acc = edge_mass(edge_weight, edge_src, train_edge_mask, n)
                valid_acc = edge_mass(edge_weight, edge_src, valid_edge_mask, n)
                test_acc = edge_mass(edge_weight, edge_src, test_edge_mask, n)
                train_topn = topn_neighbor_ratio(query_prime, key_prime, train_idx, out_neighbors, out_degree, args.topn_batch_size)
                valid_topn = topn_neighbor_ratio(query_prime, key_prime, valid_idx, out_neighbors, out_degree, args.topn_batch_size)
                test_topn = topn_neighbor_ratio(query_prime, key_prime, test_idx, out_neighbors, out_degree, args.topn_batch_size)

            logger.add_result(run, (train_acc, valid_acc, test_acc, loss.item()))
            metric_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'train_mass': train_acc,
                'valid_mass': valid_acc,
                'test_mass': test_acc,
                'train_topn_ratio': train_topn,
                'valid_topn_ratio': valid_topn,
                'test_topn_ratio': test_topn,
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
                f'Loss: {loss:.6f}, '
                f'Train_mass: {train_acc:.8f}, '
                f'Valid_mass: {valid_acc:.8f}, '
                f'Test_mass: {test_acc:.8f}, '
                f'Train_topN: {train_topn:.6f}, '
                f'Valid_topN: {valid_topn:.6f}, '
                f'Test_topN: {test_topn:.6f}'
                )

    save_metric_plot(metric_history, run, args)
    save_metric_history(metric_history, run, args)
    model.eval()
    with torch.no_grad():
        _, query_prime, key_prime = eval_edge_weight_and_kernel(model, dataset.vertices, dataset.edges, args.tau)
    save_representative_prob_plot(query_prime, key_prime, dataset.edges, test_idx, out_neighbors, out_degree, run, args)
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
