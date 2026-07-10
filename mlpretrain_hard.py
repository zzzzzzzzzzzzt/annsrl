"""Minimal MLP link-prediction baseline (BCE version).

Purpose: isolate whether a *plain* MLP can learn the edge (out-neighbor)
structure of the graph from node coordinates alone -- WITHOUT any NodeFormer
topology encoder, Gumbel noise, kernel attention or feature-gating fusion.

Objective: binary cross-entropy. Real edges are labelled 1, randomly sampled
node pairs are labelled 0. This mirrors a simple binary edge classifier (e.g.
the RL decision head): if this baseline's top-N recall climbs while the full
NodeFormer pipeline stays flat, the problem lives in the NodeFormer stack, not
in the supervised framing. If it also fails, the data framing is the suspect.
"""
import argparse
import os
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib.parse import parser_add_main_args
from lib.graph import pretrain_graph
from lib.logger import Logger

import warnings
warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# Model: a plain MLP link predictor. No topology, no attention, no fusion.
# --------------------------------------------------------------------------- #
class MLPLinkNet(nn.Module):
    def __init__(self, in_channels, node_hidden=128, node_layers=2, pair_hidden=128,
                 dropout=0.0, scorer='pair', undirected=False, norm='none'):
        super().__init__()
        if scorer not in ('pair', 'dot'):
            raise ValueError("scorer must be 'pair' or 'dot'")
        if norm not in ('none', 'bn', 'ln'):
            raise ValueError("norm must be 'none', 'bn' or 'ln'")
        self.scorer = scorer
        self.undirected = undirected
        self.dropout = dropout

        # Per-node encoder. node_hidden == 0 -> identity (feed raw coords,
        # exactly mirroring NodeFormer.link_mlp which sees coordinate-space z).
        if node_hidden and node_hidden > 0:
            layers, dim = [], in_channels
            for _ in range(max(1, node_layers)):
                layers.append(nn.Linear(dim, node_hidden))
                # Normalization goes after the linear map, before the ELU.
                if norm == 'bn':
                    layers.append(nn.BatchNorm1d(node_hidden))
                elif norm == 'ln':
                    layers.append(nn.LayerNorm(node_hidden))
                layers.append(nn.ELU())
                dim = node_hidden
            self.node_encoder = nn.Sequential(*layers)
            self.emb_dim = node_hidden
        else:
            self.node_encoder = nn.Identity()
            self.emb_dim = in_channels

        if scorer == 'pair':
            self.link_mlp = nn.Sequential(
                nn.Linear(2 * self.emb_dim, pair_hidden),
                nn.ELU(),
                nn.Linear(pair_hidden, 1),
            )

    def reset_parameters(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters') and m is not self:
                m.reset_parameters()

    # --- node embeddings ---
    def encode(self, x):
        z = self.node_encoder(x)
        z = F.dropout(z, p=self.dropout, training=self.training)
        # Residual: only fires when the encoder preserves the input dim
        # (i.e. node_hidden == in_channels). A no-op otherwise.
        if z.shape == x.shape:
            z = z + x
        return z

    # --- pairwise features (mirrors NodeFormer._pair_features) ---
    def _pair_features(self, src_z, dst_z):
        src_z, dst_z = torch.broadcast_tensors(src_z, dst_z)
        if self.undirected:
            return torch.cat([src_z + dst_z, (src_z - dst_z).abs()], dim=-1)
        return torch.cat([src_z, dst_z], dim=-1)

    # --- raw score (logit) for a specific set of (row, col) edges: [E] ---
    def edge_logits(self, z, edge_index):
        row, col = edge_index
        if self.scorer == 'dot':
            # cosine similarity, temperature-scaled, plus learnable bias
            z_row = F.normalize(z[row], p=2, dim=-1)
            z_col = F.normalize(z[col], p=2, dim=-1)
            return (z_row * z_col).sum(-1)
        pair = self._pair_features(z[row], z[col])
        return self.link_mlp(pair).squeeze(-1)

    # --- score a grid of src x target: returns [S, T] ---
    def all_scores(self, z, src_idx, target_idx):
        if self.scorer == 'dot':
            z_src = F.normalize(z[src_idx], p=2, dim=-1)
            z_tgt = F.normalize(z[target_idx], p=2, dim=-1)
            return (z_src @ z_tgt.t())
        src_z = z[src_idx].unsqueeze(1)          # [S, 1, D]
        dst_z = z[target_idx].unsqueeze(0)       # [1, T, D]
        pair = self._pair_features(src_z, dst_z)  # [S, T, 2D]
        return self.link_mlp(pair).squeeze(-1)    # [S, T]

    # --- scores against ALL nodes (used by top-N metric): [S, N] ---
    def score_all_targets(self, z, src_idx, target_batch_size=2048):
        num_nodes = z.shape[0]
        chunks = []
        for start in range(0, num_nodes, target_batch_size):
            target = torch.arange(start, min(start + target_batch_size, num_nodes), device=z.device)
            chunks.append(self.all_scores(z, src_idx, target))
        return torch.cat(chunks, dim=-1)


# --------------------------------------------------------------------------- #
# Loss: binary cross-entropy over positive edges (label 1) and randomly
# sampled negative node pairs (label 0). Random negatives may occasionally hit
# a true edge, but such false negatives are rare and this is standard practice
# for link-prediction BCE training.
# --------------------------------------------------------------------------- #
def bce_edge_loss(model, z, pos_edges, num_nodes, neg_per_pos):
    pos_logits = model.edge_logits(z, pos_edges)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))

    if neg_per_pos <= 0 or num_nodes <= 1:
        return pos_loss

    neg_src = pos_edges[0].repeat_interleave(neg_per_pos)
    neg_dst = torch.randint(0, num_nodes, (neg_src.numel(),), device=z.device)
    neg_edges = torch.stack([neg_src, neg_dst], dim=0)
    neg_logits = model.edge_logits(z, neg_edges)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    return pos_loss + neg_loss


# --------------------------------------------------------------------------- #
# Loss: InfoNCE. Each positive edge competes against K random negatives from the
# same source in a (1 + K)-way softmax; minimizing cross-entropy pushes the true
# neighbor above the negatives. This is a *ranking* objective, aligned with the
# top-N recall evaluation, rather than the absolute-probability BCE above.
#
# NOTE ON TEMPERATURE: model.edge_logits already scales cosine by a fixed factor
# (see MLPLinkNet.edge_logits, currently x10.0). Dividing again by `temperature`
# here compounds the two. With temperature=0.1 the effective scale is x100, which
# makes the softmax extremely peaked. Keep temperature=1.0 unless you also remove
# the x10.0 in edge_logits. Also note: an additive bias (dot_bias) added equally
# to the positive and all negatives cancels in the softmax, so dot_bias receives
# no gradient under this loss.
# --------------------------------------------------------------------------- #
def infonce_edge_loss(model, z, pos_edges, num_nodes, neg_per_pos, temperature=1.0):
    pos_logits = model.edge_logits(z, pos_edges)          # [E]
    if neg_per_pos <= 0 or num_nodes <= 1:
        return pos_logits.sum() * 0.0

    neg_src = pos_edges[0].repeat_interleave(neg_per_pos)
    neg_dst = torch.randint(0, num_nodes, (neg_src.numel(),), device=z.device)
    neg_edges = torch.stack([neg_src, neg_dst], dim=0)
    neg_logits = model.edge_logits(z, neg_edges)          # [E * K]

    logits = torch.cat([pos_logits.view(-1, 1),
                        neg_logits.view(-1, neg_per_pos)], dim=1)  # [E, 1 + K]
    logits = logits / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------- #
# Metrics (mirror pretrain.py where meaningful)
# --------------------------------------------------------------------------- #
def to_undirected(edge_index):
    """Symmetrize a directed edge_index into an undirected one: for every
    (u, v) also include (v, u), then drop duplicate directed pairs.
    Returns a [2, E'] tensor on the same device."""
    rev = edge_index.flip(0)
    both = torch.cat([edge_index, rev], dim=1)
    return torch.unique(both, dim=1)


def build_out_neighbors(edge_index, num_nodes, device):
    neighbors = [set() for _ in range(num_nodes)]
    src, dst = edge_index.detach().cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            neighbors[u].add(v)
    neighbor_tensors = [torch.tensor(sorted(v), dtype=torch.long, device=device) for v in neighbors]
    degrees = torch.tensor([len(v) for v in neighbors], dtype=torch.long, device=device)
    return neighbor_tensors, degrees


@torch.no_grad()
def topn_neighbor_ratio(model, z, node_idx, out_neighbors, out_degree, batch_size=1024):
    """Fraction of a node's true out-neighbors recovered in its top-degree predictions."""
    device = z.device
    node_idx = node_idx.to(device)
    total_ratio, total_nodes = 0.0, 0
    for src in node_idx.split(batch_size):
        degrees = out_degree[src]
        valid = degrees > 0
        if not valid.any():
            continue
        src = src[valid]
        degrees = degrees[valid]
        max_k = min(int(degrees.max().item()), out_degree.numel() - 1)
        if max_k <= 0:
            continue
        scores = model.score_all_targets(z, src)
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


@torch.no_grad()
def edge_probs(model, z, edge_index, mask, num_nodes, neg_per_pos=1, batch_size=10000):
    """Mean sigmoid(logit) on the masked true edges vs. random negatives from the
    same sources -- a quick read on how well positives separate from noise."""
    if not mask.any():
        return 0.0, 0.0
    sub = edge_index[:, mask]
    pos_sum = 0.0
    neg_sum = 0.0
    pos_count = 0
    neg_count = 0
    batch_size = max(1, int(batch_size))

    for start in range(0, sub.shape[1], batch_size):
        edge_batch = sub[:, start:start + batch_size]
        pos_prob = torch.sigmoid(model.edge_logits(z, edge_batch))
        pos_sum += float(pos_prob.sum().item())
        pos_count += int(pos_prob.numel())

        neg_src = edge_batch[0].repeat_interleave(max(1, neg_per_pos))
        neg_dst = torch.randint(0, num_nodes, (neg_src.numel(),), device=z.device)
        neg = torch.stack([neg_src, neg_dst], dim=0)
        neg_prob = torch.sigmoid(model.edge_logits(z, neg))
        neg_sum += float(neg_prob.sum().item())
        neg_count += int(neg_prob.numel())

    return pos_sum / pos_count, neg_sum / neg_count


@torch.no_grad()
def edge_scores(model, z, edge_index, mask, num_nodes, neg_per_pos=1, batch_size=10000):
    """Raw-score counterpart of edge_probs (no sigmoid): mean edge_logit on true
    edges vs. random negatives. Use with InfoNCE, which optimizes relative
    ranking rather than calibrated probabilities, so sigmoid would be misleading."""
    if not mask.any():
        return 0.0, 0.0
    sub = edge_index[:, mask]
    pos_sum = 0.0
    neg_sum = 0.0
    pos_count = 0
    neg_count = 0
    batch_size = max(1, int(batch_size))

    for start in range(0, sub.shape[1], batch_size):
        edge_batch = sub[:, start:start + batch_size]
        pos_score = model.edge_logits(z, edge_batch)
        pos_sum += float(pos_score.sum().item())
        pos_count += int(pos_score.numel())

        neg_src = edge_batch[0].repeat_interleave(max(1, neg_per_pos))
        neg_dst = torch.randint(0, num_nodes, (neg_src.numel(),), device=z.device)
        neg = torch.stack([neg_src, neg_dst], dim=0)
        neg_score = model.edge_logits(z, neg)
        neg_sum += float(neg_score.sum().item())
        neg_count += int(neg_score.numel())

    return pos_sum / pos_count, neg_sum / neg_count


# --------------------------------------------------------------------------- #
# Plot / CSV
# --------------------------------------------------------------------------- #
def save_history(history, run, args):
    if not history:
        return
    out_dir = os.path.join("results/MLP_baseline", args.output_timestamp,
                           f"scorer{args.scorer}_nodeHidden{args.node_hidden}_lr{args.lr}_bce")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.dataset}_run{run:02d}_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[METRIC_CSV] saved {path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs = [h['epoch'] for h in history]
        fig, (ax_loss, ax_topn) = plt.subplots(1, 2, figsize=(12, 4.8))
        ax_loss.plot(epochs, [h['loss'] for h in history], color='tab:red', label='Loss')
        ax_loss.plot(epochs, [h['train_pos_prob'] for h in history], color='tab:green', label='Train pos prob')
        ax_loss.plot(epochs, [h['train_neg_prob'] for h in history], color='tab:gray', label='Train neg prob')
        ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss / prob'); ax_loss.grid(True, alpha=0.25); ax_loss.legend()
        ax_topn.plot(epochs, [h['train_pos_prob'] - h['train_neg_prob'] for h in history], color='tab:blue', label='Train sep')
        ax_topn.plot(epochs, [h['valid_pos_prob'] - h['valid_neg_prob'] for h in history], color='tab:orange', label='Valid sep')
        ax_topn.set_xlabel('Epoch'); ax_topn.set_ylabel('Pos-neg prob separation'); ax_topn.set_ylim(-1, 1)
        ax_topn.grid(True, alpha=0.25); ax_topn.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"{args.dataset}_run{run:02d}_metrics.svg")
        fig.savefig(p, format='svg'); plt.close(fig)
        print(f"[METRIC_PLOT] saved {p}")
    except Exception as ex:
        print(f"[METRIC_PLOT] skipped ({ex})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='MLP link-prediction baseline (BCE)')
    parser_add_main_args(parser)
    # extra args specific to this baseline
    parser.add_argument('--scorer', type=str, default='pair', choices=['pair', 'dot'],
                        help="'pair'=MLP([z_i,z_j]), 'dot'=<z_i,z_j> two-tower")
    parser.add_argument('--node_hidden', type=int, default=128,
                        help='per-node encoder width; 0 = feed raw coords (no encoder)')
    parser.add_argument('--node_layers', type=int, default=2,
                        help='number of layers in the per-node encoder')
    parser.add_argument('--pair_hidden', type=int, default=128,
                        help='hidden width of the pairwise scoring MLP')
    parser.add_argument('--norm', type=str, default='none', choices=['none', 'bn', 'ln'],
                        help="normalization in the node encoder, applied after each "
                             "Linear and before the ELU: 'bn'=BatchNorm1d, 'ln'=LayerNorm")
    parser.add_argument('--neg_per_pos', type=int, default=5,
                        help='number of random negative pairs sampled per positive edge for BCE')
    # parser.add_argument('--undirected', action='store_true',
    #                     help='symmetrize the supervision edges: for every (u,v) also add (v,u)')
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'infonce'],
                        help="'bce'=per-pair binary cross-entropy (sigmoid prob monitoring); "
                             "'infonce'=softmax over 1 pos + K neg per source (raw-score monitoring)")
    parser.add_argument('--infonce_temp', type=float, default=0.1,
                        help='softmax temperature for the InfoNCE loss (only used when --loss infonce)')
    args = parser.parse_args()
    args.output_timestamp = os.environ.get('RUN_TIMESTAMP', time.strftime('%Y%m%d_%H%M%S'))
    print(args)

    fix_seed(args.seed)
    device = torch.device("cpu") if args.cpu else \
        (torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu"))

    dataset = pretrain_graph(args.vertices_path, args.edges_path, graph_type=args.graph_type,
                             train_prop=args.train_prop, valid_prop=args.valid_prop)
    n = dataset.vertices_size
    e = dataset.edges.shape[1]
    d = dataset.vertices.shape[1]
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d}")

    dataset.vertices = dataset.vertices.to(device)

    # Force feature standardization: skewed raw coordinates can starve the
    # encoder of gradient. Zero-mean/unit-std per feature dimension.
    mean = dataset.vertices.mean(dim=0, keepdim=True)
    std = dataset.vertices.std(dim=0, keepdim=True)
    dataset.vertices = (dataset.vertices - mean) / (std + 1e-6)

    dataset.edges = dataset.edges.to(device)
    dataset.train_edges = dataset.train_edges.to(device)
    train_idx = dataset.split_idx_lst['train'].to(device)
    valid_idx = dataset.split_idx_lst['valid'].to(device)
    test_idx = dataset.split_idx_lst['test'].to(device)

    # number of nodes in the (relabeled) train subgraph used for the loss
    num_train_nodes = int(train_idx.numel())
    train_edge_ids = torch.arange(dataset.train_edges.shape[1], dtype=torch.long)
    train_edge_dataset = TensorDataset(train_edge_ids)
    out_neighbors, out_degree = build_out_neighbors(dataset.edges, n, device)

    model = MLPLinkNet(d, node_hidden=args.node_hidden, node_layers=args.node_layers,
                       pair_hidden=args.pair_hidden, dropout=args.dropout,
                       scorer=args.scorer).to(device)
    print('MODEL:', model)
    logger = Logger(args.runs, args)

    edge_src = dataset.edges[0]
    train_mask = torch.isin(edge_src, train_idx)
    valid_mask = torch.isin(edge_src, valid_idx)
    test_mask = torch.isin(edge_src, test_idx)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')
        best_state = None
        best_epoch = -1
        history = []
        train_edge_loader = DataLoader(
            train_edge_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            # Encode all nodes, then gather train nodes because train_edges is
            # relabeled into the compact train-node index space.
            z_full = model.encode(dataset.vertices)
            z = z_full[train_idx]

            total_edges = dataset.train_edges.shape[1]
            num_batches = len(train_edge_loader)
            loss_value = 0.0

            for batch_id, (batch_idx,) in enumerate(train_edge_loader):
                batch_idx = batch_idx.to(device)
                edge_batch = dataset.train_edges[:, batch_idx]
                if args.loss == 'infonce':
                    batch_loss = infonce_edge_loss(model, z, edge_batch, num_train_nodes,
                                                   args.neg_per_pos, temperature=args.infonce_temp)
                else:
                    batch_loss = bce_edge_loss(model, z, edge_batch, num_train_nodes, args.neg_per_pos)

                # scaled_loss = batch_loss * (edge_batch.shape[1] / total_edges)
                batch_loss.backward(retain_graph=batch_id < num_batches - 1)
                loss_value += float(batch_loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % args.eval_step == 0 and epoch > 0:
                model.eval()
                # Cheap eval only: edge-probability separation. The expensive
                # top-N neighbor recall is computed once at the end on the best
                # model rather than every eval step.
                # Test edges are NOT scored during training -- only train/valid.
                # Test is held out and evaluated once at the end via top-N recall.
                with torch.no_grad():
                    z_full = model.encode(dataset.vertices)
                    monitor = edge_scores if args.loss == 'infonce' else edge_probs
                    train_pos, train_neg = monitor(model, z_full, dataset.edges, train_mask, n,
                                                   args.neg_per_pos, args.batch_size)
                    valid_pos, valid_neg = monitor(model, z_full, dataset.edges, valid_mask, n,
                                                   args.neg_per_pos, args.batch_size)

                # Model selection on validation positive score (how confidently
                # the model scores true valid edges), not the pos-neg gap.
                history.append({
                    'epoch': epoch, 'loss': loss_value,
                    'train_pos_prob': train_pos, 'train_neg_prob': train_neg,
                    'valid_pos_prob': valid_pos, 'valid_neg_prob': valid_neg,
                })
                if valid_pos > best_val:
                    best_val = valid_pos
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

                print(f'Epoch: {epoch:03d}, Loss: {loss_value:.6f}, '
                      f'PosProb(tr/va): {train_pos:.4f}/{valid_pos:.4f}, '
                      f'NegProb(tr/va): {train_neg:.4f}/{valid_neg:.4f}')

        # Compute the expensive top-N neighbor recall exactly once, on the best
        # model as selected by validation positive score.
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            z_full = model.encode(dataset.vertices)
            train_topn = topn_neighbor_ratio(model, z_full, train_idx, out_neighbors, out_degree)
            valid_topn = topn_neighbor_ratio(model, z_full, valid_idx, out_neighbors, out_degree)
            test_topn = topn_neighbor_ratio(model, z_full, test_idx, out_neighbors, out_degree)
        print(f'[BEST @ epoch {best_epoch}] Top-N recall  '
              f'train: {train_topn:.6f}  valid: {valid_topn:.6f}  test: {test_topn:.6f}')
        # Logger tracks top-N recall in the (train, valid, test, loss) slots.
        logger.add_result(run, (train_topn, valid_topn, test_topn, best_val))

        save_history(history, run, args)
        logger.print_statistics(run)

    logger.print_statistics()


if __name__ == '__main__':
    main()
