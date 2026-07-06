import argparse
import csv
import os
import random
from collections import defaultdict
from struct import unpack

import numpy as np


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def fix_torch_seed(torch_module, seed):
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def count_fvecs(filename):
    count = 0
    dim = None
    with open(filename, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            current_dim, = unpack("<i", header)
            if dim is None:
                dim = current_dim
            f.seek(4 * current_dim, os.SEEK_CUR)
            count += 1
    return count, dim


def read_edges(filename, max_size=None):
    max_size = max_size or float("inf")
    edges = defaultdict(list)
    with open(filename, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            dim, = unpack("<i", header)
            vec = unpack("i" * dim, f.read(4 * dim))
            edges[len(edges)] = vec
            if len(edges) >= max_size:
                break
    return edges


def read_nsg(filename, max_size=None):
    max_size = max_size or float("inf")
    edges = defaultdict(list)
    with open(filename, "rb") as f:
        f.read(8)
        while True:
            header = f.read(4)
            if not header:
                break
            dim, = unpack("<i", header)
            vec = unpack("i" * dim, f.read(4 * dim))
            edges[len(edges)] = vec
            if len(edges) >= max_size:
                break
    return edges


def load_edge_index(edges_path, graph_type, num_nodes):
    if graph_type == "nsw":
        edge_dict = read_edges(edges_path, num_nodes)
    elif graph_type == "nsg":
        edge_dict = read_nsg(edges_path, num_nodes)
    else:
        raise ValueError("Only ['nsw', 'nsg'] graph types are supported")

    src, dst = [], []
    for u, neighbors in edge_dict.items():
        for v in neighbors:
            src.append(u)
            dst.append(v)

    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def degree_summary(values):
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def print_degree_summary(name, values):
    summary = degree_summary(values)
    print(
        f"{name}: min={summary['min']:.0f}, max={summary['max']:.0f}, "
        f"mean={summary['mean']:.4f}, median={summary['median']:.0f}"
    )


def analyze_train_idx(num_nodes, src, dst, train_idx):
    is_train = np.zeros(num_nodes, dtype=bool)
    is_train[train_idx] = True

    original_out_degree = np.bincount(src, minlength=num_nodes)
    train_edge_mask = is_train[src] & is_train[dst]
    train_src = src[train_edge_mask]
    train_out_degree = np.bincount(train_src, minlength=num_nodes)
    missing_out_degree = original_out_degree - train_out_degree

    return {
        "train_idx": train_idx,
        "is_train": is_train,
        "original_out_degree": original_out_degree,
        "train_out_degree": train_out_degree,
        "missing_out_degree": missing_out_degree,
        "train_edge_mask": train_edge_mask,
    }


def analyze_split(num_nodes, src, dst, train_prop):
    train_num = int(num_nodes * train_prop)
    perm = np.random.permutation(num_nodes)
    train_idx = perm[:train_num]
    return analyze_train_idx(num_nodes, src, dst, train_idx)


def load_with_pretrain_graph(args):
    import torch
    from lib.graph import pretrain_graph

    fix_seed(args.seed)
    fix_torch_seed(torch, args.seed)
    dataset = pretrain_graph(
        args.vertices_path,
        args.edges_path,
        graph_type=args.graph_type,
        train_prop=args.train_prop,
        valid_prop=args.valid_prop,
    )

    edges = tensor_to_numpy(dataset.edges).astype(np.int64)
    train_idx = tensor_to_numpy(dataset.split_idx_lst["train"]).astype(np.int64)
    src, dst = edges[0], edges[1]
    num_nodes = int(dataset.vertices_size)
    num_features = int(dataset.vertices.shape[1])
    stats = analyze_train_idx(num_nodes, src, dst, train_idx)
    return num_nodes, num_features, src, dst, stats, "lib.graph.pretrain_graph"


def load_with_raw_reader(args):
    fix_seed(args.seed)
    num_nodes, num_features = count_fvecs(args.vertices_path)
    src, dst = load_edge_index(args.edges_path, args.graph_type, num_nodes)
    stats = analyze_split(num_nodes, src, dst, args.train_prop)
    return num_nodes, num_features, src, dst, stats, "raw file reader"


def load_graph_data(args):
    if args.load_mode == "raw":
        return load_with_raw_reader(args)

    try:
        return load_with_pretrain_graph(args)
    except ImportError as error:
        if args.load_mode == "graph":
            raise RuntimeError(
                "Could not import lib.graph.pretrain_graph. Please run with the same environment used for training."
            ) from error

        print(f"lib.graph.pretrain_graph unavailable ({error}); falling back to raw file reader.")
        print()
        return load_with_raw_reader(args)


def write_node_csv(path, stats):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    original_out_degree = stats["original_out_degree"]
    train_out_degree = stats["train_out_degree"]
    missing_out_degree = stats["missing_out_degree"]
    is_train = stats["is_train"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_id",
                "split",
                "original_out_degree",
                "train_out_degree",
                "missing_out_degree",
                "train_edge_keep_ratio",
            ]
        )
        for node_id in range(original_out_degree.size):
            original_degree = int(original_out_degree[node_id])
            train_degree = int(train_out_degree[node_id])
            keep_ratio = train_degree / original_degree if original_degree > 0 else 0.0
            writer.writerow(
                [
                    node_id,
                    "train" if is_train[node_id] else "heldout",
                    original_degree,
                    train_degree,
                    int(missing_out_degree[node_id]),
                    f"{keep_ratio:.8f}",
                ]
            )


def default_csv_path(args):
    dataset = args.dataset or "dataset"
    return os.path.join(
        "results",
        "split_edge_analysis",
        f"{dataset}_seed{args.seed}_train{args.train_prop:g}_valid{args.valid_prop:g}_node_degrees.csv",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how many original graph edges remain in the train-node induced subgraph."
    )
    parser.add_argument("--dataset", type=str, default="DEEP10K")
    parser.add_argument("--graph_type", type=str, default="nsw")
    parser.add_argument("--vertices_path", type=str, default="data/DEEP100K/deep10k/deep_base_random10000_seed42.fvecs")
    parser.add_argument("--edges_path", type=str, default="data/DEEP100K/deep10k/deep_hnsw_M12_efC300_random10000_seed42.ivecs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_prop", type=float, default=0.8)
    parser.add_argument("--valid_prop", type=float, default=0.2)
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save per-node original/train out-degree statistics.",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Only print summary statistics and do not write the per-node CSV.",
    )
    parser.add_argument(
        "--load_mode",
        choices=["auto", "graph", "raw"],
        default="auto",
        help="auto prefers lib.graph.pretrain_graph and falls back to raw reading if training deps are unavailable.",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"ignored training-only args: {' '.join(unknown_args)}")
        print()

    if not args.vertices_path:
        raise ValueError("--vertices_path is required")
    if not args.edges_path:
        raise ValueError("--edges_path is required")
    if not args.graph_type:
        raise ValueError("--graph_type is required")

    num_nodes, num_features, src, dst, stats, loader = load_graph_data(args)

    original_edge_count = int(src.size)
    train_edge_count = int(stats["train_edge_mask"].sum())
    missing_edge_count = original_edge_count - train_edge_count
    keep_ratio = train_edge_count / original_edge_count if original_edge_count > 0 else 0.0

    train_idx = stats["train_idx"]
    is_train = stats["is_train"]
    original_out_degree = stats["original_out_degree"]
    train_out_degree = stats["train_out_degree"]
    missing_out_degree = stats["missing_out_degree"]

    train_original_out = original_out_degree[train_idx]
    train_kept_out = train_out_degree[train_idx]
    train_missing_out = missing_out_degree[train_idx]

    print(f"dataset: {args.dataset or '(unnamed)'}")
    print(f"loader: {loader}")
    print(f"nodes: {num_nodes}")
    print(f"node features: {num_features}")
    print(f"train nodes: {train_idx.size} ({train_idx.size / num_nodes:.2%})")
    print(f"held-out nodes: {(~is_train).sum()} ({(~is_train).sum() / num_nodes:.2%})")
    print()
    print(f"original edges: {original_edge_count}")
    print(f"train-induced edges: {train_edge_count}")
    print(f"missing edges vs original: {missing_edge_count} ({1.0 - keep_ratio:.2%})")
    print(f"kept edge ratio: {keep_ratio:.2%}")
    print()
    print("Out-degree over all original nodes")
    print_degree_summary("original_out_degree", original_out_degree)
    print_degree_summary("train_out_degree", train_out_degree)
    print_degree_summary("missing_out_degree", missing_out_degree)
    print()
    print("Out-degree over train nodes only")
    print_degree_summary("original_out_degree", train_original_out)
    print_degree_summary("train_out_degree", train_kept_out)
    print_degree_summary("missing_out_degree", train_missing_out)
    print()
    zero_train_out = int((train_kept_out == 0).sum())
    print(f"train nodes with zero train out-degree: {zero_train_out} ({zero_train_out / train_idx.size:.2%})")

    if not args.no_csv:
        csv_path = args.output_csv or default_csv_path(args)
        write_node_csv(csv_path, stats)
        print(f"per-node CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
