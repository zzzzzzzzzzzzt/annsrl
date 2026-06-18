import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from lib.utils import read_fvecs, write_edges, write_fvecs


def sample_vertices(vertices, sample_size, seed, sample_method):
    if sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    if sample_size > vertices.shape[0]:
        raise ValueError(
            f"--sample_size={sample_size} is larger than dataset size {vertices.shape[0]}"
        )

    if sample_method == "first":
        indices = np.arange(sample_size, dtype=np.int64)
    elif sample_method == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(vertices.shape[0], size=sample_size, replace=False)
        indices.sort()
    else:
        raise ValueError("sample_method must be one of ['random', 'first']")

    return vertices[indices], indices


def build_knn_edges(vertices, k, n_jobs):
    from sklearn.neighbors import NearestNeighbors

    if k <= 0:
        raise ValueError("--knn_k must be positive")
    if k >= vertices.shape[0]:
        raise ValueError("--knn_k must be smaller than --sample_size")

    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=n_jobs)
    knn.fit(vertices)
    neighbors = knn.kneighbors(vertices, return_distance=False)

    edges = defaultdict(list)
    for src, row in enumerate(neighbors):
        row = [int(dst) for dst in row if int(dst) != src]
        edges[src] = row[:k]
    return edges


def build_hnsw_edges(vertices, m, ef_construction, ef, degree, seed, space, n_jobs):
    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError(
            "Building an HNSW graph requires hnswlib. Install it in the same "
            "environment you use for training, for example: pip install hnswlib"
        ) from exc

    if m <= 0:
        raise ValueError("--hnsw_m must be positive")
    if ef_construction <= 0:
        raise ValueError("--hnsw_ef_construction must be positive")
    if degree <= 0:
        raise ValueError("--hnsw_degree must be positive")
    if degree >= vertices.shape[0]:
        raise ValueError("--hnsw_degree must be smaller than --sample_size")

    vertices = np.asarray(vertices, dtype=np.float32)
    num_nodes, dim = vertices.shape

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(
        max_elements=num_nodes,
        ef_construction=ef_construction,
        M=m,
        random_seed=seed,
    )
    index.set_num_threads(n_jobs)
    index.add_items(vertices, np.arange(num_nodes))
    index.set_ef(max(ef, degree + 1))

    labels, _ = index.knn_query(vertices, k=degree + 1, num_threads=n_jobs)

    edges = defaultdict(list)
    for src, row in enumerate(labels):
        row = [int(dst) for dst in row if int(dst) != src]
        edges[src] = row[:degree]
    return edges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample points from an fvecs dataset and build an ivecs graph."
    )
    parser.add_argument("--vertices_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_method", choices=["random", "first"], default="random")
    parser.add_argument("--graph_method", choices=["hnsw", "knn"], default="hnsw")
    parser.add_argument("--knn_k", type=int, default=12)
    parser.add_argument("--hnsw_m", type=int, default=12)
    parser.add_argument("--hnsw_ef_construction", type=int, default=300)
    parser.add_argument("--hnsw_ef", type=int, default=300)
    parser.add_argument(
        "--hnsw_degree",
        type=int,
        default=None,
        help="number of outgoing edges to export per node; default is 2 * hnsw_m",
    )
    parser.add_argument("--hnsw_space", choices=["l2", "ip", "cosine"], default="l2")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild files even if the expected outputs already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"{args.sample_method}{args.sample_size}_seed{args.seed}"
    vertices_out = output_dir / f"deep_base_{name}.fvecs"
    indices_out = output_dir / f"sample_indices_{name}.npy"

    if args.graph_method == "hnsw":
        hnsw_degree = args.hnsw_degree or 2 * args.hnsw_m
        edges_out = output_dir / (
            f"deep_hnsw_M{args.hnsw_m}_efC{args.hnsw_ef_construction}_{name}.ivecs"
        )
    else:
        hnsw_degree = None
        edges_out = output_dir / f"deep_knn_k{args.knn_k}_{name}.ivecs"

    expected_outputs = [vertices_out, indices_out, edges_out]
    if not args.overwrite and all(path.exists() for path in expected_outputs):
        print("All outputs already exist. Use --overwrite to rebuild:")
        for path in expected_outputs:
            print(f"  {path}")
        return

    print(f"Reading vertices from {args.vertices_path}")
    vertices = read_fvecs(args.vertices_path)
    sampled_vertices, sample_indices = sample_vertices(
        vertices, args.sample_size, args.seed, args.sample_method
    )

    print(
        f"Sampled {sampled_vertices.shape[0]} / {vertices.shape[0]} points "
        f"with method={args.sample_method}, seed={args.seed}"
    )
    write_fvecs(vertices_out, sampled_vertices)
    np.save(indices_out, sample_indices)

    if args.graph_method == "hnsw":
        print(
            "Building HNSW graph with "
            f"M={args.hnsw_m}, efConstruction={args.hnsw_ef_construction}, "
            f"ef={args.hnsw_ef}, exported_degree={hnsw_degree}"
        )
        edges = build_hnsw_edges(
            sampled_vertices,
            args.hnsw_m,
            args.hnsw_ef_construction,
            args.hnsw_ef,
            hnsw_degree,
            args.seed,
            args.hnsw_space,
            args.n_jobs,
        )
    elif args.graph_method == "knn":
        print(f"Building exact KNN graph with k={args.knn_k}")
        edges = build_knn_edges(sampled_vertices, args.knn_k, args.n_jobs)
    else:
        raise ValueError(f"Unsupported graph_method: {args.graph_method}")

    write_edges(edges_out, edges)
    print("Wrote:")
    print(f"  vertices: {vertices_out}")
    print(f"  indices:  {indices_out}")
    print(f"  edges:    {edges_out}")
    print("Use --graph_type nsw when passing this ivecs edge file to pretrain.py.")


if __name__ == "__main__":
    main()
