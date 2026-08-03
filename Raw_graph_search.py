import argparse
import ctypes
import heapq
import os
from pathlib import Path
import sys
import time

import numpy as np


DATA_DIR = Path("./data/SIFT100K")


def read_fvecs(path, max_size=None):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    dim = int(raw[0])
    width = dim + 1
    if raw.size % width != 0:
        raise ValueError(f"{path} is not a fixed-width fvecs file")

    rows = raw.reshape(-1, width)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"{path} has inconsistent fvecs dimensions")

    if max_size is not None:
        rows = rows[:max_size]
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def read_ivecs(path, max_size=None):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    dim = int(raw[0])
    width = dim + 1
    if raw.size % width != 0:
        raise ValueError(f"{path} is not a fixed-width ivecs file")

    rows = raw.reshape(-1, width)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"{path} has inconsistent ivecs dimensions")

    if max_size is not None:
        rows = rows[:max_size]
    return np.ascontiguousarray(rows[:, 1:], dtype=np.int32)


def read_edge_ivecs(path, expected_rows):
    rows = []
    with open(path, "rb") as f:
        while True:
            header = np.fromfile(f, dtype=np.int32, count=1)
            if header.size == 0:
                break
            degree = int(header[0])
            row = np.fromfile(f, dtype=np.int32, count=degree)
            if row.size != degree:
                raise ValueError(f"{path} ended in the middle of an edge row")
            rows.append(row)

    if len(rows) != expected_rows:
        raise ValueError(
            f"{path} has {len(rows)} edge rows, but vertices have {expected_rows} rows"
        )
    return rows


def pad_edges(edge_rows):
    # Add one extra sentinel column because the C++ loop reads until -1.
    max_degree = max(len(row) for row in edge_rows) + 1
    edges = np.full((len(edge_rows), max_degree), -1, dtype=np.int32)
    for row_id, row in enumerate(edge_rows):
        edges[row_id, : len(row)] = row
    return edges


def maybe_load_system_libstdcxx():
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib64/libstdc++.so.6",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            return candidate
    return None


def load_cpp_search():
    swig_dir = Path(__file__).resolve().parent / "lib" / "search_hnsw_swig"
    sys.path.insert(0, str(swig_dir))

    try:
        import search_hnsw
        return search_hnsw.find_nearest
    except ImportError as first_error:
        sys.modules.pop("search_hnsw", None)
        sys.modules.pop("_search_hnsw", None)
        maybe_load_system_libstdcxx()
        try:
            import search_hnsw
            return search_hnsw.find_nearest
        except ImportError as second_error:
            raise RuntimeError(
                "Cannot import lib/search_hnsw_swig/_search_hnsw.so. "
                "Rebuild the SWIG extension or run with a compatible libstdc++. "
                f"First error: {first_error}; second error: {second_error}"
            ) from second_error


def recall_at_k(pred, gt, k):
    if k == 1:
        return float(np.mean(pred[:, 0] == gt[:, 0]))
    return float(
        np.mean(
            [
                len(set(pred_row[:k]) & set(gt_row[:k])) / k
                for pred_row, gt_row in zip(pred, gt)
            ]
        )
    )


def run_cpp_search(vertices, edges, queries, gt, efs, k, initial_vertex_id,
                   n_jobs, max_trajectory, batch_size):
    if vertices.shape[1] != 300 and vertices.shape[1] % 16 != 0:
        raise ValueError(
            "The bundled C++ L2 kernel assumes dim is a multiple of 16 "
            "(except dim=300, which uses the GloVe negative-dot path)."
        )

    search_hnsw = load_cpp_search()
    edge_probs = np.where(edges >= 0, 1.1, -1.0).astype(np.float32)
    max_degree = edges.shape[1]
    num_actions = max_trajectory * max_degree
    num_results = k + 2 + num_actions

    for ef in efs:
        started = time.perf_counter()
        pred_chunks = []
        dc_chunks = []
        hop_chunks = []

        for start in range(0, queries.shape[0], batch_size):
            batch_queries = queries[start:start + batch_size]
            batch_size_actual = batch_queries.shape[0]
            trajectories = np.full(
                (batch_size_actual, max_trajectory), -1, dtype=np.int32
            )
            samples = np.zeros(
                (batch_size_actual, num_actions), dtype=np.float32
            )
            results = np.full(
                (batch_size_actual, num_results), -1, dtype=np.int32
            )

            search_hnsw(
                vertices,
                edges,
                edge_probs,
                batch_queries,
                trajectories,
                samples,
                results,
                k,
                initial_vertex_id,
                ef,
                n_jobs,
            )
            pred_chunks.append(results[:, :k].copy())
            dc_chunks.append(results[:, k].copy())
            hop_chunks.append(results[:, k + 1].copy())

        pred = np.vstack(pred_chunks)
        dcs = np.concatenate(dc_chunks)
        hops = np.concatenate(hop_chunks)
        recall = recall_at_k(pred, gt, k)
        elapsed = time.perf_counter() - started
        truncated = int(np.sum(hops >= max_trajectory))

        line = (
            f"Ef {ef:3d} | Recall@{k} {recall:.4f} | "
            f"Distances: {np.mean(dcs):.1f} | Hops: {np.mean(hops):.1f} | "
            f"Time: {elapsed:.2f}s"
        )
        if truncated:
            line += f" | truncated: {truncated}"
        print(line, flush=True)


def l2sqr(query, vectors):
    diff = query - vectors
    return np.einsum("ij,ij->i", diff, diff)


def search_one_python(query, vertices, edge_rows, ef, k, initial_vertex_id):
    visited = {initial_vertex_id}
    distance = float(l2sqr(query, vertices[initial_vertex_id:initial_vertex_id + 1])[0])
    dcs = 1
    hops = 0

    top_results = [(-distance, initial_vertex_id)]
    candidates = [(distance, initial_vertex_id)]
    lower_bound = distance

    while candidates:
        dist, vertex_id = heapq.heappop(candidates)
        if dist > lower_bound:
            break

        hops += 1
        neighbor_ids = [n for n in edge_rows[vertex_id] if int(n) not in visited]
        if len(neighbor_ids) == 0:
            continue

        visited.update(int(n) for n in neighbor_ids)
        neighbor_ids = np.asarray(neighbor_ids, dtype=np.int32)
        distances = l2sqr(query, vertices[neighbor_ids])
        dcs += len(neighbor_ids)

        for neighbor_id, candidate_dist in zip(neighbor_ids, distances):
            candidate_dist = float(candidate_dist)
            neighbor_id = int(neighbor_id)
            if candidate_dist < lower_bound or len(top_results) < ef:
                heapq.heappush(candidates, (candidate_dist, neighbor_id))
                heapq.heappush(top_results, (-candidate_dist, neighbor_id))

                if len(top_results) > ef:
                    heapq.heappop(top_results)
                lower_bound = -top_results[0][0]

    best = [item[1] for item in heapq.nlargest(k, top_results)]
    if len(best) < k:
        best.extend([-1] * (k - len(best)))
    return best, dcs, hops


def run_python_search(vertices, edge_rows, queries, gt, efs, k, initial_vertex_id):
    for ef in efs:
        started = time.perf_counter()
        preds = np.empty((queries.shape[0], k), dtype=np.int32)
        dcs = np.empty(queries.shape[0], dtype=np.int32)
        hops = np.empty(queries.shape[0], dtype=np.int32)

        for i, query in enumerate(queries):
            pred, dc, hop = search_one_python(
                query, vertices, edge_rows, ef, k, initial_vertex_id
            )
            preds[i] = pred
            dcs[i] = dc
            hops[i] = hop

        recall = recall_at_k(preds, gt, k)
        elapsed = time.perf_counter() - started
        print(
            f"Ef {ef:3d} | Recall@{k} {recall:.4f} | "
            f"Distances: {np.mean(dcs):.1f} | Hops: {np.mean(hops):.1f} | "
            f"Time: {elapsed:.2f}s",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw exported NSW/HNSW graph search. This is graph search "
            "over an ivecs adjacency file, not a full hnswlib multi-layer index."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--graph", choices=["hnsw", "nsw"], default="hnsw")
    parser.add_argument("--edges_path", type=Path, default=None)
    parser.add_argument("--M", type=int, default=12)
    parser.add_argument("--efC", type=int, default=300)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--ef_min", type=int, default=12)
    parser.add_argument("--ef_max", type=int, default=300)
    parser.add_argument("--ef_step", type=int, default=4)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_trajectory", type=int, default=300)
    parser.add_argument("--initial_vertex_id", type=int, default=0)
    parser.add_argument("--test_queries_size", type=int, default=None)
    parser.add_argument("--backend", choices=["cpp", "python"], default="cpp")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ef_step <= 0:
        raise ValueError("--ef_step must be positive")
    if args.ef_min <= 0 or args.ef_max < args.ef_min:
        raise ValueError("--ef_min/--ef_max define an invalid ef range")
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.n_jobs <= 0:
        raise ValueError("--n_jobs must be positive")
    if args.max_trajectory <= 0:
        raise ValueError("--max_trajectory must be positive")

    data_dir = args.data_dir
    edges_path = args.edges_path
    if edges_path is None:
        edges_path = data_dir / f"sift_{args.graph}_M{args.M}_efC{args.efC}.ivecs"

    vertices_path = data_dir / "sift_base.fvecs"
    queries_path = data_dir / "sift_query.fvecs"
    gt_path = data_dir / "test_gt.ivecs"

    started = time.perf_counter()
    vertices = read_fvecs(vertices_path)
    queries = read_fvecs(queries_path, args.test_queries_size)
    gt = read_ivecs(gt_path, args.test_queries_size)
    edge_rows = read_edge_ivecs(edges_path, expected_rows=vertices.shape[0])
    load_time = time.perf_counter() - started

    if args.k > gt.shape[1]:
        raise ValueError(f"--k={args.k} is larger than ground truth width {gt.shape[1]}")
    if not (0 <= args.initial_vertex_id < vertices.shape[0]):
        raise ValueError("--initial_vertex_id is outside the vertex id range")

    efs = list(range(args.ef_min, args.ef_max + 1, args.ef_step))
    print(
        f"Loaded {vertices.shape[0]} vertices, {queries.shape[0]} queries, "
        f"dim={vertices.shape[1]}, graph={edges_path}, load_time={load_time:.2f}s"
    )
    print(
        f"Backend={args.backend}, k={args.k}, ef={efs[0]}..{efs[-1]}, "
        f"entry={args.initial_vertex_id}"
    )

    if args.backend == "cpp":
        edges = pad_edges(edge_rows)
        print(f"Max degree={edges.shape[1] - 1} (+ sentinel), n_jobs={args.n_jobs}")
        run_cpp_search(
            vertices,
            edges,
            queries,
            gt,
            efs,
            args.k,
            args.initial_vertex_id,
            args.n_jobs,
            args.max_trajectory,
            args.batch_size,
        )
    else:
        print("Using the slow Python backend for debugging/reference only.")
        run_python_search(
            vertices,
            edge_rows,
            queries,
            gt,
            efs,
            args.k,
            args.initial_vertex_id,
        )


if __name__ == "__main__":
    main()
