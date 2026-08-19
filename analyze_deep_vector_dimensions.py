import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_VERTICES_PATH = "data/DEEP100K/deep10k/deep_base_random10000_seed42.fvecs"
DEFAULT_OUTPUT_DIR = "results/deep_vector_dim_stats"


def infer_fvecs_shape(path):
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        first_dim = np.fromfile(f, dtype=np.int32, count=1)

    if first_dim.size != 1:
        raise ValueError(f"Empty fvecs file: {path}")

    dim = int(first_dim[0])
    if dim <= 0:
        raise ValueError(f"Invalid fvecs dimension {dim} in {path}")

    record_bytes = 4 * (dim + 1)
    if file_size % record_bytes != 0:
        raise ValueError(
            f"{path} size is not divisible by fvecs record size. "
            f"first_dim={dim}, file_size={file_size}, record_bytes={record_bytes}"
        )

    return file_size // record_bytes, dim


def read_fvecs(path, max_points=None):
    path = Path(path)
    total_points, dim = infer_fvecs_shape(path)
    if max_points is None:
        n_points = total_points
    else:
        n_points = min(int(max_points), total_points)

    raw = np.memmap(path, dtype=np.int32, mode="r", shape=(total_points, dim + 1))
    raw = raw[:n_points]
    dims = np.asarray(raw[:, 0])
    if not np.all(dims == dim):
        bad = np.where(dims != dim)[0][:5].tolist()
        raise ValueError(f"Inconsistent vector dimensions in {path}; first bad rows: {bad}")

    return raw[:, 1:].view(np.float32), total_points, dim


def parse_dim_list(value):
    if value is None or value.strip() == "":
        return None
    dims = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        dims.append(int(item))
    return dims


def compute_dimension_stats(vectors):
    quantiles = np.quantile(vectors, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], axis=0)
    mins = np.min(vectors, axis=0)
    maxs = np.max(vectors, axis=0)
    means = np.mean(vectors, axis=0)
    stds = np.std(vectors, axis=0, ddof=0)
    variances = stds ** 2
    ranges = maxs - mins
    iqrs = quantiles[4] - quantiles[2]
    finite_ratios = np.mean(np.isfinite(vectors), axis=0)
    zero_ratios = np.mean(vectors == 0, axis=0)
    near_zero_ratios = np.mean(np.abs(vectors) < 1e-6, axis=0)

    centered = vectors - means
    safe_stds = np.where(stds > 0, stds, np.nan)
    skewness = np.mean((centered / safe_stds) ** 3, axis=0)
    excess_kurtosis = np.mean((centered / safe_stds) ** 4, axis=0) - 3.0

    rows = []
    for dim in range(vectors.shape[1]):
        rows.append(
            {
                "dim": dim,
                "count": vectors.shape[0],
                "min": mins[dim],
                "q01": quantiles[0, dim],
                "q05": quantiles[1, dim],
                "q25": quantiles[2, dim],
                "median": quantiles[3, dim],
                "q75": quantiles[4, dim],
                "q95": quantiles[5, dim],
                "q99": quantiles[6, dim],
                "max": maxs[dim],
                "range": ranges[dim],
                "mean": means[dim],
                "std": stds[dim],
                "var": variances[dim],
                "iqr": iqrs[dim],
                "skewness": skewness[dim],
                "excess_kurtosis": excess_kurtosis[dim],
                "finite_ratio": finite_ratios[dim],
                "zero_ratio": zero_ratios[dim],
                "near_zero_1e-6_ratio": near_zero_ratios[dim],
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_histograms(path, vectors, bins):
    rows = []
    for dim in range(vectors.shape[1]):
        values = np.asarray(vectors[:, dim])
        counts, edges = np.histogram(values, bins=bins)
        widths = np.diff(edges)
        densities = counts / max(values.size, 1) / widths
        for bin_idx, count in enumerate(counts):
            rows.append(
                {
                    "dim": dim,
                    "bin": bin_idx,
                    "left": edges[bin_idx],
                    "right": edges[bin_idx + 1],
                    "count": int(count),
                    "density": densities[bin_idx],
                }
            )
    write_csv(path, rows)


def choose_plot_dims(stats_rows, requested_dims, limit):
    if requested_dims is not None:
        return requested_dims
    ordered = sorted(stats_rows, key=lambda row: row["range"], reverse=True)
    return [int(row["dim"]) for row in ordered[:limit]]


def save_plots(output_dir, prefix, vectors, stats_rows, plot_dims, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = np.array([row["dim"] for row in stats_rows])
    mins = np.array([row["min"] for row in stats_rows])
    maxs = np.array([row["max"] for row in stats_rows])
    means = np.array([row["mean"] for row in stats_rows])
    stds = np.array([row["std"] for row in stats_rows])
    ranges = np.array([row["range"] for row in stats_rows])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].fill_between(dims, mins, maxs, alpha=0.25, label="min-max")
    axes[0].plot(dims, means, linewidth=1.2, label="mean")
    axes[0].set_ylabel("value")
    axes[0].set_title("Per-dimension value range")
    axes[0].legend(loc="best")

    axes[1].plot(dims, ranges, linewidth=1.2, label="range")
    axes[1].plot(dims, stds, linewidth=1.2, label="std")
    axes[1].set_xlabel("dimension")
    axes[1].set_ylabel("value")
    axes[1].set_title("Range and standard deviation")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_range_overview.png", dpi=180)
    plt.close(fig)

    if not plot_dims:
        return

    plot_dims = [dim for dim in plot_dims if 0 <= dim < vectors.shape[1]]
    if not plot_dims:
        return

    n_cols = min(4, len(plot_dims))
    n_rows = int(np.ceil(len(plot_dims) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, dim in zip(axes, plot_dims):
        ax.hist(np.asarray(vectors[:, dim]), bins=bins, color="#4C78A8", alpha=0.85)
        ax.set_title(f"dim {dim}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    for ax in axes[len(plot_dims):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_selected_histograms.png", dpi=180)
    plt.close(fig)


def format_top_rows(rows, metric, limit, reverse=True):
    ordered = sorted(rows, key=lambda row: row[metric], reverse=reverse)[:limit]
    return ", ".join(
        f"dim {int(row['dim'])} ({metric}={float(row[metric]):.6g})"
        for row in ordered
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize per-dimension ranges and distributions for DEEP fvecs vectors."
    )
    parser.add_argument("--vertices_path", default=DEFAULT_VERTICES_PATH, help="Path to .fvecs vectors.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/PNG outputs.")
    parser.add_argument("--prefix", default=None, help="Output file prefix. Defaults to input file stem.")
    parser.add_argument("--max_points", type=int, default=None, help="Only read the first N points.")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bin count per dimension.")
    parser.add_argument(
        "--plot_dims",
        default=None,
        help="Comma-separated dimensions to plot. Defaults to dimensions with largest ranges.",
    )
    parser.add_argument("--plot_dim_limit", type=int, default=12, help="Number of dimensions in histogram plot.")
    parser.add_argument("--no_plots", action="store_true", help="Skip PNG plot generation.")
    args = parser.parse_args()

    vertices_path = Path(args.vertices_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or vertices_path.stem

    vectors, total_points, dim = read_fvecs(vertices_path, args.max_points)
    stats_rows = compute_dimension_stats(vectors)

    summary_path = output_dir / f"{prefix}_dim_summary.csv"
    hist_path = output_dir / f"{prefix}_dim_histograms.csv"
    write_csv(summary_path, stats_rows)
    write_histograms(hist_path, vectors, args.bins)

    plot_dims = choose_plot_dims(stats_rows, parse_dim_list(args.plot_dims), args.plot_dim_limit)
    if not args.no_plots:
        save_plots(output_dir, prefix, vectors, stats_rows, plot_dims, args.bins)

    print(f"Input: {vertices_path}")
    print(f"Loaded points: {vectors.shape[0]} / {total_points}, dimensions: {dim}")
    print(f"Summary CSV: {summary_path}")
    print(f"Histogram CSV: {hist_path}")
    if not args.no_plots:
        print(f"Range plot: {output_dir / f'{prefix}_range_overview.png'}")
        print(f"Histogram plot dims: {plot_dims}")
        print(f"Histogram plot: {output_dir / f'{prefix}_selected_histograms.png'}")
    print("Largest ranges:", format_top_rows(stats_rows, "range", 8))
    print("Largest std:", format_top_rows(stats_rows, "std", 8))
    print("Smallest ranges:", format_top_rows(stats_rows, "range", 8, reverse=False))


if __name__ == "__main__":
    main()
