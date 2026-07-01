import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-30
SETTING_RE = re.compile(r"topologyActivation([^_/]+)_topology_factor([^_/]+)_lossFunction")
PLOT_ACTIVATIONS = ["Sigmoid", "Tanh"]
METRICS = [
    "train_mass", "valid_mass", "test_mass", "loss",
    "train_topn_ratio", "valid_topn_ratio", "test_topn_ratio",
]


def read_metric_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def find_metric_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith("_metrics.csv"):
                yield os.path.join(dirpath, name)


def parse_setting(path):
    match = SETTING_RE.search(path)
    if match is None:
        return None
    activation, factor = match.groups()
    return activation, float(factor)


def stable_slice(rows, metric, window, warmup):
    values = np.array([row[metric] for row in rows], dtype=float)
    if len(values) == 0:
        return slice(0, 0)

    start = min(int(len(values) * warmup), len(values) - 1)
    window = min(window, len(values) - start)
    if window <= 1:
        return slice(start, len(values))

    scaled = np.log10(np.maximum(values, EPS))
    x = np.arange(window, dtype=float)
    best_start, best_score = start, float("inf")

    for i in range(start, len(values) - window + 1):
        y = scaled[i:i + window]
        slope = np.polyfit(x, y, 1)[0]
        score = abs(slope) + y.std()
        if score < best_score:
            best_start, best_score = i, score

    return slice(best_start, best_start + window)


def summarize_run(rows, metric, select_metric, window, warmup):
    chosen = stable_slice(rows, select_metric, window, warmup)
    values = np.array([row[metric] for row in rows], dtype=float)
    epochs = np.array([row["epoch"] for row in rows], dtype=float)
    stable_values = values[chosen]
    return stable_values.mean(), stable_values.std(), epochs[chosen.start], epochs[chosen.stop - 1]


def factor_label(value):
    return "w/o" if abs(value) < 1e-12 else f"{value:g}"


def write_summary(summary, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "activation", "topology_factor", "runs", "mean", "std",
        "window_start_epoch_mean", "window_end_epoch_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def plot_summary(summary, metric, out_path):
    grouped = defaultdict(list)
    for item in summary:
        grouped[item["activation"]].append(item)

    activations = [name for name in PLOT_ACTIVATIONS if name in grouped]
    fig, axes = plt.subplots(1, len(activations), figsize=(4.2 * len(activations), 3.2), sharey=True)
    if len(activations) == 1:
        axes = [axes]

    all_factors = sorted({item["topology_factor"] for item in summary})

    for ax, activation in zip(axes, activations):
        rows = sorted(grouped[activation], key=lambda x: x["topology_factor"])
        x = np.array([row["topology_factor"] for row in rows], dtype=float)
        mean = np.array([row["mean"] for row in rows], dtype=float)
        std = np.array([row["std"] for row in rows], dtype=float)
        ax.plot(x, mean, marker="o", linewidth=1.4, color="#2f6fb5")
        ax.fill_between(x, mean - std, mean + std, color="#1f4e8c", alpha=0.30)
        ax.set_title(activation)
        ax.set_xlabel("topology_factor")
        ax.set_xticks(all_factors)
        ax.set_xticklabels([factor_label(x) for x in all_factors])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(f"stable {metric}")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot stable topology_factor/activation metrics.")
    parser.add_argument("--root", default="results/Topology_factor&Activation_Experiment", help="directory that contains *_metrics.csv files")
    parser.add_argument("--metric", default="test_mass", choices=METRICS)
    parser.add_argument("--select_metric", default="valid_mass", choices=METRICS)
    parser.add_argument("--window", type=int, default=8, help="stable window length in logged eval points")
    parser.add_argument("--warmup", type=float, default=0.3, help="ignore this early ratio before searching windows")
    parser.add_argument("--out", default="results/Topology_factor&Activation_Experiment/topology_activation_factor_stable_window8.svg")
    parser.add_argument("--summary", default="results/Topology_factor&Activation_Experiment/topology_activation_factor_stable_window8.csv")
    args = parser.parse_args()

    groups = defaultdict(list)
    skipped = 0
    for path in find_metric_files(args.root):
        setting = parse_setting(path)
        if setting is None or setting[0] not in PLOT_ACTIVATIONS:
            continue

        rows = read_metric_csv(path)
        if rows and args.metric in rows[0] and args.select_metric in rows[0]:
            groups[setting].append(rows)
        else:
            skipped += 1

    summary = []
    for (activation, factor), runs in sorted(groups.items()):
        means, stds, starts, ends = [], [], [], []
        for rows in runs:
            mean, std, start, end = summarize_run(rows, args.metric, args.select_metric, args.window, args.warmup)
            means.append(mean)
            stds.append(std)
            starts.append(start)
            ends.append(end)

        summary.append({
            "activation": activation,
            "topology_factor": factor,
            "runs": len(means),
            "mean": float(np.mean(means)),
            "std": float(np.sqrt(np.mean(np.square(stds)))),
            "window_start_epoch_mean": float(np.mean(starts)),
            "window_end_epoch_mean": float(np.mean(ends)),
        })

    if not summary:
        print(f"No Sigmoid/Tanh metric CSV files found under {args.root}")
        return

    summary_path = args.summary
    if summary_path is None:
        summary_path = os.path.splitext(args.out)[0] + ".csv"

    if skipped > 0:
        print(f"[SKIP] {skipped} CSV files do not contain {args.metric} or {args.select_metric}")

    write_summary(summary, summary_path)
    plot_summary(summary, args.metric, args.out)
    print(f"[SUMMARY] saved {summary_path}")
    print(f"[PLOT] saved {args.out}")


if __name__ == "__main__":
    main()
