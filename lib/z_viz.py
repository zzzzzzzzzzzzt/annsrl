import html
import os
from pathlib import Path

import numpy as np
import torch


def _env_flag(name):
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _should_plot(epoch):
    epochs = os.environ.get("PLOT_EDGE_PI_EPOCHS", "1")
    if epochs.lower() == "all":
        return True

    selected = set()
    for part in epochs.split(","):
        part = part.strip()
        if not part:
            continue
        selected.add(int(part))
    if epoch in selected:
        return True

    step = _env_int("PLOT_EDGE_PI_STEP", 0)
    return step > 0 and epoch % step == 0


def _fmt(value):
    if not np.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _sample_node_indices(num_nodes, count, seed):
    count = min(count, num_nodes)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_nodes, generator=generator)[:count]


def _stage_matrix(z_stage):
    z_all = z_stage[0].detach().cpu().float()
    return torch.where(torch.isfinite(z_all), z_all, torch.zeros_like(z_all))


def _z_summary(z_stage, sample_indices):
    z_all = _stage_matrix(z_stage)
    nonfinite = torch.numel(z_stage) - torch.isfinite(z_stage).sum().item()
    z_sample = z_all[sample_indices]
    mean = z_all.mean(dim=0, keepdim=True)
    centered_all = z_all - mean
    distances = torch.norm(centered_all, dim=1)
    dim_std = z_all.std(dim=0, unbiased=False)
    centered_sample = z_sample - z_sample.mean(dim=0, keepdim=True)

    if centered_sample.abs().sum() > 0 and z_sample.shape[0] > 1:
        try:
            singular = torch.linalg.svdvals(centered_sample)
            eig = singular.square()
            eig_sum = eig.sum()
            if eig_sum > 0:
                probs = eig / eig_sum
                positive = probs[probs > 0]
                effective_rank = torch.exp(-(positive * positive.log()).sum()).item()
            else:
                effective_rank = 0.0
        except RuntimeError:
            effective_rank = float("nan")
    else:
        effective_rank = 0.0

    collapse_hint = (dim_std.mean().item() < 1e-8) or (distances.mean().item() < 1e-6)
    return {
        "nonfinite": nonfinite,
        "mean_dim_std": dim_std.mean().item(),
        "mean_l2_to_mean": distances.mean().item(),
        "p95_l2_to_mean": torch.quantile(distances, 0.95).item(),
        "effective_rank": effective_rank,
        "collapse_hint": int(collapse_hint),
    }


def _shared_z_projection(z_stages, sample_indices):
    stages = list(z_stages.keys())
    samples = [_stage_matrix(z_stages[stage])[sample_indices] for stage in stages]

    # Fit PCA once on all stages together, then reuse the same two axes for
    # every panel. This makes all three panels comparable in one coordinate
    # system instead of fitting a separate projection per layer.
    stacked = torch.cat(samples, dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    if centered.abs().sum() == 0 or stacked.shape[0] < 2:
        projected = torch.zeros(stacked.shape[0], 2)
    else:
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            projected = centered @ vh[:2].t()
        except RuntimeError:
            projected = centered[:, :2]

    if projected.shape[1] == 1:
        projected = torch.cat([projected, torch.zeros(projected.shape[0], 1)], dim=1)

    split = {}
    start = 0
    for stage, sample in zip(stages, samples):
        end = start + sample.shape[0]
        split[stage] = projected[start:end, :2]
        start = end

    x = projected[:, 0]
    y = projected[:, 1]
    bounds = (x.min(), x.max(), y.min(), y.max())
    return split, bounds


def _scale_coords(coords, x0, y0, width, height, bounds):
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max, y_min, y_max = bounds
    x_range = (x_max - x_min).item()
    y_range = (y_max - y_min).item()
    x_range = x_range if x_range > 1e-12 else 1.0
    y_range = y_range if y_range > 1e-12 else 1.0
    px = x0 + 18 + (x - x_min) / x_range * (width - 36)
    py = y0 + 18 + (1 - (y - y_min) / y_range) * (height - 36)
    return px, py


def _z_distribution_panel(stage, z_stage, sample_indices, coords, bounds, x0, y0, width, height):
    summary = _z_summary(z_stage, sample_indices)
    px, py = _scale_coords(coords, x0, y0, width, height, bounds)
    color = "#dc2626" if summary["collapse_hint"] else "#2563eb"
    lines = [
        f'<text x="{x0}" y="{y0 - 48}" font-size="16" font-weight="700">{html.escape(stage)}</text>',
        f'<text x="{x0}" y="{y0 - 30}" font-size="11">meanL2={_fmt(summary["mean_l2_to_mean"])}, p95L2={_fmt(summary["p95_l2_to_mean"])}, std={_fmt(summary["mean_dim_std"])}</text>',
        f'<text x="{x0}" y="{y0 - 14}" font-size="11">effRank={_fmt(summary["effective_rank"])}, collapse={summary["collapse_hint"]}, bad={summary["nonfinite"]}</text>',
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db" />',
    ]
    radius = 2.2 if len(sample_indices) <= 1000 else 1.6
    for idx in range(len(sample_indices)):
        lines.append(f'<circle cx="{px[idx].item():.2f}" cy="{py[idx].item():.2f}" r="{radius}" fill="{color}" opacity="0.28" />')
    return "\n".join(lines)


def _write_z_layers_svg(path, z_stages, sample_indices, tau):
    panel_width = 390
    panel_height = 250
    margin_x = 48
    y0 = 148
    width = panel_width * len(z_stages) + margin_x * 2
    height = 470
    coords_by_stage, bounds = _shared_z_projection(z_stages, sample_indices)

    panels = []
    for idx, (stage, z_stage) in enumerate(z_stages.items()):
        x0 = margin_x + idx * panel_width
        panels.append(_z_distribution_panel(stage, z_stage, sample_indices, coords_by_stage[stage], bounds, x0, y0, 320, panel_height))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff" />\n'
        f'<text x="{margin_x}" y="34" font-size="20" font-weight="700">Layer-wise z distribution snapshot, tau={tau:.2f}</text>\n'
        f'<text x="{margin_x}" y="62" font-size="12">All panels use the same sampled nodes, one shared PCA fit, and one shared coordinate scale.</text>\n'
        f'<text x="{margin_x}" y="82" font-size="12">If z collapses to one coordinate, points overlap and meanL2/std approach 0; collapse=1 marks a near-collapse.</text>\n'
        f'{chr(10).join(panels)}\n'
        '</svg>\n'
    )
    path.write_text(svg)


def maybe_plot_z_layers(z_stages, tau, run, epoch, seed=0):
    if not _env_flag("PLOT_EDGE_PI") or not _should_plot(epoch):
        return []

    svg_dir = Path(os.environ.get("PLOT_EDGE_PI_DIR", "results/z_viz2")) / f"tau_{tau:.2f}" / "z_layers_svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    z_sample_size = _env_int("PLOT_Z_NODES", 2000)
    snapshot_seed = seed + run * 1000003 + epoch * 9176
    num_nodes = next(iter(z_stages.values())).shape[1]
    sample_indices = _sample_node_indices(num_nodes, z_sample_size, snapshot_seed + 4099)
    path = svg_dir / f"z_layers_run{run:02d}_epoch{epoch:04d}.svg"
    _write_z_layers_svg(path, z_stages, sample_indices, tau)
    print(f"[PLOT_Z] saved {path}")
    return [path]
