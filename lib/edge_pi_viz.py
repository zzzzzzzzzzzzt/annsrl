import html
import math
import os
from pathlib import Path

import numpy as np
import torch

from lib.Nodeformer import BIG_CONSTANT, create_projection_matrix


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


def _safe_softmax(logits):
    logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e30))
    return torch.softmax(logits, dim=0)


def _sample_gumbel_weights(shape, seed, device, tau, normalize_dim):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.empty(shape, memory_format=torch.legacy_contiguous_format)
    sample.exponential_(generator=generator)
    gumbel = -sample.clamp_min(1e-12).log().to(device) / tau
    gumbel = gumbel - gumbel.max(dim=normalize_dim, keepdim=True)[0]
    return gumbel.exp()


def _sample_sources(num_nodes, count, seed):
    count = min(count, num_nodes)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_nodes, generator=generator)[:count].tolist()


def _existing_targets(edge_index, source):
    src, dst = edge_index
    return set(dst[src == source].detach().cpu().tolist())


def _projection_matrix(nb_random_features, dim, seed_tensor, device):
    seed = torch.ceil(torch.abs(seed_tensor * BIG_CONSTANT)).to(torch.int32)
    return create_projection_matrix(nb_random_features, dim, seed=seed).to(device)


def _conv_gumbel_distribution(model, z, edge_index, tau, source, layer_idx, seed):
    conv = model.convs[layer_idx]
    device = z.device
    bsz, num_nodes = z.shape[0], z.shape[1]
    query = conv.Wq(z).reshape(bsz, num_nodes, conv.num_heads, conv.out_channels)
    key = conv.Wk(z).reshape(bsz, num_nodes, conv.num_heads, conv.out_channels)
    value = conv.Wv(z).reshape(bsz, num_nodes, conv.num_heads, conv.out_channels)

    dim = query.shape[-1]
    projection_matrix = _projection_matrix(conv.nb_random_features, dim, torch.sum(query), device)
    query_scaled = query / math.sqrt(tau)
    key_scaled = key / math.sqrt(tau)
    query_prime = conv.kernel_transformation(query_scaled, True, projection_matrix).permute(1, 0, 2, 3)
    key_prime = conv.kernel_transformation(key_scaled, False, projection_matrix).permute(1, 0, 2, 3)

    # Use one deterministic Gumbel draw and average both directions for the
    # fixed source-candidate pair.
    gumbel_weights = _sample_gumbel_weights(
        (num_nodes, bsz, conv.num_heads, conv.nb_gumbel_sample),
        seed,
        device,
        tau,
        normalize_dim=0,
    )
    key_t_gumbel = key_prime.unsqueeze(3) * gumbel_weights.unsqueeze(-1)  # [N, B, H, K, M]
    key_sum = key_t_gumbel.sum(dim=0)  # [B, H, K, M]

    query_source = query_prime[source]  # [B, H, M]
    source_to_candidate_num = torch.einsum("bhm,nbhkm->nbhk", query_source, key_t_gumbel)
    source_to_candidate_den = torch.einsum("bhm,bhkm->bhk", query_source, key_sum).unsqueeze(0)
    source_to_candidate = source_to_candidate_num / source_to_candidate_den.clamp_min(1e-30)

    key_source = key_t_gumbel[source]  # [B, H, K, M]
    candidate_to_source_num = torch.einsum("nbhm,bhkm->nbhk", query_prime, key_source)
    candidate_to_source_den = torch.einsum("nbhm,bhkm->nbhk", query_prime, key_sum)
    candidate_to_source = candidate_to_source_num / candidate_to_source_den.clamp_min(1e-30)

    probs = 0.5 * (source_to_candidate + candidate_to_source)
    return probs[:, 0].mean(dim=(1, 2))


def _global_gumbel_distribution(model, z, tau, source, seed):
    device = z.device
    query = z.unsqueeze(2) / math.sqrt(tau)
    key = z.unsqueeze(2) / math.sqrt(tau)
    # query = torch.matmul(z, model.link_W1).unsqueeze(2) / math.sqrt(tau)
    # key = torch.matmul(z, model.link_W2.t()).unsqueeze(2) / math.sqrt(tau)

    projection_matrix = _projection_matrix(model.nb_random_features, query.shape[-1], torch.sum(query), device)
    query_prime = model.kernel_transformation(query, True, projection_matrix)[0, :, 0]
    key_prime = model.kernel_transformation(key, False, projection_matrix)[0, :, 0]
    gumbel_weights = _sample_gumbel_weights(
        (key_prime.shape[0],), seed, device, tau, normalize_dim=0
    )

    key_t_gumbel = key_prime * gumbel_weights.unsqueeze(-1)
    key_sum = key_t_gumbel.sum(dim=0)

    source_to_candidate_num = (query_prime[source].unsqueeze(0) * key_t_gumbel).sum(dim=-1)
    source_to_candidate_den = (query_prime[source] * key_sum).sum(dim=-1)
    source_to_candidate = source_to_candidate_num / source_to_candidate_den.clamp_min(1e-30)

    candidate_to_source_num = (query_prime * key_t_gumbel[source].unsqueeze(0)).sum(dim=-1)
    candidate_to_source_den = (query_prime * key_sum.unsqueeze(0)).sum(dim=-1)
    candidate_to_source = candidate_to_source_num / candidate_to_source_den.clamp_min(1e-30)

    return 0.5 * (source_to_candidate + candidate_to_source)


def _distribution_stats(values):
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    sorted_values, _ = torch.sort(values, descending=True)
    max_value = sorted_values[0].item() if sorted_values.numel() else 0.0
    top5 = sorted_values[:5].sum().item() if sorted_values.numel() else 0.0
    ratio = (max_value / sorted_values[1].item()) if sorted_values.numel() > 1 and sorted_values[1].item() > 0 else float("inf")
    denom = values.square().sum().item()
    effn = (values.sum().item() ** 2 / denom) if denom > 0 else float("inf")
    bad = torch.numel(values) - torch.isfinite(values).sum().item()
    return max_value, top5, ratio, effn, bad


def _existing_stats(values, existing_targets):
    if not existing_targets:
        return None, 0.0, 0

    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    existing = torch.tensor(sorted(existing_targets), dtype=torch.long)
    existing = existing[existing < values.numel()]
    if existing.numel() == 0:
        return None, 0.0, 0

    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, order.numel() + 1, dtype=order.dtype)
    edge_ranks = ranks[existing]
    edge_mass = values[existing].sum().item()
    return int(edge_ranks.min().item()), edge_mass, int(existing.numel())


def _bar_panel(title, values, existing_targets, topk, x0, y0, width, height):
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    top_values, top_indices = torch.topk(values, k=min(topk, values.numel()))
    max_value = top_values[0].item() if top_values.numel() else 1.0
    max_value = max(max_value, 1e-12)
    stat_max, stat_top5, stat_ratio, stat_effn, stat_bad = _distribution_stats(values)
    best_edge_rank, edge_mass, edge_count = _existing_stats(values, existing_targets)
    rank_text = "none" if best_edge_rank is None else str(best_edge_rank)

    lines = [
        f'<text x="{x0}" y="{y0 - 78}" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{x0}" y="{y0 - 56}" font-size="12">max={_fmt(stat_max)}, top5={_fmt(stat_top5)}, ratio={_fmt(stat_ratio)}</text>',
        f'<text x="{x0}" y="{y0 - 38}" font-size="12">effN={_fmt(stat_effn)}, bad={int(stat_bad)}</text>',
        f'<text x="{x0}" y="{y0 - 20}" font-size="12">bestEdgeRank={rank_text}, edgeMass={_fmt(edge_mass)}, edgeCount={edge_count}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#111827" stroke-width="1" />',
    ]
    gap = 3
    bar_width = max(2, (width - gap * (len(top_values) - 1)) / max(len(top_values), 1))
    for rank, (value, node_id) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
        bar_height = max(1.0, value / max_value * height)
        color = "#f97316" if node_id in existing_targets else "#2563eb"
        x = x0 + rank * (bar_width + gap)
        y = y0 + height - bar_height
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" opacity="0.90" />')

    for rank, (value, node_id) in enumerate(zip(top_values[:5].tolist(), top_indices[:5].tolist()), start=1):
        lines.append(f'<text x="{x0}" y="{y0 + height + 22 + (rank - 1) * 18}" font-size="12">#{rank}: j={node_id}, w={value:.3e}</text>')
    return "\n".join(lines)


def _write_attention_svg(path, panels, source, tau, topk):
    panel_width = 430
    panel_gap = 70
    margin_x = 48
    y0 = 190
    bar_height = 230
    width = margin_x * 2 + len(panels) * panel_width + (len(panels) - 1) * panel_gap
    height = 620

    panel_svg = []
    for idx, (title, values, existing_targets) in enumerate(panels):
        x0 = margin_x + idx * (panel_width + panel_gap)
        panel_svg.append(_bar_panel(title, values, existing_targets, topk, x0, y0, panel_width, bar_height))

    legend_y = height - 56
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff" />\n'
        f'<text x="{margin_x}" y="34" font-size="22" font-weight="700">Gumbel attention snapshot: fixed i={source}, tau={tau:.2f}</text>\n'
        f'<text x="{margin_x}" y="64" font-size="13">Conv panels show bidirectional edge attention in each NodeFormer layer; the right panel shows global_gumbel.</text>\n'
        f'<text x="{margin_x}" y="84" font-size="13">Each panel sorts all candidate j by probability and displays top-{topk}. Orange means existing outgoing edge i -> j.</text>\n'
        f'<text x="{margin_x}" y="104" font-size="13">Weights average both directions, then average over heads and K Gumbel samples.</text>\n'
        f'{chr(10).join(panel_svg)}\n'
        f'<rect x="{margin_x}" y="{legend_y}" width="14" height="14" fill="#f97316" />\n'
        f'<text x="{margin_x + 22}" y="{legend_y + 12}" font-size="13">existing edge</text>\n'
        f'<rect x="{margin_x + 150}" y="{legend_y}" width="14" height="14" fill="#2563eb" />\n'
        f'<text x="{margin_x + 172}" y="{legend_y + 12}" font-size="13">candidate</text>\n'
        '</svg>\n'
    )
    path.write_text(svg)


def maybe_plot_edge_attention(model, edge_index, z_stages, tau, run, epoch, seed=0):
    if not _env_flag("PLOT_EDGE_PI") or not _should_plot(epoch):
        return []

    num_nodes = next(iter(z_stages.values())).shape[1]
    topk = _env_int("PLOT_EDGE_PI_TOPK", 30)
    count = _env_int("PLOT_EDGE_PI_NODES", 3)
    snapshot_seed = seed + run * 1000003 + epoch * 9176
    sources = _sample_sources(num_nodes, count, snapshot_seed + 811)
    output_dir = Path(os.environ.get("PLOT_EDGE_PI_DIR", "results/z_viz/All_data_cross_loss")) / f"tau_{tau:.2f}" / "attention_svg"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for order, source in enumerate(sources):
        existing = _existing_targets(edge_index, source)
        panels = []
        current_z = z_stages["input_z"]
        for layer_idx in range(len(model.convs)):
            values = _conv_gumbel_distribution(model, current_z, edge_index, tau, source, layer_idx, snapshot_seed + order * 1009 + layer_idx * 97)
            panels.append((f"conv_layer{layer_idx}_gumbel", values.detach().cpu(), existing))
            current_z = z_stages[f"after_layer{layer_idx}"]
        global_values = _global_gumbel_distribution(model, z_stages[f"after_layer{len(model.convs) - 1}"], tau, source, snapshot_seed + order * 1009 + 503)
        panels.append(("global_gumbel", global_values.detach().cpu(), existing))

        path = output_dir / f"edge_attention_run{run:02d}_epoch{epoch:04d}_i{source}.svg"
        _write_attention_svg(path, panels, source, tau, topk)
        print(f"[PLOT_EDGE_PI] saved {path}")
        paths.append(path)
    return paths
