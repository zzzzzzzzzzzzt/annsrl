import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


FIELDNAMES = [
    'dataset', 'run', 'epoch', 'loss', 'module', 'rows', 'dims',
    'finite_ratio', 'min', 'max', 'mean', 'std', 'near_zero_ratio',
    'norm_mean', 'norm_std', 'dim_std_mean', 'dim_std_min', 'dim_std_max',
    'cosine_mean', 'cosine_std', 'cosine_p99', 'near_duplicate_ratio',
    'pca_top1_ratio', 'pca_top2_ratio', 'pca_top10_ratio', 'effective_rank',
    'gate_saturation_ratio', 'warnings',
]


def safe_path_name(value):
    value = str(value or 'unnamed')
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in value)


def tensor_to_matrix(tensor):
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.dim() == 1:
        tensor = tensor.reshape(-1, 1)
    else:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor.numpy()


def clean_matrix(matrix):
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def sample_rows(matrix, max_points, seed):
    if matrix.shape[0] <= max_points:
        return matrix
    rng = np.random.default_rng(seed)
    idx = rng.choice(matrix.shape[0], max_points, replace=False)
    return matrix[idx]


def pca_ratios(matrix, max_points, seed):
    matrix = sample_rows(clean_matrix(matrix), max_points, seed)
    if min(matrix.shape) < 2:
        return 0.0, 0.0, 0.0, 0.0
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigvals = singular_values ** 2
    total = eigvals.sum()
    if total <= 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    ratios = eigvals / total
    entropy = -(ratios * np.log(ratios + 1e-30)).sum()
    return (
        float(ratios[0]),
        float(ratios[:2].sum()),
        float(ratios[:10].sum()),
        float(np.exp(entropy)),
    )


def cosine_stats(matrix, max_points, seed):
    matrix = sample_rows(clean_matrix(matrix), max_points, seed)
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > 1e-12
    if valid.sum() < 2:
        return 0.0, 0.0, 0.0, 0.0
    normalized = matrix[valid] / norms[valid, None]
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
    return (
        float(upper.mean()),
        float(upper.std()),
        float(np.quantile(upper, 0.99)),
        float((upper > 0.999).mean()),
    )


def embedding_stats(name, matrix, args, run, epoch, loss_value, seed):
    finite = np.isfinite(matrix)
    values = matrix[finite]
    clean = clean_matrix(matrix)
    norms = np.linalg.norm(clean, axis=1)
    dim_std = clean.std(axis=0)
    pca_top1, pca_top2, pca_top10, effective_rank = pca_ratios(
        clean, args.embedding_check_max_points, seed
    )
    cosine_mean, cosine_std, cosine_p99, near_duplicate_ratio = cosine_stats(
        clean, args.embedding_check_max_points, seed + 17
    )

    gate_saturation_ratio = 0.0
    if name == 'gate':
        gate_saturation_ratio = float(((clean < 0.01) | (clean > 0.99)).mean())

    warnings = []
    if float(finite.mean()) < 1.0:
        warnings.append('nonfinite')
    if float(dim_std.mean()) < 1e-8:
        warnings.append('near_constant')
    if cosine_mean > 0.98:
        warnings.append('high_cosine')
    if matrix.shape[1] > 1 and pca_top1 > 0.90:
        warnings.append('low_rank')
    if matrix.shape[1] >= 10 and 0 < effective_rank < 3:
        warnings.append('tiny_effective_rank')
    if gate_saturation_ratio > 0.5:
        warnings.append('gate_saturated')

    return {
        'dataset': args.dataset or 'dataset',
        'run': run,
        'epoch': epoch,
        'loss': '' if loss_value is None else float(loss_value),
        'module': name,
        'rows': int(matrix.shape[0]),
        'dims': int(matrix.shape[1]),
        'finite_ratio': float(finite.mean()),
        'min': float(values.min()) if values.size else 0.0,
        'max': float(values.max()) if values.size else 0.0,
        'mean': float(values.mean()) if values.size else 0.0,
        'std': float(values.std()) if values.size else 0.0,
        'near_zero_ratio': float((np.abs(clean) < 1e-8).mean()),
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
        'dim_std_mean': float(dim_std.mean()),
        'dim_std_min': float(dim_std.min()) if dim_std.size else 0.0,
        'dim_std_max': float(dim_std.max()) if dim_std.size else 0.0,
        'cosine_mean': cosine_mean,
        'cosine_std': cosine_std,
        'cosine_p99': cosine_p99,
        'near_duplicate_ratio': near_duplicate_ratio,
        'pca_top1_ratio': pca_top1,
        'pca_top2_ratio': pca_top2,
        'pca_top10_ratio': pca_top10,
        'effective_rank': effective_rank,
        'gate_saturation_ratio': gate_saturation_ratio,
        'warnings': '|'.join(warnings),
    }


def fit_pca_ref(matrix, max_points, seed):
    if matrix.shape[1] < 2 or matrix.shape[0] < 2:
        return None
    matrix = sample_rows(clean_matrix(matrix), max_points, seed)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    if (singular_values ** 2).sum() <= 1e-30 or components.shape[0] < 2:
        return None
    return {
        'mean': mean.astype(np.float32),
        'components': components[:min(20, components.shape[0])].astype(np.float32),
    }


def save_pca_plot(name, matrix, pca_ref, path, max_points, seed):
    if pca_ref is None:
        return
    matrix = sample_rows(clean_matrix(matrix), max_points, seed)
    centered = matrix - pca_ref['mean']
    projected = centered @ pca_ref['components'].T
    coords = projected[:, :2]
    ratios = projected.var(axis=0) / max(centered.var(axis=0).sum(), 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.35)
    axes[0].set_title(f'{name} PCA scatter')
    axes[0].set_xlabel('fixed PC1')
    axes[0].set_ylabel('fixed PC2')
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(np.arange(1, ratios.size + 1), np.cumsum(ratios), marker='o', linewidth=1.2)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Variance on fixed PCs')
    axes[1].set_xlabel('PC count')
    axes[1].set_ylabel('Cumulative ratio')
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_rows(path, rows, append=False):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    mode = 'a' if append else 'w'
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def artifact_name(run, epoch):
    return f'run{run:02d}_epoch{epoch:03d}'


@torch.no_grad()
def collect_embedding_tensors(model, vertices, edges, tau):
    was_training = model.training
    model.eval()

    x = vertices.unsqueeze(0)
    adjs = edges.unsqueeze(0)
    topology = model._encode_topology(x, adjs, tau)
    x_out = model.output_proj(topology)
    gate = model.feature_gate(topology)
    fused_z = model._fuse_features(x, topology)
    query = model.link_query(fused_z)
    key = model.link_key(fused_z)
    query_prime, key_prime = model._link_kernel(fused_z, tau)
    edge_weight = model.normalized_edge_prob(model._link_state(fused_z, tau), edges)

    if was_training:
        model.train()

    return {
        'raw_x': vertices,
        'topology': topology.squeeze(0),
        'x_out': x_out.squeeze(0),
        'gate': gate.squeeze(0),
        'fused_z': fused_z.squeeze(0),
        'fused_delta': (fused_z - x).squeeze(0),
        'query': query.squeeze(0),
        'key': key.squeeze(0),
        'query_prime': query_prime.squeeze(0).squeeze(1),
        'key_prime': key_prime.squeeze(0).squeeze(1),
        'edge_weight': edge_weight.squeeze(0),
    }


def run_embedding_check(model, vertices, edges, args, run, epoch, loss_value, pca_refs):
    tensors = collect_embedding_tensors(model, vertices, edges, args.tau)
    dataset_name = safe_path_name(args.dataset or 'dataset')
    dataset_dir = os.path.join(args.embedding_check_dir, dataset_name)
    artifact = artifact_name(run, epoch)
    rows = []

    for idx, (name, tensor) in enumerate(tensors.items()):
        module_dir = os.path.join(dataset_dir, name)
        os.makedirs(module_dir, exist_ok=True)
        torch.save(tensor.detach().cpu(), os.path.join(module_dir, f'{artifact}.pt'))

        matrix = tensor_to_matrix(tensor)
        seed = args.seed + run * 1009 + epoch * 9176 + idx
        row = embedding_stats(name, matrix, args, run, epoch, loss_value, seed)
        rows.append(row)
        write_rows(os.path.join(module_dir, f'{artifact}.csv'), [row])
        write_rows(os.path.join(module_dir, 'summary.csv'), [row], append=True)

        if name not in pca_refs:
            pca_refs[name] = fit_pca_ref(matrix, args.embedding_check_max_points, seed)
        save_pca_plot(
            name,
            matrix,
            pca_refs[name],
            os.path.join(module_dir, f'{artifact}_pca.png'),
            args.embedding_check_max_points,
            seed + 31,
        )

    write_rows(os.path.join(dataset_dir, f'{artifact}_summary.csv'), rows)
    write_rows(os.path.join(dataset_dir, 'summary.csv'), rows, append=True)
    write_rows(os.path.join(args.embedding_check_dir, 'summary.csv'), rows, append=True)
    print(f'[EMBEDDING_CHECK] saved {dataset_dir} ({artifact})')
