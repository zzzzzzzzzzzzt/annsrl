""" In-degree distribution of NSW vs kNN.

The swap operator preserves OUT-degree exactly (17 for every node), so any structural
difference NSW gets from degree heterogeneity has to live in the IN-degree. This checks
whether that heterogeneity is real and how big it is, rather than asserting it.
"""
import numpy as np
import os.path as osp

DATA_DIR = 'data/SIFT100K'


from lib.utils import read_edges


def in_degree_stats(name, adj, pad=-1):
    flat = adj.ravel()
    flat = flat[(flat != pad) & (flat >= 0)]
    counts = np.bincount(flat, minlength=adj.shape[0])
    q = np.percentile(counts, [50, 90, 99, 99.9])
    print('%-14s in-deg: mean %5.1f  sd %5.1f  max %5d  p50 %3.0f p90 %3.0f p99 %4.0f p99.9 %4.0f  zero-in %5d'
          % (name, counts.mean(), counts.std(), counts.max(), q[0], q[1], q[2], q[3],
             int((counts == 0).sum())))
    return counts


nsw_path = osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs')
print('NSW file:', nsw_path, 'exists:', osp.exists(nsw_path))
if osp.exists(nsw_path):
    e = read_edges(nsw_path)
    n = len(e)
    out_deg = np.array([len(e[i]) for i in range(n)])
    print('NSW nodes %d  OUT-deg: mean %.2f sd %.2f min %d max %d'
          % (n, out_deg.mean(), out_deg.std(), out_deg.min(), out_deg.max()))
    flat = np.concatenate([np.asarray(e[i], dtype=np.int64) for i in range(n)])
    counts = np.bincount(flat[flat >= 0], minlength=n)
    q = np.percentile(counts, [50, 90, 99, 99.9])
    print('NSW            in-deg: mean %5.1f  sd %5.1f  max %5d  p50 %3.0f p90 %3.0f p99 %4.0f p99.9 %4.0f  zero-in %5d'
          % (counts.mean(), counts.std(), counts.max(), q[0], q[1], q[2], q[3],
             int((counts == 0).sum())))

import torch
snap = 'runs/knn_fresh_frozen_fixedpoint/dynamic_edges.700.pth'
print('peak snapshot:', snap, 'exists:', osp.exists(snap))
if osp.exists(snap):
    e = torch.load(snap, weights_only=False)
    n = len(e)
    out_deg = np.array([len(e[i]) for i in range(n)])
    print('PEAK nodes %d  OUT-deg: mean %.2f sd %.2f min %d max %d'
          % (n, out_deg.mean(), out_deg.std(), out_deg.min(), out_deg.max()))
    flat = np.concatenate([np.asarray(e[i], dtype=np.int64) for i in range(n)])
    counts = np.bincount(flat[flat >= 0], minlength=n)
    q = np.percentile(counts, [50, 90, 99, 99.9])
    print('PEAK (learned)  in-deg: mean %5.1f  sd %5.1f  max %5d  p50 %3.0f p90 %3.0f p99 %4.0f p99.9 %4.0f  zero-in %5d'
          % (counts.mean(), counts.std(), counts.max(), q[0], q[1], q[2], q[3],
             int((counts == 0).sum())))

knn_path = osp.join(DATA_DIR, 'knn_cache', 'init_knn_k18.npz')
print('kNN file:', knn_path, 'exists:', osp.exists(knn_path))
if osp.exists(knn_path):
    z = np.load(knn_path)
    print('kNN npz keys:', list(z.keys()))
    arr = z[list(z.keys())[0]]
    print('kNN arr shape', arr.shape)
    in_degree_stats('kNN k18', arr[:, 1:] if arr.shape[1] == 18 else arr)
