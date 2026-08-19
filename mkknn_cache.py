""" Build a degree-matched kNN s_0 cache.

The kNN start ships at degree 24 (init_knn_k25.npz) while NSW M12 averages 16.7, and the
graph-edit swap PRESERVES degree -- so a kNN run that reaches NSW's recall does it with
44% more edges. Under a DCS budget the search cost is equalized but the memory footprint
is not, which makes "kNN beat NSW" an unfair claim. This builds the k=18 (degree 17)
cache so the comparison can be run degree-matched.

Normalization must match lib.Graph's: it applies 'global' (divide by mean norm) BEFORE
building kNN, and that is not a monotone per-point rescale across points, so building in
the raw space could pick different neighbours.
"""
import os.path as osp
import sys

import torch

from lib.graph import build_knn_graph, read_fvecs

DATA_DIR = './data/SIFT100K'


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    v = torch.tensor(read_fvecs(osp.join(DATA_DIR, 'sift_base.fvecs')))
    v = v / ((v ** 2).sum(-1) ** 0.5).mean().item()
    out = osp.join(DATA_DIR, 'knn_cache', 'init_knn_k%d.npz' % k)
    idx, _ = build_knn_graph(v, k, cache_path=out)
    print('built %s shape %s -> out-degree %d after dropping self'
          % (out, tuple(idx.shape), k - 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
