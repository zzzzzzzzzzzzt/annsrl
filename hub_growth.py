""" Does in-degree concentration track the recall peak?

If super-hubs are what turns the run over at step ~699, then max/sd of in-degree should
keep climbing straight through the peak while recall stops -- i.e. the graph goes on
concentrating after it stops paying. Reads the periodic snapshots so this is measured, not
inferred from the recall curve alone.
"""
import os.path as osp
import numpy as np
import torch

RUN = 'runs/knn_fresh_frozen_fixedpoint'
STEPS = [50, 150, 300, 450, 550, 650, 700, 800, 900, 1000]

print('%6s %9s %8s %8s %9s %9s' % ('step', 'in-sd', 'in-max', 'zero-in', 'p99.9', 'top1%share'))
for s in STEPS:
    p = osp.join(RUN, 'dynamic_edges.%d.pth' % s)
    if not osp.exists(p):
        continue
    e = torch.load(p, weights_only=False)
    n = len(e)
    flat = np.concatenate([np.asarray(e[i], dtype=np.int64) for i in range(n)])
    counts = np.bincount(flat[flat >= 0], minlength=n)
    # share of all in-edges absorbed by the top 1% of nodes: a scale-free concentration
    # measure that does not hinge on one outlier the way max does.
    top = np.sort(counts)[::-1][:max(1, n // 100)]
    print('%6d %9.1f %8d %8d %9.0f %9.3f'
          % (s, counts.std(), counts.max(), int((counts == 0).sum()),
             np.percentile(counts, 99.9), top.sum() / counts.sum()))
