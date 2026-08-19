""" Compare ANN runs at MATCHED search cost.

The ef sweep at the end of a run lifts the training DCS budget, and the distance count at
a given ef depends on the graph, so two runs' `Ef 32` lines are NOT the same operating
point. Everything here is interpolated onto a common distance grid instead.
"""
import re
import numpy as np

RUNS = [
    ('s_0  kNN deg17',           'log/expI_knn_s0_ref.log'),
    ('s_0  NSW M12  (target)',   'log/expI_nsw_s0_ref.log'),
    ('frozen  accept=OFF',       'log/expI_knn_frozen_acceptoff_500.log'),
    ('fresh frozen accept=OFF',  'log/expJ_knn_fresh_frozen_acceptoff_500.log'),
    ('  ^ same, at peak s700',   'log/expL_peak700.log'),
    ('  ^ past peak, s850',      'log/expL_peak850.log'),
    ('frozen  accept=node',      'log/expH_knn_xfer_frozen_s42.log'),
    ('xfer+train  accept=node',  'log/expH_knn_xfer_train_s42.log'),
    ('fresh train  accept=node', 'log/expH_knn_fresh_train_s42.log'),
]
GRID = [350, 400, 450, 500, 550, 600]
PAT = re.compile(r'Ef (\d+) \| Recall@10 ([\d.]+) \| Distances: ([\d.]+)')


def parse(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            m = PAT.match(line.strip())
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return rows


def interp(rows, grid):
    """ NaN outside the measured range: np.interp would silently clamp instead. """
    d = np.array([r[2] for r in rows])
    rc = np.array([r[1] for r in rows])
    return np.array([float(np.interp(g, d, rc)) if d.min() <= g <= d.max() else np.nan
                     for g in grid])


curves = {}
for name, path in RUNS:
    rows = parse(path)
    if not rows:
        print('NO SWEEP: %s (%s)' % (name, path))
        continue
    curves[name] = interp(rows, GRID)
    print('%-26s ef %d..%d  dcs %.0f..%.0f  (%d pts)'
          % (name, rows[0][0], rows[-1][0], rows[0][2], rows[-1][2], len(rows)))

print('\n%-26s %s' % ('recall@10 @ matched DCS', ' '.join('%8d' % g for g in GRID)))
for name, c in curves.items():
    print('%-26s %s' % (name, ' '.join('%8.4f' % v for v in c)))

base = curves.get('s_0  kNN deg17')
if base is not None:
    print('\n%-26s %s' % ('GAIN over s_0', ' '.join('%8d' % g for g in GRID)))
    for name, c in curves.items():
        if not name.startswith('s_0'):
            print('%-26s %s' % (name, ' '.join('%+8.4f' % v for v in c - base)))
