""" Iso-recall cost ratio: at the SAME recall, how many more distance computations does
the learned graph need than NSW? Inverts the frontier instead of reading a vertical gap,
which is the number that matters operationally -- nobody runs at a fixed DCS, they run at
a target recall and pay whatever it costs.
"""
import re
import numpy as np

PAT = re.compile(r'Ef (\d+) \| Recall@10 ([\d.]+) \| Distances: ([\d.]+)')


def curve(path):
    rc, d = [], []
    with open(path) as fh:
        for line in fh:
            m = PAT.match(line.strip())
            if m:
                rc.append(float(m.group(2)))
                d.append(float(m.group(3)))
    return np.array(rc), np.array(d)


def dcs_at(rc, d, target):
    """ recall is monotone increasing in dcs, so np.interp on (recall -> dcs) is valid. """
    if not (rc.min() <= target <= rc.max()):
        return np.nan
    return float(np.interp(target, rc, d))


nsw_rc, nsw_d = curve('log/expI_nsw_s0_ref.log')
runs = [
    ('learned, peak s699', 'log/expL_peak700.log'),
    ('learned, 500 steps',  'log/expJ_knn_fresh_frozen_acceptoff_500.log'),
    ('s_0 kNN deg17',       'log/expI_knn_s0_ref.log'),
]

print('%-22s %9s %11s %11s %8s' % ('', 'recall@10', 'NSW dcs', 'ours dcs', 'ratio'))
for name, path in runs:
    rc, d = curve(path)
    for target in (0.60, 0.70, 0.80, 0.90):
        nd = dcs_at(nsw_rc, nsw_d, target)
        od = dcs_at(rc, d, target)
        ratio = ('%.2fx' % (od / nd)) if np.isfinite(od) and np.isfinite(nd) else '--'
        print('%-22s %9.2f %11s %11s %8s'
              % (name, target,
                 '%.0f' % nd if np.isfinite(nd) else 'n/a',
                 '%.0f' % od if np.isfinite(od) else 'out of range',
                 ratio))
    print()
