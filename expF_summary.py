""" Paired summary of the accept x adv_ref grid on the NSW start.

The claim to settle: does adv_ref='noop' beat 'batch' when accept='node', and is the
whole combination reliably positive on the NSW start? Single-seed differences are not
enough -- the seed-to-seed spread observed so far is ~0.004 (the gate's two seeds gave
+0.0079/+0.0076, the argmin ablation -0.0015/-0.0004), which is the same order as the
effect being claimed (+0.0092).

Pairing: on an NSW start s_0 is loaded from a file, so it is IDENTICAL across seeds --
the seed only moves the node sample, the probe set and the Gumbel noise. Every run must
therefore report the same starting recall, which is checked below; a mismatch would mean
the runs are not paired and the differences mix in an s_0 gap.
"""
import glob
import os.path as osp
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEY = 'val/recall@10'
SEEDS = (1234, 777, 42, 2024)
CELLS = (('off', 'batch'), ('node', 'batch'), ('node', 'noop'))


def series(run):
    """(steps, values) for KEY, or None when the run has no such scalar."""
    paths = glob.glob(osp.join('runs', run, 'events.out.tfevents.*'))
    if not paths:
        return None
    steps, vals = [], []
    for p in sorted(paths):
        acc = EventAccumulator(p, size_guidance={'scalars': 0})
        acc.Reload()
        if KEY not in acc.Tags().get('scalars', []):
            continue
        for e in acc.Scalars(KEY):
            steps.append(e.step)
            vals.append(e.value)
    if not vals:
        return None
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(vals)[order]


def run_name(accept, ref, seed):
    return 'nsw_ar%s_acc%s_s%d' % (ref, accept, seed)


def find(accept, ref, seed):
    """series() for a cell, falling back to the pre-adv_ref naming.

    The accept-only runs were launched before --adv_ref existed, so they are named
    nsw_acc_<mode>_s<seed>. They ARE the adv_ref='batch' cells (that is the default),
    so they count as data rather than as missing runs.
    """
    s = series(run_name(accept, ref, seed))
    if s is None and ref == 'batch':
        s = series('nsw_acc_%s_s%d' % (accept, seed))
    return s


def main():
    table = {}
    starts = []
    print('%-18s %8s %8s %8s %8s' % ('cell', 'seed', 'start', 'end', 'delta'))
    for accept, ref in CELLS:
        for seed in SEEDS:
            s = find(accept, ref, seed)
            if s is None:
                print('%-18s %8d   (missing)' % ('%s/%s' % (accept, ref), seed))
                continue
            _, v = s
            table[(accept, ref, seed)] = v[-1] - v[0]
            starts.append(v[0])
            print('%-18s %8d %8.4f %8.4f %+8.4f'
                  % ('%s/%s' % (accept, ref), seed, v[0], v[-1], v[-1] - v[0]))

    if starts and (max(starts) - min(starts)) > 1e-9:
        print('\n[warn] runs do NOT share a starting recall (%.4f..%.4f). On an NSW start'
              % (min(starts), max(starts)))
        print('       s_0 comes from a file and must be identical, so the differences'
              ' below mix in an s_0 gap.')

    print('\nper-cell mean improvement over s_0:')
    for accept, ref in CELLS:
        d = [table[(accept, ref, s)] for s in SEEDS if (accept, ref, s) in table]
        if not d:
            continue
        d = np.asarray(d)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
        print('  %-12s n=%d  %+.4f +- %.4f   (%d/%d seeds positive)'
              % ('%s/%s' % (accept, ref), len(d), d.mean(), se, int((d > 0).sum()), len(d)))

    print('\npaired differences (same seed, so s_0 and the query splits cancel):')
    for (a1, r1), (a2, r2) in ((('node', 'noop'), ('node', 'batch')),
                               (('node', 'batch'), ('off', 'batch')),
                               (('node', 'noop'), ('off', 'batch'))):
        pairs = [(table[(a1, r1, s)] - table[(a2, r2, s)]) for s in SEEDS
                 if (a1, r1, s) in table and (a2, r2, s) in table]
        if not pairs:
            continue
        p = np.asarray(pairs)
        se = p.std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else float('nan')
        sigma = abs(p.mean()) / se if se and np.isfinite(se) and se > 0 else float('nan')
        print('  %-13s - %-13s n=%d  %+.4f +- %.4f  (%.1f sigma, %d/%d same sign)'
              % ('%s/%s' % (a1, r1), '%s/%s' % (a2, r2), len(p), p.mean(), se, sigma,
                 int((np.sign(p) == np.sign(p.mean())).sum()), len(p)))


if __name__ == '__main__':
    sys.exit(main())
