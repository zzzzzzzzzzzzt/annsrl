""" Paired summary of --actor_ctx (item 3) on the NSW start.

The claim to settle: does conditioning the pair scorer on the source node's current
neighbourhood beat the endpoint-only scorer? Both arms use the new defaults
(accept=node, adv_ref auto->noop), so the architecture change is the only difference.

Pairing: on an NSW start s_0 is loaded from a file, so it is IDENTICAL across seeds --
the seed only moves the node sample, the probe set and the Gumbel noise. Every run must
report the same starting recall, which is checked below.

Secondary keys are reported because a null result has two very different causes: the
head never moved (ctx_head is zero-init, so if the policy's behaviour is identical the
architecture was never exercised) versus the head moved and did not help. accept_frac
and logit_row_std separate those.
"""
import glob
import os.path as osp
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEY = 'val/recall@10'
SECONDARY = ('train/accept_frac', 'train/logit_row_std', 'train/noop/frac_acting',
             'train/advantage_frac_positive', 'graph/edge_len_mean')
SEEDS = (1234, 777, 42, 2024)
CELLS = ('off', 'on')


def load(run):
    """{tag: (steps, values)} for KEY + SECONDARY, or None if the run dir has no events."""
    paths = glob.glob(osp.join('runs', run, 'events.out.tfevents.*'))
    if not paths:
        return None
    want = (KEY,) + SECONDARY
    buf = {t: ([], []) for t in want}
    for p in sorted(paths):
        acc = EventAccumulator(p, size_guidance={'scalars': 0})
        acc.Reload()
        tags = acc.Tags().get('scalars', [])
        for t in want:
            if t not in tags:
                continue
            for e in acc.Scalars(t):
                buf[t][0].append(e.step)
                buf[t][1].append(e.value)
    out = {}
    for t, (s, v) in buf.items():
        if not v:
            continue
        order = np.argsort(s)
        out[t] = (np.asarray(s)[order], np.asarray(v)[order])
    return out or None


def main():
    table, secondary, starts = {}, {}, []
    print('%-8s %6s %8s %8s %9s' % ('ctx', 'seed', 'start', 'end', 'delta'))
    for ctx in CELLS:
        for seed in SEEDS:
            run = 'nsw_ctx%s_s%d' % (ctx, seed)
            d = load(run)
            if d is None or KEY not in d:
                print('%-8s %6d   (missing)' % (ctx, seed))
                continue
            v = d[KEY][1]
            table[(ctx, seed)] = v[-1] - v[0]
            secondary[(ctx, seed)] = d
            starts.append(v[0])
            print('%-8s %6d %8.4f %8.4f %+9.4f' % (ctx, seed, v[0], v[-1], v[-1] - v[0]))

    if starts and (max(starts) - min(starts)) > 1e-9:
        print('\n[warn] runs do NOT share a starting recall (%.4f..%.4f); on an NSW start'
              ' s_0 comes from a file and must be identical, so the differences below'
              ' mix in an s_0 gap.' % (min(starts), max(starts)))

    print('\nper-arm mean improvement over s_0:')
    for ctx in CELLS:
        d = np.asarray([table[(ctx, s)] for s in SEEDS if (ctx, s) in table])
        if not d.size:
            continue
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
        print('  ctx=%-4s n=%d  %+.4f +- %.4f   (%d/%d seeds positive)'
              % (ctx, len(d), d.mean(), se, int((d > 0).sum()), len(d)))

    pairs = np.asarray([table[('on', s)] - table[('off', s)] for s in SEEDS
                        if ('on', s) in table and ('off', s) in table])
    if pairs.size:
        se = pairs.std(ddof=1) / np.sqrt(len(pairs)) if len(pairs) > 1 else float('nan')
        sigma = abs(pairs.mean()) / se if se and np.isfinite(se) and se > 0 else float('nan')
        print('\npaired difference (same seed, so s_0 and the splits cancel):')
        print('  ctx=on - ctx=off  n=%d  %+.4f +- %.4f  (%.1f sigma, %d/%d same sign)'
              % (len(pairs), pairs.mean(), se, sigma,
                 int((np.sign(pairs) == np.sign(pairs.mean())).sum()), len(pairs)))

    # COMPLETE PAIRS ONLY. Averaging over whatever happens to be on disk silently mixes
    # an in-flight run into one arm, which moves the diff by more than the effect being
    # measured (seen live: a half-finished baseline shifted accept_frac by +0.010).
    done = [s for s in SEEDS if ('on', s) in secondary and ('off', s) in secondary]
    print('\nwas the new head exercised at all? (mean over the run, then over the %d'
          ' complete pairs: %s)' % (len(done), done))
    print('%-28s %10s %10s %10s' % ('key', 'ctx=off', 'ctx=on', 'diff'))
    for t in SECONDARY:
        row = []
        for ctx in CELLS:
            vals = [secondary[(ctx, s)][t][1].mean() for s in done
                    if t in secondary[(ctx, s)]]
            row.append(np.mean(vals) if vals else float('nan'))
        print('%-28s %10.4f %10.4f %+10.4f' % (t, row[0], row[1], row[1] - row[0]))


if __name__ == '__main__':
    sys.exit(main())
