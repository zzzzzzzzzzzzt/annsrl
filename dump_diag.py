"""Dump the diagnostic scalars from a TensorBoard run to a compact table.

Usage:  python dump_diag.py <run_dir> [step ...]
        python dump_diag.py runs/<exp_name> 0 50 100 200
        python dump_diag.py latest            # most recently modified run
        python dump_diag.py --list            # show available runs
        python dump_diag.py --compare <run_a> <run_b>   # phase-1 gate verdict

exp_name is generated from the hyperparameters, so pass 'latest' or copy the name
the training script prints at startup rather than inventing one. Use
--run_name on train_sift100k_ppo.py to pick the directory name yourself.
"""
import os
import sys
import glob
import os.path as osp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# The metrics that discriminate between the competing explanations for bad results.
KEYS = [
    # Optimizer health first: if no gradient reaches the weights, nothing below it
    # means anything -- the recall curve is then just the random-swap baseline.
    'train/grad_norm',
    'train/grad_norm_max',
    'train/steps_taken',
    'train/steps_skipped',
    'train/scaler_scale',
    'train/param_delta_rel',
    'train/policy_loss',
    # Why the gradient is that size: signal strength into the loss, and whether the
    # policy can express a preference at all.
    'train/advantage_std_raw',
    'train/reward_frac_exactly_zero',
    'train/logit_row_std',
    'train/logit_mean',
    # Drop-side entropy, logged only under drop_mode='policy'. Read against
    # log(graph/degree_mean) (=3.18 at degree 24): at that ceiling the drop softmax is
    # uniform, i.e. indistinguishable from drop_mode='random' however healthy the
    # gradient looks. train/drop_entropy_reg records the coefficient in force.
    'train/entropy_drop',
    'train/drop_entropy_reg',
    # Critic, logged only at gamma>0. value_td_shift is the one that decides whether
    # gamma means anything: it is |V(i,s_{t+1}) - V(i,s_t)|, and since z ignores
    # topology the whole shift comes from the neighbour mean. At ~0 the TD target is
    # r + (gamma-1)*V, a per-node constant, and the critic is inert.
    'train/value_loss',
    'train/value_mean',
    'train/value_td_shift',
    # Fraction of acting nodes whose edit survived the acceptance test. Always 1.0 at
    # --accept off. Under 'step' it is 0 or 1 per step, so its mean is the accept RATE.
    'train/accept_frac',
    # Fraction of the batch with positive advantage. Pinned at ~0.5 by construction
    # under adv_ref='batch' (centering forces mean 0), so it is only informative under
    # 'noop', where it should track how often an edit actually beat doing nothing.
    'train/advantage_frac_positive',
    # Upstream of the logits: if the node embeddings themselves are collapsed
    # (ratio << 1e-01), no temperature can rescue the softmax.
    'train/embed_spread_ratio',
    'train/logit_scale',
    # Fraction of sampled nodes the no-op gate decided to edit. Starts at
    # sigmoid(act_bias) and should MOVE if the gate is learning; frozen at the
    # prior means the gate gets no useful gradient.
    'train/noop/frac_acting',
    'train/recall',
    'train/reward_terms/r_target',
    'train/reward_terms/r_dense',
    'train/reward_terms/r_cost',
    'train/reward_terms/total',
    'train/search/num_hops',
    'train/search/dcs',
    'train/search/frac_hops_le2',
    'graph/reachable_frac',
    'graph/edge_len_mean',
    'graph/degree_mean',
    'val/recall@1',
    # Logged only when the reward's k > 1. Preferred over recall@1 where present:
    # measured on SIFT100K at equal cost, recall/SE is 5 at k=1 and 18 at k=10, so
    # recall@1 near the floor is mostly quantization noise.
    'val/recall@10',
    'val/mean_reward',
    'val/distance_computations',
    'train/credit/visits_median',
    'train/credit/frac_visits_le2',
    'train/credit/frac_observed',
    'train/credit/frac_observed_total',
    'train/nodes_credited',
    'train/graph_reward_delta',
    'train/kl',
    'train/entropy',
    'train/advantage',
]


def fmt(v):
    """Format so small-but-nonzero never renders as a flat 0.0000.

    grad_norm and KL live around 1e-6..1e-10 in this setup; %.4f turns all of them
    into '0.0000', which reads as "no gradient at all" and is a different diagnosis
    entirely from "gradient present but vanishing".
    """
    if v == 0:
        return '0'
    if abs(v) < 1e-3 or abs(v) >= 1e6:
        return '%.3e' % v
    return '%.4f' % v


def list_runs():
    """Run directories that actually contain event files, newest first."""
    runs = {osp.dirname(p) for p in glob.glob('runs/*/events.out.tfevents.*')}
    return sorted(runs, key=osp.getmtime, reverse=True)


def resolve(run_dir):
    """Map a user-supplied argument to a real run directory, or None."""
    if run_dir == 'latest':
        runs = list_runs()
        return runs[0] if runs else None
    if osp.isdir(run_dir):
        return run_dir
    # Tolerate a bare exp_name, and a unique substring of one.
    if osp.isdir(osp.join('runs', run_dir)):
        return osp.join('runs', run_dir)
    matches = [r for r in list_runs() if run_dir in osp.basename(r)]
    return matches[0] if len(matches) == 1 else None


def read_series(run_dir, keys):
    """{key: {step: value}} for the keys present in this run."""
    acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    acc.Reload()
    available = set(acc.Tags().get('scalars', []))
    return {k: {e.step: e.value for e in acc.Scalars(k)}
            for k in keys if k in available}


def trend(d):
    """(first, best, last) of a {step: value} series, or None if empty."""
    if not d:
        return None
    steps = sorted(d)
    return d[steps[0]], max(d.values()), d[steps[-1]]


# Each check is (label, key, how to judge). 'gate' is the decisive one: the policy
# must beat random edge addition on held-out recall. The others explain a failure
# but cannot substitute for it -- a healthy gradient that improves nothing means
# the defect is in the reward or the credit assignment, not the hyperparameters.
HEALTH = [
    ('embedding spread', 'train/embed_spread_ratio',
     lambda last: (last > 0.1, 'collapsed (<0.1) -- feat stats wrong again')),
    ('gradient norm', 'train/grad_norm',
     lambda last: (1e-4 < last < 1e3, 'outside 1e-4..1e3 -- vanished or diverging')),
    ('policy moved (|kl|)', 'train/kl',
     lambda last: (abs(last) > 1e-5, 'still ~0 -- a second blockage remains')),
    # 5.4 is 85% of the ADDITION side's uniform ceiling, log(574) = 6.353 (measured 574
    # 2-hop candidates per node on the random s_0 at degree 24). It used to be 57% of a
    # combined ceiling: before 2026-08-06 train/entropy silently included the drop term,
    # so the ceiling was log(574) + log(24) = 9.53 and observed values ran ~4.33 where
    # they now run ~3.10. The threshold still fires correctly -- a uniform addition
    # softmax reads ~6.35 -- but it is anchored to the narrower quantity now. The drop
    # side has its own row: train/entropy_drop against log(24) = 3.178.
    ('entropy', 'train/entropy',
     lambda last: (last < 5.4, 'at the uniform ceiling -- no preference expressed')),
]


def compare(name_a, name_b):
    dir_a, dir_b = resolve(name_a), resolve(name_b)
    for name, d in ((name_a, dir_a), (name_b, dir_b)):
        if d is None:
            print('No run directory matching %r.' % name)
            return 1

    keys = ['val/recall@1', 'val/recall@10', 'val/mean_reward',
            'train/graph_reward_delta'] + [k for _, k, _ in HEALTH]
    sa, sb = read_series(dir_a, keys), read_series(dir_b, keys)

    print('=' * 74)
    print('PHASE 1 GATE   A = %s (policy swaps)' % name_a)
    print('               B = %s (uniform swaps, frozen)' % name_b)
    print('=' * 74)

    # Judge on recall@k when the run logged it: at equal cost recall@10 carries ~3x
    # the signal/noise of recall@1, which from a weak s_0 sits close enough to the
    # floor that its differences are quantization noise.
    recall_key = 'val/recall@10' if (sa.get('val/recall@10')
                                     and sb.get('val/recall@10')) else 'val/recall@1'
    va, vb = sa.get(recall_key, {}), sb.get(recall_key, {})
    ta, tb = trend(va), trend(vb)
    if ta is None or tb is None:
        print('%s missing from one of the runs; cannot judge.' % recall_key)
        return 1
    # Validation runs every 10 steps, so a short run has one or two points and
    # both curves are still sitting on the shared s_0 value. Refuse to render a
    # verdict from that rather than reporting a meaningless FAIL.
    n_val = min(len(va), len(vb))
    too_short = n_val < 5

    print('\n%-19s %-10s %-10s %-10s' % (recall_key, 'start', 'best', 'end'))
    print('  A policy          %-10.4f %-10.4f %-10.4f' % ta)
    print('  B uniform         %-10.4f %-10.4f %-10.4f' % tb)
    print('  A-B (best)        %+.4f' % (ta[1] - tb[1]))
    print('  A vs its own s_0  %+.4f' % (ta[1] - ta[0]))
    # A and B now share --seed, so they start from the SAME s_0 and the two 'start'
    # values must agree. A gap there means the pairing broke and A-B is confounded by
    # the starting graph -- previously measured at 0.0418, larger than the advantage
    # the gate credited to A.
    if abs(ta[0] - tb[0]) > 1e-9:
        print('  [warn] A and B start at different %s (%.4f vs %.4f): runs are NOT'
              % (recall_key, ta[0], tb[0]))
        print('         paired, so A-B mixes the policy effect with the s_0 gap.')

    # From a random s_0, recall@1 starts at ~0.0036 -- close enough to the floor that
    # its differences are quantization noise, while val/mean_reward is the quantity the
    # policy actually maximizes and moves by 1e-02. Print both, and let the shaped
    # reward decide the verdict when recall has not left its starting value.
    ra, rb = trend(sa.get('val/mean_reward', {})), trend(sb.get('val/mean_reward', {}))
    reward_gap = None
    if ra is not None and rb is not None:
        print('\nval/mean_reward     %-10s %-10s %-10s' % ('start', 'best', 'end'))
        print('  A policy          %-10.4f %-10.4f %-10.4f' % ra)
        print('  B uniform         %-10.4f %-10.4f %-10.4f' % rb)
        print('  A-B (best)        %+.4f' % (ra[1] - rb[1]))
        print('  A vs its own s_0  %+.4f' % (ra[1] - ra[0]))
        reward_gap = (ra[1] - rb[1], ra[1] - ra[0])

    print('\nhealth of run A')
    problems = []
    for label, key, judge in HEALTH:
        t = trend(sa.get(key, {}))
        if t is None:
            print('  %-22s (not logged)' % label)
            continue
        ok, why = judge(t[2])
        print('  %-22s %-12s %s' % (label, fmt(t[2]), 'ok' if ok else 'BAD: ' + why))
        if not ok:
            problems.append(label)

    beats_random = ta[1] > tb[1]
    beats_start = ta[1] > ta[0]
    # recall pinned to its start value means it carries no information; fall back to
    # the shaped reward rather than reporting a FAIL that only reflects the floor.
    recall_flat = abs(ta[1] - ta[0]) < 1e-4 and abs(tb[1] - tb[0]) < 1e-4
    judged_on = recall_key
    if recall_flat and reward_gap is not None:
        beats_random, beats_start = reward_gap[0] > 0, reward_gap[1] > 0
        judged_on = 'val/mean_reward (%s never left s_0)' % recall_key
    print('\njudged on: %s' % judged_on)
    print('\n' + '-' * 74)
    if too_short:
        print('TOO SHORT TO JUDGE: only %d validation point(s) per run (need >= 5).'
              % n_val)
        print('      val/recall@1 is logged every 10 steps, so both curves are')
        print('      still on the shared s_0 value. Re-run with MAX_STEPS >= 100.')
        print('      The health block above IS meaningful -- it is per-step.')
    elif beats_random and beats_start:
        print('PASS: policy beats uniform swaps AND improves on s_0.')
        print('      -> proceed to phase 2 (credit noise) / phase 3 (reward terms).')
    elif beats_random:
        print('PARTIAL: policy beats random, but never exceeds its own starting graph.')
        print('      -> the reward is likely steering away from recall; do phase 3')
        print('         (--alpha 0 --beta 0 --k 10) before anything else.')
    else:
        print('FAIL: policy does not beat random edge addition.')
        if problems:
            print('      unhealthy: %s -- fix these first.' % ', '.join(problems))
        else:
            print('      optimizer looks healthy, so the defect is downstream:')
            print('      reward exchange rate (phase 3) or credit noise (phase 2).')
        print('      Do NOT tune hyperparameters against this.')
    print('-' * 74)
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    if sys.argv[1] in ('--list', '-l'):
        for run in list_runs():
            print(run)
        return 0

    if sys.argv[1] == '--compare':
        if len(sys.argv) != 4:
            print('Usage: python dump_diag.py --compare <run_a> <run_b>')
            return 1
        return compare(sys.argv[2], sys.argv[3])

    requested = sys.argv[1]
    run_dir = resolve(requested)
    if run_dir is None:
        # exp_name is derived from the hyperparameters, so a hand-typed name almost
        # never matches. Show what exists instead of letting tensorboard raise.
        print("No run directory matching %r." % requested)
        runs = list_runs()
        if not runs:
            print('No runs with event files found under ./runs.')
            return 1
        print('\nAvailable runs (newest first):')
        for run in runs[:15]:
            print('   ', run)
        if len(runs) > 15:
            print('    ... and %d more (use --list)' % (len(runs) - 15))
        print("\nTry:  python %s latest %s" % (osp.basename(sys.argv[0]),
                                               ' '.join(sys.argv[2:])))
        return 1

    want = [int(s) for s in sys.argv[2:]] or None

    acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    acc.Reload()
    available = set(acc.Tags().get('scalars', []))

    series = {}
    for key in KEYS:
        if key in available:
            series[key] = {e.step: e.value for e in acc.Scalars(key)}

    if not series:
        print('No known scalars found in %s. Tags present:' % run_dir)
        for tag in sorted(available):
            print('   ', tag)
        return 1

    all_steps = sorted({s for d in series.values() for s in d})
    if want is None:
        # first, last, and three evenly spaced points in between
        picks = [all_steps[0], all_steps[-1]]
        for frac in (0.25, 0.5, 0.75):
            picks.append(all_steps[int(frac * (len(all_steps) - 1))])
        steps = sorted(set(picks))
    else:
        steps = []
        for w in want:
            # nearest recorded step at or before w, else the earliest available
            candidates = [s for s in all_steps if s <= w] or all_steps
            steps.append(candidates[-1])
        steps = sorted(set(steps))

    width = max(len(k) for k in series) + 2
    print('run: %s' % osp.abspath(run_dir))
    print()
    print('metric'.ljust(width) + ''.join('%13s' % ('step %d' % s) for s in steps))
    print('-' * (width + 13 * len(steps)))
    for key in KEYS:
        if key not in series:
            continue
        row = key.ljust(width)
        for s in steps:
            v = series[key].get(s)
            if v is None:
                # carry the most recent earlier value; metrics log at different cadences
                earlier = [k for k in series[key] if k <= s]
                v = series[key][max(earlier)] if earlier else None
            row += '%13s' % ('--' if v is None else fmt(v))
        print(row)

    missing = [k for k in KEYS if k not in series]
    if missing:
        print('\nnot logged in this run (expected if it predates the diagnostics):')
        for key in missing:
            print('   ', key)
    return 0


if __name__ == '__main__':
    sys.exit(main())
