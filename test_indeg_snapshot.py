"""Does the PPO update score under s_t's in-degrees, or the live graph's?

sample_swaps stores in_deg_np alongside adj_ids for the same reason adj_ids is
stored at all: by the time the update runs, several epochs later, the graph holds
s_{t+1}. If the update recomputed the counts instead, the PPO ratio would compare
two policies evaluated on DIFFERENT states, which biases the gradient silently --
no crash, just a worse policy.

Proving the snapshot is correct needs it to actually DIFFER from the live graph at
update time. If it never differs, the code path is untested by construction, so
this asserts a difference is observed rather than just that the key exists.
"""
import atexit
import runpy
import sys

import numpy as np

import lib

stats = {'calls': 0, 'present': 0, 'differs': 0, 'max_gap': 0}
_orig = lib.GraphEditPPO.train_on_batch


def patched(self, action, *a, **kw):
    stats['calls'] += 1
    snap = action.get('in_deg_np')
    if snap is not None:
        stats['present'] += 1
        live = self.compute_in_degrees()
        gap = int(np.abs(np.asarray(snap, dtype=np.int64) - live).max())
        if gap:
            stats['differs'] += 1
            stats['max_gap'] = max(stats['max_gap'], gap)
    return _orig(self, action, *a, **kw)


lib.GraphEditPPO.train_on_batch = patched


@atexit.register
def report():
    print('\n[snapshot] update calls=%d, snapshot present=%d, '
          'differed from live graph=%d (max per-node gap %d)'
          % (stats['calls'], stats['present'], stats['differs'], stats['max_gap']))
    if not stats['calls']:
        print('[FAIL] train_on_batch never ran')
        return
    if stats['present'] != stats['calls']:
        print('[FAIL] snapshot missing on %d of %d update calls'
              % (stats['calls'] - stats['present'], stats['calls']))
        return
    if not stats['differs']:
        print('[FAIL] snapshot never differed from the live graph -- the s_t path '
              'is untested, a recompute would have passed too')
        return
    print('[ok] every update carried an s_t snapshot, and it genuinely differs '
          'from the live graph, so the stored counts are load-bearing')


sys.argv = [
    'train_sift100k_ppo.py',
    '--graph_type', 'nsw',
    '--pretrained_path', 'models/mlplink_SIFT100K_dot_best.pth',
    '--max_steps', '4', '--commit_stride', '1', '--max_grad_norm', '1.0',
    '--nodes_per_step', '128', '--dcs_budget', '300', '--ef', '32', '--k', '10',
    '--seed', '1234', '--init_seed', '1234', '--beta', '0',
    '--drop_mode', 'policy', '--no_plot', '--logit_scale', '20',
    '--accept', 'off',   # off so every edit commits and the graph really moves
    '--actor_ctx', '--indeg_ctx', '--indeg_noop',
    '--run_name', 'smoke_indeg_snap',
]
runpy.run_path('train_sift100k_ppo.py', run_name='__main__')
