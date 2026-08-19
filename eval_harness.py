"""One harness, every arm, both controls. Use this instead of ad-hoc eval scripts.

Enforces the three methodology rules this project learned the hard way:
  1. Never compare in-training recalls across runs (each evaluates its own batch;
     measured discrepancy: ctrl_s789 printed 0.8096, actual 0.4997). Everything here
     is re-scored on the same val_queries[:NQ] with the same ef/k/budget.
  2. Report the learning-free control alongside every arm, at BOTH a fixed 5% dose
     and matched to the arm's own edit count. An arm that loses to matched-random
     has contributed nothing regardless of how it compares to s_0.
  3. Report coverage beside recall. Coverage ordered recall at rho=+1.0 across the
     Tier 0 graphs; the loop's unbounded-BFS `reachable` saturates at ~0.994.

Usage:
    python eval_harness.py runs/tier1a_ctrl_s42/dynamic_edges.500.pth [more...]
    python eval_harness.py --glob 'runs/b1_scratch_s*/dynamic_edges.500.pth'
"""
import argparse
import glob as globmod
import os.path as osp
import numpy as np
import torch
import lib
import lib_control
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build,
                          in_degrees, visit_counts, recall_at, gini)


def make_hnsw(graph, edges=None):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges is not None:
        hnsw.dynamic_edges = edges
        hnsw.adj = hnsw.build_adjacency()
    return hnsw


def score(graph, edges=None, want_adj=False):
    hnsw = make_hnsw(graph, edges)
    res = hnsw.search_deterministic(graph.val_queries[:NQ])
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
    ind = in_degrees(hnsw)
    out = dict(recall=rec, se=se, coverage=float((vis > 0).mean()),
               dcs=float(res['total_distance_computations'].mean()),
               ind_max=int(ind.max()), ind_gini=gini(ind))
    if want_adj:
        out['adj'] = hnsw.adj.copy()
        out['pad'] = hnsw.service_labels['pad']
    return out


HDR = '%-30s %8s %8s %9s %9s %8s %9s'
ROW = '%-30s %8.4f %8.4f %9.4f %9.1f %8d %9s'

N_PAIRS = 200000        # random pairs defining the length-percentile reference CDF
N_LEN_NODES = 20000     # nodes sampled for the added-edge length statistic


def _pair_dists(V, a, b, chunk=200000):
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), chunk):
        e = min(s + chunk, len(a))
        d = V[a[s:e]].astype(np.float64) - V[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt((d * d).sum(1))
    return out


def length_ref(graph, seed=7):
    """Reference CDF + a sampled node set, so edge lengths can be quoted as a
    percentile of random vertex-pair distances. A percentile is comparable across
    graphs; a raw L2 in a 128-dim normalized space carries no intuition."""
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    rng = np.random.default_rng(seed)
    n = V.shape[0]
    pa, pb = rng.integers(0, n, size=N_PAIRS), rng.integers(0, n, size=N_PAIRS)
    keep = pa != pb
    ref = np.sort(_pair_dists(V, pa[keep], pb[keep]))
    nodes = rng.choice(n, size=min(N_LEN_NODES, n), replace=False)
    return V, ref, nodes


def added_edge_pct(V, ref, nodes, base_adj, adj, pad):
    """Mean length percentile of the edges an arm ADDED relative to s_0.

    This is the quantity the short-edge attractor is defined in: every policy run so
    far lands at 1.6-2.0 regardless of where it starts. An intervention that does not
    move this number has not changed what the policy does, whatever it did to recall.
    """
    src, dst = [], []
    for i in nodes:
        b = base_adj[i]; b = set(b[b != pad].tolist())
        l = adj[i]; l = set(l[l != pad].tolist())
        for t in (l - b):
            src.append(i); dst.append(t)
    if not src:
        return float('nan')
    d = _pair_dists(V, np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64))
    return float((100.0 * np.searchsorted(ref, d) / len(ref)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*', help='dynamic_edges snapshots to score')
    ap.add_argument('--glob', default=None, help='glob pattern for snapshots')
    ap.add_argument('--graph_type', default='knn')
    ap.add_argument('--seeds', type=int, default=3,
                    help='seeds for the fixed-dose control')
    ap.add_argument('--no_len', action='store_true',
                    help='skip the added-edge length percentile (the slow column)')
    args = ap.parse_args()

    paths = list(args.paths)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    paths = [p for p in paths if osp.exists(p)]

    graph = build(args.graph_type)
    base = score(graph, None, want_adj=True)
    base_adj, pad = base['adj'], base['pad']

    print('== eval_harness: NQ=%d ef=%d k=%d dcs_budget=%d s_0=%s =='
          % (NQ, EF, K, DCS_BUDGET, args.graph_type))
    lref = None if args.no_len else length_ref(graph)
    print(HDR % ('graph', 'recall', '+-SE', 'coverage', 'dcs/q', 'ind_max', 'n_moved')
          + ('' if args.no_len else '%9s' % 'add_pct'))
    print(ROW % ('s_0 (%s)' % args.graph_type, base['recall'], base['se'],
                 base['coverage'], base['dcs'], base['ind_max'], '-')
          + ('' if args.no_len else '%9s' % '-'))

    # Control 1: the tuned fixed dose, averaged over seeds. Independent of any arm.
    seeds = (2, 3, 5)[:args.seeds]
    ctrl_seeds = (2, 3, 5, 7, 11)[:max(args.seeds, 3)]
    mean, se, k = lib_control.control_row(
        graph, base_adj, pad, lambda e: score(graph, e)['recall'], seeds=seeds)
    print(('%-30s %8.4f %8.4f %9s %9s %8s %9s')
          % ('CONTROL random %.0f%% (n=%d)' % (100 * lib_control.DEFAULT_DOSE, len(seeds)),
             mean, se, '-', '-', '-', '%d' % k))

    rows = []
    for p in paths:
        edges = torch.load(p, weights_only=False)
        r = score(graph, edges, want_adj=True)
        moved = lib_control.changed_slots(base_adj, r['adj'], pad)
        label = p.replace('runs/', '').replace('/dynamic_edges', '@').replace('.pth', '')
        tail = ''
        if lref is not None:
            V, ref, nodes = lref
            tail = '%9.2f' % added_edge_pct(V, ref, nodes, base_adj, r['adj'], pad)
        print(ROW % (label, r['recall'], r['se'], r['coverage'], r['dcs'],
                     r['ind_max'], '%d' % moved) + tail)
        rows.append((label, r, moved))

    # Control 2: matched to each arm's own edit budget. This is the one that matters.
    if rows:
        print()
        print('--- vs COUNT-MATCHED random control (same edits, numpy targets) ---')
        # Averaged over seeds, because the control is NOT a fixed number: which slots get
        # rewired and where they point are two separate draws, and at one seed the matched
        # control has swung 0.5023-0.5522 between two arms whose edit counts differ by 1%.
        # Every single-seed d(arm-ctrl) in this project carries that unreported +-0.025.
        print('%-30s %9s %9s %9s %11s %9s' %
              ('arm', 'recall', 'ctrl', 'ctrl SE', 'd(arm-ctrl)', 'n_moved'))
        deltas = []
        for label, r, moved in rows:
            cv = []
            for s in ctrl_seeds:
                edges, k = lib_control.rewire(base_adj, pad, n_edges=moved, seed=s)
                cv.append(score(graph, edges)['recall'])
            cv = np.asarray(cv, dtype=np.float64)
            cse = cv.std(ddof=1) / np.sqrt(len(cv)) if len(cv) > 1 else 0.0
            d = r['recall'] - cv.mean()
            deltas.append(d)
            print('%-30s %9.4f %9.4f %9.4f %+11.4f %9d' %
                  (label, r['recall'], cv.mean(), cse, d, moved))
        v = np.asarray(deltas, dtype=np.float64)
        if len(v) > 1:
            print('mean d = %+.4f +- %.4f  (%d/%d arms beat matched random)'
                  % (v.mean(), v.std(ddof=1) / np.sqrt(len(v)),
                     int((v > 0).sum()), len(v)))
        print()
        print('An arm only contributes something if d > 0. Beating s_0 is not enough:')
        print('random rewiring beats s_0 too, and by more.')
        print('The real bar is the C2 length rule -- kNN + 5%% of slots rewired to length')
        print('percentile 65 -- at 0.5566 +- 0.0080 over 3 slot seeds. Reproduce with:')
        print('  python c2_lengthsweep.py --dose 0.05 --pcts 65 --slot_seed {11,23,37}')


if __name__ == '__main__':
    main()
