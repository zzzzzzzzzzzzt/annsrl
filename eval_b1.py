"""B1: is the short-edge bias inherited from the pretrained scorer, or learned by PPO?

B0 established that the policy converges to an edge-length attractor at ~pct 1.6-2
and removes rewiring's long edges 5.8x faster than kNN's short ones. Two candidate
causes remain, and they imply opposite next steps:

  (a) The kNN-pretrained scorer cannot score a long edge highly, because its
      pretraining positives were nearest-neighbour pairs. Fix = change supervision.
  (b) The PPO reward itself prefers short edges (any local structure is rewarded
      because the batch recall responds to it). Fix = change the reward.

This runs the identical config with --scratch (random-init scorer) and asks whether
the attractor moves. If scratch lands at the SAME add_pct, the pretrained weights
are exonerated and (b) is the story. If scratch reaches a materially higher add_pct,
the pretrained init is the blocker and B2's restricted action space is the fix.

Reports recall/coverage under the shared harness AND the edge-length percentile, so
one table answers both "did it do better" and "did it change what it does".
"""
import os.path as osp
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build,
                          in_degrees, visit_counts, recall_at, gini)

STEPS = [100, 200, 300, 400, 500]
N_NODES = 20000
N_PAIRS = 200000
SEED = 7
SEEDS = [42, 123, 456]

ARMS = [('b1_scratch', 'runs/b1_scratch_s%d/dynamic_edges.%d.pth'),
        ('tier1a_ctrl', 'runs/tier1a_ctrl_s%d/dynamic_edges.%d.pth')]


def dists(V, a, b, chunk=200000):
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), chunk):
        e = min(s + chunk, len(a))
        d = V[a[s:e]].astype(np.float64) - V[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt((d * d).sum(1))
    return out


def load(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    return hnsw


def score(graph, edges_path):
    hnsw = load(graph, edges_path)
    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)
    ind = in_degrees(hnsw)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)
    seen = vis > 0
    return dict(recall=rec, se=se, coverage=float(seen.mean()),
                ind_max=int(ind.max()), ind_gini=gini(ind),
                adj=hnsw.adj.copy(), pad=hnsw.service_labels['pad'])


def diff_edges(base, learned, pad, nodes):
    a_s, a_d, r_s, r_d = [], [], [], []
    for i in nodes:
        b = base[i]; b = set(b[b != pad].tolist())
        l = learned[i]; l = set(l[l != pad].tolist())
        for t in (l - b):
            a_s.append(i); a_d.append(t)
        for t in (b - l):
            r_s.append(i); r_d.append(t)
    return (np.array(a_s, dtype=np.int64), np.array(a_d, dtype=np.int64),
            np.array(r_s, dtype=np.int64), np.array(r_d, dtype=np.int64))


def main():
    rng = np.random.default_rng(SEED)
    print('== B1: does a random-init scorer escape the short-edge attractor? ==')
    print('NQ=%d ef=%d k=%d dcs_budget=%d' % (NQ, EF, K, DCS_BUDGET))
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    n = V.shape[0]

    pa, pb = rng.integers(0, n, size=N_PAIRS), rng.integers(0, n, size=N_PAIRS)
    keep = pa != pb
    ref = np.sort(dists(V, pa[keep], pb[keep]))

    def pct(d):
        return 100.0 * np.searchsorted(ref, d) / len(ref)

    nodes = rng.choice(n, size=min(N_NODES, n), replace=False)
    base = score(graph, None)
    base_adj, pad = base['adj'], base['pad']
    print('\n%-18s %8s %9s %9s %9s %8s' %
          ('run', 'recall', 'coverage', 'add_pct', 'rem_pct', 'ind_max'))
    print('%-18s %8.4f %9.4f %9s %9s %8d' %
          ('kNN s_0', base['recall'], base['coverage'], '-', '-', base['ind_max']))

    out = {}
    for arm, tmpl in ARMS:
        for seed in SEEDS:
            path = tmpl % (seed, 500)
            if not osp.exists(path):
                print('%-18s MISSING %s' % ('%s_s%d' % (arm, seed), path))
                continue
            r = score(graph, path)
            a_s, a_d, r_s, r_d = diff_edges(base_adj, r['adj'], pad, nodes)
            r['add_pct'] = float(pct(dists(V, a_s, a_d)).mean()) if len(a_s) else float('nan')
            r['rem_pct'] = float(pct(dists(V, r_s, r_d)).mean()) if len(r_s) else float('nan')
            r['n_add'] = len(a_s)
            out[(arm, seed)] = r
            print('%-18s %8.4f %9.4f %9.2f %9.2f %8d' %
                  ('%s_s%d' % (arm, seed), r['recall'], r['coverage'],
                   r['add_pct'], r['rem_pct'], r['ind_max']))

    print('\n--- paired scratch - pretrained (same seed) ---')
    print('%-8s %10s %11s %10s' % ('seed', 'd_recall', 'd_coverage', 'd_add_pct'))
    dr, dc, dp = [], [], []
    for seed in SEEDS:
        a, b = out.get(('b1_scratch', seed)), out.get(('tier1a_ctrl', seed))
        if a is None or b is None:
            continue
        dr.append(a['recall'] - b['recall'])
        dc.append(a['coverage'] - b['coverage'])
        dp.append(a['add_pct'] - b['add_pct'])
        print('%-8d %+10.4f %+11.4f %+10.2f' % (seed, dr[-1], dc[-1], dp[-1]))

    if len(dr) > 1:
        m = len(dr)
        for name, v in (('recall', dr), ('coverage', dc), ('add_pct', dp)):
            v = np.asarray(v, dtype=np.float64)
            print('paired d_%-9s = %+.5f +- %.5f  (%d/%d positive)'
                  % (name, v.mean(), v.std(ddof=1) / np.sqrt(m), int((v > 0).sum()), m))

    print('\nd_add_pct ~ 0  => the pretrained weights are NOT the cause; the reward is.')
    print('d_add_pct >> 0 => the pretrained init carries the bias, and B2 (restricting')
    print('the candidate pool to long edges) is the right next intervention.')


if __name__ == '__main__':
    main()
