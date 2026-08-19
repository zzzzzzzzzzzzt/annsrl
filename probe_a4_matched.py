"""A4: the policy vs random rewiring at MATCHED edge-change count.

The 5%-rewiring comparison moves 120k directed edges. If the policy moved fewer
than that in 500 steps, the comparison flattered random rewiring by giving it a
bigger budget of changes. This measures how many out-edge slots each learned
graph actually repointed away from its kNN start, then hands uniform random
rewiring exactly that many.

Only then is "did the policy pick better edges than chance?" a fair question:
same start, same number of edges moved, same evaluation harness.
"""
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build, in_degrees,
                          visit_counts, recall_at)

STEP = 500
RUNS = [('ctrl_s42', 42), ('ctrl_s123', 123), ('ctrl_s456', 456), ('ctrl_s789', 789),
        ('ideg_s42', 42), ('ideg_s123', 123), ('ideg_s456', 456), ('ideg_s789', 789)]
REWIRE_SEEDS = [1, 2, 3]


def load(graph, edges_path):
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()
    return hnsw


def measure(hnsw, graph):
    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)
    vis, _ = visit_counts(res, hnsw.num_vertices, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)
    seen = vis > 0
    return dict(recall=rec, se=se, coverage=float(seen.mean()),
                pops=float(vis[seen].mean()) if seen.any() else 0.0)


def changed_slots(base, learned, pad):
    """Out-edge slots whose target is NOT in that node's original kNN neighbour set.

    Counted per node as a set difference, so re-adding an edge the node already had
    is not counted as a change and a node edited twice is not double-counted.
    """
    n, total = base.shape[0], 0
    for i in range(n):
        b = base[i]
        bs = set(b[b != pad].tolist())
        l = learned[i]
        total += sum(1 for t in l[l != pad].tolist() if t not in bs)
    return total


def rewire_n(base, pad, k, rng, n_vert):
    adj = base.copy()
    valid = np.argwhere(base != pad)
    k = min(k, len(valid))
    pick = valid[rng.choice(len(valid), size=k, replace=False)]
    adj[pick[:, 0], pick[:, 1]] = rng.integers(0, n_vert, size=k).astype(adj.dtype)
    return adj


def main():
    print('== A4: matched edge-change count ==')
    print('NQ=%d ef=%d k=%d budget=%d step=%d' % (NQ, EF, K, DCS_BUDGET, STEP))
    graph = build('knn')
    hnsw = load(graph, None)
    pad, n = hnsw.service_labels['pad'], hnsw.num_vertices
    base = hnsw.adj.copy()
    n_edges = int((base != pad).sum())
    b = measure(hnsw, graph)
    print('\nkNN s_0: recall %.4f  coverage %.4f  (%d directed edges)'
          % (b['recall'], b['coverage'], n_edges))

    print('\n--- how many edges did each learned run actually move? ---')
    print('%-12s %10s %8s %9s %9s' % ('run', 'moved', 'of all', 'recall', 'coverage'))
    counts = {}
    for name, seed in RUNS:
        arm = 'ctrl' if name.startswith('ctrl') else 'ideg'
        p = 'runs/tier1a_%s_s%d/dynamic_edges.%d.pth' % (arm, seed, STEP)
        h = load(graph, p)
        c = changed_slots(base, h.adj, pad)
        r = measure(h, graph)
        counts[name] = (c, r)
        print('%-12s %10d %7.2f%% %9.4f %9.4f'
              % (name, c, 100.0 * c / n_edges, r['recall'], r['coverage']))

    med = int(np.median([c for c, _ in counts.values()]))
    print('\nmedian edges moved by the policy: %d (%.2f%% of all edges)'
          % (med, 100.0 * med / n_edges))

    print('\n--- random rewiring at THAT SAME count ---')
    print('%-12s %10s %9s %9s %9s' % ('seed', 'moved', 'recall', '+-SE', 'coverage'))
    recs = []
    for s in REWIRE_SEEDS:
        hnsw.adj = rewire_n(base, pad, med, np.random.default_rng(s), n)
        r = measure(hnsw, graph)
        recs.append(r['recall'])
        print('%-12d %10d %9.4f %9.4f %9.4f'
              % (s, med, r['recall'], r['se'], r['coverage']))
    hnsw.adj = base

    pol = [r['recall'] for _, r in counts.values()]
    print('\n%-34s %.4f +- %.4f' % ('policy (n=8)', np.mean(pol), np.std(pol, ddof=1)))
    print('%-34s %.4f +- %.4f' % ('random @ matched count (n=3)',
                                  np.mean(recs), np.std(recs, ddof=1)))
    print('%-34s %.4f' % ('kNN s_0', b['recall']))
    print('\nIf random at the matched count still wins, the policy is not choosing')
    print('better edges than chance. If the policy wins here but lost at 5%%, then')
    print('it picks better edges but far too few of them -- a different problem,')
    print('and a fixable one.')


if __name__ == '__main__':
    main()
