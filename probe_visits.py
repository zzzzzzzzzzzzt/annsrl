"""Does in-degree uniformity buy VISIT-frequency uniformity, and does that buy recall?

The hub-explosion diagnosis stopped at structure: the learned policy drives max
in-degree to 8039 against NSW's 71. The proposed mechanism goes one step further --
if in-degree were uniform, visit frequency would be too, and a uniform walk wastes
less of the DCS budget. That is two claims, and they are separable:

  arrow 1:  in-degree skew  ->  visit-frequency skew
  arrow 2:  visit-frequency skew  ->  recall

Arrow 1 is NOT free. In-degree counts edges that exist; visit frequency counts
vertices greedy search actually pops. A high-in-degree vertex far from the entry
point, or one whose in-edges never satisfy "closer to the query", is rarely visited.
So this measures both arrows on the same four graphs rather than assuming the first.

Why visit frequency is the more interesting target: with a hard DCS budget, every
re-pop of the same hub is budget NOT spent reaching a new region. Visit skew is
therefore wasted budget directly, whereas in-degree skew only proxies for it.

The four graphs span both axes -- uniform/good (NSW), hub-heavy/weak (kNN s_0), and
the policy at its recall peak (step 700) vs well past it (step 1100), where in-degree
has run away. If arrow 2 holds, visit entropy should track recall ACROSS all four
more tightly than in-degree entropy does.
"""
import os.path as osp

import numpy as np
import torch
from scipy.stats import spearmanr

import lib

DATA_DIR = './data/SIFT100K'
NQ = 10000          # SE on recall@10 ~2e-3, small against the gaps between these graphs
NJ = 32
INIT_DEGREE = 24
EF = 32
K = 10
DCS_BUDGET = 300    # same fixed-cost regime the gate uses, so recall is comparable
FIXEDPOINT = 'runs/knn_fresh_frozen_fixedpoint/dynamic_edges.%d.pth'

GRAPHS = [
    ('NSW s_0',           'nsw', None),
    ('kNN s_0',           'knn', None),
    ('policy @700 peak',  'knn', FIXEDPOINT % 700),
    ('policy @1100 late', 'knn', FIXEDPOINT % 1100),
]


def build(graph_type, seed=1234):
    """Same construction as probe_regime.build, so numbers are comparable to it."""
    kw = dict(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        initial_vertex_id=0, graph_type=graph_type, normalization='global',
        # val_queries_size=NQ, not NQ//2: the caller slices val_queries[:NQ], so a
        # smaller pool silently halves the sample and the reported SE would be wrong.
        train_queries_size=NQ, val_queries_size=NQ,
        ground_truth_n_neighbors=100,
    )
    if graph_type == 'nsw':
        kw['edges_path'] = osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs')
    else:
        kw['edges_path'] = None
        kw['init_degree'] = INIT_DEGREE
        kw['init_seed'] = seed
    return lib.Graph(**kw)


def norm_entropy(counts, support=None):
    """H(p)/log(support) for p = counts/sum, in [0, 1]. 1 == perfectly uniform.

    The direct formalization of "uniform": maximized exactly when every bin carries
    equal mass, and unlike max/p99 it responds to the whole distribution, not the tail.

    :param support: number of bins to normalize against, defaulting to len(counts).
        For VISIT counts this must be the number of visited vertices, not the graph
        size. ANN search is supposed to touch a tiny fraction of the graph per query,
        so ~98% of vertices are never popped; normalizing by 100k makes the measure
        report that (unavoidable, desirable) sparsity instead of the concentration we
        actually care about, and it then reads ~0.62 for every graph regardless of how
        skewed the used part is. Coverage is a separate number, reported separately.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    n = len(counts) if support is None else support
    if n <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(n))


def gini(counts):
    """Inequality of the distribution, 0 == uniform, ->1 == all mass on one node.

    Reported alongside entropy because the two disagree in a useful way: entropy is
    dominated by the bulk, Gini by the tail. A graph with one 8000-degree hub and an
    otherwise flat profile moves Gini far more than entropy.
    """
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    if x.sum() <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def in_degrees(hnsw):
    """In-degree of every vertex under the current adjacency, [N] int64."""
    adj, pad = hnsw.adj, hnsw.service_labels['pad']
    flat = adj.reshape(-1)
    return np.bincount(flat[flat != pad].astype(np.int64), minlength=adj.shape[0])


def visit_counts(res, num_vertices, pad):
    """How many times greedy search POPPED each vertex, summed over queries, [N].

    Same extraction as GraphEditPPO.credit_nodes: a vertex is popped at most once per
    search, so no trajectory row holds duplicates and the batch can be bincounted whole.
    """
    traj = res['trajectories']
    hops = np.asarray(res['num_hops'])
    valid = (np.arange(traj.shape[1])[None, :] < hops[:, None]) & (traj != pad)
    visited = traj[valid].astype(np.int64)
    return np.bincount(visited, minlength=num_vertices), int(hops.sum())


def recall_at(pred, gt, k):
    """Mean recall@k and its standard error."""
    gt = gt.numpy() if torch.is_tensor(gt) else np.asarray(gt)
    hit = np.array([len(set(p[:k]) & set(g[:k])) / k for p, g in zip(pred, gt)])
    return float(hit.mean()), float(hit.std(ddof=1) / np.sqrt(len(hit)))


def analyse(label, graph_type, edges_path):
    graph = build(graph_type)
    # max_trajectory must exceed the hops the budget permits, or the walk is cut short
    # by the buffer instead of by the budget and visit counts are silently truncated.
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                             max_dcs=DCS_BUDGET)
    if edges_path is not None:
        hnsw.dynamic_edges = torch.load(edges_path, weights_only=False)
        hnsw.adj = hnsw.build_adjacency()

    queries, gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(queries)

    n = hnsw.num_vertices
    ind = in_degrees(hnsw)
    vis, total_hops = visit_counts(res, n, hnsw.service_labels['pad'])
    rec, se = recall_at(res['best_vertex_ids'], gt, K)

    # Arrow 1, over ALL vertices: does in-degree predict whether a vertex is reached?
    rho_all = spearmanr(ind, vis).statistic
    # ...and restricted to reached vertices: among those search does find, does
    # in-degree predict HOW OFTEN? The two can differ a lot -- in-degree can gate
    # reachability without governing frequency, which would make it the wrong knob
    # for evening out the budget even if arrow 1 looks strong overall.
    seen = vis > 0
    rho_seen = spearmanr(ind[seen], vis[seen]).statistic if seen.sum() > 2 else float('nan')

    # Concentration is measured over the VISITED subset; coverage is reported apart
    # from it. Mixing the two is what made every graph look identically skewed.
    vis_seen = vis[seen]
    n_seen = int(seen.sum())
    # A fixed absolute count, not a percentage of the graph: 1% of 100k is 1000, which
    # can exceed half the visited set, so "top 1%" would report coverage again rather
    # than concentration.
    hottest = np.sort(vis_seen)[::-1][:100]
    return dict(
        label=label, recall=rec, se=se,
        dcs=float(res['total_distance_computations'].mean()),
        ind_max=int(ind.max()), ind_p99=float(np.percentile(ind, 99)),
        ind_zero=float((ind == 0).mean()), ind_H=norm_entropy(ind), ind_gini=gini(ind),
        # Coverage: how much of the graph the query set reaches at all.
        coverage=float(seen.mean()), n_seen=n_seen,
        pops_per_seen=float(vis_seen.mean()) if n_seen else 0.0,
        # Hops per query. With the budget binding this is the cost actually spent, and
        # it is what makes pops/reached interpretable: coverage can be identical while
        # one graph re-pops far more, which is the budget-waste claim in raw form.
        hops_per_query=total_hops / max(1, len(queries)),
        # Concentration among used vertices.
        vis_max=int(vis.max()),
        vis_p99=float(np.percentile(vis_seen, 99)) if n_seen else 0.0,
        vis_H=norm_entropy(vis_seen, support=n_seen),
        vis_gini=gini(vis_seen),
        rho_all=float(rho_all), rho_seen=float(rho_seen),
        # Share of the whole DCS budget spent on the 100 hottest vertices -- the
        # "wasted budget" claim in its most direct form.
        top100_share=float(hottest.sum() / max(1, total_hops)),
    )


def main():
    print('== in-degree uniformity vs visit-frequency uniformity vs recall ==')
    print('NQ=%d ef=%d k=%d dcs_budget=%d (fixed cost, so recall is comparable)'
          % (NQ, EF, K, DCS_BUDGET))
    print('H = normalized entropy in [0,1], 1 == perfectly uniform. gini: 0 == uniform.')
    print()

    rows = []
    for label, gtype, path in GRAPHS:
        print('[run] %s ...' % label, flush=True)
        try:
            rows.append(analyse(label, gtype, path))
        except Exception as exc:      # a missing snapshot must not lose the other rows
            print('  [skip] %s: %s: %s' % (label, type(exc).__name__, exc))

    if not rows:
        print('no graph could be analysed')
        return

    print()
    print('--- in-degree (structure: edges that EXIST) ---')
    print('%-19s %8s %8s %8s %7s %7s' % ('graph', 'max', 'p99', 'zero%', 'H', 'gini'))
    for r in rows:
        print('%-19s %8d %8.1f %7.2f%% %7.4f %7.4f'
              % (r['label'], r['ind_max'], r['ind_p99'], 100 * r['ind_zero'],
                 r['ind_H'], r['ind_gini']))

    print()
    print('--- coverage and cost (budget=%d, so check whether it binds) ---' % DCS_BUDGET)
    print('%-19s %9s %10s %12s %9s %9s'
          % ('graph', 'reached%', 'n_reached', 'pops/reached', 'hops/q', 'DCS/q'))
    for r in rows:
        print('%-19s %8.2f%% %10d %12.2f %9.1f %9.1f'
              % (r['label'], 100 * r['coverage'], r['n_seen'], r['pops_per_seen'],
                 r['hops_per_query'], r['dcs']))

    print()
    print('--- visit concentration AMONG reached vertices ---')
    print('%-19s %8s %8s %7s %7s %10s'
          % ('graph', 'max', 'p99', 'H', 'gini', 'top100DCS'))
    for r in rows:
        print('%-19s %8d %8.1f %7.4f %7.4f %9.2f%%'
              % (r['label'], r['vis_max'], r['vis_p99'],
                 r['vis_H'], r['vis_gini'], 100 * r['top100_share']))

    print()
    print('--- arrow 1: does in-degree predict visits? (Spearman) ---')
    print('%-19s %10s %10s' % ('graph', 'rho(all)', 'rho(seen)'))
    for r in rows:
        print('%-19s %10.4f %10.4f' % (r['label'], r['rho_all'], r['rho_seen']))
    print('rho(all) high  => in-degree governs WHETHER a vertex is reached.')
    print('rho(seen) low  => among reached vertices it does NOT govern how often,')
    print('                  so evening out in-degree would not even out the budget.')

    print()
    print('--- arrow 2: which uniformity tracks recall? ---')
    print('%-19s %8s %8s %8s %8s' % ('graph', 'recall', '+-SE', 'ind_H', 'vis_H'))
    for r in rows:
        print('%-19s %8.4f %8.4f %8.4f %8.4f'
              % (r['label'], r['recall'], r['se'], r['ind_H'], r['vis_H']))

    # With only len(GRAPHS) points this is descriptive, not inferential -- it says
    # which candidate ORDERS these graphs correctly, which is what decides whether the
    # idea is worth building on, not whether the coefficient is significant.
    if len(rows) >= 3:
        rec = np.array([r['recall'] for r in rows])
        print()
        for key, name in (('ind_H', 'in-degree H'), ('vis_H', 'visit H (reached)'),
                          ('ind_gini', 'in-degree gini'), ('vis_gini', 'visit gini'),
                          ('top100_share', 'top100 DCS share'),
                          ('coverage', 'coverage'), ('ind_max', 'in-degree max'),
                          ('pops_per_seen', 'pops per reached')):
            vals = np.array([r[key] for r in rows])
            if np.ptp(vals) == 0:
                continue
            print('  rho(recall, %-19s) = %+.4f' % (name, spearmanr(rec, vals).statistic))
        print()
        print('n=%d graphs, so read these as ordering checks, not significance tests.'
              % len(rows))
        print('The idea is worth building on if visit-side uniformity orders recall')
        print('at least as well as in-degree does AND rho(seen) is high -- otherwise')
        print('in-degree is the wrong lever for the effect you are after.')


if __name__ == '__main__':
    main()
