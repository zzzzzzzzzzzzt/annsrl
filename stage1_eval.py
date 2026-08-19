"""Stage 1 acceptance test: did the learned p_u carry information, or re-derive Stage 0?

Stage 0 found per-node adaptivity null, but only for LINEAR maps of three closed-form
features ([[stage0-per-node-p-is-null]]). A learned head on raw z_u could in principle
find what those miss. This is the test that decides it, and it is deliberately harsh:
all three must pass.

  1. BEATS THE RULE       the arm must beat delta-at-65 (0.5520 +- 0.0103, C2/sigma sweep).
                          Absolute gate.

  2. BEATS ITS OWN SHUFFLE take the LEARNED p_u, permute it across nodes, rebuild the graph
                          with the same everything else, and score. The multiset of p
                          values is identical, so this cancels the "spread tax" the sigma
                          sweep measured (a p distribution with sd 15 costs -0.005, sd 25
                          costs -0.018) and isolates the assignment. Unlike Stage 0's
                          version this is FUNCTION-CLASS-FREE: it judges whatever the head
                          learned, whatever form it took.

  3. IS NOT JUST STAGE 0   correlate p_u with knn_radius / centroid / lid. High correlation
                          means the head re-derived what Stage 0 already swept, so winning
                          would only mean my slope grid was too coarse -- not a new finding.

Reported together with the spread of p_u itself: a head that collapsed to a constant has
learned "the rule was right", which is a legitimate (negative) outcome and must not be
confused with a head that varies but carries no information.
"""
import argparse
import numpy as np
import torch
import lib
from probe_visits import (NQ, NJ, EF, K, DCS_BUDGET, build,
                          visit_counts, recall_at)
from stage0_adaptive_p import node_features, zscore

DRAW_SEED0 = 1000
PERM_SEED = 4242


def load(graph, path):
    h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                          max_dcs=DCS_BUDGET)
    if path:
        h.dynamic_edges = torch.load(path, weights_only=False)
        h.adj = h.build_adjacency()
    return h


def score(h, graph, adj):
    h.adj = adj
    res = h.search_deterministic(graph.val_queries[:NQ])
    rec, se = recall_at(res['best_vertex_ids'], graph.val_gt[:NQ], K)
    vis, _ = visit_counts(res, h.num_vertices, h.service_labels['pad'])
    return rec, se, float((vis > 0).mean())


def learned_p(agent_path, graph, h, device):
    """p_u for EVERY vertex under the trained head and the final graph."""
    agent = torch.load(agent_path, weights_only=False, map_location=device)
    agent.eval()
    state = agent.prepare_state(graph, device=device, training=False)
    pad = h.service_labels['pad']
    n = h.num_vertices
    out = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n, 4096):
            e = min(s + 4096, n)
            ids = torch.arange(s, e, dtype=torch.int64, device=device)
            rows = h.adj[s:e]
            valid = rows != pad
            a_ids = torch.as_tensor(np.where(valid, rows, 0).astype(np.int64), device=device)
            a_mask = torch.as_tensor(valid, dtype=torch.bool, device=device)
            d = agent.get_len_delta(ids, a_ids, a_mask, state=state, device=device)
            out[s:e] = d.float().cpu().numpy()
    return out


def targets_for(V, slots, arms, draws, device, jitter_pct=0.5):
    """Targets for every (arm, draw) in ONE chunked argsort pass over the sources.

    The ordering depends only on the source set, not on p, so recomputing it per arm and
    per draw would multiply the cost by len(arms)*draws for no reason -- at 3 arms x 5
    draws x 3 slot seeds that is 45 passes over 120k sources instead of 3.
    """
    srcs = slots[:, 0]
    k, n = len(srcs), V.shape[0]
    ranks = {}
    for name, p_u in arms.items():
        for d in range(draws):
            rng = np.random.default_rng(DRAW_SEED0 + d)
            p = np.clip(p_u[srcs] + jitter_pct * rng.normal(size=k), 2.0, 98.0)
            ranks[(name, d)] = np.clip(np.rint(p / 100.0 * (n - 1)).astype(np.int64),
                                       1, n - 1)
    out = {key: np.empty(k, dtype=np.int64) for key in ranks}
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    for s in range(0, k, 256):
        e = min(s + 256, k)
        su = torch.as_tensor(srcs[s:e].astype(np.int64), device=device)
        order = torch.cdist(Vt[su], Vt).argsort(dim=1)
        for key, r in ranks.items():
            out[key][s:e] = order.gather(
                1, torch.as_tensor(r[s:e], device=device)[:, None])[:, 0].cpu().numpy()
        del order
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('agent', help='agent.N.pth from a --len_head run')
    ap.add_argument('--edges', default=None,
                    help='dynamic_edges snapshot the head should be evaluated ON. '
                         'Default: kNN s_0, so p_u is judged as a construction rule '
                         'rather than as a patch to one particular graph')
    ap.add_argument('--dose', type=float, default=0.05)
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--slot_seeds', type=int, nargs='*', default=[11, 23, 37])
    ap.add_argument('--p0', type=float, default=65.0)
    ap.add_argument('--span', type=float, default=25.0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    h = load(graph, args.edges)
    adj, pad = h.adj.copy(), h.service_labels['pad']

    delta = learned_p(args.agent, graph, h, device)
    p_u = args.p0 + args.span * np.tanh(delta)
    print('== Stage 1 acceptance test: %s ==' % args.agent)
    print('learned p_u: mean %.2f  sd %.2f  p5 %.2f  p95 %.2f  (p0=%.0f span=%.0f)'
          % (p_u.mean(), p_u.std(), *np.percentile(p_u, [5, 95]), args.p0, args.span))
    if p_u.std() < 0.5:
        print('  -> the head COLLAPSED to a constant: it learned that the global rule was'
              '\n     right. That is a legitimate negative, not a failure to train.')

    # Test 3 first: it costs no searches.
    rad, lid = node_features(V, device)
    cent = np.sqrt(((V.astype(np.float64) - V.astype(np.float64).mean(0)) ** 2).sum(1))
    print('\ncorrelation of p_u with the features Stage 0 already swept:')
    for name, f in (('knn_radius', rad), ('centroid', cent), ('lid', lid)):
        z = zscore(f)
        r = float(np.corrcoef(zscore(p_u), z)[0, 1]) if p_u.std() > 1e-9 else float('nan')
        print('  %-12s pearson %+.3f' % (name, r))
    print('  |r| > ~0.7 on any of these => the head re-derived Stage 0, so a win would')
    print('  only mean the slope grid was too coarse.')

    print('\n%-26s %8s %8s %9s' % ('variant', 'recall', 'SE', 'coverage'))
    agg = {'p0': [], 'learned': [], 'shuffled': []}
    for ss in args.slot_seeds:
        valid = np.argwhere(adj != pad)
        k = int(round(args.dose * len(valid)))
        slots = valid[np.random.default_rng(ss).choice(len(valid), size=k, replace=False)]
        perm = np.random.default_rng(PERM_SEED + ss).permutation(len(p_u))
        arms = {'p0': np.full(len(p_u), args.p0),
                'learned': p_u,
                'shuffled': p_u[perm]}
        tgt = targets_for(V, slots, arms, args.draws, device)
        for name in arms:
            v = []
            for d in range(args.draws):
                a = adj.copy()
                a[slots[:, 0], slots[:, 1]] = tgt[(name, d)].astype(adj.dtype)
                v.append(score(h, graph, a)[0])
            agg[name].append(float(np.mean(v)))
            print('%-26s %8.4f %8.4f %9s'
                  % ('slot %d  %s' % (ss, name), np.mean(v),
                     np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0, '-'))

    def line(tag, a, b):
        d = np.asarray(agg[a]) - np.asarray(agg[b])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
        ok = 'PASS' if d.mean() > 0 and d.mean() > 2 * se else 'FAIL'
        print('%-34s %+.4f +- %.4f  (%d/%d)  %s'
              % (tag, d.mean(), se, int((d > 0).sum()), len(d), ok))

    print()
    for name in ('p0', 'learned', 'shuffled'):
        v = np.asarray(agg[name])
        print('%-26s %8.4f +- %.4f' % (name, v.mean(), v.std(ddof=1) / np.sqrt(len(v))))
    print()
    line('TEST 1  learned vs p=65', 'learned', 'p0')
    line('TEST 2  learned vs own shuffle', 'learned', 'shuffled')
    print('\nTEST 2 is the one that matters: it holds the p multiset fixed, so it cancels')
    print('the spread tax (sd 15 costs -0.005, sd 25 costs -0.018) and judges only the')
    print('assignment. All three tests must pass for the head to have contributed.')


if __name__ == '__main__':
    main()
