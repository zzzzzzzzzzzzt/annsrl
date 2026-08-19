"""C1b: is the policy's +0.014 per-node target matching, or cross-node coordination?

C1 destroyed both at once ([[c1-policy-target-choice-carries-real-information]]): after an
independent same-length reshuffle, each source picks its own target, so the ~10 shared
targets stop being shared AND each source loses the specific node it chose. This splits
the two.

A naive "permute targets among sources" does NOT work as a length-preserving test: a node
that sits at percentile 93 from u_1 sits at a random-pair distance from u_2, so permuting
silently converts the test into the uniform-target one. The two variants below each hold
one factor exactly fixed instead, and are read against DIFFERENT references:

  perm      targets permuted within the hub SET. In-degree per hub is exactly preserved
            and the target set is unchanged; only WHICH source uses WHICH hub changes.
            Read against `intact` -> isolates per-node matching.

  relabel   each of the 10 hubs is mapped to one random node, consistently. In-degree and
            the whole share-a-target pattern are preserved EXACTLY; only the identity of
            the hub nodes changes, and lengths go to ~percentile 100.
            Read against `uniform` (same length, no sharing) -> isolates coordination.

`c1 reshuffle` and `uniform` are re-measured here on the hub edge set so all five rows are
on one edge selection. Everything is averaged over paired draws
([[ablation-needs-draw-averaging]]).
"""
import argparse
import numpy as np
import torch
import lib
from c1_reshuffle import (load, measure, edge_lengths, rank_neighbourhoods,
                          apply_reshuffle, apply_uniform, run_draws, DRAW_SEED0)
from probe_visits import build, in_degrees

BAND = 100
N_HUB = 10


def apply_perm(adj, idx, rng):
    """Shuffle the target column in place: same multiset, so in-degree is exact."""
    a = adj.copy()
    t = adj[idx[:, 0], idx[:, 1]].copy()
    rng.shuffle(t)
    a[idx[:, 0], idx[:, 1]] = t
    return a


def apply_relabel(adj, idx, hubs, rng):
    """Map each hub to one random node. Sharing pattern and in-degree exactly preserved."""
    a = adj.copy()
    new = rng.choice(adj.shape[0], size=len(hubs), replace=False)
    lut = {int(h): int(n) for h, n in zip(hubs, new)}
    t = adj[idx[:, 0], idx[:, 1]]
    a[idx[:, 0], idx[:, 1]] = np.array([lut[int(x)] for x in t], dtype=adj.dtype)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--band', type=int, default=BAND)
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--n_hub', type=int, default=N_HUB)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)

    print('== C1b decomposition on the top-%d hub in-edges (%d draws, band +-%d) =='
          % (args.n_hub, args.draws, args.band))
    print('%-26s %8s %8s %9s %9s %9s'
          % ('variant', 'recall', 'SE(draw)', 'coverage', 'sel_len', 'ind_max'))
    acc = {k: [] for k in ('c1', 'perm', 'relabel', 'uniform')}

    for p in args.paths:
        h = load(graph, p)
        adj, pad = h.adj.copy(), h.service_labels['pad']
        ind = in_degrees(h)
        hubs = np.argsort(ind)[::-1][:args.n_hub].astype(adj.dtype)
        idx = np.argwhere(np.isin(adj, hubs) & (adj != pad))

        def sel_len(a):
            return edge_lengths(V, idx, a).mean()

        def ind_max_of(a):
            return int(np.bincount(a[a != pad].ravel().astype(np.int64),
                                   minlength=adj.shape[0]).max())

        name = p.split('/')[-2]
        r, c = measure(h, graph, adj.copy())
        print('%-26s %8.4f %8s %9.4f %9.4f %9d'
              % (name + ' intact', r, '-', c, sel_len(adj), int(ind.max())))

        cand = rank_neighbourhoods(V, idx[:, 0], adj[idx[:, 0], idx[:, 1]], args.band, device)
        rows = [
            ('c1', '  c1 reshuffle (len fixed)', lambda g: apply_reshuffle(adj, idx, cand, g)),
            ('perm', '  perm within hub set', lambda g: apply_perm(adj, idx, g)),
            ('relabel', '  relabel hubs -> random', lambda g: apply_relabel(adj, idx, hubs, g)),
            ('uniform', '  uniform targets', lambda g: apply_uniform(adj, idx, g)),
        ]
        got = {}
        for key, label, fn in rows:
            m, se, cov = run_draws(h, graph, fn, args.draws)
            a1 = fn(np.random.default_rng(DRAW_SEED0))
            got[key] = m
            print('%-26s %8.4f %8.4f %9.4f %9.4f %9d'
                  % (label, m, se, cov, sel_len(a1), ind_max_of(a1)))
        for k in acc:
            acc[k].append(got[k] - r)
        print('%-26s n_edges=%d' % ('', len(idx)))

    def line(tag, v, ref=0.0):
        v = np.asarray(v) - ref
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
        print('%-34s %+.4f +- %.4f  (%d/%d positive)'
              % (tag, v.mean(), se, int((v > 0).sum()), len(v)))

    print()
    for k in ('c1', 'perm', 'relabel', 'uniform'):
        line('%s vs intact' % k, acc[k])
    print()
    line('perm    - c1       (sharing kept)', np.array(acc['perm']) - np.array(acc['c1']))
    line('relabel - uniform  (sharing kept)', np.array(acc['relabel']) - np.array(acc['uniform']))
    print("""
Read:
  perm ~ intact                -> per-node matching is worthless; the value is the hub SET
  perm ~ c1 (both << intact)   -> per-node matching IS the value
  relabel > uniform            -> coordination (sharing a target) is worth something on its own
  relabel ~ uniform            -> which nodes are hubs, and sharing, are both irrelevant""")


if __name__ == '__main__':
    main()
