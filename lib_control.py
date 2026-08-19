"""The learning-free control every graph-edit result must be reported against.

Rationale: the random-rewiring baseline was missing for the whole first half of this
project, and when it was finally run it BEAT the learned policy (0.521 vs 0.339).
Every "the policy improves on s_0" claim made before that was uninterpretable,
because s_0 is not the relevant comparison -- "s_0 plus one line of numpy" is.
Keeping it in a module means no future eval script can quietly omit it.

Two matching modes, and picking the wrong one is how a control gets rigged:

  dose=f      rewire f * |E| edges. Use to reproduce the tuned optimum (f=0.05).
  n_edges=m   rewire exactly m edges. Use when comparing against a policy run,
              with m = that run's own changed-edge count. The policy moves ~409k
              edges (17% of E, 3.4x the tuned optimum), so a 5% control is a
              WEAKER intervention and flatters the policy by 3.4x in edit budget.

Always report which mode was used. `rewire_matched_to` reads the count off a saved
policy graph so the caller cannot get it wrong.
"""
import numpy as np
import torch

DEFAULT_DOSE = 0.05      # tuned optimum: 0.5210 +- 0.0391 over 3 seeds
DEFAULT_SEED = 2         # middle of the three measured seeds, not the luckiest


def _compact_rows(adj, pad, n):
    """Per-row dedup + self-loop removal, emitted as tail-compact lists.

    search_hnsw.cc stops reading a neighbour row at the first -1, so an interior pad
    silently truncates the list. Building plain Python lists keeps that invariant by
    construction. Random targets can collide with an existing neighbour or the node
    itself; dropping those here stops some nodes from ending up with a lower
    effective degree for reasons unrelated to the rewiring.
    """
    edges = {}
    for i in range(n):
        row = adj[i]
        row = row[row != pad]
        seen, out = set(), []
        for t in row.tolist():
            if t != i and t not in seen:
                seen.add(t)
                out.append(int(t))
        edges[i] = out
    return edges


def rewire(adj, pad, *, dose=None, n_edges=None, seed=DEFAULT_SEED):
    """Redirect a random subset of edges to uniformly random targets.

    Out-degree is preserved (each picked slot is overwritten, not deleted), so the
    result is degree-matched to the input and the comparison isolates WHERE edges
    point rather than how many there are.

    :return: (edges dict for dynamic_edges, n_rewired)
    """
    if (dose is None) == (n_edges is None):
        raise ValueError('pass exactly one of dose= or n_edges=')
    adj = adj.copy()
    n = adj.shape[0]
    valid = np.argwhere(adj != pad)
    rng = np.random.default_rng(seed)

    k = int(round(dose * len(valid))) if dose is not None else int(n_edges)
    k = min(k, len(valid))
    pick = valid[rng.choice(len(valid), size=k, replace=False)]
    adj[pick[:, 0], pick[:, 1]] = rng.integers(0, n, size=k).astype(adj.dtype)
    return _compact_rows(adj, pad, n), k


def changed_slots(base_adj, learned_adj, pad):
    """How many directed edges a learned graph moved, as a per-node set difference.

    Counting |added| only (not added+removed) so the number is directly usable as
    n_edges for a matched control: a swap is one edge moved, not two.
    """
    total = 0
    for i in range(base_adj.shape[0]):
        b = base_adj[i]; b = set(b[b != pad].tolist())
        l = learned_adj[i]; l = set(l[l != pad].tolist())
        total += len(l - b)
    return total


def rewire_matched_to(base_adj, learned_adj, pad, *, seed=DEFAULT_SEED):
    """Control that moves exactly as many edges as `learned_adj` did.

    This is the comparison a policy run has to win to have contributed anything:
    same start, same edit budget, targets chosen by numpy instead of by a network.
    """
    m = changed_slots(base_adj, learned_adj, pad)
    edges, k = rewire(base_adj, pad, n_edges=m, seed=seed)
    return edges, k


def control_row(graph, base_adj, pad, score_fn, *, dose=DEFAULT_DOSE,
                n_edges=None, seeds=(2, 3, 5)):
    """Score the control over several seeds and return (mean, se, n_rewired).

    `score_fn(edges_dict) -> float` lets the caller reuse whatever harness it
    already has, so the control is measured under identical conditions to the arms
    it is being compared against -- the mistake that made the in-training recalls
    non-comparable in the first place.
    """
    vals, k_used = [], None
    for s in seeds:
        edges, k = rewire(base_adj, pad, dose=dose, n_edges=n_edges, seed=s)
        k_used = k
        vals.append(float(score_fn(edges)))
    v = np.asarray(vals, dtype=np.float64)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(se), k_used
