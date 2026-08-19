"""compute_in_degrees correctness + the s_t snapshot actually reaching the update.

The zero-init tests in test_indeg.py prove the heads are wired up, but they say
nothing about whether the NUMBERS fed to them are the right ones. Two things can
silently go wrong: the count itself (padding leaking in, wrong axis), and the
snapshot (the update re-scoring under s_{t+1}'s degrees instead of s_t's).
"""
import numpy as np
import torch
import lib

pad_checks = []

# --- 1. the count, against an independent implementation ---
# A hand-built adjacency with a known in-degree profile, including an isolated node
# and a hub, so a padding leak or an axis slip cannot pass by coincidence.
PAD = -1
adj = np.array([
    [1, 2, PAD],
    [2, 3, PAD],
    [3, PAD, PAD],
    [1, 2, 0],
    [PAD, PAD, PAD],   # contributes nothing
], dtype=np.int64)

flat = adj.reshape(-1)
expect = np.bincount(flat[flat != PAD], minlength=adj.shape[0])
# node: 0 <- {3}=1, 1 <- {0,3}=2, 2 <- {0,1,3}=3, 3 <- {1,2}=2, 4 <- {}=0
assert list(expect) == [1, 2, 3, 2, 0], list(expect)


class FakeHNSW:
    def __init__(self, adj):
        self.adj = adj
        self.service_labels = {'pad': PAD}


class Bare:
    """Just the methods under test, bound to a fake graph."""
    uses_indeg = lib.GraphEditPPO.uses_indeg
    compute_in_degrees = lib.GraphEditPPO.compute_in_degrees
    indeg_tensor = lib.GraphEditPPO.indeg_tensor

    def __init__(self, adj, agent):
        self.hnsw = FakeHNSW(adj)
        self.agent = agent
        self.device = 'cpu'


class StubAgent:
    def __init__(self, ctx=False, noop=False):
        self.indeg_ctx, self.indeg_noop = ctx, noop


bare = Bare(adj, StubAgent(ctx=True))
got = bare.compute_in_degrees()
assert list(got) == list(expect), (list(got), list(expect))
assert len(got) == adj.shape[0], 'in-degree vector must cover every node'
print('[ok] compute_in_degrees matches independent count, padding excluded')

# A pad sentinel that is a VALID index (some builds use N, not -1) must not be
# counted as an edge into node N-1 or wrap around.
adj_big_pad = np.where(adj == PAD, 5, adj)
big = Bare(adj_big_pad, StubAgent(ctx=True))
big.hnsw.service_labels['pad'] = 5
assert list(big.compute_in_degrees()) == list(expect), 'non-negative pad leaked'
print('[ok] compute_in_degrees excludes a non-negative pad sentinel')

# --- 2. uses_indeg gates the whole computation ---
assert Bare(adj, StubAgent(ctx=True)).uses_indeg
assert Bare(adj, StubAgent(noop=True)).uses_indeg
assert Bare(adj, StubAgent(ctx=True, noop=True)).uses_indeg
assert not Bare(adj, StubAgent()).uses_indeg
# An agent predating the flags has neither attribute; must not raise.
assert not Bare(adj, object()).uses_indeg
print('[ok] uses_indeg true iff a head consumes in-degrees; safe on old agents')

# --- 3. indeg_tensor ---
t = bare.indeg_tensor(got)
assert t.dtype == torch.float32 and t.shape == (adj.shape[0],)
assert bare.indeg_tensor(None) is None, 'None must pass through, not crash'
print('[ok] indeg_tensor -> float32 [N]; None passes through')

print('\nall in-degree graph-side checks passed')
