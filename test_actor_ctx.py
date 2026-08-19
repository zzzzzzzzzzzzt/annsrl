"""Checks for --actor_ctx: the topology-conditioned pair scorer.

1. chunk alignment  -- score_candidates chunks over flattened (node, candidate) pairs
   and slices ctx by start//width. If that slice is off by a row, every node is scored
   with a NEIGHBOUR's topology and the bug is invisible (logits still look fine).
2. state responsiveness -- the whole point: the same pair must score differently in two
   graph states. Requires perturbing the zero-init output layer first.
3. zero-init equivalence -- an untouched ctx_head must reproduce the baseline scorer
   exactly, so an A/B starts from the same policy.
"""
import numpy as np
import torch

import lib

torch.manual_seed(0)
np.random.seed(0)

N, D, EMB, DEG, WIDTH = 64, 8, 16, 6, 5


def make_agent(actor_ctx):
    torch.manual_seed(0)
    return lib.MLPLinkAgent(D, node_hidden=EMB, node_layers=1, pair_hidden=16,
                            scorer='pair', actor_ctx=actor_ctx)


class FakeState:
    def __init__(self, z):
        self.vertices = z


def rand_adj(pad_last=True):
    adj = torch.randint(0, N, (N, DEG))
    mask = torch.ones(N, DEG, dtype=torch.bool)
    if pad_last:
        # A ragged degree distribution, including one node with no neighbours at all.
        for i in range(N):
            d = 1 + (i % DEG)
            mask[i, d:] = False
        mask[0, :] = False
    return adj, mask


# ---------------------------------------------------------------- test 1
agent = make_agent(True)
x = torch.randn(N, D)
z = agent.encode(x)
state = FakeState(z)
adj, mask = rand_adj()
ctx = agent.get_node_ctx(torch.arange(N), adj, mask, state=state)
assert ctx.shape == (N, EMB), ctx.shape
# node 0 has no neighbours -> all-zero row, not NaN
assert torch.isfinite(ctx).all()
assert float(ctx[0].abs().max()) == 0.0

# Manual reference for the neighbour mean.
for i in (1, 7, 31):
    d = int(mask[i].sum())
    ref = z[adj[i, :d]].mean(0)
    assert torch.allclose(ctx[i], ref, atol=1e-5), i

# Now the chunking. Break the zero-init so ctx actually changes the logits, then score
# with a chunk size that forces multiple passes and compare against a single pass.
torch.nn.init.normal_(agent.ctx_head[-1].weight, std=1.0)
torch.nn.init.normal_(agent.ctx_head[-1].bias, std=1.0)

cand = torch.randint(0, N, (N, WIDTH))
cmask = torch.ones(N, WIDTH, dtype=torch.bool)
cmask[:, -1] = False


class Dummy(lib.GraphEditPPO):
    """score_candidates/node_ctx only need .agent, .device, .nodes_in_batch."""
    def __init__(self, agent, nodes_in_batch):
        self.agent = agent
        self.device = 'cpu'
        self.nodes_in_batch = nodes_in_batch


whole = Dummy(agent, N).score_candidates(state, torch.arange(N), cand, cmask, ctx=ctx)
for nib in (1, 3, 7, N - 1):
    part = Dummy(agent, nib).score_candidates(state, torch.arange(N), cand, cmask, ctx=ctx)
    finite = torch.isfinite(whole)
    assert torch.allclose(whole[finite], part[finite], atol=1e-5), \
        'chunk size %d disagrees, max diff %.3e' % (
            nib, float((whole[finite] - part[finite]).abs().max()))
    assert (~torch.isfinite(part[~finite])).all()
print('test1 chunk alignment: ctx slice is row-aligned at every chunk size, '
      'empty-neighbourhood row is 0 not NaN -- OK')

# A per-node ctx must actually reach the right node: shuffling ctx must change the
# logits. (If score_candidates dropped ctx, or expanded it wrong, this passes silently.)
shuffled = ctx[torch.randperm(N)]
alt = Dummy(agent, N).score_candidates(state, torch.arange(N), cand, cmask, ctx=shuffled)
assert not torch.allclose(whole[torch.isfinite(whole)], alt[torch.isfinite(whole)]), \
    'ctx has no effect on the logits'

# ---------------------------------------------------------------- test 2
# Same pair, two graph states -> different score. This is the limitation being fixed.
pair_from = torch.tensor([5, 9, 17])
pair_to = torch.tensor([12, 3, 40])
adj_a = adj[pair_from].clone()
mask_a = mask[pair_from].clone()
adj_b = adj_a.clone()
adj_b[:, 0] = (adj_b[:, 0] + 13) % N       # rewire one edge each
mask_b = mask_a.clone()

ctx_a = agent.get_node_ctx(pair_from, adj_a, mask_a, state=state)
ctx_b = agent.get_node_ctx(pair_from, adj_b, mask_b, state=state)
la = agent.get_pair_logits(pair_from, pair_to, state=state, ctx=ctx_a)
lb = agent.get_pair_logits(pair_from, pair_to, state=state, ctx=ctx_b)
base = agent.get_pair_logits(pair_from, pair_to, state=state, ctx=None)
assert (la - lb).abs().min() > 1e-4, \
    'score did not move when the neighbourhood changed: %s' % (la - lb)
print('test2 state responsiveness: one rewired edge moves the same pair\'s logit by '
      '%s (baseline scorer would give 0)' % ['%.4f' % v for v in (la - lb).tolist()])

# ---------------------------------------------------------------- test 3
# Zero-init equivalence, and that actor_ctx=False leaves the old path untouched.
fresh = make_agent(True)
plain = make_agent(False)
zf = fresh.encode(x)
sf = FakeState(zf)
c = fresh.get_node_ctx(torch.arange(N), adj, mask, state=sf)
with_ctx = fresh.get_pair_logits(torch.arange(N), cand[:, 0], state=sf, ctx=c)
no_ctx = fresh.get_pair_logits(torch.arange(N), cand[:, 0], state=sf, ctx=None)
assert torch.allclose(with_ctx, no_ctx, atol=1e-6), \
    'zero-init ctx_head is not a no-op: max diff %.3e' % float((with_ctx - no_ctx).abs().max())

assert plain.get_node_ctx(torch.arange(N), adj, mask, state=FakeState(plain.encode(x))) is None
assert not hasattr(plain, 'ctx_head')
# ...and the algorithm-side helper reports None for an agent without the head, so
# score_candidates takes exactly the pre-change path.
assert Dummy(plain, N).node_ctx(FakeState(plain.encode(x)), torch.arange(N), adj, mask) is None
print('test3 zero-init equivalence: fresh ctx_head is an exact no-op; actor_ctx=False '
      'has no ctx_head and node_ctx() returns None -- OK')

# ---------------------------------------------------------------- test 4
# Gradients reach the new head through the chunked scorer.
def grads(agent_):
    zg = agent_.encode(x)
    sg = FakeState(zg)
    cg = Dummy(agent_, N).node_ctx(sg, torch.arange(N), adj, mask, grad=True)
    lg = Dummy(agent_, 7).score_candidates(sg, torch.arange(N), cand, cmask,
                                           grad=True, ctx=cg)
    lg[torch.isfinite(lg)].sum().backward()
    return (float(agent_.ctx_head[-1].weight.grad.abs().sum()),
            float(agent_.ctx_head[0].weight.grad.abs().sum()),
            float(agent_.node_encoder[0].weight.grad.abs().sum()))


g = make_agent(True)
out_g, hid_g, enc_g = grads(g)
# The output layer is zero-init, so d(loss)/d(hidden) = W_out = 0 on the FIRST step:
# the hidden layer legitimately gets no gradient until W_out moves off zero. Same
# behaviour as value_head; it is not a broken graph.
assert out_g > 0, 'no gradient into the ctx_head output layer'
assert hid_g == 0.0, 'expected zero hidden-layer grad at zero-init, got %.3e' % hid_g
assert enc_g > 0, 'encoder detached'

g2 = make_agent(True)
torch.nn.init.normal_(g2.ctx_head[-1].weight, std=1.0)
out2, hid2, enc2 = grads(g2)
assert hid2 > 0, 'hidden layer still has no gradient once W_out is nonzero'
print('test4 gradient flow: ctx_head output layer gets grad %.3e at zero-init (hidden '
      'layer 0 by construction), hidden layer %.3e once W_out is nonzero, encoder '
      'always fed -- OK' % (out_g, hid2))
print('all actor_ctx tests passed')
