"""Plumbing checks for --indeg_ctx / --indeg_noop.

The heads are zero-init, so the property that matters is: enabling the flags must
not change any score until the weights move. If it does, the A/B against the
no-indeg control is confounded from step 0.
"""
import numpy as np
import torch
import lib

# state.vertices holds ENCODED embeddings (z = node_encoder(x)), so the fake state's
# width is node_hidden, not vertex_size. act_head is Linear(emb_dim, 1) and will
# reject anything else.
D, EMB, N = 8, 16, 200
torch.manual_seed(0)
np.random.seed(0)


def make_agent(**kw):
    return lib.MLPLinkAgent(D, node_hidden=EMB, node_layers=1, pair_hidden=EMB,
                            scorer='dot', logit_scale=10.0, act_bias=2.0, **kw)


class FakeState:
    def __init__(self, z):
        self.vertices = z


z = torch.randn(N, EMB)
state = FakeState(z)
from_idx = torch.randint(0, N, (64,))
to_idx = torch.randint(0, N, (64,))
in_deg = torch.randint(0, 500, (N,)).float()
indeg_pairs = torch.log1p(torch.stack([in_deg[from_idx], in_deg[to_idx]], -1))

# --- 1. zero-init means the flag is a no-op on the forward pass ---
base = make_agent()
withdeg = make_agent(indeg_ctx=True)
withdeg.load_state_dict(base.state_dict(), strict=False)

l_base = base.get_pair_logits(from_idx, to_idx, state=state)
l_deg = withdeg.get_pair_logits(from_idx, to_idx, state=state, indeg=indeg_pairs)
assert torch.allclose(l_base, l_deg, atol=1e-6), (l_base - l_deg).abs().max()
print('[ok] indeg_ctx zero-init: pair logits unchanged, max diff %.2e'
      % (l_base - l_deg).abs().max())

# ...and that it is not simply ignoring the input: perturb the weight.
# Column 1 is dst; drive it alone so the correlation isolates the dst term rather
# than mixing in the src contribution.
with torch.no_grad():
    withdeg.indeg_head.zero_()
    withdeg.indeg_head[1] = -0.5
l_moved = withdeg.get_pair_logits(from_idx, to_idx, state=state, indeg=indeg_pairs)
assert not torch.allclose(l_base, l_moved, atol=1e-3), 'indeg_head has no effect'
# A negative weight must DISCOUNT high-in-degree candidates, i.e. the change is
# monotone decreasing in log1p(dst_indeg). This is the sign convention the whole
# fix rests on, so check it rather than assume it.
delta = (l_moved - l_base).detach()
corr = np.corrcoef(delta.numpy(), indeg_pairs[:, 1].numpy())[0, 1]
assert corr < -0.999, 'expected negative weight to penalise popular dst, corr=%.3f' % corr
print('[ok] indeg_head sign: negative weight discounts popular dst (corr %.4f)' % corr)

# The src column must be a separate degree of freedom, not a shared one.
with torch.no_grad():
    withdeg.indeg_head.zero_()
    withdeg.indeg_head[0] = -0.5
l_src = withdeg.get_pair_logits(from_idx, to_idx, state=state, indeg=indeg_pairs)
corr_src = np.corrcoef((l_src - l_base).detach().numpy(),
                       indeg_pairs[:, 0].numpy())[0, 1]
assert corr_src < -0.999, 'src column does not track src in-degree, corr=%.3f' % corr_src
print('[ok] indeg_head src/dst are separate columns (corr %.4f)' % corr_src)

# --- 2. same for the no-op gate ---
adj_ids = torch.randint(0, N, (32, 6))
adj_mask = torch.ones(32, 6, dtype=torch.bool)
adj_mask[:, 4:] = False
node_ids = torch.arange(32)

gate_base = make_agent()
gate_deg = make_agent(indeg_noop=True)
gate_deg.load_state_dict(gate_base.state_dict(), strict=False)

a_base = gate_base.get_act_logits(node_ids, state=state)
a_deg = gate_deg.get_act_logits(node_ids, state=state, adj_ids=adj_ids,
                                adj_mask=adj_mask, in_deg=in_deg)
assert torch.allclose(a_base, a_deg, atol=1e-6), (a_base - a_deg).abs().max()
print('[ok] indeg_noop zero-init: act logits unchanged, max diff %.2e'
      % (a_base - a_deg).abs().max())

with torch.no_grad():
    gate_deg.act_indeg_w.fill_(-1.0)
a_moved = gate_deg.get_act_logits(node_ids, state=state, adj_ids=adj_ids,
                                  adj_mask=adj_mask, in_deg=in_deg)
# The mean must respect the mask: only the 4 real slots may contribute.
expect = torch.log1p(in_deg[adj_ids[:, :4]]).mean(1)
assert torch.allclose(a_moved - a_base, -expect, atol=1e-5), 'mask ignored in nb mean'
print('[ok] indeg_noop masked mean: padding slots excluded')

# A node with no valid neighbour must not divide by zero.
empty_mask = torch.zeros(32, 6, dtype=torch.bool)
a_empty = gate_deg.get_act_logits(node_ids, state=state, adj_ids=adj_ids,
                                  adj_mask=empty_mask, in_deg=in_deg)
assert torch.isfinite(a_empty).all(), 'all-padding row produced non-finite logit'
assert torch.allclose(a_empty, a_base, atol=1e-6), 'empty row should contribute 0'
print('[ok] indeg_noop empty row: finite, contributes 0')

# --- 3. gradients actually reach both new parameters ---
grad_agent = make_agent(indeg_ctx=True, indeg_noop=True)
loss = grad_agent.get_pair_logits(from_idx, to_idx, state=state,
                                  indeg=indeg_pairs).sum() \
    + grad_agent.get_act_logits(node_ids, state=state, adj_ids=adj_ids,
                                adj_mask=adj_mask, in_deg=in_deg).sum()
loss.backward()
assert grad_agent.indeg_head.grad.abs().sum() > 0, 'no grad into indeg_head'
assert grad_agent.act_indeg_w.grad.abs().sum() > 0, 'no grad into act_indeg_w'
print('[ok] gradients reach indeg_head and act_indeg_w')

# --- 4. enabling either flag must consume no RNG ---
# Otherwise the flag shifts the node sample and the Gumbel noise, and the paired A/B
# against the no-indeg control measures that shift as well as the feature itself.
def rng_after(**kw):
    torch.manual_seed(1234)
    make_agent(**kw)
    return torch.rand(4)


ref = rng_after()
for kw in ({'indeg_ctx': True}, {'indeg_noop': True},
           {'indeg_ctx': True, 'indeg_noop': True}):
    got = rng_after(**kw)
    assert torch.equal(ref, got), 'flags %s shift the RNG stream' % kw
print('[ok] indeg flags are RNG-neutral: construction draws nothing')

print('\nall indeg plumbing checks passed')
