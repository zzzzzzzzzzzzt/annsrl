"""B1-direct: does the PRETRAINED scorer, before any RL, prefer short edges?

The B2 pool pre-check ruled out the action space: on the rewired graph the pool's
median per-node max is pct 95, yet the policy adds at pct 1.61. So the ranking is
the scorer's doing. But every measurement so far was taken after 500 PPO steps, so
"the pretrained weights carry the bias" is still confounded with "PPO learned it".

This removes the confound entirely: load the pretrained checkpoint, score the 2-hop
candidate pool on the UNTOUCHED s_0, and correlate score with edge length. No
training, no graph drift -- just the initial scorer's preference.

Reports, per graph:
  - Spearman rho(score, length): negative => shorter is scored higher
  - the length percentile of the argmax candidate vs a uniform draw vs the pool max
  - the same for a RANDOM-INIT scorer, which is B1's treatment measured in one shot

If the pretrained scorer shows a strong negative rho and random-init shows ~0, the
bias is inherited from pretraining and the 3 running B1 jobs should escape it.
"""
import numpy as np
import torch
import lib
from probe_visits import NJ, EF, K, DCS_BUDGET, build

N_NODES = 2000
N_PAIRS = 200000
SEED = 7
CKPT = 'models/mlplink_SIFT100K_dot_best.pth'
REWIRED = 'runs/rewired_s0/dynamic_edges.0.pth'
LOGIT_SCALE = 20.0


def dists(V, a, b, chunk=200000):
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), chunk):
        e = min(s + chunk, len(a))
        d = V[a[s:e]].astype(np.float64) - V[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt((d * d).sum(1))
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum() + 1e-30))


def make_agent(graph, pretrained):
    """MLPLinkAgent built exactly as train_sift100k_ppo.py builds it."""
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    agent = lib.MLPLinkAgent(
        graph.vertices.shape[1],
        node_hidden=ckpt.get('node_hidden', 256),
        node_layers=ckpt.get('node_layers', 2),
        pair_hidden=ckpt.get('pair_hidden', 256),
        scorer=ckpt.get('scorer', 'dot'),
        norm=ckpt.get('norm', 'layer'),
        dropout=0.0,
        feat_mean=graph.vertices.mean(dim=0, keepdim=True),
        feat_std=graph.vertices.std(dim=0, keepdim=True),
        logit_scale=LOGIT_SCALE, learn_logit_scale=False,
        act_bias=2.0, actor_ctx=True,
    )
    if pretrained:
        agent.load_state_dict(ckpt['state_dict'], strict=False)
    return agent.eval()


def score_pool(agent, graph, hnsw, nodes, device='cpu'):
    """Pair logits for every valid 2-hop candidate, plus the pool itself."""
    cand_ids, cand_mask = hnsw.get_candidates(nodes)
    state = agent.prepare_state(graph, device=device)

    node_t = torch.as_tensor(nodes, dtype=torch.int64, device=device)
    cand_t = torch.as_tensor(cand_ids, device=device)
    mask_t = torch.as_tensor(cand_mask, dtype=torch.bool, device=device)

    adj_rows = hnsw.adj[nodes]
    adj_valid = adj_rows != hnsw.service_labels['pad']
    adj_t = torch.as_tensor(np.where(adj_valid, adj_rows, 0).astype(np.int64), device=device)
    adjm_t = torch.as_tensor(adj_valid, dtype=torch.bool, device=device)

    with torch.no_grad():
        ctx = agent.get_node_ctx(node_t, adj_t, adjm_t, state=state, device=device)
        B, C = cand_ids.shape
        frm = node_t[:, None].expand(B, C).reshape(-1)
        to = cand_t.reshape(-1)
        ctx_f = ctx.unsqueeze(1).expand(-1, C, -1).reshape(-1, ctx.shape[-1])
        out = []
        chunk = 200000
        for s in range(0, frm.numel(), chunk):
            e = min(s + chunk, frm.numel())
            out.append(agent.get_pair_logits(frm[s:e], to[s:e], state=state,
                                            device=device, ctx=ctx_f[s:e]).float())
        logits = torch.cat(out).reshape(B, C)
    return cand_ids, cand_mask, logits.cpu().numpy()


def main():
    rng = np.random.default_rng(SEED)
    print('== B1-direct: is the short-edge preference already in the pretrained weights? ==')
    graph = build('knn')
    V = graph.vertices.numpy() if torch.is_tensor(graph.vertices) else np.asarray(graph.vertices)
    n = V.shape[0]

    pa, pb = rng.integers(0, n, size=N_PAIRS), rng.integers(0, n, size=N_PAIRS)
    keep = pa != pb
    ref = np.sort(dists(V, pa[keep], pb[keep]))

    def pct(d):
        return 100.0 * np.searchsorted(ref, d) / len(ref)

    nodes = rng.choice(n, size=N_NODES, replace=False)

    for gname, gpath in (('kNN s_0', None), ('rewired s_0', REWIRED)):
        hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400,
                                 max_dcs=DCS_BUDGET)
        if gpath is not None:
            hnsw.dynamic_edges = torch.load(gpath, weights_only=False)
            hnsw.adj = hnsw.build_adjacency()

        print('\n########## %s ##########' % gname)
        for tag, pre in (('PRETRAINED', True), ('RANDOM-INIT', False)):
            torch.manual_seed(SEED)
            agent = make_agent(graph, pretrained=pre)
            cand_ids, cand_mask, logits = score_pool(agent, graph, hnsw, nodes)

            rhos, argmax_pct, unif_pct, max_pct = [], [], [], []
            for i in range(len(nodes)):
                m = cand_mask[i]
                if m.sum() < 5:
                    continue
                d = dists(V, np.full(int(m.sum()), nodes[i], dtype=np.int64),
                          cand_ids[i][m].astype(np.int64))
                q = pct(d)
                s = logits[i][m]
                rhos.append(spearman(s, q))
                argmax_pct.append(q[np.argmax(s)])
                unif_pct.append(q.mean())
                max_pct.append(q.max())

            print('  %-12s rho(score,len) = %+.4f   argmax_pct %6.3f   '
                  'uniform_pct %6.3f   pool_max_pct %6.3f'
                  % (tag, float(np.mean(rhos)), float(np.mean(argmax_pct)),
                     float(np.mean(unif_pct)), float(np.mean(max_pct))))

    print('\nrho << 0 for PRETRAINED and ~0 for RANDOM-INIT => pretraining installed the')
    print('bias, so the 3 running B1 jobs should add materially longer edges. If BOTH')
    print('are negative, the geometry of the encoder (not its training) is responsible.')


if __name__ == '__main__':
    main()
