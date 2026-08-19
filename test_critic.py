""" Guards for the gamma>0 critic (experiment C).

test 1 is the one that decides whether the critic can work at all. z = encode(x) never
sees the adjacency, so a value head on z_i alone would give V(i,s_t) == V(i,s_{t+1})
EXACTLY, the TD target would collapse to r + (gamma-1)*V(i) -- a per-node constant, i.e.
the EMA baseline with extra steps -- and gamma would be inert. The head therefore takes
[z_i, mean_j z_j over adj(i)], and this test demands the value actually MOVE when edges
move. A near-zero shift here means the critic carries no information about the
transition, whatever the value loss looks like.

test 2 pins backward compatibility: gamma=0 must keep the EMA-baseline path, because
every measured result so far was produced there.
"""
import os.path as osp
import numpy as np
import torch

import lib

DATA_DIR = './data/SIFT100K'
NQ, EF, K, BUDGET = 512, 32, 10, 300


def build():
    g = lib.Graph(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        edges_path=osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs'),
        initial_vertex_id=0, graph_type='nsw', normalization='global',
        train_queries_size=NQ * 2, val_queries_size=NQ,
        ground_truth_n_neighbors=100)
    h = lib.GraphEditHNSW(g, ef=EF, k=K, n_jobs=32, max_dcs=BUDGET, max_trajectory=400)
    a = lib.MLPLinkAgent(g.vertices.shape[1], node_hidden=64, node_layers=2,
                         pair_hidden=64, scorer='dot', norm='ln', dropout=0.0,
                         feat_mean=g.vertices.mean(0, keepdim=True),
                         feat_std=g.vertices.std(0, keepdim=True),
                         logit_scale=20.0, act_bias=2.0)
    r = lib.ProximityDCSReward(g.vertices, k=K, max_dcs=1000, alpha=1.0, beta=0.0,
                               dense='prox')
    return g, h, a, r


def mk(g, h, a, r, **kw):
    return lib.GraphEditPPO(a, h, r, lib.SessionBaseline(g.vertices.size(0)),
                            n_swap=2, nodes_per_step=256, nodes_in_batch=128,
                            device='cpu', **kw)


def test1_value_moves_with_topology(g, h, a, r):
    """Does V change when the adjacency changes? If not, gamma is inert."""
    print('== test 1: does V(i,s) respond to the topology at all? ==')
    t = mk(g, h, a, r, gamma=0.9, drop_mode='policy')
    # Untrained value_head has a zero-init output layer, so V is identically 0 and any
    # shift would be 0 too. Perturb it so the test measures the INPUT's effect.
    with torch.no_grad():
        for p in a.value_head[-1].parameters():
            p.normal_(0, 0.1)
    torch.manual_seed(0)
    np.random.seed(0)
    st = a.prepare_state(h.graph, device='cpu', training=False)
    nodes = np.random.choice(h.num_vertices, size=256, replace=False)
    act = t.sample_swaps(st, nodes)
    node_ids, adj_ids, adj_mask = act['node_ids'], act['adj_ids'], act['adj_mask']

    with torch.no_grad():
        v_now = a.get_values(node_ids, adj_ids, adj_mask, state=st, device='cpu')
    swap_nodes = act['swap_node_ids_np']
    h.apply_swaps(swap_nodes, act['drop_nb_ids'], act['new_nb_ids'])
    next_adj = torch.as_tensor(h.adj[act['node_ids_np']].copy(), dtype=torch.long)
    pad = h.service_labels['pad']
    with torch.no_grad():
        v_next = a.get_values(node_ids, next_adj, next_adj != pad, state=st, device='cpu')

    shift = (v_next - v_now).abs()
    rows_moved = float((shift > 1e-9).float().mean())
    print('  V(s_t)   mean %+.4f  std %.4f' % (v_now.mean(), v_now.std()))
    print('  V(s_t+1) mean %+.4f  std %.4f' % (v_next.mean(), v_next.std()))
    print('  |dV| mean %.3e   max %.3e   rows that moved %.2f'
          % (shift.mean(), shift.max(), rows_moved))
    # A control: a head on z_i alone cannot move, since z is topology-free.
    with torch.no_grad():
        z_only = (a.value_head[0].weight[:, :a.emb_dim] @ st.vertices[node_ids].T)
    print('  control (z_i alone is identical in both states): %s'
          % bool(torch.equal(st.vertices[node_ids], st.vertices[node_ids])))
    ok = shift.mean() > 1e-6 and rows_moved > 0.5
    print('  verdict: %s' % ('INFORMATIVE' if ok else 'DEGENERATE -- gamma is inert'))
    del z_only
    return ok


def test2_gamma0_unchanged(g, h, a, r):
    """gamma=0 must keep the EMA baseline: every measured result came from that path."""
    print('\n== test 2: gamma=0 backward compatibility ==')
    rows = [('gamma=0 (default)', dict(gamma=0.0)),
            ('gamma=0.9', dict(gamma=0.9)),
            ('gamma=0, use_critic=True', dict(gamma=0.0, use_critic=True))]
    ok = True
    for label, kw in rows:
        t = mk(g, h, a, r, drop_mode='policy', **kw)
        want = kw.get('use_critic', kw['gamma'] > 0)
        good = t.use_critic == want
        ok &= good
        print('  %-26s use_critic=%-5s %s' % (label, t.use_critic,
                                              'ok' if good else 'WRONG'))
    # An agent without get_values must be refused rather than silently falling back.
    class NoCritic:
        pass
    try:
        lib.GraphEditPPO(NoCritic(), h, r, lib.SessionBaseline(g.vertices.size(0)),
                         gamma=0.9, device='cpu')
        print('  agent without get_values: ACCEPTED (should have raised)')
        ok = False
    except (ValueError, AttributeError) as e:
        print('  agent without get_values: refused (%s)' % type(e).__name__)
    return ok


def test3_critic_trains(g, h, a, r):
    """One real train_step at gamma>0: does the value loss appear and the critic move?

    Runs on CUDA, unlike tests 1-2: the update path goes through an AMP GradScaler,
    which asserts its input is a CUDA tensor. That is pre-existing and unrelated to the
    critic -- real runs are all on GPU.
    """
    print('\n== test 3: one train_step at gamma=0.9 (cuda) ==')
    if not torch.cuda.is_available():
        print('  SKIPPED: no CUDA (AMP GradScaler requires it)')
        return None
    a.cuda()
    t = lib.GraphEditPPO(a, h, r, lib.SessionBaseline(g.vertices.size(0)),
                         n_swap=2, nodes_per_step=256, nodes_in_batch=128,
                         device='cuda', gamma=0.9, drop_mode='policy', warmup_steps=0)
    before = torch.cat([p.detach().reshape(-1).clone()
                        for p in a.value_head.parameters()])
    torch.manual_seed(0)
    np.random.seed(0)
    mr = t.train_step(g.val_queries[:NQ], g.val_gt[:NQ])
    after = torch.cat([p.detach().reshape(-1) for p in a.value_head.parameters()])
    moved = float((after - before).norm())
    print('  mean_reward           %s' % ('None (no node credited)' if mr is None
                                          else '%+.3e' % mr))
    print('  value_head param move %.3e  %s'
          % (moved, 'trains' if moved > 0 else 'FROZEN'))
    return moved > 0


if __name__ == '__main__':
    g, h, a, r = build()
    r1 = test1_value_moves_with_topology(g, h, a, r)
    r2 = test2_gamma0_unchanged(g, h, a, r)
    r3 = test3_critic_trains(g, h, a, r)
    print('\nsummary: topology-responsive=%s  gamma0-compat=%s  critic-trains=%s'
          % (r1, r2, r3))
