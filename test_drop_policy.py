"""Checks on the trainable drop side (GraphEditPPO.drop_mode).

The failure this guards against is silent: if the log-prob the UPDATE computes does not
reproduce the one `sample_swaps` recorded, then ratio = exp(logp - old_logp) compares two
different distributions and PPO optimizes nothing recognizable -- while every logged
metric (loss, KL, grad norm) still looks healthy. So test 1 rebuilds old_logp through the
update's own code path and demands an exact match.
"""
import numpy as np
import os.path as osp
import torch
import lib

DATA_DIR = './data/SIFT100K'
NQ, NJ = 512, 32
EF, K = 32, 10
N_NODES, N_SWAP = 256, 2


def setup(seed=1234):
    graph = lib.Graph(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        edges_path=osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs'),
        initial_vertex_id=0, graph_type='nsw', normalization='global',
        train_queries_size=NQ * 2, val_queries_size=NQ,
        ground_truth_n_neighbors=100,
    )
    hnsw = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_dcs=300,
                             max_trajectory=400)
    agent = lib.MLPLinkAgent(
        graph.vertices.shape[1], node_hidden=64, node_layers=2, pair_hidden=64,
        scorer='dot', norm='ln', dropout=0.0,
        feat_mean=graph.vertices.mean(dim=0, keepdim=True),
        feat_std=graph.vertices.std(dim=0, keepdim=True),
        logit_scale=20.0, act_bias=2.0)
    reward = lib.ProximityDCSReward(graph.vertices, k=K, max_dcs=1000,
                                    alpha=1.0, beta=0.0, dense='prox')
    return graph, hnsw, agent, reward


def make_trainer(agent, hnsw, reward, graph, drop_mode):
    return lib.GraphEditPPO(
        agent, hnsw, reward, lib.SessionBaseline(graph.vertices.size(0)),
        n_swap=N_SWAP, nodes_per_step=N_NODES, nodes_in_batch=128,
        drop_mode=drop_mode, device='cpu')


def test1_logp_roundtrip(graph, hnsw, agent, reward):
    """The update's log-prob must reproduce sample_swaps' EXACTLY.

    ratio = exp(logp - old_logp) is only a PPO ratio if both sides describe the same
    distribution over the same action. A mismatch (wrong sign on the drop logits, a
    stale adj row, a missing act_f factor) leaves every logged metric healthy while the
    objective becomes meaningless, so this is checked numerically rather than by eye.
    """
    print('== test 1: update log-prob reproduces sample_swaps log-prob ==')
    print('%-8s %-14s %-14s %-12s %s'
          % ('mode', 'max|d logp|', 'mean|d logp|', 'max|ratio-1|', 'verdict'))
    for mode in ('policy', 'argmin', 'random'):
        torch.manual_seed(0)
        np.random.seed(0)
        trainer = make_trainer(agent, hnsw, reward, graph, mode)
        state = agent.prepare_state(hnsw.graph, device='cpu')
        nodes = np.random.choice(hnsw.num_vertices, size=N_NODES, replace=False)
        action = trainer.sample_swaps(state, nodes)
        assert action is not None, 'no node had usable candidates'

        # Rebuild the log-prob the way the update does, at the SAME parameters, so any
        # difference is a bug in the mirror rather than a real parameter change.
        act = action['act']
        logits = trainer.score_candidates(state, action['node_ids'], action['cand_ids'],
                                          action['cand_mask'], grad=False)
        act_logits = agent.get_act_logits(action['node_ids'], state=state,
                                          device='cpu').float()
        import torch.nn.functional as F
        logp = F.logsigmoid(torch.where(act, act_logits, -act_logits))
        logp = logp + trainer.masked_logp(
            act, trainer.plackett_luce_logp(logits, action['chosen']))
        if mode == 'policy':
            nb = trainer.score_candidates(state, action['node_ids'], action['adj_ids'],
                                          action['adj_mask'], grad=False)
            d_logits = (-nb).masked_fill(~action['adj_mask'], float('-inf'))
            logp = logp + trainer.masked_logp(
                act, trainer.plackett_luce_logp(d_logits, action['drop_slots']))
        d = (logp - action['old_logp']).abs()
        ratio_err = (torch.exp(logp - action['old_logp']) - 1.0).abs().max().item()
        ok = d.max().item() < 1e-4
        print('%-8s %-14.3e %-14.3e %-12.3e %s'
              % (mode, d.max().item(), d.mean().item(), ratio_err,
                 'MATCH' if ok else 'MISMATCH -- ratio is not a PPO ratio'))
        if not ok:
            return False
    return True


def test2_drop_is_trainable(graph, hnsw, agent, reward):
    """Does the drop decision carry gradient, and does its distribution have spread?

    Two separate ways the drop side can be dead. 'argmin' fails the first: it puts all
    mass on one outcome, so d logp / d theta of the taken action is zero by construction
    -- which is exactly why it never trained. Near-zero row spread fails the second: a
    uniform drop softmax means the scorer cannot express which edge is least worth
    keeping, so sampling from it is indistinguishable from 'random'.
    """
    print()
    print('== test 2: is the drop choice trainable? ==')
    print('%-8s %-16s %-18s %-16s %s'
          % ('mode', 'grad(drop)', 'drop logit row std', 'drop entropy', 'verdict'))
    for mode in ('policy', 'argmin'):
        torch.manual_seed(0)
        np.random.seed(0)
        trainer = make_trainer(agent, hnsw, reward, graph, mode)
        state = agent.prepare_state(hnsw.graph, device='cpu', training=True)
        nodes = np.random.choice(hnsw.num_vertices, size=N_NODES, replace=False)
        action = trainer.sample_swaps(state, nodes)

        agent.zero_grad(set_to_none=True)
        adj_ids, adj_mask = action['adj_ids'], action['adj_mask']
        nb = trainer.score_candidates(state, action['node_ids'], adj_ids, adj_mask,
                                      grad=True)
        d_logits = (-nb).masked_fill(~adj_mask, float('-inf'))
        if mode == 'policy':
            # The quantity PPO differentiates: log-prob of the drops actually taken.
            obj = trainer.plackett_luce_logp(d_logits, action['drop_slots']).sum()
        else:
            # argmin's "log-prob" is log(1) = 0 for the taken action -- a constant, so
            # there is nothing to differentiate. Represent that literally.
            obj = torch.zeros((), requires_grad=True).sum()
        obj.backward()
        gnorm = float(torch.cat([p.grad.reshape(-1) for p in agent.parameters()
                                 if p.grad is not None]).norm().item()) \
            if any(p.grad is not None for p in agent.parameters()) else 0.0

        with torch.no_grad():
            rows = d_logits.masked_fill(~adj_mask, float('nan'))
            row_std = float(((rows - rows.nanmean(-1, keepdim=True)) ** 2)
                            .nanmean().sqrt().item())
            lp = torch.log_softmax(d_logits, dim=-1)
            ent = float(-(lp.exp() * lp.masked_fill(~adj_mask, 0.0)).sum(-1).mean().item())
            # Reference point: a uniform draw over this many valid slots.
            max_ent = float(torch.log(adj_mask.sum(-1).float()).mean().item())
        print('%-8s %-16.3e %-18.4f %-16s %s'
              % (mode, gnorm, row_std, '%.3f / %.3f max' % (ent, max_ent),
                 'trainable' if gnorm > 1e-8 else 'FROZEN: no gradient'))
    return True


def test3_recall_effect(graph, hnsw, agent, reward, n_batches=20, seed=0):
    """Does the untrained drop ranking actually hurt, on the same node batches?

    Reproduces the earlier finding that argmin is worse than uniform. Each mode gets the
    SAME node batches and the same s_0 (snapshot/restore between trials), so the
    comparison is paired and the between-batch variance cancels.
    """
    print()
    print('== test 3: recall effect of each drop mode (paired, %d batches) ==' % n_batches)
    probe, probe_gt = graph.val_queries[:NQ], graph.val_gt[:NQ]
    res = hnsw.search_deterministic(probe)
    base = float(np.mean(reward.reward_terms_batch(
        best_vertex_ids=res['best_vertex_ids'], ground_truth_ids=probe_gt,
        total_distance_computations=res['total_distance_computations'],
        trajectories=res['trajectories'], num_hops=res['num_hops'],
        queries=probe)['r_target']))
    print('  s_0 recall@%d = %.4f' % (K, base))

    node_batches = []
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        node_batches.append(rng.choice(hnsw.num_vertices, size=N_NODES, replace=False))

    print('%-8s %-16s %-14s %s' % ('mode', 'mean d_recall', 'SE', 'frac>0'))
    out = {}
    for mode in ('policy', 'argmin', 'random'):
        trainer = make_trainer(agent, hnsw, reward, graph, mode)
        deltas = []
        for i, nodes in enumerate(node_batches):
            # Same seed per (mode, batch): the addition side then draws identically
            # across modes, so the only difference is which edges were given up.
            torch.manual_seed(1000 + i)
            np.random.seed(1000 + i)
            state = agent.prepare_state(hnsw.graph, device='cpu')
            action = trainer.sample_swaps(state, nodes)
            if action is None or not len(action['swap_node_ids_np']):
                continue
            swap_nodes = action['swap_node_ids_np']
            undo = hnsw.snapshot(swap_nodes)
            hnsw.apply_swaps(swap_nodes, action['drop_nb_ids'], action['new_nb_ids'])
            r = hnsw.search_deterministic(probe)
            rt = reward.reward_terms_batch(
                best_vertex_ids=r['best_vertex_ids'], ground_truth_ids=probe_gt,
                total_distance_computations=r['total_distance_computations'],
                trajectories=r['trajectories'], num_hops=r['num_hops'],
                queries=probe)['r_target']
            hnsw.restore(undo)
            deltas.append(float(np.mean(rt)) - base)
        arr = np.array(deltas)
        out[mode] = arr
        se = arr.std() / max(np.sqrt(len(arr)), 1.0)
        print('%-8s %-16.3e %-14.3e %.2f' % (mode, arr.mean(), se, (arr > 0).mean()))

    # Paired difference is the honest comparison: same batches, so the per-batch noise
    # that dominates the absolute numbers cancels.
    n = min(len(out['argmin']), len(out['random']))
    if n > 1:
        for mode in ('policy', 'argmin'):
            d = out[mode][:n] - out['random'][:n]
            se = d.std() / np.sqrt(n)
            print('  paired %-7s - random: %+.3e +- %.3e (%.1f sigma)'
                  % (mode, d.mean(), se, abs(d.mean()) / max(se, 1e-30)))
    return True


def main():
    graph, hnsw, agent, reward = setup()
    agent.eval()   # freeze dropout so the two log-prob paths are comparable
    ok = test1_logp_roundtrip(graph, hnsw, agent, reward)
    if not ok:
        print('\nlog-prob mismatch: stopping, the rest would measure a broken objective')
        return 1
    test2_drop_is_trainable(graph, hnsw, agent, reward)
    test3_recall_effect(graph, hnsw, agent, reward)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
