""" Did --actor_ctx's new head actually train, and how much does it change the ranking?

A null A/B result on actor_ctx has two very different causes. ctx_head's output layer is
zero-initialized, so if the policy never moved it the architecture was never exercised at
all and the experiment measured nothing. This reads the saved agent and reports:

  1. how far ctx_head moved off zero (weight norms; the output layer starts at exactly 0),
  2. the magnitude of the ctx residual relative to the base pair score, on real 2-hop
     candidate rows -- a residual that is 1e-4 of the base score cannot reorder anything,
  3. how much the ranking actually changed: Spearman-style rank agreement and top-1
     agreement between the base score and base+ctx over each node's candidate row.

Usage: probe_ctx_head.py <run_name> [step]
"""
import glob
import os.path as osp
import re
import sys

import numpy as np
import torch

import lib

DATA_DIR = './data/SIFT100K'
NODES = 512


def latest_agent(run, step=None):
    if step is not None:
        return 'runs/%s/agent.%s.pth' % (run, step), int(step)
    paths = glob.glob(osp.join('runs', run, 'agent.*.pth'))
    if not paths:
        raise SystemExit('no agent checkpoint in runs/%s' % run)
    best, best_step = None, -1
    for p in paths:
        m = re.search(r'agent\.(\d+)\.pth$', p)
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best, best_step


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run = sys.argv[1]
    step = sys.argv[2] if len(sys.argv) > 2 else None
    path, resolved = latest_agent(run, step)
    agent = torch.load(path, weights_only=False)
    print('loaded %s (step %d)' % (path, resolved))

    if not getattr(agent, 'actor_ctx', False):
        print('this agent has no ctx_head (actor_ctx was off) -- nothing to probe')
        return 0

    print('\n1) how far ctx_head moved off its init')
    for name, p in agent.ctx_head.named_parameters():
        print('   ctx_head.%-8s norm %.4e  max|.| %.4e' % (name, p.norm().item(),
                                                           p.abs().max().item()))
    print('   (the output layer, index 2, starts at EXACTLY 0; a norm still ~0 there'
          ' means\n    the head never trained and the A/B measured the baseline scorer)')

    # Real candidate rows, so the numbers describe the distribution the policy actually
    # samples from rather than random pairs. Params must match the sweep's: the NSW edge
    # file is M=12/efC=300 and lib.Graph normalizes, so a mismatch here would feed the
    # agent embeddings in a different scale than it trained on.
    graph = lib.Graph(
        vertices_path=osp.join(DATA_DIR, 'sift_base.fvecs'),
        edges_path=osp.join(DATA_DIR, 'sift_nsw_M12_efC300.ivecs'),
        train_queries_path=osp.join(DATA_DIR, 'sift_learn_1m.fvecs'),
        test_queries_path=osp.join(DATA_DIR, 'sift_query.fvecs'),
        train_gt_path=osp.join(DATA_DIR, 'train_1m_gt.ivecs'),
        test_gt_path=osp.join(DATA_DIR, 'test_gt.ivecs'),
        train_queries_size=200000 + 20000, val_queries_size=20000,
        ground_truth_n_neighbors=100, graph_type='nsw', initial_vertex_id=0,
    )
    hnsw = lib.GraphEditHNSW(graph, ef=32, k=10, n_jobs=8, max_dcs=300)
    dyn = 'runs/%s/dynamic_edges.%d.pth' % (run, resolved)
    if osp.exists(dyn):
        hnsw.dynamic_edges = torch.load(dyn, weights_only=False)
        print('\n   using the run\'s own edited graph (%s)' % osp.basename(dyn))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = agent.to(device=device)
    rs = np.random.RandomState(0)
    nodes_np = rs.choice(hnsw.num_vertices, size=NODES, replace=False)
    cand_np, cmask_np = hnsw.get_candidates(nodes_np)
    keep = cmask_np.sum(-1) >= 2
    nodes_np, cand_np, cmask_np = nodes_np[keep], cand_np[keep], cmask_np[keep]

    nodes = torch.as_tensor(nodes_np, dtype=torch.int64, device=device)
    cand = torch.as_tensor(cand_np, dtype=torch.int64, device=device)
    cmask = torch.as_tensor(cmask_np, dtype=torch.bool, device=device)
    adj_rows = hnsw.adj[nodes_np]
    adj_valid = adj_rows != hnsw.service_labels['pad']
    adj_ids = torch.as_tensor(np.where(adj_valid, adj_rows, 0).astype(np.int64), device=device)
    adj_mask = torch.as_tensor(adj_valid, dtype=torch.bool, device=device)

    state = agent.prepare_state(graph, device=device)
    with torch.no_grad():
        ctx = agent.get_node_ctx(nodes, adj_ids, adj_mask, state=state, device=device)
        width = cand.shape[1]
        flat_from = nodes[:, None].expand(-1, width).reshape(-1)
        flat_to = cand.reshape(-1)
        flat_ctx = ctx.unsqueeze(1).expand(-1, width, -1).reshape(-1, ctx.shape[-1])
        base = agent.get_pair_logits(flat_from, flat_to, state=state, device=device,
                                     ctx=None).reshape(cand.shape).float()
        full = agent.get_pair_logits(flat_from, flat_to, state=state, device=device,
                                     ctx=flat_ctx).reshape(cand.shape).float()
    resid = (full - base)
    valid = cmask

    print('\n2) ctx residual vs base score, over %d nodes x %d candidates (%d valid pairs)'
          % (cand.shape[0], width, int(valid.sum())))
    b, r = base[valid], resid[valid]
    print('   base  score      mean %+.4f  std %.4f' % (b.mean(), b.std()))
    print('   ctx   residual   mean %+.4f  std %.4f  max|.| %.4f'
          % (r.mean(), r.std(), r.abs().max()))
    print('   residual std / base within-row std   %.4f' % _row_std_ratio(base, resid, valid))
    print('   (a ratio near 0 means the residual cannot reorder a candidate row; the'
          ' softmax\n    only sees within-row differences, so the WITHIN-ROW spread is'
          ' the comparison)')

    print('\n3) did the ranking change?')
    bm = base.masked_fill(~valid, float('-inf'))
    fm = full.masked_fill(~valid, float('-inf'))
    top1 = (bm.argmax(-1) == fm.argmax(-1)).float().mean()
    print('   top-1 candidate unchanged in %.1f%% of rows' % (100.0 * top1))
    # Fraction of valid within-row pairs whose relative order flipped.
    flips, total = 0, 0
    for i in range(bm.shape[0]):
        m = valid[i]
        if int(m.sum()) < 2:
            continue
        bi, fi = base[i][m], full[i][m]
        db = bi[:, None] - bi[None, :]
        df = fi[:, None] - fi[None, :]
        iu = torch.triu(torch.ones_like(db, dtype=torch.bool), diagonal=1)
        flips += int(((db[iu] > 0) != (df[iu] > 0)).sum())
        total += int(iu.sum())
    if total:
        print('   %.2f%% of within-row candidate pairs changed order (%d/%d)'
              % (100.0 * flips / total, flips, total))
    return 0


def _row_std_ratio(base, resid, valid):
    """std of the residual against the base score's WITHIN-ROW std, both masked."""
    def row_std(t):
        m = t.masked_fill(~valid, float('nan'))
        c = m - m.nanmean(-1, keepdim=True)
        return float((c ** 2).nanmean().sqrt())
    denom = row_std(base)
    return row_std(resid) / denom if denom > 0 else float('nan')


if __name__ == '__main__':
    sys.exit(main())
