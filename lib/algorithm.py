"""
Algorithms are training methods that
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
from .utils import compute_flat_grad, get_flat_params_from, set_flat_params_to
from torch.utils.tensorboard import SummaryWriter
from pandas import DataFrame


class BaseAlgorithm:
    """ A trainer class that updates agent parameters and draws logs """
    def __init__(self, agent, hnsw, reward, baseline=None, writer=None, device='cuda',
                 warmup_steps=0):
        """
        :type agent: lib.agent.BaseAgent
        :type hnsw: lib.hnsw.HNSW
        :type reward: function **session_records: vector of rewards for each action in session
        :type baseline: lib.baseline.BaselineInterface
        :param warmup_steps: number of steps to collect experience and update the baseline
            without performing any policy gradient update. Resolves the cold-start problem
            where the baseline starts at all-zeros and produces meaningless advantages.
        """
        self.hnsw = hnsw
        self.agent = agent
        self.reward = reward
        self.device = device
        self.baseline = baseline
        self.writer = writer or SummaryWriter()
        self.step = 0
        self.warmup_steps = warmup_steps

        self.tensor_dtypes = {
            'from_vertex_ids': torch.int64,
            'to_vertex_ids': torch.int64,
            'actions': torch.int64,
            'session_index': torch.int64,
            'rewards': torch.float32,
        }

    def get_session_batch(self, queries, ground_truth_ids, summarize=True,
                          sample_device=None, is_evaluate=False, **kwargs):
        """
        plays one session per query and ground_truth_id
        :param queries: vectors to find nearest neighbor to, [batch_size, vertex_size]
        :param ground_truth_ids: indices of actual nearest neighbors, [batch_size]
        :returns: a dict with session details
         - state: output of agent.prepare_state, common for all sessions
         - from_ix, to_ix: indices of vectors from which edge was predicted, each int32 [batch_size]
         - actions: whether edge was allowed(1) or not(0), int32 [batch_size]
         - rewards: individual rewards for each action, float32 [batch_size]
         - session_index: session index for each sample in batch, int32[batch_size]
        """
        sample_device = sample_device or self.device
        state_device = 'cpu' if is_evaluate else sample_device
        self.agent.to(device=sample_device)
        state = self.agent.prepare_state(self.hnsw.graph, device=state_device, **kwargs)
        session_records = self.hnsw.record_sessions(self.agent, queries, state=state,
                                                    **dict(kwargs, is_evaluate=is_evaluate, device=sample_device))
        self.agent.to(device=self.device)

        tensors = {name: [] for name in self.tensor_dtypes.keys()}
        for i, (query, gt, rec) in enumerate(zip(queries, ground_truth_ids, session_records)):
            rec['query'] = query
            rec['ground_truth_id'] = gt
            rec['rewards'] = self.reward(**rec)
            rec['session_index'] = [i] * len(rec['rewards'])

            if rec['actions'][0] != self.hnsw.service_labels['no_actions']:
                for col in tensors.keys():
                    tensors[col].extend(rec[col])

        tensors = {name: torch.tensor(value, dtype=self.tensor_dtypes[name], device=self.device)
                   for name, value in tensors.items()}

        results = dict(tensors, state=state)
        if summarize:
            results['summary'] = self.summarize(session_records, **kwargs)
        return results

    def summarize(self, session_records, prefix='train', write_logs=True, **kwargs):
        """ logs all information about session records """
        counters = {
            prefix + '/mean_reward': np.mean([np.mean(rec['rewards']) for rec in session_records]),
            prefix + '/recall@1': np.mean([rec['best_vertex_id'] == rec['ground_truth_id'][0]
                                           for rec in session_records]),
            prefix + '/distance_computations': np.mean([rec['total_distance_computations']
                                                        for rec in session_records]),
            prefix + '/num_hops': np.mean([rec['num_hops'] for rec in session_records]),
            prefix + '/recall@1_per_distance_computation' : \
                np.mean([int(rec['best_vertex_id'] == rec['ground_truth_id'][0]) / rec['total_distance_computations']
                         for rec in session_records])
        }
        k = len(session_records[0]['best_vertex_ids'])
        n_gt = len(session_records[0]['ground_truth_id'])
      
        if k > 1:
            assert k <= n_gt    
            recall_all = np.mean([float(len(set(rec['best_vertex_ids']) & set(rec['ground_truth_id'][:k].tolist()))) / k 
                                  for rec in session_records])
            counters[prefix + '/recall@%i' % k] = recall_all
            counters[prefix + '/recall@%i_per_distance_computation' % k] = \
                np.mean([float(len(set(rec['best_vertex_ids']) & set(rec['ground_truth_id'][:k].tolist()))) / (rec['total_distance_computations'] * k)
                         for rec in session_records])
        if write_logs:
            for key, value in counters.items():
                self.writer.add_scalar(key, value, global_step=self.step)
        return counters

    def train_step(self, batch_queries, batch_ground_truth_ids, **kwargs):
        """ samples sessions and performs update step
        :param batch_queries: vectors to find nearest neighbor to, [batch_size, vertex_size]
        :param batch_ground_truth_ids: indices of actual nearest neighbors, [batch_size]
        :returns: mean reward
        """
        batch_records = self.get_session_batch(batch_queries, batch_ground_truth_ids, **kwargs)

        # Warmup phase: update baseline only, skip policy gradient
        if self.warmup_steps > 0 and self.step < self.warmup_steps:
            if self.baseline is not None:
                self.baseline.update(
                    rewards=batch_records['rewards'],
                    session_index=batch_records['session_index'],
                    device=self.device,
                    **kwargs,
                )
            # self.writer.add_scalar('train/warmup_step', self.step, global_step=self.step)
            mean_reward = batch_records['rewards'].mean().item()
            self.step += 1
            return mean_reward

        mean_reward = self.train_on_batch(**batch_records, **kwargs)
        self.step += 1
        return mean_reward

    def train_on_batch(self, **rec_kwargs):
        """ updates agent parameters on sampled sessions"""
        raise NotImplementedError()

    def evaluate(self, batch_queries, batch_ground_truth_ids, prefix='dev', **kwargs):
        """ Compute metrics and write logs for current agent state
        :param batch_queries: vectors to find nearest neighbor to, [batch_size, vertex_size]
        :param batch_ground_truth_ids: indices of actual nearest neighbors, [batch_size]
        :param prefix: prefix for metric names
        """
        summary = self.get_session_batch(batch_queries, batch_ground_truth_ids, greedy=True, sample_device='cpu', 
                                   summarize=True, prefix=prefix, is_evaluate=True, **kwargs)['summary']
        mean_reward = np.mean(summary[prefix + '/mean_reward'])
        return mean_reward

    @staticmethod
    def aggregate_samples(from_vertex_ids, to_vertex_ids, actions, advantages, device='cuda'):
        """ Merge the same samples """
        df = DataFrame({'from_vertex_ids': from_vertex_ids.cpu(), 'to_vertex_ids': to_vertex_ids.cpu(),
                        'actions': actions.cpu(), 'advantages': advantages.cpu(),
                        'freqs': torch.ones(len(actions)).type(torch.float32)})
        df = df.groupby(['from_vertex_ids', 'to_vertex_ids', 'actions'], sort=False).sum().reset_index()

        # Use of Gumbel Max Trick for unbiased Fvp estimate when train on subset of samples
        df['probs'] = np.log(df['freqs']) + np.random.gumbel(0, 1, len(df))
        df = df.sort_values(by=['probs'], ascending=False)
        del df['probs']
        return [torch.tensor(df[column].values, device=device) for column in df.columns]


class TRPO(BaseAlgorithm):
    """ Trust Region Policy Optimization, see https://arxiv.org/pdf/1502.05477.pdf.
        Deprecated, use EfficientTRPO instead
    """
    def __init__(self, agent, hnsw, reward, baseline, max_kl=0.01, damping=0.1, entropy_reg=0.0, **kwargs):
        super().__init__(agent, hnsw, reward, baseline, **kwargs)
        self.max_kl = max_kl
        self.damping = damping
        self.entropy_reg = entropy_reg

    def linesearch(self, f, x, fullstep):
        max_backtracks = 10
        loss, _, _ = f(x)
        powers = torch.arange(max_backtracks, dtype=torch.float32).cuda()
        for stepfrac in .5 ** powers:
            xnew = x + stepfrac * fullstep
            new_loss, kl, _ = f(xnew)
            actual_improve = new_loss - loss
            if kl.item() <= self.max_kl and actual_improve.item() < 0:
                x = xnew
                loss = new_loss
        return x

    def conjugate_gradient(self, f_Ax, b, cg_iters=10, residual_tol=1e-10):
        p = b.clone()
        r = b.clone()
        x = torch.zeros(b.size()).cuda()
        rdotr = torch.sum(r * r)
        for i in range(cg_iters):
            z = f_Ax(p)
            v = rdotr / (torch.sum(p * z) + 1e-8)
            x += v * p
            r -= v * z
            newrdotr = torch.sum(r * r)
            mu = newrdotr / (rdotr + 1e-8)
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x

    def train_on_batch(self, state, from_vertex_ids, to_vertex_ids, actions, rewards, session_index, **kwargs):
        baseline = self.baseline.get(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                     rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        advantage = rewards - baseline

        # Update baseline for the next iteration
        mean_reward = self.baseline.update(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                           rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        # if not train mode, exit
        if self.max_kl == 0.0:
            return mean_reward

        # Aggregate samples
        from_vertex_ids, to_vertex_ids, actions, advantage, freqs = \
            self.aggregate_samples(from_vertex_ids, to_vertex_ids, actions, advantage, device=self.device)

        state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)
        logp = self.agent.get_edge_logp(from_vertex_ids, to_vertex_ids, state=state, device=self.device)
        logp_action = torch.gather(logp, dim=-1, index=actions[:, None])[:, 0]

        old_logp = logp.detach()
        old_logp_action = logp_action.detach()

        ratio = torch.exp(logp_action - old_logp_action)  # pi(a|s) / pi_old(a|s)
        loss = -(ratio * advantage).sum() / freqs.sum()

        grads = torch.autograd.grad(loss, self.agent.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach_()

        def Fvp(v):
            # Here we compute Fx to do solve Fx = g using conjugate gradients
            state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)
            logp = self.agent.get_edge_logp(from_vertex_ids, to_vertex_ids, state=state, device=self.device)
            probs = logp.exp()
            kl = (freqs * (probs * (logp - old_logp)).sum(-1)).sum() / freqs.sum()
            assert (kl > -0.0001).all() and (kl < 10000).all()

            grads = torch.autograd.grad(kl, self.agent.parameters(), create_graph=True)

            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.agent.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach_()
            return flat_grad_grad_kl + v * self.damping

        stepdir = self.conjugate_gradient(Fvp, -loss_grad, 10)

        # Here we compute the initial vector to do linear search
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]

        # Here we get the start point
        prev_params = get_flat_params_from(self.agent)

        @torch.no_grad()
        def get_loss_kl_ent(params):
            # Helper for linear search
            set_flat_params_to(self.agent, params)
            state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)

            logp = self.agent.get_edge_logp(from_vertex_ids, to_vertex_ids, state=state, device=self.device)
            logp_action = torch.gather(logp, dim=-1, index=actions[:, None])[:, 0]
            probs = torch.exp(logp)
            kl = (freqs * (probs * (logp - old_logp)).sum(-1)).sum() / freqs.sum()

            ratio = torch.exp(logp_action - old_logp_action)  # pi(a|s) / pi_old(a|s)
            loss = -(ratio * advantage).sum() / freqs.sum()
            ent = (freqs * (-probs * logp).sum(-1)).sum() / freqs.sum()
            assert (kl > -0.0001).all() and (kl < 10000).all()
            return [loss, kl, ent]

        # Here we find our new parameters
        new_params = self.linesearch(get_loss_kl_ent, prev_params, fullstep)
        del state  # state becomes obsolete at this point

        # And we set it to our network
        set_flat_params_to(self.agent, new_params)

        # Summarize
        loss, kl, ent = get_loss_kl_ent(new_params)
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/advantage', advantage.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/entropy', ent.item(), global_step=self.step)
        self.writer.add_scalar('train/kl', kl.item(), global_step=self.step)
        self.writer.add_scalar('train/loss', loss.item(), global_step=self.step)
        return mean_reward


class PPO(BaseAlgorithm):
    """ Proximal Policy Optimization.
        Replaces TRPO's conjugate gradient / Fisher matrix with a clipped surrogate objective,
        enabling multiple mini-batch gradient steps per session batch.
    """

    def __init__(self, agent, hnsw, reward, baseline,
                 optimizer=None, lr=3e-4,
                 clip_eps=0.2, ppo_epochs=4, samples_in_batch=4096,
                 entropy_reg=0.01, **kwargs):
        """
        :param clip_eps: PPO clipping epsilon
        :param ppo_epochs: number of gradient update passes over each session batch
        :param samples_in_batch: mini-batch size for each gradient step
        :param entropy_reg: entropy bonus coefficient
        :param optimizer: optional pre-built optimizer; if None, Adam with lr is used
        """
        super().__init__(agent, hnsw, reward, baseline, **kwargs)
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.samples_in_batch = samples_in_batch
        self.entropy_reg = entropy_reg
        self.optimizer = optimizer or torch.optim.Adam(agent.parameters(), lr=lr)

    def train_on_batch(self, state, from_vertex_ids, to_vertex_ids, actions, rewards, session_index, **kwargs):
        baseline = self.baseline.get(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                     rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        mean_reward = self.baseline.update(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                           rewards=rewards, session_index=session_index, device=self.device, **kwargs)

        advantage = (rewards - baseline).detach()

        # Compute old log-probs once (no grad)
        with torch.no_grad():
            old_logp = self.agent.get_edge_logp(from_vertex_ids, to_vertex_ids,
                                                state=state, device=self.device)
            old_logp_action = torch.gather(old_logp, dim=-1, index=actions[:, None])[:, 0]

        n = len(from_vertex_ids)
        total_loss = total_ent = total_kl = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.samples_in_batch):
                idx = perm[start:start + self.samples_in_batch]

                logp = self.agent.get_edge_logp(from_vertex_ids[idx], to_vertex_ids[idx],
                                                state=state, device=self.device)
                logp_action = torch.gather(logp, dim=-1, index=actions[idx, None])[:, 0]

                ratio = torch.exp(logp_action - old_logp_action[idx])
                adv = advantage[idx]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                ent = (-logp.exp() * logp).sum(-1).mean()
                loss = policy_loss - self.entropy_reg * ent

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    kl = (old_logp[idx].exp() * (old_logp[idx] - logp.detach())).sum(-1).mean()

                total_loss += loss.item()
                total_ent += ent.item()
                total_kl += kl.item()
                n_updates += 1

        self.writer.add_scalar('train/loss', total_loss / n_updates, global_step=self.step)
        self.writer.add_scalar('train/entropy', total_ent / n_updates, global_step=self.step)
        self.writer.add_scalar('train/kl', total_kl / n_updates, global_step=self.step)
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/advantage', advantage.mean().item(), global_step=self.step)
        return mean_reward

class OptimizedPPO(BaseAlgorithm):
    """ 
    Optimized Proximal Policy Optimization.
    引入了样本聚合去重，并修复了全量前向传播导致的 OOM 问题。
    """

    def __init__(self, agent, hnsw, reward, baseline,
                 optimizer=None, lr=3e-4,
                 clip_eps=0.2, ppo_epochs=4, samples_in_batch=40000,
                 entropy_reg=0.01, target_kl=0.015, **kwargs):
        super().__init__(agent, hnsw, reward, baseline, **kwargs)
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        # 建议稍微调小一点，比如 40000，防止反向传播时的隐层矩阵过大 OOM
        self.samples_in_batch = samples_in_batch 
        self.entropy_reg = entropy_reg
        self.target_kl = target_kl
        self.optimizer = optimizer or torch.optim.Adam(agent.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()

    def train_on_batch(self, state, from_vertex_ids, to_vertex_ids, actions, rewards, session_index, **kwargs):
        # 1. 获取 Baseline 并计算 Advantage
        baseline = self.baseline.get(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                     rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        mean_reward = self.baseline.update(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                           rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        nonzero_baseline = self.baseline.get_nonzero_baselines()

        advantage = (rewards - baseline).detach()
        adv_mean_log = advantage.mean().item()

        # 2. Advantage 归一化 (Standardization)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # 3. 核心优化：样本聚合去重 (Sample Aggregation)
        from_vertex_ids, to_vertex_ids, actions, advantage, freqs = \
            self.aggregate_samples(from_vertex_ids, to_vertex_ids, actions, advantage, device=self.device)

        n_unique = len(from_vertex_ids)

        # 4. 【修复 OOM 的关键】分块计算 old_logp，避免一次性生成 25GB 的巨型张量
        old_logp_list = []
        old_logp_action_list =[]
        with torch.no_grad():
            for start in range(0, n_unique, self.samples_in_batch):
                end = min(start + self.samples_in_batch, n_unique)
                
                # 仅对一小块数据做前向传播
                with torch.cuda.amp.autocast():
                    chunk_logp = self.agent.get_edge_logp(
                        from_vertex_ids[start:end], 
                        to_vertex_ids[start:end],
                        state=state, device=self.device
                    )
                chunk_action = actions[start:end]
                
                old_logp_list.append(chunk_logp)
                old_logp_action_list.append(torch.gather(chunk_logp, dim=-1, index=chunk_action[:, None])[:, 0])

        # 将分块计算的结果拼接起来 (拼接后的张量极小，不会爆显存)
        old_logp = torch.cat(old_logp_list, dim=0)
        old_logp_action = torch.cat(old_logp_action_list, dim=0)

        total_loss = total_ent = total_kl = 0.0
        n_updates = 0
        total_freqs = freqs.sum()

        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(n_unique, device=self.device)
            epoch_kl = 0.0

            # Encode once per epoch; share graph across mini-batches via gradient accumulation
            state = self.agent.prepare_state(self.hnsw.graph, device=self.device, training=True, **kwargs)
            self.optimizer.zero_grad()

            for start in range(0, n_unique, self.samples_in_batch):
                idx = perm[start:start + self.samples_in_batch]

                batch_freqs = freqs[idx]
                freqs_sum = batch_freqs.sum()

                with torch.cuda.amp.autocast():
                    logp = self.agent.get_edge_logp(from_vertex_ids[idx], to_vertex_ids[idx],
                                                    state=state, device=self.device).float()
                logp_action = torch.gather(logp, dim=-1, index=actions[idx, None])[:, 0]

                ratio = torch.exp(logp_action - old_logp_action[idx])
                adv = advantage[idx]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -(torch.min(surr1, surr2) * batch_freqs).sum() / freqs_sum

                ent = (-logp.exp() * logp).sum(-1)
                ent_loss = (ent * batch_freqs).sum() / freqs_sum

                loss = policy_loss - self.entropy_reg * ent_loss
                # 缩放比例为当前 batch 的权重占整个数据集权重的比例  
                loss_scale = freqs_sum / total_freqs  
                loss = loss * loss_scale  

                # Accumulate gradients; retain graph since state is shared across mini-batches
                self.scaler.scale(loss).backward(retain_graph=True)

                # KL (no grad)
                with torch.no_grad():
                    old_lp = old_logp[idx]
                    kl = (batch_freqs * (old_lp.exp() * (old_lp - logp.detach())).sum(-1)).sum() / freqs_sum

                total_loss += (loss.item() / loss_scale.item()) 
                total_ent += ent_loss.item()
                total_kl += kl.item()
                epoch_kl += kl.item() * (freqs_sum.item() / freqs.sum().item())
                n_updates += 1

            # Single optimizer step per epoch after all gradients are accumulated
            self.scaler.step(self.optimizer)  
            self.scaler.update() 

            # KL early stopping
            if self.target_kl is not None and epoch_kl > 1.5 * self.target_kl:
                break

        # 写日志
        if n_updates > 0:
            self.writer.add_scalar('train/loss', total_loss / n_updates, global_step=self.step)
            self.writer.add_scalar('train/entropy', total_ent / n_updates, global_step=self.step)
            self.writer.add_scalar('train/kl', total_kl / n_updates, global_step=self.step)
            
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/nonzero_baseline', nonzero_baseline, global_step=self.step)
        self.writer.add_scalar('train/advantage', adv_mean_log, global_step=self.step)
        
        return mean_reward
    
    
class MemoryEfficientPPO(PPO):
    """ PPO with reduced GPU memory footprint.
        Key differences from PPO:
        - old_logp computed in chunks and stored on CPU
        - advantage normalized per batch
        - torch.cuda.empty_cache() called after each mini-batch
    """

    def train_on_batch(self, state, from_vertex_ids, to_vertex_ids, actions, rewards, session_index, **kwargs):
        baseline = self.baseline.get(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                     rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        mean_reward = self.baseline.update(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                           rewards=rewards, session_index=session_index, device=self.device, **kwargs)

        advantage = (rewards - baseline).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        n = len(from_vertex_ids)

        # Keep ids on CPU for state.vertices indexing (state may be on a different device)
        from_vertex_ids_cpu = from_vertex_ids.cpu()
        to_vertex_ids_cpu = to_vertex_ids.cpu()
        actions_cpu = actions.cpu()

        # Compute old log-probs in chunks on CPU to avoid holding full tensor on GPU
        old_logp_cpu = torch.empty(n, 2, dtype=torch.float32)
        old_logp_action_cpu = torch.empty(n, dtype=torch.float32)
        with torch.no_grad():
            for s in range(0, n, self.samples_in_batch):
                e = min(s + self.samples_in_batch, n)
                chunk = self.agent.get_edge_logp(
                    from_vertex_ids_cpu[s:e], to_vertex_ids_cpu[s:e],
                    state=state, device=self.device).cpu()
                old_logp_cpu[s:e] = chunk
                old_logp_action_cpu[s:e] = torch.gather(
                    chunk, dim=-1, index=actions_cpu[s:e, None])[:, 0]
        torch.cuda.empty_cache()

        total_loss = total_ent = total_kl = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.samples_in_batch):
                idx = perm[s:s + self.samples_in_batch]

                logp = self.agent.get_edge_logp(from_vertex_ids_cpu[idx], to_vertex_ids_cpu[idx],
                                                state=state, device=self.device)
                logp_action = torch.gather(logp, dim=-1, index=actions_cpu[idx, None].to(self.device))[:, 0]

                old_lpa = old_logp_action_cpu[idx].to(self.device)
                ratio = torch.exp(logp_action - old_lpa)
                adv = advantage[idx.to(self.device)]

                surr = torch.min(ratio * adv,
                                 torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv)
                ent = (-logp.exp() * logp).sum(-1).mean()
                loss = -surr.mean() - self.entropy_reg * ent

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    old_lp = old_logp_cpu[idx].to(self.device)
                    kl = (old_lp.exp() * (old_lp - logp.detach())).sum(-1).mean()

                total_loss += loss.item()
                total_ent += ent.item()
                total_kl += kl.item()
                n_updates += 1
                torch.cuda.empty_cache()

        self.writer.add_scalar('train/loss', total_loss / n_updates, global_step=self.step)
        self.writer.add_scalar('train/entropy', total_ent / n_updates, global_step=self.step)
        self.writer.add_scalar('train/kl', total_kl / n_updates, global_step=self.step)
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/advantage', advantage.mean().item(), global_step=self.step)
        return mean_reward


class EfficientTRPO(TRPO):
    """ Optimized Trust Region Policy Optimization """

    def __init__(self, agent, hnsw, reward, baseline, samples_in_batch=100000,
                 Fvp_speedup=5, Fvp_type='fim', Fvp_min_batches=10, **kwargs):
        super().__init__(agent, hnsw, reward, baseline, **kwargs)
        self.samples_in_batch = samples_in_batch
        self.Fvp_min_batches = Fvp_min_batches
        self.Fvp_speedup = Fvp_speedup
        self.Fvp_type = Fvp_type

    def train_on_batch(self, state, from_vertex_ids, to_vertex_ids, actions, rewards, session_index, **kwargs):
        baseline = self.baseline.get(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                     rewards=rewards, session_index=session_index, device=self.device, **kwargs)

        # Update baseline for the next iteration
        mean_reward = self.baseline.update(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                           rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        # if not train mode, exit
        if self.max_kl == 0.0:
            return mean_reward

        advantage = rewards - baseline

        from_vertex_ids, to_vertex_ids, actions, advantage, freqs = \
            self.aggregate_samples(from_vertex_ids, to_vertex_ids, actions, advantage, device=self.device)

        batches_from_vertex_ids = from_vertex_ids.split(self.samples_in_batch)
        batches_to_vertex_ids = to_vertex_ids.split(self.samples_in_batch)
        batches_advantage = advantage.split(self.samples_in_batch)
        batches_actions = actions.split(self.samples_in_batch)
        batches_freqs = freqs.split(self.samples_in_batch)
        batches_old_logp = []

        loss_grad = 0
        for batch_from_vertex_ids, batch_to_vertex_ids, batch_actions, batch_advantage, batch_freqs in \
                zip(batches_from_vertex_ids, batches_to_vertex_ids, batches_actions, batches_advantage, batches_freqs):
            batch_logp = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                  state=state, device=self.device)
            batch_logp_actions = torch.gather(batch_logp, dim=-1, index=batch_actions[:, None])[:, 0]
            batch_old_logp = batch_logp.detach()
            batch_old_logp_actions = batch_logp_actions.detach()

            batches_old_logp.append(batch_old_logp)

            # Here ratio is always 1
            ratio = torch.exp(batch_logp_actions - batch_old_logp_actions)  # pi(a|s) / pi_old(a|s)
            loss = -(ratio * batch_advantage).sum() / freqs.sum()
            # Entropy requlizer
            ent = (-batch_logp.exp() * batch_logp).sum(-1)
            loss -= self.entropy_reg * (batch_freqs * ent).sum() / freqs.sum()  # add entropy regularization

            grads = torch.autograd.grad(loss, self.agent.parameters())
            loss_grad += torch.cat([grad.view(-1) for grad in grads]).detach_()

        # Forward Fisher vector product
        def Fvp_forward(v):
            # Here we compute Fx to do solve Fx = g using conjugate gradients
            flat_grad_grad_kl = 0
            min_n_batches = min(self.Fvp_min_batches, len(batches_freqs))
            nbatches = max(len(batches_freqs) // self.Fvp_speedup, min_n_batches)

            for i_batch, (batch_from_vertex_ids, batch_to_vertex_ids, batch_old_logp, batch_freqs) in \
                enumerate(zip(batches_from_vertex_ids, batches_to_vertex_ids, batches_old_logp, batches_freqs)):
                if i_batch == nbatches: break
                batch_logp = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                      state=state, device=self.device)
                batch_probs = batch_logp.exp()
                kl = (batch_freqs * (batch_probs * (batch_logp - batch_old_logp)).sum(-1)).sum()
                assert (kl > -0.0001).all()

                flat_grad_kl = compute_flat_grad(kl, self.agent.parameters(), create_graph=True)

                kl_v = (flat_grad_kl * v).sum()
                flat_grad_grad_kl += compute_flat_grad(kl_v, self.agent.parameters()).detach_()
            flat_grad_grad_kl /= sum([batch_freqs.sum() for batch_freqs in batches_freqs[:nbatches]])
            return flat_grad_grad_kl + v * self.damping

        # Fisher vector product Requires ~15% less memory
        def Fvp_fim(v):
            JTMJv = 0
            min_n_batches = min(self.Fvp_min_batches, len(batches_freqs))
            nbatches = max(len(batches_freqs) // self.Fvp_speedup, min_n_batches)

            for (batch_from_vertex_ids, batch_to_vertex_ids, batch_freqs) in \
                zip(batches_from_vertex_ids[:nbatches], batches_to_vertex_ids[:nbatches], batches_freqs[:nbatches]):
                batch_probs = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                       state=state, device=self.device).exp()
                mu = batch_probs.view(-1)  # mu = mu.view(-1)
                M = mu.pow(-1).detach()
                weighted_mu = (batch_freqs[:, None] * batch_probs).view(-1)
                # M is the second derivative of the KL distance wrt network output
                # (M*M diagonal matrix compressed into a M*1 vector)
                # mu is the network output (M*1 vector)
                t = torch.ones(mu.size(), device=self.device, requires_grad=True)
                mu_t = mu @ t
                Jt = compute_flat_grad(mu_t, self.agent.parameters(), create_graph=True)
                Jtv = Jt @ v
                Jv = torch.autograd.grad(Jtv, t)[0]
                mu_MJv = (weighted_mu * M * Jv.detach_()).sum()
                JTMJv += compute_flat_grad(mu_MJv, self.agent.parameters()).detach_()
            sum_freqs = sum([batch_freqs.sum() for batch_freqs in batches_freqs[:nbatches]])
            JTMJv /= sum_freqs
            return JTMJv + v * self.damping

        Fvp = Fvp_fim if self.Fvp_type == 'fim' else Fvp_forward
        stepdir = self.conjugate_gradient(Fvp, -loss_grad, 10)

        # Here we compute the initial vector to do linear search
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]

        # Here we get the start point
        prev_params = get_flat_params_from(self.agent)

        @torch.no_grad()
        def get_loss_kl_ent(params):
            # Helper for linear search
            set_flat_params_to(self.agent, params)

            loss, kl, ent = 0, 0, 0
            for batch_from_vertex_ids, batch_to_vertex_ids, batch_actions, \
                batch_advantage, batch_old_logp, batch_freqs in \
                    zip(batches_from_vertex_ids, batches_to_vertex_ids, batches_actions,
                        batches_advantage, batches_old_logp, batches_freqs):
                batch_logp = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                      state=state, device=self.device)
                batch_probs = batch_logp.exp()
                batch_logp_action = torch.gather(batch_logp, dim=-1, index=batch_actions[:, None])[:, 0]
                batch_old_logp_action = torch.gather(batch_old_logp, dim=-1, index=batch_actions[:, None])[:, 0]
                ratio = torch.exp(batch_logp_action - batch_old_logp_action)

                loss += -torch.sum(ratio * batch_advantage)
                kl += (batch_freqs * (batch_probs * (batch_logp - batch_old_logp)).sum(-1)).sum()
                ent += (batch_freqs * (-batch_probs * batch_logp).sum(-1)).sum()
            n_samples = freqs.sum()
            loss /= n_samples
            kl /= n_samples
            ent /= n_samples
            assert (kl > -0.0001).all() and (kl < 10000).all()
            return [loss - self.entropy_reg * ent, kl, ent]

        # Here we find our new parameters
        state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)
        new_params = self.linesearch(get_loss_kl_ent, prev_params, fullstep)

        # And we set it to our network
        set_flat_params_to(self.agent, new_params)

        # Summarize
        loss, kl, ent = get_loss_kl_ent(new_params)
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/advantage', advantage.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/entropy', ent.item(), global_step=self.step)
        self.writer.add_scalar('train/kl', kl.item(), global_step=self.step)
        self.writer.add_scalar('train/loss', loss.item(), global_step=self.step)
        return mean_reward


class GraphEditPPO(BaseAlgorithm):
    """ PPO over the graph-editing MDP of framework.md.

    State s_t is the whole graph; the action is a bounded edge swap per node
    (drop n_swap current neighbours, insert n_swap sampled from the 2-hop pool N2),
    so degree is invariant. The reward is the *graph* cost delta
    r_t = c(s_t) - c(s_{t+1}), measured by deterministic probe searches and
    credited to the nodes those probes actually visited.

    Both halves of the swap are policy decisions. A swap's effect on recall depends
    on the pair -- adding a good edge while giving up a better one is a net loss --
    so training only the addition side leaves half the action untrained. The drop
    side used to be a deterministic argmin over the same scorer with no log-prob at
    all, i.e. frozen.

    Whether training it actually helps is UNRESOLVED. On the random s_0 (6 runs, 2 seeds
    x 3 drop_entropy_reg values) 'policy' and 'argmin' are indistinguishable -- the
    difference has no stable sign across seeds. An earlier single-run reading that
    uniform drops beat argmin by +0.000488 did not survive: repeated with more seeds the
    same comparison moved to +4.9e-05 (0.3 sigma), within its own SE of ~1.4e-04. That
    s_0's per-batch |mean|/SE is 0.26, i.e. it cannot resolve an effect of this size in
    either direction, so the question needs the NSW start (gamma>0 + a critic) to settle.

    :param hnsw: must be a lib.hnsw.GraphEditHNSW
    :param n_swap: edges replaced per node per step
    :param nodes_per_step: nodes sampled for editing each step
    :param nodes_in_batch: nodes per forward chunk (memory knob)
    :param drop_mode: how the dropped neighbours are chosen.
           'policy' -- sampled from softmax(-logits) over current neighbours and
                       TRAINED, symmetric to the addition side: one edge-quality
                       score, high => likely added and unlikely dropped.
           'argmin'  -- legacy deterministic argmin, untrained (reproduces old runs)
           'random'  -- uniform over current neighbours, untrained. The chance-level
                       control for whether training the drop side buys anything.
    :param commit_stride: how many steps a swap stays tentative. 1 keeps every swap
           (the state advances every step); N > 1 measures and trains on N trial
           swaps from the same s_t and only keeps the last, so the graph moves N
           times more slowly while the policy sees N times more rollouts per state.
    """

    def __init__(self, agent, hnsw, reward, baseline,
                 optimizer=None, lr=3e-4, n_swap=2, nodes_per_step=4096,
                 clip_eps=0.2, ppo_epochs=4, nodes_in_batch=512,
                 entropy_reg=0.01, drop_entropy_reg=None, target_kl=0.015,
                 commit_stride=1, max_grad_norm=1.0, use_noop=True,
                 drop_mode='policy', gamma=0.0, value_coef=0.5,
                 use_critic=None, accept='node', adv_ref=None,
                 long_frac=0.0, cand_band=None, n_rand_cand=0, max_periph_pct=100.0,
                 len_head=False, len_p0=65.0, len_span=25.0, len_sigma=6.0,
                 len_half=10.0, len_fixed=None, **kwargs):
        super().__init__(agent, hnsw, reward, baseline, **kwargs)
        if drop_mode not in ('policy', 'argmin', 'random'):
            raise ValueError("drop_mode must be 'policy', 'argmin' or 'random', got %r"
                             % (drop_mode,))
        if accept not in ('off', 'step', 'node'):
            raise ValueError("accept must be 'off', 'step' or 'node', got %r" % (accept,))
        if adv_ref not in (None, 'batch', 'noop'):
            raise ValueError("adv_ref must be None, 'batch' or 'noop', got %r"
                             % (adv_ref,))
        # What an advantage is measured AGAINST. 'batch' (pre-2026-08-06) subtracts the
        # EMA baseline and then the batch mean, which forces mean(advantage)==0 and so
        # makes ~half the batch positive even when every sampled edit was harmful.
        # 'noop' references the no-op's exactly-0 reward instead and skips centering, so
        # a positive advantage means "better than leaving this node alone". Measured on
        # NSW at accept='node', 4 seeds: +0.0058 +- 0.0024 recall@10 for noop over batch.
        # None resolves to 'noop', except under a critic, which brings its own learned
        # reference and cannot be combined with this one (see the check below). The
        # sentinel exists so that asking for gamma>0 does not become an error.
        self._adv_ref_arg = adv_ref
        # Roll back an edit the probe measured as harmful. 'off' is the historical
        # unconditional commit, which has no lower bound at all: commit_stride's rollback
        # is unconditional too (it discards trial N-1 whatever it measured), so nothing
        # anywhere in the loop consulted the reward before keeping a change.
        self.accept = accept
        self.drop_mode = drop_mode
        # B2: keep only the longest `long_frac` of each node's 2-hop candidate pool,
        # so the scorer cannot pick a short edge because none is offered. 0 disables.
        #
        # This exists because the short-edge bias turned out to be ARCHITECTURAL, not
        # learned: rho(pair score, edge length) is -0.80 at RANDOM init and only -0.58
        # after pretraining, i.e. a dot product over a smooth encoder is a proximity
        # oracle at every weight setting. No retraining escapes it (B1: 3 seeds, null
        # on add_pct, -0.029 on recall), so the remaining lever is the menu, not the
        # ranker. The pool does contain long edges -- on the rewired graph the median
        # node's longest available candidate sits at percentile 92 while the policy
        # adds at 1.6 -- so filtering has something to bite on.
        if not 0.0 <= long_frac <= 1.0:
            raise ValueError('long_frac must be in [0, 1], got %r' % (long_frac,))
        self.long_frac = long_frac
        # D1: place the menu where the length optimum actually is, instead of at the
        # extreme long_frac reaches. C2 swept the target length percentile directly and
        # found an INTERIOR optimum at ~65 (kNN's own edges sit at 7.6, the policy's
        # runaway hubs at 93.2, and a uniform random target at 46.9 -- not 100, which is
        # what this project assumed for a long time). A concentrated band beat uniform
        # targets on 6/6 slot-seed x dose cells.
        #
        # long_frac cannot express this: it takes the longest fraction of the 2-HOP pool,
        # and that pool tops out at per-node percentile 8.95 on kNN, so a band at 65 is
        # simply not on the menu. Hence n_rand_cand -- uniformly sampled nodes appended
        # to each row before filtering, which is the only way the band becomes reachable.
        #
        # The division of labour is the one C1 measured: the RULE sets the length (the
        # policy demonstrably cannot -- its bias is architectural), and the policy chooses
        # WITHIN the band (worth +0.010 to +0.014, confirmed against a same-length
        # reshuffle and a placebo).
        if cand_band is not None:
            lo, hi = cand_band
            if not 0.0 <= lo < hi <= 100.0:
                raise ValueError('cand_band must be 0 <= lo < hi <= 100, got %r' % (cand_band,))
            cand_band = (float(lo), float(hi))
        self.cand_band = cand_band
        self.n_rand_cand = int(n_rand_cand)
        # Anti-peripherality. C1b isolated the hub problem to the IDENTITY of the chosen
        # nodes: permuting which source uses which hub is null (+0.0003), and relabelling
        # the hubs to random nodes while holding in-degree exactly gains +0.030. Those 10
        # nodes sit at distance-to-centroid percentile 98.8-99.8 on every seed -- data-cloud
        # outliers, which are dead-end highway exits. An in-degree brake does NOT fix this
        # and has measured null three separate times; excluding outliers from the menu is
        # the intervention the evidence actually points at. 100 disables.
        self.max_periph_pct = float(max_periph_pct)
        # D2: learn the length percentile per node instead of fixing it at C2's global
        # optimum. Stage 0 measured the learning-free version of this and found NULL --
        # shuffling the feature across nodes reproduced the whole gain -- but that null
        # is conditional on the LINEAR form and the three closed-form features it used.
        # This head is the function-class-free version of the same question, and it
        # carries its own acceptance test (see stage1_eval.py): beat p=65, beat its OWN
        # shuffled p_u, and not merely re-derive knn_radius / centroid / lid.
        #
        # p_u = p0 + span * tanh(delta_u), so the head starts at exactly p0 (zero-init)
        # and can never leave (p0-span, p0+span) -- which keeps it out of the p>90 region
        # that C1b showed is full of dead-end outliers.
        self.len_head = bool(len_head)
        self.len_p0 = float(len_p0)
        self.len_span = float(len_span)
        self.len_sigma = float(len_sigma)
        self.len_half = float(len_half)
        # The matched control for len_head: the SAME per-node rank band, centred at a
        # constant. Without it the only available comparison would be D1's global-CDF
        # band, which is a different rule -- so a head-vs-D1 difference could not be
        # attributed to the head rather than to the change of percentile definition.
        self.len_fixed = None if len_fixed is None else float(len_fixed)
        self._len_ref = None            # sorted random pair distances, built on first use
        self._periph_pct = None         # per-vertex centroid-distance percentile
        self._cand_rng_seed = 0         # advanced per call so rows differ across steps

        self.n_swap = n_swap
        self.nodes_per_step = nodes_per_step
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.nodes_in_batch = nodes_in_batch
        self.entropy_reg = entropy_reg
        # Separate coefficient for the DROP softmax, because the bonus and the drop
        # side's purpose pull against each other: the bonus pushes that distribution
        # toward uniform, and a uniform drop is exactly the 'random' baseline that
        # drop_mode='policy' is supposed to beat. Sharing entropy_reg made the two
        # inseparable. None means "follow entropy_reg", which reproduces the shared
        # behaviour of every run before 2026-08-06.
        self.drop_entropy_reg = (entropy_reg if drop_entropy_reg is None
                                 else drop_entropy_reg)
        # Discount on the per-node credited reward. gamma=0 is the historical greedy
        # bandit: each step is scored only by its own immediate delta, which cannot
        # leave a graph that is already a local optimum of the bounded swap (NSW is).
        # With r_t = c(s_t) - c(s_{t+1}) the return telescopes, so gamma->1 makes the
        # objective the TOTAL improvement c(s_0) - c(s_T) rather than one step of it,
        # and an edit that costs now but enables a better later state can be credited.
        self.gamma = gamma
        self.value_coef = value_coef
        # The critic replaces the EMA baseline. Defaults to on exactly when gamma>0,
        # because at gamma=0 a critic and the EMA baseline estimate the same quantity
        # (E[r_i]) and the EMA one is already validated -- so this keeps every existing
        # gamma=0 result bit-reproducible while making gamma>0 impossible to request
        # without the bootstrap it needs.
        self.use_critic = (gamma > 0) if use_critic is None else use_critic
        if self.use_critic and not hasattr(agent, 'get_values'):
            raise ValueError('use_critic/gamma>0 needs an agent with get_values(); %s '
                             'has none' % type(agent).__name__)
        # Resolve the adv_ref sentinel now that use_critic is known. A critic supplies
        # its own learned reference V(i, s_t), so "positive" there means "better than the
        # critic expected", NOT "better than not touching this node" -- stacking the two
        # would make the sign uninterpretable. An EXPLICIT 'noop' plus a critic is a
        # contradiction and is refused; the auto default just steps aside.
        if self._adv_ref_arg is None:
            self.adv_ref = 'batch' if self.use_critic else 'noop'
        else:
            self.adv_ref = self._adv_ref_arg
            if self.use_critic and self.adv_ref == 'noop':
                raise ValueError("adv_ref='noop' is incompatible with a critic: the "
                                 'critic subtracts its own learned baseline V(i,s_t), '
                                 'so the reference is V, not 0')
        self.target_kl = target_kl
        # Rewards here are differences of recall estimates, so a single unlucky
        # probe batch can produce a large advantage; with a temperature-scaled
        # scorer the resulting gradient norm reaches ~1e+01. None disables
        # clipping (the norm is still reported).
        self.max_grad_norm = max_grad_norm
        # Let each node decline to be edited. Requires agent.get_act_logits.
        self.use_noop = use_noop and hasattr(agent, 'get_act_logits')
        if use_noop and not self.use_noop:
            print('[GraphEditPPO] agent has no get_act_logits; no-op action disabled')
        assert commit_stride >= 1
        self.commit_stride = commit_stride
        self.last_opt_diagnostics = None
        self.optimizer = optimizer or torch.optim.Adam(agent.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()
        # Cost of the current *committed* graph. Reused as the pre-swap cost of every
        # trial in the stride, since a reverted trial leaves the graph unchanged.
        self._probe_cache = None
        # Trials taken since the last commit; only steps that actually swapped count.
        self._trials_since_commit = 0

    def node_ctx(self, state, node_ids, adj_ids, adj_mask, grad=False):
        """ Per-node topology summary for the actor, [B, emb_dim], or None if the agent
        has no actor_ctx head. Computed once per batch and shared by both the addition
        and the removal scoring pass, which condition on the same neighbourhood.
        """
        fn = getattr(self.agent, 'get_node_ctx', None)
        if fn is None:
            return None
        grad_ctx = torch.enable_grad() if grad else torch.no_grad()
        with grad_ctx:
            return fn(node_ids, adj_ids, adj_mask, state=state, device=self.device)

    @property
    def uses_indeg(self):
        """ Whether the agent has any head that consumes in-degrees. """
        return bool(getattr(self.agent, 'indeg_ctx', False)
                    or getattr(self.agent, 'indeg_noop', False))

    def filter_long_candidates(self, node_ids_np, cand_ids_np, cand_mask_np):
        """ B2: mask out all but the longest `self.long_frac` of each node's pool.

        Operates on the mask only -- cand_ids is left untouched, so padded slots keep
        holding a valid index and stay safe to embed (get_candidates' contract).

        Rows are filtered independently and each keeps at least `n_swap` candidates:
        a node whose pool is small must still be able to act, and zeroing a row would
        make its log_softmax NaN. Distances are computed against the node's own
        coordinates rather than a global percentile table, because "long" only means
        anything relative to the node's own neighbourhood scale.
        """
        if not self.long_frac:
            return cand_mask_np

        V = self.hnsw.graph.vertices
        V = V.numpy() if torch.is_tensor(V) else np.asarray(V)
        keep_mask = np.zeros_like(cand_mask_np)
        floor = max(self.n_swap, 1)

        for i, node in enumerate(node_ids_np):
            valid = np.flatnonzero(cand_mask_np[i])
            if valid.size == 0:
                continue
            n_keep = max(floor, int(np.ceil(self.long_frac * valid.size)))
            if n_keep >= valid.size:
                keep_mask[i, valid] = True
                continue
            d = V[cand_ids_np[i, valid]].astype(np.float64) - V[node].astype(np.float64)
            d = (d * d).sum(-1)
            # argpartition puts the n_keep largest distances in the tail.
            keep_mask[i, valid[np.argpartition(d, valid.size - n_keep)[valid.size - n_keep:]]] = True

        return keep_mask

    def _length_ref(self):
        """ Sorted random pair distances, so a distance can be read as a percentile.

        A raw L2 in 128-dim normalised space carries no intuition and is not comparable
        across graphs; every length statement in this project is a percentile of THIS
        distribution. Built once and cached -- 200k pairs is enough to resolve the band
        edges to well under a percentile.
        """
        if self._len_ref is None:
            V = self.hnsw.graph.vertices
            V = V.numpy() if torch.is_tensor(V) else np.asarray(V)
            rng = np.random.default_rng(7)
            n = V.shape[0]
            a, b = rng.integers(0, n, size=200000), rng.integers(0, n, size=200000)
            keep = a != b
            d = V[a[keep]].astype(np.float64) - V[b[keep]].astype(np.float64)
            self._len_ref = np.sort(np.sqrt((d * d).sum(1)))
        return self._len_ref

    def _peripherality(self):
        """ Per-vertex distance-to-centroid, as a percentile over all vertices. """
        if self._periph_pct is None:
            V = self.hnsw.graph.vertices
            V = V.numpy() if torch.is_tensor(V) else np.asarray(V)
            V = V.astype(np.float64)
            d = np.sqrt(((V - V.mean(0)) ** 2).sum(1))
            self._periph_pct = 100.0 * d.argsort().argsort() / (len(d) - 1)
        return self._periph_pct

    def augment_candidates(self, node_ids_np, cand_ids_np, cand_mask_np, p_center=None):
        """ D1/D2: append random candidates, then keep only those inside the length band.

        Returns possibly-widened (cand_ids, cand_mask). cand_ids stays fully populated
        with valid vertex indices even where the mask is False, because downstream code
        embeds every column regardless of the mask (get_candidates' contract).

        Random candidates are drawn per row and screened against the row's existing
        neighbours and the node itself, so a sampled "addition" is always a real new edge.
        Roughly (hi-lo)% of them survive the band, which is why n_rand_cand needs to be
        a few hundred for a 20-point band to leave a usable menu.

        Two ways of saying "percentile", and they are NOT the same:

          p_center is None   the band is fixed and measured against the GLOBAL random
                             pair-distance CDF. This is what D1 shipped with.
          p_center given     per-node band centred on p_center[i], and the percentile is
                             each candidate's RANK among the row's own uniform sample --
                             i.e. the percentile of the distance distribution CONDITIONED
                             ON THIS NODE. This is the definition C2 optimised (it takes
                             the node at rank p% of the source's own ordering), and it
                             differs from the global one for any node whose neighbourhood
                             is denser or sparser than average. The uniform candidates
                             double as an unbiased sample of that conditional distribution,
                             so the rank costs nothing extra to estimate.

        Rows that end up with fewer than n_swap in-band candidates fall back to the
        candidates whose percentile is CLOSEST to the band centre rather than to an
        arbitrary set: an all-masked row makes log_softmax NaN, and silently reverting
        such a row to the unfiltered pool would put short edges back on the menu for
        exactly the nodes the band is meant to help.
        """
        banded = self.cand_band is not None or p_center is not None
        if not banded and not self.n_rand_cand and self.max_periph_pct >= 100.0:
            return cand_ids_np, cand_mask_np
        if p_center is not None and not self.n_rand_cand:
            raise ValueError('per-node rank bands need --n_rand_cand: the rank is '
                             'estimated from the row\'s own uniform sample')

        V = self.hnsw.graph.vertices
        V = V.numpy() if torch.is_tensor(V) else np.asarray(V)
        pad = self.hnsw.service_labels['pad']
        n_vert = V.shape[0]
        rng = np.random.default_rng(self._cand_rng_seed)
        self._cand_rng_seed += 1

        n_2hop = cand_ids_np.shape[1]
        if self.n_rand_cand:
            extra = rng.integers(0, n_vert, size=(len(node_ids_np), self.n_rand_cand))
            cand_ids_np = np.concatenate(
                [cand_ids_np, extra.astype(cand_ids_np.dtype)], axis=1)
            cand_mask_np = np.concatenate(
                [cand_mask_np, np.ones(extra.shape, dtype=cand_mask_np.dtype)], axis=1)

        ref = self._length_ref() if (self.cand_band is not None and p_center is None) else None
        periph = self._peripherality() if self.max_periph_pct < 100.0 else None
        floor = max(self.n_swap, 1)
        adj = self.hnsw.adj
        half = self.len_half

        for i, node in enumerate(node_ids_np):
            row = cand_ids_np[i]
            keep = cand_mask_np[i].copy()
            # never offer the node itself or an edge it already has
            nbrs = adj[node]
            keep &= (row != node) & ~np.isin(row, nbrs[nbrs != pad])
            if periph is not None:
                keep &= periph[row] <= self.max_periph_pct
            if not banded:
                cand_mask_np[i] = keep if keep.sum() >= floor else cand_mask_np[i]
                continue

            valid = np.flatnonzero(keep)
            if valid.size == 0:
                cand_mask_np[i] = False
                cand_mask_np[i, :floor] = True
                continue
            d = V[row[valid]].astype(np.float64) - V[node].astype(np.float64)
            d = np.sqrt((d * d).sum(-1))
            if p_center is None:
                pct = 100.0 * np.searchsorted(ref, d) / len(ref)
                lo, hi = self.cand_band
            else:
                # Rank among this row's own uniform sample: an unbiased estimate of the
                # candidate's percentile in the distance distribution conditioned on
                # `node`, at resolution 100/n_rand_cand.
                samp = np.sort(np.sqrt(((V[row[n_2hop:]].astype(np.float64)
                                         - V[node].astype(np.float64)) ** 2).sum(-1)))
                pct = 100.0 * np.searchsorted(samp, d) / len(samp)
                lo, hi = p_center[i] - half, p_center[i] + half
            inb = (pct >= lo) & (pct <= hi)
            sel = valid[inb] if inb.sum() >= floor else \
                valid[np.argsort(np.abs(pct - 0.5 * (lo + hi)))[:floor]]
            cand_mask_np[i] = False
            cand_mask_np[i, sel] = True

        return cand_ids_np, cand_mask_np

    def adj_rows_of(self, node_ids_np):
        """ (adj_ids, adj_mask) tensors for the given nodes under the CURRENT graph. """
        rows = self.hnsw.adj[node_ids_np]
        valid = rows != self.hnsw.service_labels['pad']
        return (torch.as_tensor(np.where(valid, rows, 0).astype(np.int64), device=self.device),
                torch.as_tensor(valid, dtype=torch.bool, device=self.device))

    def len_p_mean(self, state, node_ids, adj_ids, adj_mask, grad=False):
        """ p0 + span*tanh(delta): the head's mean length percentile per node. """
        ctx = torch.enable_grad if grad else torch.no_grad
        with ctx():
            delta = self.agent.get_len_delta(node_ids, adj_ids, adj_mask,
                                             state=state, device=self.device)
        return self.len_p0 + self.len_span * torch.tanh(delta)

    @staticmethod
    def gaussian_logp(x, mean, sigma):
        """ log N(x; mean, sigma) elementwise. Analytic, so no rsample bookkeeping. """
        return (-0.5 * ((x - mean) / sigma) ** 2
                - math.log(sigma) - 0.5 * math.log(2.0 * math.pi))

    def compute_in_degrees(self):
        """ In-degree of every graph node under the CURRENT adjacency, [N] int64.

        The policy's only view of the graph was z_i (vertex features) and, with
        actor_ctx, the mean of its neighbours' features -- all functions of coordinates,
        none of connectivity. So nothing in the input distinguished a candidate that
        1000 other nodes already point at from one nobody points at, and since the
        attractive targets are attractive to EVERY node, each independent per-node
        decision piled onto the same few destinations: measured on SIFT100K, max
        in-degree 8039 vs NSW's 71. The fix has to be an input, not a penalty -- a
        reward term cannot tell the policy WHICH candidate is the saturated one.

        Returns raw counts; log1p is applied by whichever head consumes them.
        """
        adj = self.hnsw.adj
        pad = self.hnsw.service_labels['pad']
        flat = adj.reshape(-1)
        return np.bincount(flat[flat != pad].astype(np.int64),
                           minlength=adj.shape[0])

    def indeg_tensor(self, in_deg_np):
        """ Raw in-degree counts -> float32 device tensor for the scoring heads. """
        if in_deg_np is None:
            return None
        return torch.as_tensor(np.asarray(in_deg_np, dtype=np.float32),
                               device=self.device)

    def score_candidates(self, state, node_ids, cand_ids, cand_mask, grad=False, ctx=None,
                         in_deg=None):
        """ framework.md step 4: score every (node, 2-hop candidate) pair.

        :param ctx: optional [B, emb_dim] per-node topology summary (agent.get_node_ctx).
                    Broadcast to one row per pair so the scorer sees the neighbourhood
                    of the node the candidate would attach to.
        :param in_deg: optional [N] float tensor of raw in-degrees over ALL graph nodes.
                       Sliced per pair into [log1p(src), log1p(dst)] so the scorer can
                       discount a candidate that is already over-subscribed.
        :return: logits [B, C] float32 with masked slots at -inf
        """
        num_nodes, width = cand_ids.shape
        from_flat = node_ids[:, None].expand(num_nodes, width).reshape(-1)
        to_flat = cand_ids.reshape(-1)

        chunk = max(1, self.nodes_in_batch) * width
        logits = []
        # Action sampling never backprops, and the pair count here is large
        # (nodes x candidates), so skip building the graph entirely in that case.
        grad_ctx = torch.enable_grad() if grad else torch.no_grad()
        with grad_ctx:
            for start in range(0, from_flat.numel(), chunk):
                end = min(start + chunk, from_flat.numel())
                # `chunk` is a whole multiple of `width`, so a chunk always covers
                # complete node rows and this slice of ctx lines up exactly. Expanded
                # per chunk rather than once up front so the [B*C, emb_dim] copy is
                # never materialized in full.
                ctx_part = None
                if ctx is not None:
                    ctx_part = ctx[start // width:-(-end // width)]
                    ctx_part = ctx_part.unsqueeze(1).expand(-1, width, -1).reshape(
                        -1, ctx.shape[-1])
                # Padding slots index in_deg at whatever id the mask will discard;
                # the row is overwritten with -inf below, so the value is irrelevant.
                indeg_part = None
                if in_deg is not None:
                    indeg_part = torch.log1p(torch.stack([
                        in_deg[from_flat[start:end]],
                        in_deg[to_flat[start:end].clamp_min(0)],
                    ], dim=-1))
                with torch.cuda.amp.autocast():
                    part = self.agent.get_pair_logits(from_flat[start:end], to_flat[start:end],
                                                      state=state, device=self.device,
                                                      ctx=ctx_part, indeg=indeg_part)
                logits.append(part.float())
            logits = torch.cat(logits, dim=0).reshape(num_nodes, width)
            return logits.masked_fill(~cand_mask, float('-inf'))

    @staticmethod
    def masked_logp(act, logp):
        """ log-prob of draws that only exist where `act` is true, as 0 elsewhere.

        `act.float() * logp` is the obvious spelling and it is wrong: a no-op node's
        logp can be -inf (its row was all-padding, or the draw had vanishing
        probability), and 0 * -inf is NaN, which then poisons old_logp, the PPO ratio
        and every gradient downstream. A node that did not act has no draw, so its
        contribution is exactly 0 by construction -- select it rather than scale it.
        """
        return torch.where(act, logp, torch.zeros_like(logp))

    @staticmethod
    def gumbel_like(logits):
        """ Standard Gumbel(0,1) noise shaped like `logits`.

        Written out because the one-liner form is a trap:
            -log(-log(u.clamp_min(eps)).clamp_min(eps))
        binds .clamp_min to log(u), NOT to its negation. log(u) is negative, clamp_min
        raises it to +eps, the unary minus makes it -eps, and log of a negative is NaN --
        so EVERY sample came back NaN. topk over an all-NaN row returns a fixed slot
        position, which looks like sampling but ignores the logits completely: measured
        on SIFT100K, 143 of 256 rows all picked the identical slots [15, 17], including
        padding slots that -inf should have excluded. Clamp the positive quantity
        instead, after negating.
        """
        u = torch.rand_like(logits)
        return -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))

    @staticmethod
    def plackett_luce_logp(logits, chosen):
        """ log-prob of drawing `chosen` (ordered, without replacement) from softmax(logits).

        logp = sum_j [ logits[c_j] - logsumexp(logits over slots not yet drawn) ].
        Equivalent to Gumbel-top-k sampling, and differentiable w.r.t. logits.

        :param logits: [B, C] with -inf on invalid slots
        :param chosen: [B, n_swap] indices into C, in draw order
        :return: [B] log-probabilities
        """
        available = torch.ones_like(logits, dtype=torch.bool)
        logp = torch.zeros(logits.size(0), device=logits.device, dtype=logits.dtype)
        for j in range(chosen.size(1)):
            idx = chosen[:, j:j + 1]
            masked = logits.masked_fill(~available, float('-inf'))
            logp = logp + (torch.gather(masked, 1, idx)[:, 0] - torch.logsumexp(masked, dim=1))
            available = available.scatter(1, idx, False)
        return logp

    def sample_swaps(self, state, node_ids_np, greedy=False, **kwargs):
        """ framework.md step 4: pick which edges to add and which to drop.

        Additions: Gumbel-top-k over the candidate softmax (top-k when greedy).
        Removals: the n_swap lowest-scoring *current* neighbours of each node, so
        the agent's own edge scores decide what is least worth keeping.

        :return: dict with the tensors the PPO update needs, or None if no node in
                 the batch has any usable 2-hop candidate.
        """
        hnsw = self.hnsw
        cand_ids_np, cand_mask_np = hnsw.get_candidates(node_ids_np)

        # D2: the length head runs FIRST, because its output defines the menu. Sampled
        # (not argmax) unless greedy, so the PPO ratio has a distribution to move.
        p_np = p_sample = p_mean_t = None
        if self.len_fixed is not None:
            p_np = np.full(len(node_ids_np), self.len_fixed)
        if self.len_head:
            all_ids = torch.as_tensor(node_ids_np, dtype=torch.int64, device=self.device)
            a_ids, a_mask = self.adj_rows_of(node_ids_np)
            p_mean_t = self.len_p_mean(state, all_ids, a_ids, a_mask, grad=False)
            p_sample = p_mean_t if greedy else \
                p_mean_t + self.len_sigma * torch.randn_like(p_mean_t)
            p_np = p_sample.detach().cpu().numpy()

        # D1: widen the pool with random vertices and restrict it to the length band,
        # BEFORE the B2 filter and before the `enough` check, so both see the menu the
        # policy will really sample from.
        cand_ids_np, cand_mask_np = self.augment_candidates(
            node_ids_np, cand_ids_np, cand_mask_np, p_center=p_np)

        # B2: restrict the menu to long edges before anything is scored or counted,
        # so the `enough` check below sees the pool the policy will actually sample
        # from and no node survives the filter with an unusable row.
        cand_mask_np = self.filter_long_candidates(node_ids_np, cand_ids_np, cand_mask_np)

        # Drop nodes with fewer candidates than we need to insert. At least one valid
        # candidate is required regardless of n_swap: an all-masked row is all -inf,
        # and log_softmax over it is NaN, which would poison the entropy term.
        enough = cand_mask_np.sum(-1) >= max(self.n_swap, 1)
        # ...and nodes whose degree is too small to give edges up.
        degrees = (hnsw.adj[node_ids_np] != hnsw.service_labels['pad']).sum(-1)
        keep = enough & (degrees > self.n_swap)
        if not keep.any():
            return None

        node_ids_np, cand_ids_np, cand_mask_np = node_ids_np[keep], cand_ids_np[keep], cand_mask_np[keep]
        if self.len_head:
            keep_t = torch.as_tensor(keep, dtype=torch.bool, device=self.device)
            p_sample, p_mean_t = p_sample[keep_t], p_mean_t[keep_t]

        node_ids = torch.as_tensor(node_ids_np, dtype=torch.int64, device=self.device)
        cand_ids = torch.as_tensor(cand_ids_np, dtype=torch.int64, device=self.device)
        cand_mask = torch.as_tensor(cand_mask_np, dtype=torch.bool, device=self.device)

        # Current neighbour rows, read BEFORE any scoring: they are both the drop
        # candidates and (via get_node_ctx) the topology the addition side conditions
        # on, and the swap has not been applied yet so this is s_t's adjacency.
        adj_rows = hnsw.adj[node_ids_np]
        adj_valid = adj_rows != hnsw.service_labels['pad']
        adj_ids = torch.as_tensor(np.where(adj_valid, adj_rows, 0).astype(np.int64), device=self.device)
        adj_mask = torch.as_tensor(adj_valid, dtype=torch.bool, device=self.device)
        ctx = self.node_ctx(state, node_ids, adj_ids, adj_mask, grad=False)

        # Read from s_t's adjacency, like adj_rows above, and snapshotted into the
        # action dict: the PPO update re-scores these pairs several epochs later, by
        # which time the graph holds s_{t+1} and the counts have moved.
        in_deg_np = self.compute_in_degrees() if self.uses_indeg else None
        in_deg = self.indeg_tensor(in_deg_np)

        logits = self.score_candidates(state, node_ids, cand_ids, cand_mask, grad=False,
                                       ctx=ctx, in_deg=in_deg)

        if greedy:
            chosen = logits.topk(self.n_swap, dim=-1).indices
        else:
            chosen = (logits + self.gumbel_like(logits)).topk(self.n_swap, dim=-1).indices
        new_nb_ids = torch.gather(cand_ids, 1, chosen)

        # The no-op gate. Sampled here so the environment only sees the swaps of the
        # nodes that actually chose to act; the rest keep their edges untouched and
        # their reward is therefore exactly 0 by construction rather than measured.
        if self.use_noop:
            with torch.no_grad(), torch.cuda.amp.autocast():
                act_logits = self.agent.get_act_logits(node_ids, state=state,
                                                       device=self.device,
                                                       adj_ids=adj_ids, adj_mask=adj_mask,
                                                       in_deg=in_deg).float()
            if greedy:
                act = act_logits > 0
            else:
                act = torch.rand_like(act_logits) < torch.sigmoid(act_logits)
        else:
            act_logits = torch.zeros_like(node_ids, dtype=torch.float32)
            act = torch.ones_like(act_logits, dtype=torch.bool)

        # Which current neighbours to give up. Scored by the same pair scorer, so one
        # edge-quality function serves both halves: a high score means "worth having",
        # hence likely to be ADDED and unlikely to be DROPPED. (adj_ids/adj_mask/ctx
        # were built above, before the addition side needed the same rows.)
        if self.drop_mode == 'random':
            # Uniform over valid slots. Untrained, but its log-prob is a constant that
            # cancels in the PPO ratio, so it is excluded from old_logp entirely.
            drop_logits = torch.zeros_like(adj_mask, dtype=torch.float32)
        else:
            nb_logits = self.score_candidates(state, node_ids, adj_ids, adj_mask,
                                              grad=False, ctx=ctx, in_deg=in_deg)
            # Negate FIRST, then re-mask: score_candidates leaves padding at -inf,
            # which negation would turn into +inf and topk would then pick padding.
            drop_logits = -nb_logits
        drop_logits = drop_logits.masked_fill(~adj_mask, float('-inf'))

        if self.drop_mode == 'argmin' or (greedy and self.drop_mode == 'policy'):
            drop_slots = drop_logits.topk(self.n_swap, dim=-1).indices
        else:
            # Gumbel-top-k, the same sampler the addition side uses. Sampling rather
            # than argmin is what makes this trainable at all: an argmin puts all mass
            # on one outcome, so there is no distribution to move.
            drop_slots = (drop_logits
                          + self.gumbel_like(drop_logits)).topk(self.n_swap, dim=-1).indices
        drop_nb_ids = torch.gather(adj_ids, 1, drop_slots)

        # Joint log-prob of (gate decision, and the draws only if it acted). The three
        # factorize, so this is exact: log P(act) + log P(adds | act) + log P(drops |
        # act) for acting nodes, log P(no-op) for the rest. A no-op node's action is a
        # real decision with a real log-prob, so it belongs in the PPO ratio like any
        # other. The drop term is omitted unless the drops are a trained decision: for
        # 'argmin' and 'random' it is a constant in theta and cancels in the ratio.
        gate_logp = F.logsigmoid(torch.where(act, act_logits, -act_logits))
        old_logp = gate_logp + self.masked_logp(act, self.plackett_luce_logp(logits, chosen))
        if self.drop_mode == 'policy':
            old_logp = old_logp + self.masked_logp(
                act, self.plackett_luce_logp(drop_logits, drop_slots))
        if self.len_head:
            # The length choice is a real action with a real log-prob, so it enters the
            # ratio like the others. It is NOT gated by `act`: p_u was drawn (and shaped
            # the menu) before the no-op gate was consulted, so masking it by act would
            # make old_logp and the update's logp factorize differently.
            old_logp = old_logp + self.gaussian_logp(p_sample, p_mean_t, self.len_sigma)
            # The decisive training diagnostic. p_sd ~ 0 means the head stayed at the
            # global rule (a legitimate negative); p_sd growing means it is trying to
            # differentiate nodes, which is what Stage 1 is testing for.
            self.writer.add_scalar('train/len/p_mean', float(p_mean_t.mean()),
                                   global_step=self.step)
            self.writer.add_scalar('train/len/p_sd',
                                   float(p_mean_t.std()) if p_mean_t.numel() > 1 else 0.0,
                                   global_step=self.step)

        act_np = act.cpu().numpy()
        return dict(
            node_ids=node_ids, node_ids_np=node_ids_np,
            cand_ids=cand_ids, cand_mask=cand_mask,
            chosen=chosen,
            # The update re-scores the neighbour rows, so it needs the ids/mask that
            # were scored here -- adj changes as soon as the swap is applied.
            adj_ids=adj_ids, adj_mask=adj_mask, drop_slots=drop_slots,
            # s_t's in-degrees, for the same reason adj_ids is carried: the update
            # must score the pairs under the state the action was sampled from.
            in_deg_np=in_deg_np,
            act=act, act_np=act_np,
            # D2: the sampled percentile, needed to recompute its log-prob in the update.
            p_sample=p_sample,
            # The environment must only be handed the acting nodes' swaps.
            swap_node_ids_np=node_ids_np[act_np],
            new_nb_ids=new_nb_ids.cpu().numpy()[act_np],
            drop_nb_ids=drop_nb_ids.cpu().numpy()[act_np],
            old_logp=old_logp.detach(),
        )

    def probe_costs(self, probe_queries, probe_gt):
        """ Deterministic search on the current graph -> per-query performance R(s).

        c(s) = -R(s), so r_t = c(s_t) - c(s_{t+1}) = R(s_{t+1}) - R(s_t): the reward
        is the improvement produced by the swap. See ProximityDCSReward.cost_scalar.

        :return: (rewards [nq], res dict, terms dict) -- terms holds the unweighted
                 r_target / r_dense / r_cost arrays for diagnostics, or None if the
                 reward does not decompose.
        """
        res = self.hnsw.search_deterministic(probe_queries)
        kw = dict(best_vertex_ids=res['best_vertex_ids'],
                  ground_truth_ids=probe_gt,
                  total_distance_computations=res['total_distance_computations'],
                  trajectories=res['trajectories'],
                  num_hops=res['num_hops'],
                  # ProximityDCSReward(dense='prox') scores how close the ANSWERS are,
                  # so it needs the query vectors, not just the returned ids.
                  queries=probe_queries)
        if hasattr(self.reward, 'reward_terms_batch'):
            terms = self.reward.reward_terms_batch(**kw)
            return terms['total'], res, terms
        return self.reward.reward_batch(**kw), res, None

    def resolve_acceptance(self, action, rewards, observed, delta, undo,
                           pre_rewards, pre_res, pre_terms,
                           post_rewards, post_res, post_terms, probe_refresh):
        """ Keep or roll back the committing trial, per self.accept.

        Without this the transition is unconditional: every sampled edit lands and stays
        no matter what the probe measured, so from a graph that is already a local
        optimum of the bounded swap the only possible trajectory is downhill. Measured on
        NSW: 300 steps x 512 nodes x ~0.87 acting is ~134k committed edits, each on
        average harmful, for -0.045 recall@10.

        This changes the MDP rather than just the bookkeeping: the transition becomes
        s_{t+1} = accepted ? edited : s_t, i.e. a proposal-acceptance process. The reward
        the policy learns from is still the measured delta of the PROPOSAL, so a rejected
        proposal is still a training signal -- it just no longer damages the graph.

        :returns: fraction of acting nodes whose edit was kept
        """
        if self.accept == 'off':
            # s_{t+1} is the new committed state, so its cost is the next pre-cost.
            self._probe_cache = (post_rewards, post_res, post_terms) if probe_refresh else None
            return 1.0

        n_acting = max(int(np.asarray(action['act_np']).sum()), 1)
        if self.accept == 'step':
            # All-or-nothing on the aggregate probe delta. This is the only mode with a
            # clean guarantee: the state that survives is exactly the one measured to be
            # better, so the probe cost is monotone non-increasing by construction.
            if float(delta.mean()) > 0:
                self._probe_cache = (post_rewards, post_res, post_terms) if probe_refresh else None
                return 1.0
            if undo is not None:
                self.hnsw.restore(undo)
            return 0.0  # graph is s_t again, so _probe_cache still describes it

        # self.accept == 'node': keep only the nodes whose own credited reward was
        # positive. Higher throughput than 'step' (from a local optimum the JOINT edit of
        # 512 nodes almost never improves, so 'step' rejects nearly everything), but it
        # carries NO monotonicity guarantee: credit_nodes attributes a jointly-measured
        # delta to individual nodes, so a subset's true effect is not the sum of parts.
        keep = np.asarray(observed) & (np.asarray(rewards) > 0)
        node_ids_np = action['node_ids_np']
        revert_ids = set(int(v) for v in node_ids_np[~keep])
        # A node with no probe evidence is reverted too: "not measured to help" is the
        # conservative reading, and keeping it would be pure unmeasured drift.
        partial = {v: nbrs for v, nbrs in (undo or {}).items() if v in revert_ids}
        if partial:
            self.hnsw.restore(partial)
        kept = int(keep.sum())
        if not partial:
            # Nothing was reverted, so the graph really is s_{t+1} and post still
            # describes it. Worth special-casing: at a high accept rate this is the
            # common path and it halves the searches.
            self._probe_cache = (post_rewards, post_res, post_terms) if probe_refresh else None
        else:
            # After a partial revert the graph is neither s_t nor s_{t+1}, so BOTH
            # cached costs are stale. Invalidating forces a fresh pre-cost search next
            # step -- the price of this mode, and why 'step' stays cheaper.
            self._probe_cache = None
        return kept / n_acting

    def credit_nodes(self, node_ids_np, delta, pre_res, post_res, act=None):
        """ framework.md step 6, "local" performance difference.

        A node only gets credit from probes whose walk touched it, before or after
        the swap. Nodes no probe visited are masked out of the update entirely --
        their reward is unobserved, not zero.

        :param delta: per-query r_t = R(s_{t+1}) - R(s_t), float array [nq]
        :return: (rewards [B] float32, observed [B] bool)
        """
        num_vertices = self.hnsw.num_vertices
        accum = np.zeros(num_vertices, dtype=np.float64)
        counts = np.zeros(num_vertices, dtype=np.float64)
        pad = self.hnsw.service_labels['pad']

        for res in (pre_res, post_res):
            traj = res['trajectories']
            hops = np.asarray(res['num_hops'])
            # A vertex is popped at most once per search, so no row holds duplicates
            # and we can scatter-add the whole batch at once.
            valid = (np.arange(traj.shape[1])[None, :] < hops[:, None]) & (traj != pad)
            visited = traj[valid].astype(np.int64)
            if visited.size == 0:
                continue
            weights = np.broadcast_to(delta[:, None], traj.shape)[valid]
            accum += np.bincount(visited, weights=weights, minlength=num_vertices)
            counts += np.bincount(visited, minlength=num_vertices)

        counts_b = counts[node_ids_np]
        observed = counts_b > 0
        rewards = np.zeros(len(node_ids_np), dtype=np.float32)
        rewards[observed] = (accum[node_ids_np][observed] / counts_b[observed]).astype(np.float32)

        # A node that chose the no-op changed nothing, so its contribution to delta is
        # exactly 0 -- known by construction, not estimated from probe walks. Marking
        # it observed is what lets PPO compare "act (measured, usually negative from a
        # local optimum)" against "do nothing (exactly 0)" inside one normalization.
        # A no-op node that probes DID visit gets its measured value overwritten with
        # 0 on purpose: that value was contamination from the other nodes edited this
        # step, since this node's own swap did not happen.
        probe_observed = observed
        if act is not None:
            noop = ~np.asarray(act, dtype=bool)
            rewards[noop] = 0.0
            observed = probe_observed | noop

        # How many probe walks backed each node's reward. A node credited by a single
        # probe has a reward dominated by the other nodes edited this step, not by its
        # own swap -- the update weighs it the same as a node seen by thousands.
        # Measured on probe_observed, not observed: a no-op node has 0 visits by
        # definition, and folding those zeros in would drag visits_mean down and push
        # frac_visits_le2 to ~1 no matter how well the probe covers the acting nodes.
        if probe_observed.any():
            seen = counts_b[probe_observed]
            self.writer.add_scalar('train/credit/visits_mean', float(seen.mean()),
                                   global_step=self.step)
            self.writer.add_scalar('train/credit/visits_median', float(np.median(seen)),
                                   global_step=self.step)
            self.writer.add_scalar('train/credit/frac_visits_le2', float((seen <= 2).mean()),
                                   global_step=self.step)
            # Probe coverage of the sampled nodes, kept comparable across runs with and
            # without the no-op; frac_observed_total is the share that actually enters
            # the update (probe-covered acting nodes plus all no-op nodes).
            self.writer.add_scalar('train/credit/frac_observed',
                                   float(probe_observed.mean()), global_step=self.step)
            self.writer.add_scalar('train/credit/frac_observed_total',
                                   float(observed.mean()), global_step=self.step)
        return rewards, observed

    def invalidate_probe_cache(self):
        """ Forget the cached cost of the current graph.

        Must be called whenever the probe set changes: a cost measured on other
        queries is not comparable, so reusing it would corrupt every reward.
        """
        self._probe_cache = None

    def log_diagnostics(self, post_res, post_terms, pre_terms=None):
        """ Per-term reward breakdown + search-shape stats.

        The composite reward R = r_target + alpha*r_dense + beta*r_cost is only safe
        while its terms agree about what "better" means. Once the action edits the
        graph, they can disagree: r_cost rewards fewer distance computations, and a
        degenerate graph earns that by making the search terminate early -- often on
        a short, straight walk into a local minimum, which r_dense also rewards. Only
        r_target actually pins the optimum to "find the true neighbour".

        So the signature of a reward being gamed is r_cost (and usually r_dense)
        climbing while r_target is flat or falling. Watch train/reward_terms/* and
        train/search/num_hops together; recall going down while total reward goes up
        means alpha/beta need to be cut, not tuned.
        """
        hops = np.asarray(post_res['num_hops'], dtype=np.float64)
        dcs = np.asarray(post_res['total_distance_computations'], dtype=np.float64)
        self.writer.add_scalar('train/search/num_hops', float(hops.mean()), global_step=self.step)
        self.writer.add_scalar('train/search/dcs', float(dcs.mean()), global_step=self.step)
        # A collapsing graph shows up here first: the walk dies after a hop or two.
        self.writer.add_scalar('train/search/frac_hops_le2', float((hops <= 2).mean()),
                               global_step=self.step)

        if post_terms is None:
            return
        for name, values in post_terms.items():
            self.writer.add_scalar('train/reward_terms/' + name, float(np.mean(values)),
                                   global_step=self.step)
        # r_target is the only term whose optimum is "answer the query correctly",
        # so log it unweighted as the headline metric: this IS recall@k.
        self.writer.add_scalar('train/recall', float(np.mean(post_terms['r_target'])),
                               global_step=self.step)
        if pre_terms is not None:
            # Per-term contribution to this step's reward. If the swap is being
            # driven by cost rather than recall, delta_r_cost dominates.
            for name in ('r_target', 'r_dense', 'r_cost'):
                weight = 1.0 if name == 'r_target' else \
                    (self.reward.alpha if name == 'r_dense' else self.reward.beta)
                delta_term = float(np.mean(post_terms[name] - pre_terms[name]))
                self.writer.add_scalar('train/reward_delta/' + name, delta_term,
                                       global_step=self.step)
                self.writer.add_scalar('train/reward_delta/' + name + '_weighted',
                                       weight * delta_term, global_step=self.step)

    def train_step(self, probe_queries, probe_ground_truth_ids, probe_refresh=True, **kwargs):
        """ One iteration t of framework.md: steps 3 (features) -> 7 (RL update).

        With commit_stride > 1 the swap is applied *tentatively*: the post-swap cost
        is measured, the reward is credited and the policy updated, and then the graph
        is rolled back to s_t. Only the last trial of each stride is kept, so the state
        advances once per commit_stride steps while every trial still trains the agent.
        Because the graph is identical across a stride's trials, they all share one
        pre-swap cost and are effectively independent rollouts from the same s_t.

        :param probe_queries: queries used to measure c(s), [probe_size, vertex_size]
        :param probe_ground_truth_ids: their true neighbours, [probe_size, >=k]
        :param probe_refresh: carry the committed graph's cost across commits instead
               of re-measuring it, halving the searches per step. Only valid while the
               probe set is unchanged. Within a stride the cost is always reused, since
               a reverted trial provably leaves the graph untouched.
        :returns: mean reward over credited nodes, or None if the step was skipped
        """
        # step 3: one full-graph MLP forward gives features H of every node
        self.agent.to(device=self.device)
        state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)

        # step 4: sample the nodes to edit and their swaps
        num_nodes = min(self.nodes_per_step, self.hnsw.num_vertices)
        node_ids_np = np.random.choice(self.hnsw.num_vertices, size=num_nodes, replace=False)
        action = self.sample_swaps(state, node_ids_np, **kwargs)
        if action is None:
            self.step += 1
            return None

        # step 6a: cost of s_t. Cached across a stride's trials, and across commits
        # too when probe_refresh is on.
        if self._probe_cache is not None:
            pre_rewards, pre_res, pre_terms = self._probe_cache
        else:
            pre_rewards, pre_res, pre_terms = self.probe_costs(probe_queries, probe_ground_truth_ids)
            self._probe_cache = (pre_rewards, pre_res, pre_terms)

        # step 5: environment transition s_t -> s_{t+1}, tentatively unless this trial
        # closes the stride. Snapshot first so a rejected trial can be rolled back.
        self._trials_since_commit += 1
        commit = self._trials_since_commit >= self.commit_stride
        # Only the nodes whose gate fired touch the graph, so only those need
        # snapshotting -- and if none fired, s_{t+1} == s_t and delta is exactly 0.
        swap_nodes = action['swap_node_ids_np']
        # With an acceptance test a COMMITTING trial can also be rolled back, so the
        # snapshot can no longer be skipped on the strength of `commit` alone.
        undo = None if (commit and self.accept == 'off') \
            else self.hnsw.snapshot(swap_nodes)
        num_changed = self.hnsw.apply_swaps(swap_nodes, action['drop_nb_ids'],
                                            action['new_nb_ids']) if len(swap_nodes) else 0

        # step 6b: cost of s_{t+1}, and r_t = c(s_t) - c(s_{t+1}) = R(s_{t+1}) - R(s_t)
        post_rewards, post_res, post_terms = self.probe_costs(probe_queries, probe_ground_truth_ids)
        delta = post_rewards - pre_rewards

        # s_{t+1}'s neighbour rows for the TD target, read HERE because the restore()
        # below puts s_t back on a non-committing trial. Copy, not a view: self.adj is
        # updated in place, so a view would silently become s_t after the restore.
        if self.use_critic:
            action['next_adj_np'] = self.hnsw.adj[action['node_ids_np']].copy()

        # Moved ahead of the commit decision because per-node acceptance needs the
        # per-node rewards to decide with. It is a pure function of the two searches
        # and the sampled action, so it does not care what the graph currently holds.
        rewards, observed = self.credit_nodes(action['node_ids_np'], delta, pre_res,
                                              post_res, act=action['act_np'])

        accept_frac = 1.0
        if not commit:
            self.hnsw.restore(undo)  # back to s_t; _probe_cache still describes it
        else:
            self._trials_since_commit = 0
            accept_frac = self.resolve_acceptance(action, rewards, observed, delta, undo,
                                                  pre_rewards, pre_res, pre_terms,
                                                  post_rewards, post_res, post_terms,
                                                  probe_refresh)
        self.writer.add_scalar('train/accept_frac', accept_frac, global_step=self.step)
        self.writer.add_scalar('train/noop/frac_acting',
                               float(action['act_np'].mean()), global_step=self.step)

        self.writer.add_scalar('train/graph_reward_delta', float(delta.mean()), global_step=self.step)
        self.writer.add_scalar('train/graph_performance', float(post_rewards.mean()), global_step=self.step)
        self.writer.add_scalar('train/edges_changed', num_changed, global_step=self.step)
        self.writer.add_scalar('train/edges_committed', num_changed if commit else 0, global_step=self.step)
        self.writer.add_scalar('train/nodes_credited', int(observed.sum()), global_step=self.step)
        self.log_diagnostics(post_res, post_terms, pre_terms)

        if not observed.any():
            self.step += 1
            return None

        # Warmup phase: let the all-zero baseline fill in before trusting advantages
        if self.warmup_steps > 0 and self.step < self.warmup_steps:
            observed_rewards = torch.as_tensor(rewards[observed], dtype=torch.float32,
                                               device=self.device)
            node_ids = action['node_ids'][torch.as_tensor(observed, dtype=torch.bool,
                                                          device=self.device)]
            self.baseline.update(
                rewards=observed_rewards,
                session_index=torch.arange(node_ids.numel(), dtype=torch.int64, device=self.device),
                query_index=node_ids.cpu(), device=self.device)
            self.step += 1
            return float(rewards[observed].mean())

        # step 7: RL update on the credited nodes only
        mean_reward = self.train_on_batch(action=action, rewards=rewards, observed=observed, **kwargs)
        self.step += 1
        return mean_reward

    def train_on_batch(self, action, rewards, observed, **kwargs):
        """ framework.md step 7: clipped-surrogate update of the Rule-Picker.

        The policy is the per-node candidate softmax; the log-prob of an action is
        the Plackett-Luce log-prob of the n_swap draws, so the PPO ratio is
        exp(logp_new - logp_old) exactly as in the per-edge case.
        """
        keep = torch.as_tensor(observed, dtype=torch.bool, device=self.device)
        node_ids = action['node_ids'][keep]
        cand_ids = action['cand_ids'][keep]
        cand_mask = action['cand_mask'][keep]
        chosen = action['chosen'][keep]
        old_logp = action['old_logp'][keep]
        act = action['act'][keep]
        # The neighbour rows as they were when the drops were sampled. Re-reading
        # hnsw.adj here would read s_{t+1}, whose rows no longer contain the dropped
        # edges, so the log-prob of the action taken would be unrecoverable.
        adj_ids = action['adj_ids'][keep]
        adj_mask = action['adj_mask'][keep]
        drop_slots = action['drop_slots'][keep]
        p_sample_b = action['p_sample'][keep] if self.len_head else None
        # NOT indexed by `keep`: this is a whole-graph vector indexed by node id, and
        # the scorer looks up arbitrary candidate ids in it, not just the kept rows.
        in_deg = self.indeg_tensor(action.get('in_deg_np'))
        rewards_t = torch.as_tensor(rewards[observed], dtype=torch.float32, device=self.device)

        num_nodes = node_ids.size(0)
        # SessionBaseline keyed by node id: one sample per node, so session_index is
        # just a contiguous range and query_index carries the node ids. It follows
        # the same device split as BaseAlgorithm.get_session_batch -- session_index
        # and rewards on the compute device (it bincounts them and indexes a device
        # tensor with them), query_index on CPU (it indexes the CPU EMA buffer).
        session_index = torch.arange(num_nodes, dtype=torch.int64, device=self.device)
        query_index = node_ids.cpu()
        base_kwargs = dict(rewards=rewards_t, session_index=session_index,
                           query_index=query_index, device=self.device)

        baseline = self.baseline.get(**base_kwargs)
        mean_reward = self.baseline.update(**base_kwargs)
        nonzero_baseline = self.baseline.get_nonzero_baselines() \
            if hasattr(self.baseline, 'get_nonzero_baselines') else 0.0

        value_target = None
        if self.use_critic:
            # TD(0): A_i = r_i + gamma * V(i, s_{t+1}) - V(i, s_t), with the target
            # detached so the critic's own error does not flow into the policy.
            # Both states use the SAME z (prepare_state ignores the adjacency), so the
            # entire difference between V(i,s_t) and V(i,s_{t+1}) comes from the
            # neighbour rows -- which is precisely why get_values takes them as args.
            next_adj = torch.as_tensor(action['next_adj_np'], dtype=torch.long,
                                       device=self.device)[keep]
            pad = self.hnsw.service_labels['pad']
            next_mask = next_adj != pad
            with torch.no_grad():
                v_state = self.agent.prepare_state(self.hnsw.graph, device=self.device,
                                                   training=False)
                v_next = self.agent.get_values(node_ids, next_adj, next_mask,
                                               state=v_state, device=self.device)
                value_target = (rewards_t + self.gamma * v_next).detach()
                v_now = self.agent.get_values(node_ids, adj_ids, adj_mask,
                                              state=v_state, device=self.device)
            advantage = (value_target - v_now).detach()
            self.writer.add_scalar('train/value_mean', float(v_now.mean()),
                                   global_step=self.step)
            self.writer.add_scalar('train/value_next_mean', float(v_next.mean()),
                                   global_step=self.step)
            # The decisive critic diagnostic. z is topology-free, so if the neighbour
            # mean carries nothing this is ~0 and the TD target degenerates to
            # r + (gamma-1)*V, i.e. the critic is a per-node constant and gamma is
            # inert. Non-zero is necessary (not sufficient) for gamma to mean anything.
            self.writer.add_scalar('train/value_td_shift',
                                   float((v_next - v_now).abs().mean()),
                                   global_step=self.step)
        elif self.adv_ref == 'noop':
            # Reference 0, i.e. the no-op's known reward -- so the EMA baseline is
            # skipped too. Subtracting it would put the reference back at "this node's
            # own recent average", which destroys the sign just as centering does: a
            # node whose edits are consistently harmful would still show a positive
            # advantage as soon as one edit was less harmful than its average.
            advantage = rewards_t.detach()
        else:
            advantage = (rewards_t - baseline.to(self.device)).detach()
        adv_mean_log = advantage.mean().item()
        # Spread BEFORE normalization. The division below rescales to unit variance,
        # so this is the only place the true signal strength is visible: if it is at
        # or below the 1e-8 epsilon, normalization divides ~0 by ~0 and the policy
        # gradient dies no matter how healthy the optimizer looks.
        adv_std_raw = advantage.std(unbiased=False).item() if advantage.numel() > 1 else 0.0
        frac_zero = float((rewards_t == 0).float().mean().item())
        self.writer.add_scalar('train/advantage_std_raw', adv_std_raw, global_step=self.step)
        self.writer.add_scalar('train/reward_frac_exactly_zero', frac_zero,
                               global_step=self.step)
        if self.adv_ref == 'noop':
            # Scale WITHOUT re-centering, so the sign of the advantage survives.
            #
            # Subtracting the batch mean forces mean(advantage) == 0, which makes
            # roughly half the batch positive NO MATTER WHAT -- including when every
            # sampled edit was harmful. Measured on NSW: mean_reward is ~-0.001 at every
            # step, i.e. all actions are bad, yet PPO still reinforces "the least-bad
            # half" and recall falls 0.045 over 300 steps.
            #
            # The no-op is the fixed reference that makes the absolute sign meaningful:
            # its reward is exactly 0 by construction (a node that changed nothing
            # contributed nothing), not an estimate. So r_i > 0 means "better than
            # leaving this node alone" and r_i < 0 means "worse", and only the former
            # should be reinforced. Centering destroys exactly that distinction.
            #
            # A reference that does not depend on the sampled action leaves the policy
            # gradient unbiased, and 0 qualifies -- it is a property of the no-op branch,
            # not of what was drawn.
            advantage = advantage / (advantage.std(unbiased=False) + 1e-8)
        else:
            # unbiased std is NaN for a single sample, which can happen when only one
            # node in the batch was credited by the probe searches.
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        self.writer.add_scalar('train/advantage_frac_positive',
                               float((advantage > 0).float().mean()), global_step=self.step)

        total_loss = total_ent = total_kl = 0.0
        # Accumulated apart from total_ent so the two bonuses can be read separately.
        # This NARROWS train/entropy: before 2026-08-06 it silently included the drop
        # term, so its magnitude (and dump_diag.py's HEALTH threshold) referred to a sum
        # over two distributions with a correspondingly higher uniform ceiling.
        total_drop_ent = 0.0
        n_drop_updates = 0
        total_value_loss = 0.0
        n_value_updates = 0
        n_updates = 0
        # Optimizer-health probes. A flat KL says the policy never moved, but not why:
        # zero/absent gradient, a GradScaler silently skipping every step on inf/nan,
        # or a real-but-tiny gradient losing to the entropy bonus all look identical
        # from the outside. These three separate them.
        params = [p for p in self.agent.parameters() if p.requires_grad]
        before = torch.cat([p.detach().reshape(-1) for p in params]) if params else None
        grad_norms, policy_losses = [], []
        steps_taken = steps_skipped = 0

        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(num_nodes, device=self.device)
            epoch_kl = 0.0

            # Re-encode once per epoch; mini-batches accumulate gradients into it.
            state = self.agent.prepare_state(self.hnsw.graph, device=self.device, training=True, **kwargs)
            self.optimizer.zero_grad()

            for start in range(0, num_nodes, self.nodes_in_batch):
                idx = perm[start:start + self.nodes_in_batch]
                batch_frac = idx.numel() / num_nodes

                # Recomputed from the STORED s_t rows, not from the live graph: the swap
                # may already be committed by now, and the ratio has to compare two
                # policies evaluated on the same state.
                ctx_b = self.node_ctx(state, node_ids[idx], adj_ids[idx], adj_mask[idx],
                                      grad=True)
                logits = self.score_candidates(state, node_ids[idx], cand_ids[idx],
                                               cand_mask[idx], grad=True, ctx=ctx_b,
                                               in_deg=in_deg)
                # Must mirror sample_swaps' factorization exactly, or the ratio
                # exp(logp - old_logp) compares two different distributions.
                act_b = act[idx]
                if self.use_noop:
                    with torch.cuda.amp.autocast():
                        act_logits = self.agent.get_act_logits(
                            node_ids[idx], state=state, device=self.device,
                            adj_ids=adj_ids[idx], adj_mask=adj_mask[idx], in_deg=in_deg)
                    act_logits = act_logits.float()
                    gate_logp = F.logsigmoid(torch.where(act_b, act_logits, -act_logits))
                else:
                    gate_logp = torch.zeros_like(old_logp[idx])
                logp = gate_logp + self.masked_logp(
                    act_b, self.plackett_luce_logp(logits, chosen[idx]))
                drop_logits = None
                if self.drop_mode == 'policy':
                    # Same scorer, same negate-then-mask order as sample_swaps.
                    nb_logits = self.score_candidates(state, node_ids[idx], adj_ids[idx],
                                                      adj_mask[idx], grad=True, ctx=ctx_b,
                                                      in_deg=in_deg)
                    drop_logits = (-nb_logits).masked_fill(~adj_mask[idx], float('-inf'))
                    logp = logp + self.masked_logp(
                        act_b, self.plackett_luce_logp(drop_logits, drop_slots[idx]))
                if self.len_head:
                    # Mirrors sample_swaps exactly: ungated, same p0/span/sigma, and
                    # scored on the STORED s_t neighbour rows so the ratio compares two
                    # policies on one state.
                    p_mean_b = self.len_p_mean(state, node_ids[idx], adj_ids[idx],
                                               adj_mask[idx], grad=True)
                    logp = logp + self.gaussian_logp(p_sample_b[idx], p_mean_b,
                                                     self.len_sigma)

                if epoch == 0 and start == 0:
                    # Softmax only sees logit *differences*, so the within-row spread
                    # is what decides whether the policy can express a preference at
                    # all. Near zero => uniform distribution => entropy sits at its
                    # maximum, where its gradient also vanishes.
                    with torch.no_grad():
                        finite = logits[cand_mask[idx]]
                        rows = logits.masked_fill(~cand_mask[idx], float('nan'))
                        row_std = (rows - rows.nanmean(-1, keepdim=True)) ** 2
                        self.writer.add_scalar('train/logit_row_std',
                                               float(row_std.nanmean().sqrt().item()),
                                               global_step=self.step)
                        if finite.numel():
                            self.writer.add_scalar('train/logit_mean',
                                                   float(finite.mean().item()),
                                                   global_step=self.step)
                        # Embedding spread relative to embedding magnitude. A tiny
                        # ratio means every node encodes to nearly the same point,
                        # so no scorer built on those embeddings can separate
                        # candidates -- which shows up downstream as a uniform
                        # softmax that looks identical to a badly tuned temperature.
                        # (This is exactly how the feat_mean/feat_std scale
                        # mismatch presented: ratio ~1e-03, cosine pinned at 1.0.)
                        z = state.vertices
                        per_dim = z.std(dim=0).mean()
                        magnitude = z.norm(dim=-1).mean() / (z.shape[-1] ** 0.5)
                        self.writer.add_scalar('train/embed_spread_ratio',
                                               float((per_dim / (magnitude + 1e-12)).item()),
                                               global_step=self.step)
                        scale = getattr(self.agent, 'logit_scale', None)
                        if scale is not None:
                            self.writer.add_scalar('train/logit_scale',
                                                   float(scale.item()),
                                                   global_step=self.step)

                ratio = torch.exp(logp - old_logp[idx])
                adv = advantage[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy of the candidate distribution, over valid slots only.
                logprobs = torch.log_softmax(logits, dim=-1)
                probs = logprobs.exp()
                cand_ent = -(probs * logprobs.masked_fill(~cand_mask[idx], 0.0)).sum(-1)
                # The drop softmax is a trained distribution too, so it also needs a
                # bonus to keep it from collapsing onto one neighbour (which is the
                # frozen argmin this replaced). Kept as its OWN term rather than folded
                # into cand_ent: at a shared coefficient the bonus pushes the drop
                # distribution toward uniform, and uniform IS the 'random' baseline that
                # drop_mode='policy' exists to beat, so the two could not be separated.
                drop_ent = None
                if drop_logits is not None:
                    d_logprobs = torch.log_softmax(drop_logits, dim=-1)
                    drop_ent = -(d_logprobs.exp()
                                 * d_logprobs.masked_fill(~adj_mask[idx], 0.0)).sum(-1)
                if self.use_noop:
                    # Bernoulli entropy of the gate, plus the candidate entropy only
                    # where a draw actually happens -- matching the log-prob split.
                    # Without the gate term the entropy bonus cannot keep the gate
                    # from saturating to all-act or all-noop.
                    p_act = torch.sigmoid(act_logits)
                    gate_ent = -(p_act * F.logsigmoid(act_logits)
                                 + (1 - p_act) * F.logsigmoid(-act_logits))
                    ent_loss = (gate_ent + p_act * cand_ent).mean()
                    # Same p_act gating: a node that did not act performed no drop.
                    drop_ent_loss = ((p_act * drop_ent).mean()
                                     if drop_ent is not None else None)
                else:
                    ent_loss = cand_ent.mean()
                    drop_ent_loss = drop_ent.mean() if drop_ent is not None else None

                loss = policy_loss - self.entropy_reg * ent_loss
                if drop_ent_loss is not None:
                    loss = loss - self.drop_entropy_reg * drop_ent_loss
                value_loss = None
                if value_target is not None:
                    # Fitted on s_t's rows: V(i, s_t) is what the target is a target
                    # FOR. Recomputed per epoch so the critic actually trains, rather
                    # than reusing the no-grad v_now from the advantage above.
                    v_pred = self.agent.get_values(node_ids[idx], adj_ids[idx],
                                                   adj_mask[idx], state=state,
                                                   device=self.device)
                    value_loss = F.mse_loss(v_pred, value_target[idx])
                    loss = loss + self.value_coef * value_loss
                self.scaler.scale(loss * batch_frac).backward(retain_graph=True)

                with torch.no_grad():
                    kl = (old_logp[idx] - logp.detach()).mean()

                total_loss += loss.item()
                total_ent += ent_loss.item()
                if value_loss is not None:
                    total_value_loss += value_loss.item()
                    n_value_updates += 1
                if drop_ent_loss is not None:
                    total_drop_ent += drop_ent_loss.item()
                    n_drop_updates += 1
                total_kl += kl.item()
                epoch_kl += kl.item() * batch_frac
                policy_losses.append(policy_loss.item())
                n_updates += 1

            # Unscale before reading the norm, otherwise it reflects the AMP loss
            # scale (which is ~65536) rather than the true gradient magnitude.
            self.scaler.unscale_(self.optimizer)
            # clip_grad_norm_ returns the norm measured BEFORE clipping, so this
            # both clips and reports. max_grad_norm=None disables clipping while
            # still reporting (float('inf') never clips).
            max_norm = float('inf') if self.max_grad_norm is None else self.max_grad_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm).item() \
                if params else 0.0
            grad_norms.append(grad_norm)

            # scaler.step() is a no-op when the unscaled grads hold inf/nan. Comparing
            # the scale across the call is the only way to observe that from outside.
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < scale_before:
                steps_skipped += 1
            else:
                steps_taken += 1

            if self.target_kl is not None and epoch_kl > 1.5 * self.target_kl:
                break

        if n_updates > 0:
            self.writer.add_scalar('train/loss', total_loss / n_updates, global_step=self.step)
            self.writer.add_scalar('train/entropy', total_ent / n_updates, global_step=self.step)
            self.writer.add_scalar('train/kl', total_kl / n_updates, global_step=self.step)
            self.writer.add_scalar('train/policy_loss',
                                   float(np.mean(policy_losses)), global_step=self.step)
        if n_value_updates > 0:
            self.writer.add_scalar('train/value_loss',
                                   total_value_loss / n_value_updates, global_step=self.step)
        if n_drop_updates > 0:
            # Only logged under drop_mode='policy'. Read against log(degree_mean) --
            # sitting at that ceiling means the drop softmax is uniform, i.e.
            # indistinguishable from drop_mode='random' whatever the gradient says.
            self.writer.add_scalar('train/entropy_drop',
                                   total_drop_ent / n_drop_updates, global_step=self.step)
            self.writer.add_scalar('train/drop_entropy_reg', self.drop_entropy_reg,
                                   global_step=self.step)

        # The decisive numbers. grad_norm == 0 means nothing reached the parameters;
        # steps_skipped == ppo_epochs means AMP threw every update away.
        #
        # grad_norm is the one to trust: param_delta can be non-zero even with an
        # exactly-zero gradient, because Adam keeps stepping on its stored momentum.
        # A moving param_delta is therefore NOT evidence that learning is happening.
        if grad_norms:
            self.writer.add_scalar('train/grad_norm', float(np.mean(grad_norms)),
                                   global_step=self.step)
            self.writer.add_scalar('train/grad_norm_max', float(np.max(grad_norms)),
                                   global_step=self.step)
        self.writer.add_scalar('train/steps_taken', steps_taken, global_step=self.step)
        self.writer.add_scalar('train/steps_skipped', steps_skipped, global_step=self.step)
        self.writer.add_scalar('train/scaler_scale', self.scaler.get_scale(),
                               global_step=self.step)
        delta = delta_rel = 0.0
        if before is not None:
            after = torch.cat([p.detach().reshape(-1) for p in params])
            delta = (after - before).norm().item()
            delta_rel = delta / (before.norm().item() + 1e-12)
            self.writer.add_scalar('train/param_delta', delta, global_step=self.step)
            # Relative movement, so the number is comparable across layer widths.
            self.writer.add_scalar('train/param_delta_rel', delta_rel, global_step=self.step)

        # Exposed for the training loop to print, so the run is diagnosable without
        # opening TensorBoard.
        self.last_opt_diagnostics = dict(
            grad_norm=float(np.mean(grad_norms)) if grad_norms else 0.0,
            steps_taken=steps_taken, steps_skipped=steps_skipped,
            scaler_scale=self.scaler.get_scale(),
            param_delta_rel=delta_rel,
            policy_loss=float(np.mean(policy_losses)) if policy_losses else 0.0,
            kl=total_kl / n_updates if n_updates else 0.0,
            entropy=total_ent / n_updates if n_updates else 0.0,
        )
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/nonzero_baseline', nonzero_baseline, global_step=self.step)
        self.writer.add_scalar('train/advantage', adv_mean_log, global_step=self.step)
        return mean_reward

    def evaluate(self, batch_queries, batch_ground_truth_ids, prefix='dev',
                 write_logs=True, ef=None, **kwargs):
        """ Deterministic evaluation of the current graph s_t.

        There is no per-edge sampling to evaluate here -- the policy's effect is
        entirely baked into the topology -- so we simply search the graph as it is.

        NB: unlike BaseAlgorithm.evaluate, which returns the bare mean reward, this
        returns the full counters dict (the ef sweep at the end of training reuses
        the recall/DCS entries). Callers wanting the scalar want
        counters[prefix + '/mean_reward'].
        """
        res = self.hnsw.search_deterministic(batch_queries, ef=ef)
        best = res['best_vertex_ids']
        dcs = res['total_distance_computations'].astype(np.float64)
        gt = np.asarray(batch_ground_truth_ids)
        k = best.shape[1]

        rewards = self.reward.reward_batch(
            best_vertex_ids=best, ground_truth_ids=gt,
            total_distance_computations=res['total_distance_computations'],
            trajectories=res['trajectories'], num_hops=res['num_hops'],
            queries=batch_queries,
        )
        recall_1 = (best[:, 0] == gt[:, 0]).astype(np.float64)
        matches = (best[:, :, None] == gt[:, None, :k]).any(-1).sum(-1) / float(k)

        counters = {
            prefix + '/mean_reward': float(rewards.mean()),
            prefix + '/recall@1': float(recall_1.mean()),
            prefix + '/distance_computations': float(dcs.mean()),
            prefix + '/num_hops': float(res['num_hops'].mean()),
            prefix + '/recall@1_per_distance_computation': float((recall_1 / dcs).mean()),
        }
        if k > 1:
            counters[prefix + '/recall@%i' % k] = float(matches.mean())
            counters[prefix + '/recall@%i_per_distance_computation' % k] = float((matches / dcs).mean())

        if write_logs:
            for key, value in counters.items():
                self.writer.add_scalar(key, value, global_step=self.step)
        return counters