import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from .Nodeformer import NodeFormer


class BaseAgent(nn.Module):
    # State is arbitrary information that agent needs to compute about its graph. Immutable.
    State = namedtuple("AgentState", ['vertices'])

    def prepare_state(self, graph, device='cpu', **kwargs):
        """ Pre-computes graph representation for further use in edge prediction """
        return self.State(vertices=graph.vertices.to(device=device))

    def predict_edges(self, vertex_id, neighbor_ids, state, **kwargs):
        """
        For each node in neighbor_ids, predicts whether it is available from vertex_idx
        :param vertex_id: vertex index (0-based)
        :param neighbor_ids: neighbor vertex indices, a list [num_neighbors]
        :param state: output of prepare_state function
        :return: 0/1 vector for each neighbor in neighbor ids
            1 if agent allows an edge between vertex_id and and that neighbor,
            0 if there is no edge
        """
        return [1] * len(neighbor_ids)


class ProbabilisticAgent(BaseAgent):
    """ Agent with:
        * `get_edge_logp` method which should be implemented in your successor class.
        * `predict_edges` which uses edges' logprobs to predict them.
            Normally, you shouldn't override it.
    """
    def get_edge_logp(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        """ Take vertices and predict probability of an edge between them. """
        raise NotImplementedError

    def get_pair_logits(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        """
        Raw (unsquashed) score for each (from, to) pair, shape [batch].

        The graph-edit MDP (lib.algorithm.GraphEditPPO) needs logits rather than
        the [batch, 2] log-probs of `get_edge_logp`: its policy is a softmax over
        a *set* of candidate neighbors per node, so the scores must be free to
        span the whole real line. Going through get_edge_logp would first squash
        each score into a per-edge probability clipped to [min_prob, max_prob],
        flattening exactly the differences the candidate softmax needs.

        The default implementation falls back to log p(edge) so any existing
        ProbabilisticAgent keeps working; subclasses with a natural logit should
        override (see MLPLinkAgent / SimpleNeuralAgent).
        """
        return self.get_edge_logp(from_vertex_ids, to_vertex_ids,
                                  state=state, device=device, **kwargs)[:, 1]

    def predict_edges(self, vertex_id, neighbor_ids, greedy=False, state=None, **kwargs):
        """
        For each node in neighbor_ids, predicts whether it is available from vertex_idx
        :param vertex_id: vertex index (0-based)
        :param neighbor_ids: neighbor vertex indices, a list [num_neighbors]
        :param logp_cache: precomputed logp for vertices in session
        :param greedy: whether to take argmax or sample according to probs
        :return: 0/1 vector for each neighbor in neighbor ids
            1 if agent allows an edge between vertex_id and and that neighbor,
            0 if there is no edge
        """
        with torch.no_grad():
            if vertex_id not in state.logp_cache.keys():
                edge_logp = self.get_edge_logp([vertex_id] * len(neighbor_ids), neighbor_ids, state=state, **kwargs)
                edge_logp = edge_logp.to(device='cpu')
                state.logp_cache[vertex_id] = edge_logp
            else:
                edge_logp = state.logp_cache[vertex_id]

            if greedy:
                return edge_logp.argmax(dim=-1)
            else:
                return torch.multinomial(torch.exp(edge_logp), 1)[:, 0]


class SimpleNeuralAgent(ProbabilisticAgent):
    """ Agent with a feedwforward neural network for edge prediction """
    def __init__(self, vertex_size, hidden_size, activation=nn.ELU(), min_prob=1e-4):
        super().__init__()

        self.min_prob, self.max_prob = min_prob, 1. - min_prob
        self.edge_network = nn.Sequential(
            nn.Linear(2 * vertex_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 1),
        )

    def get_edge_logp(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        """
        :param from_vertex_ids: indices of vertices from which there could be and edge, [batch_size]
        :param to_vertex_ids: indices of vertices to which there could be an edge, [batch_size]
        :return: log-probabilities of taking and not taking edge between vertex1 and vertex2,
            shape: [batch_size, 2]
        """
        vertices_from = state.vertices[from_vertex_ids, :].to(device=device)
        vertices_to = state.vertices[to_vertex_ids, :].to(device=device)

        nn_inputs = torch.cat([vertices_from, vertices_to], dim=-1)
        theta = torch.sigmoid(self.edge_network(nn_inputs))
        theta = theta * (self.max_prob - self.min_prob) + self.min_prob
        probs = torch.cat([theta, 1. - theta], dim=-1)
        return probs.log()

    def get_pair_logits(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        """Raw score (before sigmoid) for each (from, to) pair, shape [batch]."""
        vertices_from = state.vertices[from_vertex_ids, :].to(device=device)
        vertices_to = state.vertices[to_vertex_ids, :].to(device=device)
        nn_inputs = torch.cat([vertices_from, vertices_to], dim=-1)
        return self.edge_network(nn_inputs).squeeze(-1)


class NodeFormerAgent(ProbabilisticAgent):
    """
    Agent that encodes all graph nodes with NodeFormer once per step,
    then uses the resulting hidden vectors for edge prediction.
    State.vertices: [N, hidden_size] NodeFormer hidden representations.
    """
    State = namedtuple("AgentState", ['vertices'])

    def __init__(self, vertex_size, nf_hidden_size, mlp_hidden_size,
                 num_layers=2, num_heads=4,
                 nb_random_features=30, use_bn=True, use_residual=True,
                 min_prob=1e-4):
        super().__init__()
        self.min_prob = min_prob
        self.max_prob = 1. - min_prob

        # NodeFormer encodes raw features → hidden reps; no edge loss needed here
        self.encoder = NodeFormer(
            in_channels=vertex_size,
            hidden_channels=nf_hidden_size,
            out_channels=nf_hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            nb_random_features=nb_random_features,
            use_bn=use_bn,
            use_residual=use_residual,
            use_gumbel=True,
            use_edge_loss=False,
        )

        self.edge_network = nn.Sequential(
            nn.Linear(2 * nf_hidden_size, mlp_hidden_size),
            nn.ELU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ELU(),
            nn.Linear(mlp_hidden_size, 1),
        )

    def _build_edge_index(self, graph, device):
        src, dst = [], []
        for v, neighbors in graph.edges.items():
            for nb in neighbors:
                src.append(int(v))
                dst.append(int(nb))
        return (
            torch.tensor(src, dtype=torch.long, device=device),
            torch.tensor(dst, dtype=torch.long, device=device),
        )

    def prepare_state(self, graph, device='cuda', training=False, **kwargs):
        """Encode all nodes with NodeFormer; returned state.vertices are hidden reps."""
        x = graph.vertices.to(device)
        # edge_index = self._build_edge_index(graph, device)
        # adjs = [edge_index]
        adjs = None
        with torch.cuda.amp.autocast():
            if training:
                hidden = self.encoder(x, adjs).float()   # [N, hidden_size]
            else:
                self.eval()
                with torch.no_grad():
                    hidden = self.encoder(x, adjs).float()   # [N, hidden_size]
                    # self.train()
        return self.State(vertices=hidden)

    def get_edge_logp(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        h_from = state.vertices[from_vertex_ids].to(device)
        h_to   = state.vertices[to_vertex_ids].to(device)
        theta = torch.sigmoid(self.edge_network(torch.cat([h_from, h_to], dim=-1)))
        theta = theta * (self.max_prob - self.min_prob) + self.min_prob
        probs = torch.cat([theta, 1. - theta], dim=-1)
        return probs.log()


class MLPLinkAgent(ProbabilisticAgent):
    """
    Agent whose network mirrors MLPLinkNet from mlpretrain_hard.py: a per-node
    encoder (node_encoder) followed by a pairwise scorer (link_mlp or dot
    product). Module names/shapes are kept identical to MLPLinkNet so a
    state_dict pretrained by that script can be loaded into this agent
    directly via load_state_dict once such a checkpoint exists.
    Like NodeFormerAgent, state.vertices holds encoded node embeddings
    (z = node_encoder(x)), computed once per prepare_state call.
    """
    def __init__(self, vertex_size, node_hidden=128, node_layers=2, pair_hidden=128,
                 dropout=0.0, scorer='pair', undirected=False, norm='none', min_prob=1e-4,
                 feat_mean=None, feat_std=None, logit_scale=1.0, learn_logit_scale=False,
                 act_bias=2.0, actor_ctx=False, indeg_ctx=False, indeg_noop=False):
        super().__init__()
        if scorer not in ('pair', 'dot'):
            raise ValueError("scorer must be 'pair' or 'dot'")
        if norm not in ('none', 'bn', 'ln'):
            raise ValueError("norm must be 'none', 'bn' or 'ln'")
        self.scorer = scorer
        self.undirected = undirected
        self.dropout = dropout
        self.min_prob, self.max_prob = min_prob, 1. - min_prob

        # Temperature on the raw score. The 'dot' scorer returns a *cosine*
        # similarity, so its output is confined to [-1, 1]. A softmax over the
        # ~240 two-hop candidates of the graph-edit MDP cannot express a
        # preference in that range: with observed within-row spread ~0.09 the
        # most-preferred candidate gets 1.2x uniform probability, so sampling is
        # effectively uniform and the entropy term sits at its maximum, where
        # its own gradient vanishes. Measured on SIFT100K: scale 1 -> 1.2x
        # uniform, 10 -> 5.1x, 20 -> 16.6x. Kept in log space so it stays
        # positive (a negative scale would invert the ranking) and so a
        # multiplicative step is what gradient descent sees.
        if logit_scale <= 0:
            raise ValueError('logit_scale must be positive')
        if learn_logit_scale:
            self.log_logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))
        else:
            self.register_buffer('log_logit_scale',
                                 torch.tensor(math.log(logit_scale)))

        # Optional per-feature standardization (mirrors mlpretrain_hard.py's
        # zero-mean/unit-std preprocessing), applied ONLY to the input of this
        # agent's own encoder -- NOT to graph.vertices, which HNSW distance
        # search and the reward use directly for real L2 distances. Registered
        # as buffers so they move with .to(device) / are (de)serialized with
        # the module, but are never trained.
        if feat_mean is not None and feat_std is not None:
            self.register_buffer('feat_mean', torch.as_tensor(feat_mean, dtype=torch.float32).view(1, -1))
            self.register_buffer('feat_std', torch.as_tensor(feat_std, dtype=torch.float32).view(1, -1))
        else:
            self.feat_mean = None
            self.feat_std = None

        if node_hidden and node_hidden > 0:
            layers, dim = [], vertex_size
            for _ in range(max(1, node_layers)):
                layers.append(nn.Linear(dim, node_hidden))
                if norm == 'bn':
                    layers.append(nn.BatchNorm1d(node_hidden))
                elif norm == 'ln':
                    layers.append(nn.LayerNorm(node_hidden))
                layers.append(nn.ELU())
                dim = node_hidden
            self.node_encoder = nn.Sequential(*layers)
            self.emb_dim = node_hidden
        else:
            self.node_encoder = nn.Identity()
            self.emb_dim = vertex_size

        # Per-node "should this node be edited at all?" head, i.e. the no-op action.
        # Without it every sampled node is FORCED to swap, so from a graph that is
        # already a local optimum under the bounded swap (measured on SIFT100K: 0
        # of 20 random swap batches improved recall) the policy can only pick the
        # least-bad downhill move. The bias starts positive so the initial policy
        # still acts on most nodes and the reward signal does not vanish on step 1.
        self.act_head = nn.Linear(self.emb_dim, 1)
        nn.init.zeros_(self.act_head.weight)
        nn.init.constant_(self.act_head.bias, float(act_bias))

        # Critic: V(i, s) for the discounted sum of node i's future credited reward.
        # It takes [z_i, mean of z over adj(i)] -- NOT z_i alone. z is a function of
        # vertex features only (prepare_state never sees the adjacency), so a head on
        # z_i alone would give V(i, s_t) == V(i, s_{t+1}) exactly, the TD target would
        # collapse to r + (gamma-1)*V(i), and the critic would carry no information
        # about the transition at all. The neighbour mean is the cheapest input that
        # actually moves when an edge is swapped. Zero-init the output layer so the
        # initial V is exactly 0 and early advantages equal the raw rewards.
        self.value_head = nn.Sequential(
            nn.Linear(2 * self.emb_dim, pair_hidden),
            nn.ELU(),
            nn.Linear(pair_hidden, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

        # D2 length head: a per-node scalar saying WHERE, in length-percentile terms,
        # this node's new edge should land. Takes [z_i, mean of z over adj(i)] like the
        # critic.
        #
        # This head is NOT subject to the architectural short-edge bias that killed
        # every attempt to fix the pair scorer. That argument was about score(u,v) =
        # dot(f(z_u), f(z_v)): a smooth f makes the score track -||z_u - z_v|| at every
        # weight setting, so ranking candidates by it is a proximity oracle no matter
        # what it is trained on. A per-node scalar has no pairwise structure and ranks
        # nothing, so it can express "I want percentile 42" and "you want 78" freely.
        #
        # Zero-init the output layer so the initial delta is exactly 0 and the starting
        # policy IS the measured learning-free optimum (p = 65 for every node). Anything
        # this head does is therefore a strict improvement test over that rule, not a
        # from-scratch gamble -- the same zero-init-residual pattern as actor_ctx, which
        # is the only intervention in this project that ever helped.
        self.len_head = nn.Sequential(
            nn.Linear(2 * self.emb_dim, pair_hidden),
            nn.ELU(),
            nn.Linear(pair_hidden, 1),
        )
        nn.init.zeros_(self.len_head[-1].weight)
        nn.init.zeros_(self.len_head[-1].bias)

        if scorer == 'pair':
            self.link_mlp = nn.Sequential(
                nn.Linear(2 * self.emb_dim, pair_hidden),
                nn.ELU(),
                nn.Linear(pair_hidden, 1),
            )

        # Topology-conditioned correction to the pair score. Without it the score of
        # (i, j) is a fixed function of the two endpoints' features -- the same in every
        # graph state -- so the actor can express one global "which edges are good"
        # ranking and cannot represent the state-dependent tradeoff that actually
        # matters here: whether node i needs another close neighbour or a long-range
        # link depends on the edges it already has.
        #
        # Kept as a SEPARATE residual head rather than widening link_mlp's input so a
        # checkpoint pretrained by mlpretrain_hard.py still loads with identical shapes,
        # and zero-initialized at the output so an A/B starts from exactly the baseline
        # scorer's behaviour. |z_j - ctx_i| is included explicitly because the useful
        # quantity is how far the candidate sits from the neighbourhood's centre, and a
        # linear layer cannot form an absolute difference of its own inputs.
        self.actor_ctx = bool(actor_ctx)
        if self.actor_ctx:
            self.ctx_head = nn.Sequential(
                nn.Linear(4 * self.emb_dim, pair_hidden),
                nn.ELU(),
                nn.Linear(pair_hidden, 1),
            )
            nn.init.zeros_(self.ctx_head[-1].weight)
            nn.init.zeros_(self.ctx_head[-1].bias)

        # In-degree awareness for the pair scorer: a weight on [log1p(src), log1p(dst)].
        # Zero-init, so step-0 behaviour is the baseline scorer exactly.
        #
        # A bare zeros Parameter rather than nn.Linear(2, 1) on purpose: Linear's
        # constructor runs kaiming_uniform_ and therefore CONSUMES global RNG draws,
        # which shifts every downstream draw (the node sample, the Gumbel noise) even
        # though the weight is overwritten with zeros immediately after. Enabling the
        # flag would then change the trajectory by itself and the paired A/B against
        # the no-indeg control would be measuring the RNG shift as well as the feature.
        # torch.zeros draws nothing, so indeg_ctx is exactly RNG-neutral.
        self.indeg_ctx = bool(indeg_ctx)
        if self.indeg_ctx:
            self.indeg_head = nn.Parameter(torch.zeros(2))

        # In-degree signal for the no-op gate: mean log(1 + in_deg) of current neighbors.
        # Scalar weight, zero-init. Positive weight -> high-indeg neighbors -> stop editing.
        self.indeg_noop = bool(indeg_noop)
        if self.indeg_noop:
            self.act_indeg_w = nn.Parameter(torch.zeros(1))

    # --- node embeddings (mirrors MLPLinkNet.encode) ---
    def encode(self, x):
        if self.feat_mean is not None:
            x = (x - self.feat_mean.to(device=x.device)) / (self.feat_std.to(device=x.device) + 1e-6)
        z = self.node_encoder(x)
        z = F.dropout(z, p=self.dropout, training=self.training)
        if z.shape == x.shape:
            z = z + x
        return z

    # --- pairwise features (mirrors MLPLinkNet._pair_features) ---
    def _pair_features(self, src_z, dst_z):
        src_z, dst_z = torch.broadcast_tensors(src_z, dst_z)
        if self.undirected:
            return torch.cat([src_z + dst_z, (src_z - dst_z).abs()], dim=-1)
        return torch.cat([src_z, dst_z], dim=-1)

    # --- raw score (logit) for specific (from, to) pairs, mirrors MLPLinkNet.edge_logits ---
    def edge_logits(self, z, from_idx, to_idx, ctx=None, indeg=None):
        """Raw (unscaled) score per pair. 'dot' returns a cosine similarity.

        :param ctx: optional [P, emb_dim] neighbourhood summary of the *from* node,
                    see neighbour_mean(). Adds the ctx_head residual; ignored when the
                    agent was built with actor_ctx=False.
        :param indeg: optional [P, 2] tensor of [log1p(src_indeg), log1p(dst_indeg)]
                      per pair. Adds the indeg_head residual; ignored when the agent
                      was built with indeg_ctx=False.
        """
        if self.scorer == 'dot':
            z_from = F.normalize(z[from_idx], p=2, dim=-1)
            z_to = F.normalize(z[to_idx], p=2, dim=-1)
            base = (z_from * z_to).sum(-1)
        else:
            pair = self._pair_features(z[from_idx], z[to_idx])
            base = self.link_mlp(pair).squeeze(-1)

        # Topology context residual
        if ctx is not None and self.actor_ctx:
            src_z, dst_z = torch.broadcast_tensors(z[from_idx], z[to_idx])
            feats = torch.cat([src_z, dst_z, ctx, (dst_z - ctx).abs()], dim=-1)
            base = base + self.ctx_head(feats).squeeze(-1)

        # In-degree residual
        if indeg is not None and self.indeg_ctx:
            base = base + (indeg * self.indeg_head.to(indeg.dtype)).sum(-1)

        return base

    @staticmethod
    def neighbour_mean(z, adj_ids, adj_mask):
        """Mean of z over each row's real neighbours, [B, deg] ids -> [B, emb_dim].

        A row with no valid slot would divide by zero, so clamp the count; such a node
        gets an all-zero mean, which is the correct "no local structure" encoding.
        Padding is -1 in the caller's contract, so clamp the ids before gathering --
        the mask zeroes those rows out afterwards anyway.
        """
        safe_ids = adj_ids.clamp_min(0)
        nb_z = z[safe_ids] * adj_mask.unsqueeze(-1).to(z.dtype)
        return nb_z.sum(1) / adj_mask.sum(1, keepdim=True).clamp_min(1).to(z.dtype)

    def get_node_ctx(self, node_ids, adj_ids, adj_mask, *, state, device='cpu', **kwargs):
        """Per-node topology summary for the actor, [B, emb_dim], or None if disabled.

        The adjacency is an argument for the same reason get_values takes one: which
        state is being described has to be explicit at the call site, not inferred from
        whether the graph has been committed yet.
        """
        if not self.actor_ctx:
            return None
        z = state.vertices.to(device=device)
        return self.neighbour_mean(z, adj_ids, adj_mask)

    @property
    def logit_scale(self):
        return self.log_logit_scale.exp()

    def get_act_logits(self, node_ids, *, state, device='cpu',
                       adj_ids=None, adj_mask=None, in_deg=None, **kwargs):
        """Logit of "edit this node" vs "leave it alone", shape [batch].

        Deliberately NOT scaled by logit_scale: that temperature exists to widen a
        cosine over ~240 competing candidates, whereas this is a single binary
        choice per node and needs no such correction.

        With indeg_noop, the gate also sees the mean log(1 + in-degree) of the node's
        CURRENT out-neighbours. Without it the gate is a function of z_i alone, so it
        cannot tell a node whose neighbours are fresh from one whose neighbours have
        already become super-hubs -- which is exactly the state in which continuing to
        edit makes the graph worse. A positive act_indeg_w means "my neighbours are
        already saturated, stop", giving the policy a learnable termination rule.

        :param adj_ids: current neighbour ids, [B, deg], padding clamped to >= 0
        :param adj_mask: which slots are real, [B, deg]
        :param in_deg: in-degree per graph node, [N] float. Indexed by adj_ids.
        """
        z = state.vertices.to(device=device)
        base = self.act_head(z[node_ids]).squeeze(-1)
        if not self.indeg_noop or in_deg is None or adj_ids is None or adj_mask is None:
            return base
        # Same masked-mean contract as neighbour_mean: clamp the padding ids before
        # gathering, then zero those slots out; a node with no valid neighbour gets 0.
        nb_log_indeg = torch.log1p(in_deg[adj_ids.clamp_min(0)])
        nb_log_indeg = nb_log_indeg * adj_mask.to(nb_log_indeg.dtype)
        nb_mean = nb_log_indeg.sum(1) / adj_mask.sum(1).clamp_min(1).to(nb_log_indeg.dtype)
        return base + self.act_indeg_w.to(base.device) * nb_mean

    def get_len_delta(self, node_ids, adj_ids, adj_mask, *, state, device='cpu', **kwargs):
        """ Unbounded per-node correction to the length percentile, shape [batch].

        The caller squashes and centres this: p_u = p0 + span * tanh(delta). Returning
        the raw value keeps the bound a property of the experiment configuration rather
        than of the network, so p0 and span can be re-tuned without retraining.

        Takes the same [z_i, neighbour mean] input as the critic. z alone would make
        the head blind to what the node's edges currently look like, and "does this
        node already have a long edge" is the one piece of state that plausibly bears
        on how long the next one should be.
        """
        z = state.vertices.to(device=device)
        return self.len_head(torch.cat([z[node_ids],
                                        self.neighbour_mean(z, adj_ids, adj_mask)],
                                       dim=-1)).squeeze(-1)

    def get_values(self, node_ids, adj_ids, adj_mask, *, state, device='cpu', **kwargs):
        """ V(i, s) for each node, shape [batch].

        :param adj_ids: neighbour ids of each node IN THE STATE BEING VALUED, [B, deg]
        :param adj_mask: which of those slots are real (padding is -1), [B, deg]

        The adjacency is an argument rather than read from the graph because the two
        states a TD target needs (s_t and s_{t+1}) differ only in their edges, and by
        the time the update runs the graph already holds s_{t+1} -- or, when the trial
        was rolled back, s_t again. Passing the rows in makes which state is being
        valued explicit at the call site instead of depending on commit timing.
        """
        z = state.vertices.to(device=device)
        z_self = z[node_ids]
        z_nb = self.neighbour_mean(z, adj_ids, adj_mask)
        return self.value_head(torch.cat([z_self, z_nb], dim=-1)).squeeze(-1)

    def prepare_state(self, graph, device='cpu', training=False, **kwargs):
        """Encode all nodes once; returned state.vertices are node embeddings z."""
        # Run the encoder on whatever device this module's own parameters live
        # on, NOT necessarily `device`: BaseAlgorithm.get_session_batch pins
        # state_device='cpu' during evaluation (to avoid holding all-N node
        # embeddings on GPU) while keeping the agent's weights on GPU via a
        # prior agent.to(device=sample_device) call. Encoding on `device` in
        # that case would feed a CPU tensor into CUDA-resident Linear layers
        # and crash. Only the *output* z is moved to the requested `device`.
        encode_device = next(self.parameters()).device
        x = graph.vertices.to(device=encode_device)
        if training:
            z = self.encode(x)
        else:
            self.eval()
            with torch.no_grad():
                z = self.encode(x)
        return self.State(vertices=z.to(device=device))

    def get_edge_logp(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        z = state.vertices.to(device=device)
        logit = self.edge_logits(z, from_vertex_ids, to_vertex_ids)
        theta = torch.sigmoid(logit).unsqueeze(-1)
        theta = theta * (self.max_prob - self.min_prob) + self.min_prob
        probs = torch.cat([1. - theta, theta], dim=-1)
        return probs.log()

    def get_pair_logits(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu',
                        ctx=None, indeg=None, **kwargs):
        """Temperature-scaled score for each pair, shape [batch].

        This is the logit the graph-edit MDP's candidate softmax consumes, so
        the scale is applied HERE and not in get_edge_logp: that path feeds a
        sigmoid whose output is a per-edge traversal probability calibrated by
        pretraining, and rescaling it would break that calibration.

        :param ctx: optional [batch, emb_dim] topology summary of each from-node, as
                    produced by get_node_ctx() and sliced to match this chunk.
        :param indeg: optional [batch, 2] log1p in-degrees of (src, dst), sliced to
                      match this chunk. See edge_logits.
        """
        z = state.vertices.to(device=device)
        logit = self.edge_logits(z, from_vertex_ids, to_vertex_ids, ctx=ctx, indeg=indeg)
        return logit * self.logit_scale.to(device=logit.device)
