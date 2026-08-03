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
                 feat_mean=None, feat_std=None):
        super().__init__()
        if scorer not in ('pair', 'dot'):
            raise ValueError("scorer must be 'pair' or 'dot'")
        if norm not in ('none', 'bn', 'ln'):
            raise ValueError("norm must be 'none', 'bn' or 'ln'")
        self.scorer = scorer
        self.undirected = undirected
        self.dropout = dropout
        self.min_prob, self.max_prob = min_prob, 1. - min_prob

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

        if scorer == 'pair':
            self.link_mlp = nn.Sequential(
                nn.Linear(2 * self.emb_dim, pair_hidden),
                nn.ELU(),
                nn.Linear(pair_hidden, 1),
            )

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
    def edge_logits(self, z, from_idx, to_idx):
        if self.scorer == 'dot':
            z_from = F.normalize(z[from_idx], p=2, dim=-1)
            z_to = F.normalize(z[to_idx], p=2, dim=-1)
            return (z_from * z_to).sum(-1)
        pair = self._pair_features(z[from_idx], z[to_idx])
        return self.link_mlp(pair).squeeze(-1)

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
