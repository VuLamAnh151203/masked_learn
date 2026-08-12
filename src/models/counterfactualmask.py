# coding: utf-8
r"""
COUNTERFACTUALMASK
==================
A GLORIA/CaMuRe-compatible counterfactual graph-effect recommender.

Core idea
---------
The SAME LightGCN parameters encode both the observed graph and a learned
intervened graph:

    E_obs = f_theta(A)
    M_ui  = sigmoid(g_phi(e_u^obs, e_i^obs) / T)
    E_cf  = f_theta(A * M)
    C     = E_obs - E_cf

For a user-item pair (u, i), define

    s_obs = <e_u^obs, e_i^obs>
    s_cf  = <e_u^cf,  e_i^cf>
    delta = s_obs - s_cf

and an interaction-level gate alpha_ui. The final score is

    s_final = s_cf + alpha_ui * delta.

Thus the model starts from the stable/intervened score and restores only the
observational effect judged useful by the gate.

Important implementation choices
--------------------------------
1. Observed and counterfactual views SHARE the item embeddings, user
   preferences, and LightGCN propagation operator. Therefore E_obs-E_cf is due
   to changing graph weights, not due to two unrelated encoders.
2. The mask router sees full-graph representations only. By default its inputs
   are detached so mask-learning gradients do not manipulate the factual
   representation through the router-input path.
3. The mask is EDGE LEVEL: every observed interaction has its own learned
   weight produced by a shared MLP. Reverse edges reuse the same weight.
4. A retention-budget loss prevents M=1, a weak binary loss avoids a constant
   soft mask, stable-view BPR keeps E_cf useful, and a small gate penalty avoids
   the trivial alpha=1 solution.
5. Degree is NEVER given to the mask router. Degree is used only in diagnostics
   to compare learned item-level mask statistics with the original degree split.

First recommended defaults
--------------------------
    cf_keep_ratio: 0.70
    cf_mask_temperature: 1.0
    cf_mask_hidden_dim: feat_embed_dim
    cf_mask_detach_input: true

    cf_stable_weight: 0.10
    cf_mask_budget_weight: 0.10
    cf_mask_binary_weight: 0.001
    cf_gate_weight: 0.001

    cf_gate_hidden_dim: 32
    cf_gate_init_alpha: 0.25

    cf_use_item_item: true
    cf_freeze_shared_encoder: false
    cf_pretrained_checkpoint: null   # optional TOPKPRETRAIN checkpoint
    embedding_reg_weight: 0.0

Optional pretrained initialization
----------------------------------
If cf_pretrained_checkpoint points to a TOPKPRETRAIN checkpoint, this model
loads:
    router_item_embedding.weight -> id_embedding.weight
    router_gcn.preference        -> shared_gcn.preference

The encoder is still trainable unless cf_freeze_shared_encoder=true.

Model file / class
------------------
Place as:
    src/models/counterfactualmask.py
Use model name:
    COUNTERFACTUALMASK
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from common.abstract_recommender import GeneralRecommender


def _cfg(config, key, default):
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = checkpoint.get(key, None)
            if isinstance(value, dict):
                return value

        # Raw state_dict.
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            if any(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint

    raise ValueError('Could not find a PyTorch state_dict in checkpoint.')


def _strip_module_prefix(state_dict):
    output = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        output[key] = value
    return output


def _find_first_tensor(state_dict, keys):
    # First try exact names.
    for key in keys:
        value = state_dict.get(key, None)
        if torch.is_tensor(value):
            return value, key

    # Some trainers prefix saved parameters with strings such as `model.`.
    # Accept a unique suffix match so normal framework checkpoints still load.
    for candidate in keys:
        matches = [
            (key, value)
            for key, value in state_dict.items()
            if key.endswith(candidate) and torch.is_tensor(value)
        ]
        if len(matches) == 1:
            key, value = matches[0]
            return value, key

    return None, None


class COUNTERFACTUALMASK(GeneralRecommender):
    """Shared-encoder counterfactual graph-effect recommendation model."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset = dataset
        self.num_user = self.n_users
        self.num_item = self.n_items

        print(
            '[COUNTERFACTUALMASK] users={}, items={}'.format(
                self.num_user, self.num_item
            )
        )

        # ---------------------------------------------------------------
        # Base GLORIA/CaMuRe configuration.
        # ---------------------------------------------------------------
        self.batch_size = int(config['train_batch_size'])
        self.feat_embed_dim = int(config['feat_embed_dim'])
        self.n_layers = int(config['n_mm_layers'])
        self.knn_k = int(config['knn_k'])
        self.aggr_mode = config['aggr_mode']
        self.num_layer = 1
        self.dim_latent = 64
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1

        # ---------------------------------------------------------------
        # Counterfactual-mask hyperparameters.
        # ---------------------------------------------------------------
        self.cf_keep_ratio = float(_cfg(config, 'cf_keep_ratio', 0.70))
        if not (0.0 < self.cf_keep_ratio < 1.0):
            raise ValueError('cf_keep_ratio must be in (0, 1).')

        self.cf_mask_temperature = float(
            _cfg(config, 'cf_mask_temperature', 1.0)
        )
        if self.cf_mask_temperature <= 0.0:
            raise ValueError('cf_mask_temperature must be > 0.')

        self.cf_mask_hidden_dim = int(
            _cfg(config, 'cf_mask_hidden_dim', self.feat_embed_dim)
        )
        self.cf_mask_detach_input = _as_bool(
            _cfg(config, 'cf_mask_detach_input', True)
        )

        self.cf_stable_weight = float(
            _cfg(config, 'cf_stable_weight', 0.10)
        )
        self.cf_mask_budget_weight = float(
            _cfg(config, 'cf_mask_budget_weight', 0.10)
        )
        self.cf_mask_binary_weight = float(
            _cfg(config, 'cf_mask_binary_weight', 0.001)
        )
        self.cf_gate_weight = float(
            _cfg(config, 'cf_gate_weight', 0.001)
        )

        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        self.cf_gate_hidden_dim = int(
            _cfg(config, 'cf_gate_hidden_dim', 32)
        )
        self.cf_gate_init_alpha = float(
            _cfg(config, 'cf_gate_init_alpha', 0.25)
        )
        if not (0.0 < self.cf_gate_init_alpha < 1.0):
            raise ValueError('cf_gate_init_alpha must be in (0, 1).')

        self.cf_use_item_item = _as_bool(
            _cfg(config, 'cf_use_item_item', True)
        )
        self.cf_freeze_shared_encoder = _as_bool(
            _cfg(config, 'cf_freeze_shared_encoder', False)
        )

        self.cf_degree_overlap_ratio = float(
            _cfg(config, 'cf_degree_overlap_ratio', 0.10)
        )
        if not (0.0 < self.cf_degree_overlap_ratio < 1.0):
            raise ValueError('cf_degree_overlap_ratio must be in (0, 1).')

        self.cf_eval_item_chunk_size = int(
            _cfg(config, 'cf_eval_item_chunk_size', 2048)
        )
        if self.cf_eval_item_chunk_size <= 0:
            raise ValueError('cf_eval_item_chunk_size must be > 0.')

        # ---------------------------------------------------------------
        # Full training interaction graph.
        # ---------------------------------------------------------------
        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)
        self.num_interactions = int(train_interactions.nnz)

        forward_edges_np = self.pack_edge_index(train_interactions)
        forward_edges = torch.tensor(
            forward_edges_np,
            dtype=torch.long
        ).t().contiguous()
        reverse_edges = forward_edges[[1, 0], :]

        self.register_buffer(
            'forward_edge_index',
            forward_edges
        )
        self.register_buffer(
            'edge_index',
            torch.cat([forward_edges, reverse_edges], dim=1)
        )

        # Forward-edge user/item ids are useful for the edge router and
        # diagnostics. Item ids here are local [0, num_item).
        self.register_buffer(
            'forward_edge_users',
            torch.tensor(train_interactions.row, dtype=torch.long)
        )
        self.register_buffer(
            'forward_edge_items',
            torch.tensor(train_interactions.col, dtype=torch.long)
        )

        # Degree is diagnostic only, never a router input.
        item_degree_np = np.bincount(
            train_interactions.col,
            minlength=self.num_item
        ).astype(np.float32)
        self.register_buffer(
            'item_degree',
            torch.from_numpy(item_degree_np)
        )

        # ---------------------------------------------------------------
        # ONE shared embedding table + ONE shared LightGCN.
        # ---------------------------------------------------------------
        self.id_embedding = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.id_embedding.weight)

        self.shared_gcn = GCN(
            datasets=self.dataset,
            batch_size=self.batch_size,
            num_user=self.num_user,
            num_item=self.num_item,
            dim_id=int(config['embedding_size']),
            aggr_mode=self.aggr_mode,
            num_layer=self.num_layer,
            has_feature=False,
            dropout=self.drop_rate,
            dim_latent=self.dim_latent,
            device=self.device,
            features=self.id_embedding.weight,
        )

        # ---------------------------------------------------------------
        # Edge mask router:
        # [u, i, u*i, |u-i|] -> scalar logit.
        # No explicit degree input.
        # ---------------------------------------------------------------
        mask_input_dim = 4 * self.feat_embed_dim
        self.mask_mlp = nn.Sequential(
            nn.Linear(mask_input_dim, self.cf_mask_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.cf_mask_hidden_dim, 1),
        )
        self._initialize_mask_mlp()

        # ---------------------------------------------------------------
        # Scalable interaction-level counterfactual gate.
        #
        # Instead of materializing [u_cf, i_cf, C_u, C_i] for EVERY user-item
        # pair at full-sort evaluation, project user and item sides separately:
        #
        #   q_u = W_u [e_u_cf || C_u]
        #   q_i = W_i [e_i_cf || C_i]
        #
        #   gate_logit = <q_u, q_i>/sqrt(H)
        #                + w_cf * s_cf
        #                + w_delta * (s_obs - s_cf)
        #                + b
        #
        # This is still interaction specific but efficient for full sorting.
        # ---------------------------------------------------------------
        gate_side_dim = 2 * self.feat_embed_dim
        self.gate_user_proj = nn.Linear(
            gate_side_dim,
            self.cf_gate_hidden_dim,
            bias=False
        )
        self.gate_item_proj = nn.Linear(
            gate_side_dim,
            self.cf_gate_hidden_dim,
            bias=False
        )
        self.gate_cf_weight = nn.Parameter(torch.tensor(0.0))
        self.gate_delta_weight = nn.Parameter(torch.tensor(0.0))
        gate_bias = math.log(
            self.cf_gate_init_alpha / (1.0 - self.cf_gate_init_alpha)
        )
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias)))
        self._initialize_gate()

        # ---------------------------------------------------------------
        # Optional GLORIA item-item graph. Apply THE SAME item-item operator
        # to factual and counterfactual item representations.
        # ---------------------------------------------------------------
        self.mm_adj = None
        if self.cf_use_item_item:
            t_feat = getattr(self, 't_feat', None)
            if t_feat is None:
                print(
                    '[COUNTERFACTUALMASK] t_feat is unavailable; '
                    'disabling cf_use_item_item.'
                )
                self.cf_use_item_item = False
            else:
                _, self.mm_adj = self.get_knn_adj_mat(t_feat)

        # ---------------------------------------------------------------
        # Optional TOPKPRETRAIN initialization.
        # ---------------------------------------------------------------
        pretrained_path = _cfg(
            config,
            'cf_pretrained_checkpoint',
            None
        )
        self.cf_pretrained_checkpoint = None
        if pretrained_path is not None and str(pretrained_path).strip() != '':
            self.cf_pretrained_checkpoint = os.path.abspath(
                os.path.expanduser(str(pretrained_path))
            )
            if not os.path.isfile(self.cf_pretrained_checkpoint):
                raise FileNotFoundError(self.cf_pretrained_checkpoint)
            self._load_pretrained_encoder(self.cf_pretrained_checkpoint)

        if self.cf_freeze_shared_encoder:
            for parameter in self.id_embedding.parameters():
                parameter.requires_grad = False
            for parameter in self.shared_gcn.parameters():
                parameter.requires_grad = False

        # Cached diagnostic / forward state.
        self.result_embed = None  # kept for framework compatibility only
        self.last_observed_rep = None
        self.last_counterfactual_rep = None
        self.last_forward_edge_mask = None
        self.last_alpha_pos = None
        self.last_alpha_neg = None
        self.last_loss_components = None

    # ==================================================================
    # Initialization utilities
    # ==================================================================
    def _initialize_mask_mlp(self):
        first = self.mask_mlp[0]
        last = self.mask_mlp[2]

        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)

        # Small output weights + bias at logit(target keep ratio) gives a
        # stable non-trivial starting mask close to cf_keep_ratio.
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        init_bias = self.cf_mask_temperature * math.log(
            self.cf_keep_ratio / (1.0 - self.cf_keep_ratio)
        )
        nn.init.constant_(last.bias, float(init_bias))

    def _initialize_gate(self):
        # Small projections keep the initial interaction compatibility near 0,
        # so alpha starts mainly from cf_gate_init_alpha.
        nn.init.normal_(self.gate_user_proj.weight, mean=0.0, std=1e-2)
        nn.init.normal_(self.gate_item_proj.weight, mean=0.0, std=1e-2)

    # ==================================================================
    # Optional checkpoint initialization
    # ==================================================================
    def _load_pretrained_encoder(self, path):
        checkpoint = _safe_torch_load(path)
        state_dict = _strip_module_prefix(
            _extract_state_dict(checkpoint)
        )

        item_tensor, item_key = _find_first_tensor(
            state_dict,
            [
                'router_item_embedding.weight',
                'id_embedding.weight',
                'pretrained_item_embedding.weight',
                'item_embedding.weight',
            ]
        )
        user_tensor, user_key = _find_first_tensor(
            state_dict,
            [
                'router_gcn.preference',
                'shared_gcn.preference',
                'pretrained_gcn.preference',
                'gcn.preference',
            ]
        )

        if item_tensor is None or user_tensor is None:
            raise KeyError(
                'Could not load pretrained shared encoder. Need an item '
                'embedding tensor and a user preference tensor. Checkpoint '
                'keys include: {}'.format(list(state_dict.keys())[:20])
            )

        if tuple(item_tensor.shape) != tuple(self.id_embedding.weight.shape):
            raise ValueError(
                'Pretrained item embedding shape mismatch: checkpoint {} vs '
                'model {}.'.format(
                    tuple(item_tensor.shape),
                    tuple(self.id_embedding.weight.shape)
                )
            )
        if tuple(user_tensor.shape) != tuple(self.shared_gcn.preference.shape):
            raise ValueError(
                'Pretrained user preference shape mismatch: checkpoint {} vs '
                'model {}.'.format(
                    tuple(user_tensor.shape),
                    tuple(self.shared_gcn.preference.shape)
                )
            )

        with torch.no_grad():
            self.id_embedding.weight.copy_(item_tensor)
            self.shared_gcn.preference.copy_(user_tensor)

        print(
            '[COUNTERFACTUALMASK] initialized shared encoder from: {}\n'
            '  item key: {}\n'
            '  user key: {}'.format(path, item_key, user_key)
        )

    # ==================================================================
    # Graph utilities / GLORIA item-item graph
    # ==================================================================
    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(
                mm_embeddings,
                p=2,
                dim=-1,
                keepdim=True
            ).clamp_min(1e-12)
        )
        sim = torch.mm(
            context_norm,
            context_norm.transpose(1, 0)
        )
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim

        indices0 = torch.arange(
            knn_ind.shape[0],
            device=mm_embeddings.device
        ).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack(
            [indices0.reshape(-1), knn_ind.reshape(-1)],
            dim=0
        )
        return indices, self.compute_normalized_laplacian(
            indices,
            adj_size
        )

    def compute_normalized_laplacian(self, indices, adj_size):
        values = torch.ones(
            indices.size(1),
            device=indices.device,
            dtype=torch.float32
        )
        adj = torch.sparse_coo_tensor(
            indices,
            values,
            adj_size,
            device=indices.device
        ).coalesce()

        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        inv_sqrt = row_sum.pow(-0.5)
        norm_values = inv_sqrt[indices[0]] * inv_sqrt[indices[1]]

        return torch.sparse_coo_tensor(
            indices,
            norm_values,
            adj_size,
            device=indices.device
        ).coalesce()

    def item_item(self, item_rep):
        if not self.cf_use_item_item or self.mm_adj is None:
            return item_rep

        h = item_rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return item_rep + h

    # ==================================================================
    # Observed graph, mask, and counterfactual graph
    # ==================================================================
    def compute_observed_representation(self):
        observed_rep, _ = self.shared_gcn(
            self.edge_index,
            self.id_embedding.weight,
            edge_mask=None
        )
        return observed_rep

    def compute_forward_edge_mask(self, observed_rep):
        """Compute one mask value per ORIGINAL forward user-item interaction."""
        routing_rep = (
            observed_rep.detach()
            if self.cf_mask_detach_input
            else observed_rep
        )

        user_rep = routing_rep[self.forward_edge_users]
        item_rep = routing_rep[
            self.num_user + self.forward_edge_items
        ]

        edge_features = torch.cat(
            [
                user_rep,
                item_rep,
                user_rep * item_rep,
                torch.abs(user_rep - item_rep),
            ],
            dim=1
        )

        logits = self.mask_mlp(edge_features).squeeze(-1)
        masks = torch.sigmoid(
            logits / self.cf_mask_temperature
        )
        return masks, logits

    def make_undirected_edge_mask(self, forward_mask):
        # edge_index is [all forward edges, all reverse edges].
        return torch.cat([forward_mask, forward_mask], dim=0)

    def compute_counterfactual_representation(self, undirected_mask):
        counterfactual_rep, _ = self.shared_gcn(
            self.edge_index,
            self.id_embedding.weight,
            edge_mask=undirected_mask
        )
        return counterfactual_rep

    def apply_shared_item_item_operator(self, representation):
        """Apply the same parameter-free item-item operator to one view."""
        user_rep = representation[:self.num_user]
        item_rep = representation[self.num_user:]
        item_rep = self.item_item(item_rep)
        return torch.cat([user_rep, item_rep], dim=0)

    def compute_views(self):
        observed_raw = self.compute_observed_representation()
        forward_mask, mask_logits = self.compute_forward_edge_mask(
            observed_raw
        )
        undirected_mask = self.make_undirected_edge_mask(forward_mask)
        counterfactual_raw = self.compute_counterfactual_representation(
            undirected_mask
        )

        observed = self.apply_shared_item_item_operator(observed_raw)
        counterfactual = self.apply_shared_item_item_operator(
            counterfactual_raw
        )

        self.last_observed_rep = observed
        self.last_counterfactual_rep = counterfactual
        self.last_forward_edge_mask = forward_mask

        return observed, counterfactual, forward_mask, mask_logits

    # ==================================================================
    # Interaction-level counterfactual gate
    # ==================================================================
    def _project_gate_sides(
        self,
        user_cf,
        item_cf,
        user_effect,
        item_effect
    ):
        user_gate = self.gate_user_proj(
            torch.cat([user_cf, user_effect], dim=-1)
        )
        item_gate = self.gate_item_proj(
            torch.cat([item_cf, item_effect], dim=-1)
        )
        return user_gate, item_gate

    def interaction_gate_pairwise(
        self,
        user_cf,
        item_cf,
        user_effect,
        item_effect,
        score_cf,
        score_delta
    ):
        user_gate, item_gate = self._project_gate_sides(
            user_cf,
            item_cf,
            user_effect,
            item_effect
        )
        compatibility = (
            user_gate * item_gate
        ).sum(dim=-1) / math.sqrt(float(self.cf_gate_hidden_dim))

        gate_logits = (
            compatibility
            + self.gate_cf_weight * score_cf
            + self.gate_delta_weight * score_delta
            + self.gate_bias
        )
        return torch.sigmoid(gate_logits)

    def interaction_gate_matrix(
        self,
        user_cf,
        item_cf,
        user_effect,
        item_effect,
        score_cf,
        score_delta
    ):
        """Vectorized gate for [B users] x [C candidate items]."""
        user_gate, item_gate = self._project_gate_sides(
            user_cf,
            item_cf,
            user_effect,
            item_effect
        )
        compatibility = torch.matmul(
            user_gate,
            item_gate.t()
        ) / math.sqrt(float(self.cf_gate_hidden_dim))

        gate_logits = (
            compatibility
            + self.gate_cf_weight * score_cf
            + self.gate_delta_weight * score_delta
            + self.gate_bias
        )
        return torch.sigmoid(gate_logits)

    # ==================================================================
    # Pairwise scoring
    # ==================================================================
    def _pair_score_components(
        self,
        observed_rep,
        counterfactual_rep,
        user_ids,
        item_ids
    ):
        item_nodes = item_ids + self.num_user

        u_obs = observed_rep[user_ids]
        i_obs = observed_rep[item_nodes]
        u_cf = counterfactual_rep[user_ids]
        i_cf = counterfactual_rep[item_nodes]

        u_effect = u_obs - u_cf
        i_effect = i_obs - i_cf

        score_obs = (u_obs * i_obs).sum(dim=-1)
        score_cf = (u_cf * i_cf).sum(dim=-1)
        score_delta = score_obs - score_cf

        alpha = self.interaction_gate_pairwise(
            u_cf,
            i_cf,
            u_effect,
            i_effect,
            score_cf,
            score_delta
        )

        score_final = score_cf + alpha * score_delta

        return {
            'score_final': score_final,
            'score_obs': score_obs,
            'score_cf': score_cf,
            'score_delta': score_delta,
            'alpha': alpha,
        }

    def forward(self, interaction, return_aux=False):
        observed_rep, counterfactual_rep, forward_mask, mask_logits = (
            self.compute_views()
        )

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        pos = self._pair_score_components(
            observed_rep,
            counterfactual_rep,
            users,
            pos_items
        )
        neg = self._pair_score_components(
            observed_rep,
            counterfactual_rep,
            users,
            neg_items
        )

        self.last_alpha_pos = pos['alpha']
        self.last_alpha_neg = neg['alpha']

        if not return_aux:
            return pos['score_final'], neg['score_final']

        aux = {
            'observed_rep': observed_rep,
            'counterfactual_rep': counterfactual_rep,
            'forward_mask': forward_mask,
            'mask_logits': mask_logits,
            'pos': pos,
            'neg': neg,
        }
        return pos['score_final'], neg['score_final'], aux

    # ==================================================================
    # Losses
    # ==================================================================
    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def stable_bpr_loss(self, aux):
        return self.bpr_loss(
            aux['pos']['score_cf'],
            aux['neg']['score_cf']
        )

    def mask_budget_loss(self, forward_mask):
        target = torch.as_tensor(
            self.cf_keep_ratio,
            dtype=forward_mask.dtype,
            device=forward_mask.device
        )
        return (forward_mask.mean() - target).pow(2)

    @staticmethod
    def mask_binary_loss(forward_mask):
        return (
            forward_mask * (1.0 - forward_mask)
        ).mean()

    @staticmethod
    def gate_regularization_loss(pos_alpha, neg_alpha):
        # Small penalty against the trivial alpha=1 -> s_final=s_obs solution.
        return 0.5 * (pos_alpha.mean() + neg_alpha.mean())

    def embedding_regularization_loss(self):
        if self.embedding_reg_weight <= 0.0:
            # Keep the returned scalar connected to model device/dtype.
            return self.gate_bias * 0.0

        return (
            self.id_embedding.weight.pow(2).mean()
            + self.shared_gcn.preference.pow(2).mean()
        )

    def calculate_loss(self, interaction):
        pos_final, neg_final, aux = self.forward(
            interaction,
            return_aux=True
        )

        rec_loss = self.bpr_loss(pos_final, neg_final)
        stable_loss = self.stable_bpr_loss(aux)
        budget_loss = self.mask_budget_loss(aux['forward_mask'])
        binary_loss = self.mask_binary_loss(aux['forward_mask'])
        gate_loss = self.gate_regularization_loss(
            aux['pos']['alpha'],
            aux['neg']['alpha']
        )
        emb_reg = self.embedding_regularization_loss()

        total = (
            rec_loss
            + self.cf_stable_weight * stable_loss
            + self.cf_mask_budget_weight * budget_loss
            + self.cf_mask_binary_weight * binary_loss
            + self.cf_gate_weight * gate_loss
            + self.embedding_reg_weight * emb_reg
        )

        self.last_loss_components = {
            'total': total.detach(),
            'bpr_final': rec_loss.detach(),
            'bpr_stable_cf': stable_loss.detach(),
            'mask_budget': budget_loss.detach(),
            'mask_binary': binary_loss.detach(),
            'gate': gate_loss.detach(),
            'embedding_reg': emb_reg.detach(),
        }

        return total

    # ==================================================================
    # Full-sort prediction
    # ==================================================================
    def full_sort_predict(self, interaction):
        observed_rep, counterfactual_rep, _, _ = self.compute_views()

        user_ids = interaction[0]
        u_obs = observed_rep[user_ids]
        u_cf = counterfactual_rep[user_ids]
        u_effect = u_obs - u_cf

        all_i_obs = observed_rep[self.num_user:]
        all_i_cf = counterfactual_rep[self.num_user:]
        all_i_effect = all_i_obs - all_i_cf

        score_chunks = []
        chunk_size = self.cf_eval_item_chunk_size

        for start in range(0, self.num_item, chunk_size):
            end = min(start + chunk_size, self.num_item)

            i_obs = all_i_obs[start:end]
            i_cf = all_i_cf[start:end]
            i_effect = all_i_effect[start:end]

            score_obs = torch.matmul(u_obs, i_obs.t())
            score_cf = torch.matmul(u_cf, i_cf.t())
            score_delta = score_obs - score_cf

            alpha = self.interaction_gate_matrix(
                u_cf,
                i_cf,
                u_effect,
                i_effect,
                score_cf,
                score_delta
            )
            score_final = score_cf + alpha * score_delta
            score_chunks.append(score_final)

        return torch.cat(score_chunks, dim=1)

    # ==================================================================
    # Diagnostics
    # ==================================================================
    @torch.no_grad()
    def get_counterfactual_statistics(self):
        was_training = self.training
        self.eval()

        observed_rep, counterfactual_rep, mask, _ = self.compute_views()
        effect = observed_rep - counterfactual_rep

        obs_norm = F.normalize(observed_rep, p=2, dim=1)
        cf_norm = F.normalize(counterfactual_rep, p=2, dim=1)
        view_cosine = (obs_norm * cf_norm).sum(dim=1)

        stats = {
            'mask_mean': mask.mean().detach(),
            'mask_std': mask.std(unbiased=False).detach(),
            'mask_min': mask.min().detach(),
            'mask_max': mask.max().detach(),
            'mask_binary_measure': (
                mask * (1.0 - mask)
            ).mean().detach(),
            'effect_norm_mean': effect.norm(dim=1).mean().detach(),
            'effect_norm_user_mean': effect[:self.num_user].norm(
                dim=1
            ).mean().detach(),
            'effect_norm_item_mean': effect[self.num_user:].norm(
                dim=1
            ).mean().detach(),
            'observed_cf_cosine_mean': view_cosine.mean().detach(),
            'gate_bias_alpha': torch.sigmoid(self.gate_bias).detach(),
            'gate_cf_weight': self.gate_cf_weight.detach().clone(),
            'gate_delta_weight': self.gate_delta_weight.detach().clone(),
        }

        if self.last_alpha_pos is not None:
            alpha = torch.cat(
                [self.last_alpha_pos, self.last_alpha_neg],
                dim=0
            )
            stats.update({
                'last_gate_mean': alpha.mean().detach(),
                'last_gate_std': alpha.std(unbiased=False).detach(),
                'last_gate_min': alpha.min().detach(),
                'last_gate_max': alpha.max().detach(),
            })

        if was_training:
            self.train()
        return stats

    @torch.no_grad()
    def get_degree_overlap_statistics(self):
        """
        Post-hoc only: compare learned mask structure against top-degree items.

        We aggregate edge masks to an item mean:
            retained_score_i   = mean_{u:(u,i) in E} M_ui
            suppressed_score_i = 1 - retained_score_i

        Then compare top rho items under each learned score with top rho degree.
        Degree is NOT used during training or mask generation.
        """
        was_training = self.training
        self.eval()

        observed_raw = self.compute_observed_representation()
        forward_mask, _ = self.compute_forward_edge_mask(observed_raw)

        mask_sum = torch.zeros(
            self.num_item,
            dtype=forward_mask.dtype,
            device=forward_mask.device
        )
        edge_count = torch.zeros_like(mask_sum)
        mask_sum.index_add_(0, self.forward_edge_items, forward_mask)
        edge_count.index_add_(
            0,
            self.forward_edge_items,
            torch.ones_like(forward_mask)
        )
        item_mask_mean = mask_sum / edge_count.clamp_min(1.0)

        k = max(1, int(self.num_item * self.cf_degree_overlap_ratio))
        k = min(k, self.num_item)

        degree_top = torch.topk(self.item_degree, k=k).indices
        retained_top = torch.topk(item_mask_mean, k=k).indices
        suppressed_top = torch.topk(1.0 - item_mask_mean, k=k).indices

        degree_bool = torch.zeros(
            self.num_item,
            dtype=torch.bool,
            device=forward_mask.device
        )
        retained_bool = torch.zeros_like(degree_bool)
        suppressed_bool = torch.zeros_like(degree_bool)
        degree_bool[degree_top] = True
        retained_bool[retained_top] = True
        suppressed_bool[suppressed_top] = True

        retained_intersection = (degree_bool & retained_bool).sum()
        suppressed_intersection = (degree_bool & suppressed_bool).sum()
        retained_union = (degree_bool | retained_bool).sum().clamp_min(1)
        suppressed_union = (degree_bool | suppressed_bool).sum().clamp_min(1)

        # Pearson correlation between learned mean mask and log(1+degree).
        x = item_mask_mean
        y = torch.log1p(self.item_degree.to(x.dtype))
        x_center = x - x.mean()
        y_center = y - y.mean()
        denominator = torch.sqrt(
            x_center.pow(2).sum() * y_center.pow(2).sum()
        ).clamp_min(1e-12)
        pearson = (x_center * y_center).sum() / denominator

        stats = {
            'k': torch.tensor(k, device=forward_mask.device),
            'item_mask_mean': item_mask_mean.detach(),
            'degree_top_indices': degree_top.detach(),
            'retained_top_indices': retained_top.detach(),
            'suppressed_top_indices': suppressed_top.detach(),
            'retained_vs_degree_intersection': retained_intersection.detach(),
            'retained_vs_degree_overlap_at_k': (
                retained_intersection.float() / float(k)
            ).detach(),
            'retained_vs_degree_jaccard': (
                retained_intersection.float() / retained_union.float()
            ).detach(),
            'suppressed_vs_degree_intersection': suppressed_intersection.detach(),
            'suppressed_vs_degree_overlap_at_k': (
                suppressed_intersection.float() / float(k)
            ).detach(),
            'suppressed_vs_degree_jaccard': (
                suppressed_intersection.float() / suppressed_union.float()
            ).detach(),
            'mask_degree_pearson_log_degree': pearson.detach(),
            'mean_mask_degree_top': item_mask_mean[degree_top].mean().detach(),
            'mean_mask_all_items': item_mask_mean.mean().detach(),
        }

        if was_training:
            self.train()
        return stats

    @torch.no_grad()
    def print_degree_overlap_statistics(self):
        stats = self.get_degree_overlap_statistics()
        print('\n========== Counterfactual Mask vs Degree ==========')
        print('K items                         : {}'.format(
            int(stats['k'].item())
        ))
        print('Retained Top-K vs degree overlap: {:.4f}'.format(
            float(stats['retained_vs_degree_overlap_at_k'].item())
        ))
        print('Retained Top-K vs degree Jaccard: {:.4f}'.format(
            float(stats['retained_vs_degree_jaccard'].item())
        ))
        print('Suppressed Top-K vs degree overlap: {:.4f}'.format(
            float(stats['suppressed_vs_degree_overlap_at_k'].item())
        ))
        print('Suppressed Top-K vs degree Jaccard: {:.4f}'.format(
            float(stats['suppressed_vs_degree_jaccard'].item())
        ))
        print('Mask vs log-degree Pearson       : {:.4f}'.format(
            float(stats['mask_degree_pearson_log_degree'].item())
        ))
        print('Mean mask on degree Top-K        : {:.4f}'.format(
            float(stats['mean_mask_degree_top'].item())
        ))
        print('Mean mask over all items         : {:.4f}'.format(
            float(stats['mean_mask_all_items'].item())
        ))
        print('====================================================\n')
        return stats

    def set_mask_temperature(self, temperature):
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError('temperature must be > 0.')
        self.cf_mask_temperature = temperature


class GCN(nn.Module):
    """LightGCN-style encoder with a trainable user preference table."""

    def __init__(
        self,
        datasets,
        batch_size,
        num_user,
        num_item,
        dim_id,
        aggr_mode,
        num_layer,
        has_feature,
        dropout,
        dim_latent=None,
        device=None,
        features=None,
        user_profile=None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.has_feature = has_feature
        self.dropout = dropout
        self.device = device
        self.userprofile = user_profile

        preference_dim = self.dim_latent if self.has_feature else self.dim_feat

        self.preference = nn.Parameter(
            torch.empty(num_user, preference_dim)
        )
        nn.init.xavier_normal_(self.preference, gain=1.0)

        self.conv_embed_1 = Base_gcn(
            preference_dim,
            preference_dim,
            aggr=self.aggr_mode
        )

    def forward(self, edge_index, features, edge_mask=None):
        x = torch.cat([self.preference, features], dim=0)
        x = F.normalize(x, p=2, dim=1)

        h = self.conv_embed_1(
            x,
            edge_index,
            edge_mask=edge_mask
        )
        h_1 = self.conv_embed_1(
            h,
            edge_index,
            edge_mask=edge_mask
        )
        h_2 = self.conv_embed_1(
            h_1,
            edge_index,
            edge_mask=edge_mask
        )

        return x + h + h_1 + h_2, self.preference


class Base_gcn(MessagePassing):
    """Weighted LightGCN propagation with branch-specific normalization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True,
        bias=True,
        aggr='add',
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_mask=None, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        if size is None:
            size = (x.size(0), x.size(0))

        edge_index, edge_mask = remove_self_loops(
            edge_index,
            edge_mask
        )

        if edge_mask is None:
            edge_mask = torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype
            )
        else:
            edge_mask = edge_mask.to(
                device=x.device,
                dtype=x.dtype
            )

        if self.aggr == 'add':
            row, col = edge_index

            # Because the graph contains both directions, summing outgoing
            # weighted edges gives the weighted degree for every node.
            deg = torch.zeros(
                size[0],
                device=x.device,
                dtype=x.dtype
            )
            deg = deg.index_add(0, row, edge_mask)

            deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
            deg_inv_sqrt = torch.where(
                deg > 0,
                deg_inv_sqrt,
                torch.zeros_like(deg_inv_sqrt)
            )

            edge_weight = (
                edge_mask
                * deg_inv_sqrt[row]
                * deg_inv_sqrt[col]
            )
        else:
            edge_weight = edge_mask

        return self.propagate(
            edge_index,
            size=size,
            x=x,
            edge_weight=edge_weight
        )

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels
        )
