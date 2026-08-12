# coding: utf-8
r"""
COUNTERFACTUAL2BRANCH
=====================
A GLORIA/CaMuRe-compatible learnable two-branch graph recommender with a
shared counterfactual reference encoder.

Architecture
------------

                         Interaction graph A
                                |
                     Shared reference encoder
                                |
                         E_obs = f_ref(A)
                                |
                       Learn edge mask M
                                |
                    E_cf = f_ref(A * M)
                                |
                   C = E_obs - E_cf
                                |
                  mask/effect supervision
                                |
                                M
                  +-------------+-------------+
                  |                           |
                  v                           v
              A * M                     A * (1-M)
                  |                           |
       separate user/item table    separate user/item table
             U1, I1                      U2, I2
                  |                           |
               GCN1                        GCN2
                  |                           |
                  +--------- concat ----------+
                                |
                         item-item GCN
                                |
                               BPR

The two recommendation branches are fully independent:
    - branch 1 has its own item embedding table and its own user table
    - branch 2 has its own item embedding table and its own user table
    - branch 1 and branch 2 do NOT share trainable embeddings

The reference encoder is separate from both recommendation branches. It exists
only to produce a stable space in which the edge mask and counterfactual effect
are defined.

Two reference modes
-------------------
1) From scratch

    cf2_reference_mode: scratch

The reference item/user tables start randomly and are trained jointly.  This
mode therefore has THREE trainable embedding sets:
    reference + branch1 + branch2.

2) Pretrained reference encoder

    cf2_reference_mode: pretrained
    cf2_pretrained_checkpoint: C:/.../TOPKPRETRAIN-....pth
    cf2_freeze_reference: true

The reference item/user tables are loaded from TOPKPRETRAIN and frozen by
default.  Only the mask router and the two recommendation branches are trained.
This mode has trainable capacity much closer to a two-branch degree baseline.

Optional branch initialization
------------------------------
Set:

    cf2_init_branches_from_reference: true

to copy the reference item/user tables into BOTH independent branches before
training starts.  The copies then train independently.  The default is false so
that the main scratch-vs-pretrained comparison changes only the reference
encoder unless you explicitly request pretrained branch initialization too.

Mask
----
The router is edge-level and receives NO degree input:

    r_ui = [e_u, e_i, e_u * e_i, |e_u - e_i|]
    M_ui = sigmoid(MLP(r_ui) / temperature)

The same M_ui is used for the forward and reverse copy of an interaction.
Training uses soft masks.  Evaluation can optionally use an exact hard Top-K
edge mask with:

    cf2_eval_hard_mask: true

where K ~= cf2_keep_ratio * |E|.

Loss
----
Primary loss:
    L_rec = BPR(concat(branch1, branch2))

Reference / mask supervision:
    L_ref_obs = BPR(E_obs)                  [scratch mode only]
    L_ref_cf  = BPR(E_cf)
    L_budget  = (mean(M) - keep_ratio)^2
    L_binary  = mean(M * (1-M))

Optional specialization supervision:
    branch1 -> E_cf
    branch2 -> C = E_obs - E_cf

using cosine alignment.  It is disabled by default because it is a stronger
assumption:

    cf2_specialization_weight: 0.0

Recommended first settings
--------------------------
    cf2_reference_mode: scratch          # OR pretrained
    cf2_keep_ratio: 0.70
    cf2_mask_temperature: 1.0
    cf2_mask_hidden_dim: feat_embed_dim
    cf2_mask_detach_input: true

    cf2_reference_bpr_weight: 0.10
    cf2_counterfactual_bpr_weight: 0.10
    cf2_mask_budget_weight: 0.10
    cf2_mask_binary_weight: 0.001
    cf2_specialization_weight: 0.0

    cf2_eval_hard_mask: false
    cf2_use_item_item: true
    cf2_init_branches_from_reference: false
    embedding_reg_weight: 0.0

Place this file at:
    src/models/counterfactual2branch.py
Use model name:
    COUNTERFACTUAL2BRANCH
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
    """Read optional config values without requiring config.get()."""
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
    # PyTorch >= 2.6 changed torch.load defaults; use the normal checkpoint
    # behavior explicitly while remaining compatible with older versions.
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

    raise ValueError('Could not locate a PyTorch state_dict in checkpoint.')


def _strip_module_prefix(state_dict):
    output = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        output[key] = value
    return output


def _find_first_tensor(state_dict, keys):
    # Exact names first.
    for key in keys:
        value = state_dict.get(key, None)
        if torch.is_tensor(value):
            return value, key

    # Frameworks sometimes prefix keys with "model." etc.  Accept a unique
    # suffix match.
    for wanted in keys:
        matches = [
            (key, value)
            for key, value in state_dict.items()
            if key.endswith(wanted) and torch.is_tensor(value)
        ]
        if len(matches) == 1:
            return matches[0][1], matches[0][0]

    return None, None


class COUNTERFACTUAL2BRANCH(GeneralRecommender):
    """Counterfactual mask discovery + two independent recommendation experts."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset = dataset
        self.num_user = self.n_users
        self.num_item = self.n_items

        print(
            '[COUNTERFACTUAL2BRANCH] users={}, items={}'.format(
                self.num_user, self.num_item
            )
        )

        # ---------------------------------------------------------------
        # Base GLORIA/CaMuRe settings.
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
        # Reference mode.
        # ---------------------------------------------------------------
        self.reference_mode = str(
            _cfg(config, 'cf2_reference_mode', 'scratch')
        ).strip().lower()
        if self.reference_mode not in {'scratch', 'pretrained'}:
            raise ValueError(
                "cf2_reference_mode must be 'scratch' or 'pretrained'."
            )

        self.freeze_reference = _as_bool(
            _cfg(
                config,
                'cf2_freeze_reference',
                self.reference_mode == 'pretrained'
            )
        )
        if self.reference_mode == 'scratch' and self.freeze_reference:
            print(
                '[COUNTERFACTUAL2BRANCH] WARNING: scratch reference is '
                'configured frozen. Random frozen reference features are '
                'usually not recommended.'
            )

        self.init_branches_from_reference = _as_bool(
            _cfg(config, 'cf2_init_branches_from_reference', False)
        )
        self.cache_frozen_observed = _as_bool(
            _cfg(config, 'cf2_cache_frozen_observed', True)
        )

        # ---------------------------------------------------------------
        # Mask / optimization settings.
        # ---------------------------------------------------------------
        self.keep_ratio = float(_cfg(config, 'cf2_keep_ratio', 0.70))
        if not (0.0 < self.keep_ratio < 1.0):
            raise ValueError('cf2_keep_ratio must be in (0, 1).')

        self.mask_temperature = float(
            _cfg(config, 'cf2_mask_temperature', 1.0)
        )
        if self.mask_temperature <= 0.0:
            raise ValueError('cf2_mask_temperature must be > 0.')

        self.mask_hidden_dim = int(
            _cfg(config, 'cf2_mask_hidden_dim', self.feat_embed_dim)
        )
        self.mask_detach_input = _as_bool(
            _cfg(config, 'cf2_mask_detach_input', True)
        )

        self.reference_bpr_weight = float(
            _cfg(config, 'cf2_reference_bpr_weight', 0.10)
        )
        self.counterfactual_bpr_weight = float(
            _cfg(config, 'cf2_counterfactual_bpr_weight', 0.10)
        )
        self.mask_budget_weight = float(
            _cfg(config, 'cf2_mask_budget_weight', 0.10)
        )
        self.mask_binary_weight = float(
            _cfg(config, 'cf2_mask_binary_weight', 0.001)
        )
        self.specialization_weight = float(
            _cfg(config, 'cf2_specialization_weight', 0.0)
        )
        self.branch_aux_weight = float(
            _cfg(config, 'cf2_branch_aux_weight', 0.0)
        )
        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        self.eval_hard_mask = _as_bool(
            _cfg(config, 'cf2_eval_hard_mask', False)
        )
        self.use_item_item = _as_bool(
            _cfg(config, 'cf2_use_item_item', True)
        )
        self.degree_overlap_ratio = float(
            _cfg(config, 'cf2_degree_overlap_ratio', 0.10)
        )
        if not (0.0 < self.degree_overlap_ratio < 1.0):
            raise ValueError('cf2_degree_overlap_ratio must be in (0, 1).')

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

        self.register_buffer('forward_edge_index', forward_edges)
        self.register_buffer(
            'edge_index',
            torch.cat([forward_edges, reverse_edges], dim=1)
        )
        self.register_buffer(
            'forward_edge_users',
            torch.tensor(train_interactions.row, dtype=torch.long)
        )
        self.register_buffer(
            'forward_edge_items',
            torch.tensor(train_interactions.col, dtype=torch.long)
        )

        # Degree is diagnostic only. It is NEVER an input to the mask router.
        item_degree_np = np.bincount(
            train_interactions.col,
            minlength=self.num_item
        ).astype(np.float32)
        self.register_buffer('item_degree', torch.from_numpy(item_degree_np))

        # ---------------------------------------------------------------
        # Shared REFERENCE encoder: one item table + one user table.
        # ---------------------------------------------------------------
        self.reference_item_embedding = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.reference_item_embedding.weight)

        self.reference_gcn = GCN(
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
            features=self.reference_item_embedding.weight,
        )

        # ---------------------------------------------------------------
        # Edge mask router from E_obs.
        # ---------------------------------------------------------------
        mask_input_dim = 4 * self.feat_embed_dim
        self.mask_mlp = nn.Sequential(
            nn.Linear(mask_input_dim, self.mask_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.mask_hidden_dim, 1),
        )
        self._initialize_mask_mlp()

        # ---------------------------------------------------------------
        # BRANCH 1: completely separate item/user parameters.
        # GCN.preference is the branch-specific user table U1.
        # ---------------------------------------------------------------
        self.item_embedding_branch1 = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.item_embedding_branch1.weight)

        self.branch1_gcn = GCN(
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
            features=self.item_embedding_branch1.weight,
        )

        # ---------------------------------------------------------------
        # BRANCH 2: another completely separate item/user parameter set.
        # GCN.preference is the branch-specific user table U2.
        # ---------------------------------------------------------------
        self.item_embedding_branch2 = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.item_embedding_branch2.weight)

        self.branch2_gcn = GCN(
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
            features=self.item_embedding_branch2.weight,
        )

        # ---------------------------------------------------------------
        # Optional GLORIA item-item graph, used ONLY after branch concat.
        # ---------------------------------------------------------------
        self.mm_adj = None
        if self.use_item_item:
            t_feat = getattr(self, 't_feat', None)
            if t_feat is None:
                print(
                    '[COUNTERFACTUAL2BRANCH] t_feat unavailable; '
                    'disabling cf2_use_item_item.'
                )
                self.use_item_item = False
            else:
                _, self.mm_adj = self.get_knn_adj_mat(t_feat)

        # ---------------------------------------------------------------
        # Pretrained reference loading.
        # ---------------------------------------------------------------
        self.pretrained_checkpoint = None
        if self.reference_mode == 'pretrained':
            path = _cfg(config, 'cf2_pretrained_checkpoint', None)
            if path is None or str(path).strip() == '':
                raise ValueError(
                    'cf2_pretrained_checkpoint is required when '
                    'cf2_reference_mode=pretrained.'
                )

            self.pretrained_checkpoint = os.path.abspath(
                os.path.expanduser(str(path))
            )
            if not os.path.isfile(self.pretrained_checkpoint):
                raise FileNotFoundError(self.pretrained_checkpoint)

            self._load_pretrained_reference(self.pretrained_checkpoint)

        # Optional: initialize both independent experts from the reference.
        if self.init_branches_from_reference:
            self._copy_reference_to_branches()

        # Freeze ONLY the reference encoder if requested.  Mask and branch
        # parameters remain trainable.
        if self.freeze_reference:
            for p in self.reference_item_embedding.parameters():
                p.requires_grad = False
            for p in self.reference_gcn.parameters():
                p.requires_grad = False

        # Lazy cached E_obs for a frozen reference encoder.  Do not cache E_cf:
        # E_cf must be recomputed because it depends on the changing mask M.
        self._cached_reference_observed = None

        # Framework / diagnostics caches.
        self.result_embed = None
        self.last_reference_observed = None
        self.last_reference_counterfactual = None
        self.last_effect = None
        self.last_forward_mask = None
        self.last_mask_logits = None
        self.last_branch1_rep = None
        self.last_branch2_rep = None
        self.last_loss_components = None

        self._print_parameter_summary()

    # ==================================================================
    # Initialization / checkpoint utilities
    # ==================================================================
    def _initialize_mask_mlp(self):
        first = self.mask_mlp[0]
        last = self.mask_mlp[2]

        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)

        # Start soft mask mass close to cf2_keep_ratio.
        init_bias = self.mask_temperature * math.log(
            self.keep_ratio / (1.0 - self.keep_ratio)
        )
        nn.init.constant_(last.bias, float(init_bias))

    def _load_pretrained_reference(self, path):
        checkpoint = _safe_torch_load(path)
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

        item_tensor, item_key = _find_first_tensor(
            state_dict,
            [
                'router_item_embedding.weight',      # TOPKPRETRAIN
                'reference_item_embedding.weight',
                'id_embedding.weight',
                'item_embedding.weight',
            ]
        )
        user_tensor, user_key = _find_first_tensor(
            state_dict,
            [
                'router_gcn.preference',             # TOPKPRETRAIN
                'reference_gcn.preference',
                'shared_gcn.preference',
                'gcn.preference',
            ]
        )

        if item_tensor is None or user_tensor is None:
            raise KeyError(
                'Could not locate pretrained reference item/user tensors. '
                'Expected TOPKPRETRAIN keys such as '
                'router_item_embedding.weight and router_gcn.preference. '
                'First checkpoint keys: {}'.format(list(state_dict.keys())[:25])
            )

        if tuple(item_tensor.shape) != tuple(
            self.reference_item_embedding.weight.shape
        ):
            raise ValueError(
                'Reference item shape mismatch: checkpoint {} vs model {}.'.format(
                    tuple(item_tensor.shape),
                    tuple(self.reference_item_embedding.weight.shape)
                )
            )
        if tuple(user_tensor.shape) != tuple(self.reference_gcn.preference.shape):
            raise ValueError(
                'Reference user shape mismatch: checkpoint {} vs model {}.'.format(
                    tuple(user_tensor.shape),
                    tuple(self.reference_gcn.preference.shape)
                )
            )

        with torch.no_grad():
            self.reference_item_embedding.weight.copy_(item_tensor)
            self.reference_gcn.preference.copy_(user_tensor)

        print(
            '[COUNTERFACTUAL2BRANCH] loaded pretrained reference:\n'
            '  checkpoint: {}\n'
            '  item key : {}\n'
            '  user key : {}'.format(path, item_key, user_key)
        )

    def _copy_reference_to_branches(self):
        """Copy reference embeddings into BOTH branches; copies remain independent."""
        with torch.no_grad():
            self.item_embedding_branch1.weight.copy_(
                self.reference_item_embedding.weight
            )
            self.item_embedding_branch2.weight.copy_(
                self.reference_item_embedding.weight
            )
            self.branch1_gcn.preference.copy_(self.reference_gcn.preference)
            self.branch2_gcn.preference.copy_(self.reference_gcn.preference)

        print(
            '[COUNTERFACTUAL2BRANCH] initialized BOTH branch user/item tables '
            'from the reference encoder. They remain independent afterward.'
        )

    def _print_parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        ref_total = (
            self.reference_item_embedding.weight.numel()
            + self.reference_gcn.preference.numel()
        )
        ref_trainable = sum(
            p.numel()
            for module in (self.reference_item_embedding, self.reference_gcn)
            for p in module.parameters()
            if p.requires_grad
        )
        branch_trainable = sum(
            p.numel()
            for module in (
                self.item_embedding_branch1,
                self.branch1_gcn,
                self.item_embedding_branch2,
                self.branch2_gcn,
            )
            for p in module.parameters()
            if p.requires_grad
        )
        mask_trainable = sum(
            p.numel() for p in self.mask_mlp.parameters() if p.requires_grad
        )

        print(
            '[COUNTERFACTUAL2BRANCH] reference_mode={} | '
            'freeze_reference={}'.format(
                self.reference_mode,
                self.freeze_reference
            )
        )
        print(
            '[COUNTERFACTUAL2BRANCH] parameters total={} | trainable={} | '
            'reference(trainable/embedding-core)={}/{} | branches={} | mask={}'.format(
                total,
                trainable,
                ref_trainable,
                ref_total,
                branch_trainable,
                mask_trainable,
            )
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
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
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
        return indices, self.compute_normalized_laplacian(indices, adj_size)

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
        if not self.use_item_item or self.mm_adj is None:
            return item_rep

        h = item_rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return item_rep + h

    # ==================================================================
    # Reference encoder and learned intervention M
    # ==================================================================
    def compute_reference_observed(self):
        """E_obs = f_ref(A). Cache only when the reference is frozen."""
        if (
            self.freeze_reference
            and self.cache_frozen_observed
            and self._cached_reference_observed is not None
        ):
            return self._cached_reference_observed

        observed, _ = self.reference_gcn(
            self.edge_index,
            self.reference_item_embedding.weight,
            edge_mask=None
        )

        if self.freeze_reference and self.cache_frozen_observed:
            # No mask depends on this propagation itself.  Detaching is safe;
            # mask_mlp still trains from the resulting fixed features.
            observed = observed.detach()
            self._cached_reference_observed = observed

        return observed

    def compute_soft_forward_mask(self, reference_observed):
        routing_rep = (
            reference_observed.detach()
            if self.mask_detach_input
            else reference_observed
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
        masks = torch.sigmoid(logits / self.mask_temperature)
        return masks, logits

    def hard_topk_forward_mask(self, logits):
        """Exact hard retained-edge budget for optional evaluation."""
        n = logits.numel()
        k = int(round(self.keep_ratio * n))
        k = max(1, min(k, n - 1 if n > 1 else 1))

        selected = torch.topk(
            logits,
            k=k,
            largest=True,
            sorted=False
        ).indices
        hard = torch.zeros_like(logits)
        hard[selected] = 1.0
        return hard

    @staticmethod
    def make_undirected_edge_mask(forward_mask):
        # edge_index = [all forward edges, all reverse edges]
        return torch.cat([forward_mask, forward_mask], dim=0)

    def compute_reference_counterfactual(self, undirected_mask):
        # IMPORTANT: do not wrap this in no_grad even when reference params are
        # frozen. Gradients must pass through edge_mask -> mask_mlp.
        counterfactual, _ = self.reference_gcn(
            self.edge_index,
            self.reference_item_embedding.weight,
            edge_mask=undirected_mask
        )
        return counterfactual

    def compute_reference_views(self, hard_mask=False):
        observed = self.compute_reference_observed()
        soft_mask, logits = self.compute_soft_forward_mask(observed)

        forward_mask = (
            self.hard_topk_forward_mask(logits)
            if hard_mask
            else soft_mask
        )
        undirected_mask = self.make_undirected_edge_mask(forward_mask)
        counterfactual = self.compute_reference_counterfactual(
            undirected_mask
        )
        effect = observed - counterfactual

        self.last_reference_observed = observed
        self.last_reference_counterfactual = counterfactual
        self.last_effect = effect
        self.last_forward_mask = forward_mask
        self.last_mask_logits = logits

        return {
            'observed': observed,
            'counterfactual': counterfactual,
            'effect': effect,
            'forward_mask': forward_mask,
            'soft_forward_mask': soft_mask,
            'mask_logits': logits,
            'undirected_mask': undirected_mask,
        }

    # ==================================================================
    # Two independent recommendation branches
    # ==================================================================
    def compute_branch_representations(self, undirected_mask):
        complement_mask = 1.0 - undirected_mask

        branch1_rep, _ = self.branch1_gcn(
            self.edge_index,
            self.item_embedding_branch1.weight,
            edge_mask=undirected_mask
        )
        branch2_rep, _ = self.branch2_gcn(
            self.edge_index,
            self.item_embedding_branch2.weight,
            edge_mask=complement_mask
        )

        self.last_branch1_rep = branch1_rep
        self.last_branch2_rep = branch2_rep
        return branch1_rep, branch2_rep

    def fuse_branches(self, branch1_rep, branch2_rep):
        user1 = branch1_rep[:self.num_user]
        user2 = branch2_rep[:self.num_user]
        item1 = branch1_rep[self.num_user:]
        item2 = branch2_rep[self.num_user:]

        user_rep = torch.cat([user1, user2], dim=1)
        item_rep = torch.cat([item1, item2], dim=1)

        # Match the original GLORIA pattern: concatenate first, then run the
        # item-item propagation on the concatenated item representation.
        item_rep = self.item_item(item_rep)

        result = torch.cat([user_rep, item_rep], dim=0)
        self.result_embed = result
        return result

    def compute_model_representations(self, hard_mask=False):
        ref = self.compute_reference_views(hard_mask=hard_mask)
        branch1, branch2 = self.compute_branch_representations(
            ref['undirected_mask']
        )
        result = self.fuse_branches(branch1, branch2)
        return result, ref, branch1, branch2

    # ==================================================================
    # Pairwise scoring / auxiliary scoring
    # ==================================================================
    def pairwise_scores(self, representation, interaction):
        users = interaction[0]
        pos_items = interaction[1] + self.num_user
        neg_items = interaction[2] + self.num_user

        user_tensor = representation[users]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]

        pos = (user_tensor * pos_tensor).sum(dim=1)
        neg = (user_tensor * neg_tensor).sum(dim=1)
        return pos, neg

    def pairwise_scores_ids(self, representation, users, pos_items, neg_items):
        pos_nodes = pos_items + self.num_user
        neg_nodes = neg_items + self.num_user
        user_tensor = representation[users]
        pos = (user_tensor * representation[pos_nodes]).sum(dim=1)
        neg = (user_tensor * representation[neg_nodes]).sum(dim=1)
        return pos, neg

    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    # ==================================================================
    # Mask / effect / specialization losses
    # ==================================================================
    def mask_budget_loss(self, soft_forward_mask):
        target = torch.as_tensor(
            self.keep_ratio,
            dtype=soft_forward_mask.dtype,
            device=soft_forward_mask.device
        )
        return (soft_forward_mask.mean() - target).pow(2)

    @staticmethod
    def mask_binary_loss(soft_forward_mask):
        return (
            soft_forward_mask * (1.0 - soft_forward_mask)
        ).mean()

    @staticmethod
    def _cosine_alignment_loss(pred, target, eps=1e-8):
        """1-cosine on nodes whose target has non-negligible norm."""
        target = target.detach()
        valid = target.norm(dim=1) > eps
        if not bool(valid.any()):
            return pred.sum() * 0.0

        pred_n = F.normalize(pred[valid], p=2, dim=1, eps=eps)
        target_n = F.normalize(target[valid], p=2, dim=1, eps=eps)
        return 1.0 - (pred_n * target_n).sum(dim=1).mean()

    def specialization_loss(self, branch1, branch2, ref):
        """
        Optional semantics:
            branch1 (A*M)       should resemble E_cf
            branch2 (A*(1-M))   should resemble C=E_obs-E_cf

        Disabled by default (weight 0.0) because it is a stronger modeling
        assumption than the basic learned split.
        """
        stable = self._cosine_alignment_loss(
            branch1,
            ref['counterfactual']
        )
        effect = self._cosine_alignment_loss(
            branch2,
            ref['effect']
        )
        return stable + effect

    def embedding_regularization_loss(self):
        if self.embedding_reg_weight <= 0.0:
            return self.mask_mlp[-1].bias.sum() * 0.0

        reg = (
            self.item_embedding_branch1.weight.pow(2).mean()
            + self.item_embedding_branch2.weight.pow(2).mean()
            + self.branch1_gcn.preference.pow(2).mean()
            + self.branch2_gcn.preference.pow(2).mean()
        )

        if not self.freeze_reference:
            reg = reg + (
                self.reference_item_embedding.weight.pow(2).mean()
                + self.reference_gcn.preference.pow(2).mean()
            )
        return reg

    # ==================================================================
    # Forward / training objective
    # ==================================================================
    def forward(self, interaction, return_aux=False):
        # Training always uses the differentiable soft mask.  Evaluation may
        # choose hard masks through full_sort_predict().
        result, ref, branch1, branch2 = self.compute_model_representations(
            hard_mask=False
        )
        pos, neg = self.pairwise_scores(result, interaction)

        if not return_aux:
            return pos, neg

        return pos, neg, {
            'result': result,
            'ref': ref,
            'branch1': branch1,
            'branch2': branch2,
        }

    def calculate_loss(self, interaction):
        pos, neg, aux = self.forward(interaction, return_aux=True)
        rec_loss = self.bpr_loss(pos, neg)

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        ref = aux['ref']

        # Observed reference BPR is useful only when the scratch reference is
        # trainable.  In frozen-pretrained mode it would be a constant term.
        ref_obs_loss = rec_loss * 0.0
        if not self.freeze_reference and self.reference_bpr_weight > 0.0:
            ref_obs_pos, ref_obs_neg = self.pairwise_scores_ids(
                ref['observed'], users, pos_items, neg_items
            )
            ref_obs_loss = self.bpr_loss(ref_obs_pos, ref_obs_neg)

        # Counterfactual BPR trains the MASK even when the reference encoder is
        # frozen because E_cf depends differentiably on M.
        ref_cf_loss = rec_loss * 0.0
        if self.counterfactual_bpr_weight > 0.0:
            ref_cf_pos, ref_cf_neg = self.pairwise_scores_ids(
                ref['counterfactual'], users, pos_items, neg_items
            )
            ref_cf_loss = self.bpr_loss(ref_cf_pos, ref_cf_neg)

        budget_loss = self.mask_budget_loss(ref['soft_forward_mask'])
        binary_loss = self.mask_binary_loss(ref['soft_forward_mask'])

        spec_loss = rec_loss * 0.0
        if self.specialization_weight > 0.0:
            spec_loss = self.specialization_loss(
                aux['branch1'],
                aux['branch2'],
                ref
            )

        # Optional individual-branch ranking loss. Leave at 0 initially so the
        # main comparison mirrors the original concat-then-BPR architecture.
        branch_aux_loss = rec_loss * 0.0
        if self.branch_aux_weight > 0.0:
            b1_pos, b1_neg = self.pairwise_scores_ids(
                aux['branch1'], users, pos_items, neg_items
            )
            b2_pos, b2_neg = self.pairwise_scores_ids(
                aux['branch2'], users, pos_items, neg_items
            )
            branch_aux_loss = 0.5 * (
                self.bpr_loss(b1_pos, b1_neg)
                + self.bpr_loss(b2_pos, b2_neg)
            )

        emb_reg = self.embedding_regularization_loss()

        total = (
            rec_loss
            + self.reference_bpr_weight * ref_obs_loss
            + self.counterfactual_bpr_weight * ref_cf_loss
            + self.mask_budget_weight * budget_loss
            + self.mask_binary_weight * binary_loss
            + self.specialization_weight * spec_loss
            + self.branch_aux_weight * branch_aux_loss
            + self.embedding_reg_weight * emb_reg
        )

        self.last_loss_components = {
            'total': total.detach(),
            'bpr_final': rec_loss.detach(),
            'bpr_reference_observed': ref_obs_loss.detach(),
            'bpr_reference_counterfactual': ref_cf_loss.detach(),
            'mask_budget': budget_loss.detach(),
            'mask_binary': binary_loss.detach(),
            'specialization': spec_loss.detach(),
            'branch_aux': branch_aux_loss.detach(),
            'embedding_reg': emb_reg.detach(),
        }
        return total

    # ==================================================================
    # Full-sort evaluation
    # ==================================================================
    def full_sort_predict(self, interaction):
        result, _, _, _ = self.compute_model_representations(
            hard_mask=self.eval_hard_mask
        )
        user_tensor = result[:self.num_user]
        item_tensor = result[self.num_user:]
        selected_users = user_tensor[interaction[0]]
        return torch.matmul(selected_users, item_tensor.t())

    # ==================================================================
    # Diagnostics
    # ==================================================================
    @torch.no_grad()
    def get_counterfactual_statistics(self):
        was_training = self.training
        self.eval()

        ref = self.compute_reference_views(
            hard_mask=self.eval_hard_mask
        )
        b1, b2 = self.compute_branch_representations(ref['undirected_mask'])

        soft = ref['soft_forward_mask']
        active = ref['forward_mask']
        effect = ref['effect']

        stats = {
            'reference_mode': self.reference_mode,
            'reference_frozen': self.freeze_reference,
            'mask_mean_soft': soft.mean().detach(),
            'mask_std_soft': soft.std(unbiased=False).detach(),
            'mask_min_soft': soft.min().detach(),
            'mask_max_soft': soft.max().detach(),
            'mask_binary_measure': (
                soft * (1.0 - soft)
            ).mean().detach(),
            'active_branch1_edge_mass': active.mean().detach(),
            'active_branch2_edge_mass': (1.0 - active).mean().detach(),
            'effect_norm_mean': effect.norm(dim=1).mean().detach(),
            'effect_user_norm_mean': effect[:self.num_user].norm(
                dim=1
            ).mean().detach(),
            'effect_item_norm_mean': effect[self.num_user:].norm(
                dim=1
            ).mean().detach(),
            'reference_obs_cf_cosine': F.cosine_similarity(
                ref['observed'],
                ref['counterfactual'],
                dim=1
            ).mean().detach(),
            'branch1_norm_mean': b1.norm(dim=1).mean().detach(),
            'branch2_norm_mean': b2.norm(dim=1).mean().detach(),
            'branch_cosine_mean': F.cosine_similarity(
                b1,
                b2,
                dim=1
            ).mean().detach(),
        }

        if was_training:
            self.train()
        return stats

    @torch.no_grad()
    def get_degree_overlap_statistics(self):
        """
        Diagnostic only. Aggregate learned edge masks by item mean and compare
        retained/suppressed Top-K items with degree Top-K items.
        """
        was_training = self.training
        self.eval()

        ref = self.compute_reference_views(hard_mask=False)
        edge_mask = ref['soft_forward_mask']

        mask_sum = torch.zeros(
            self.num_item,
            dtype=edge_mask.dtype,
            device=edge_mask.device
        )
        edge_count = torch.zeros_like(mask_sum)
        mask_sum.index_add_(0, self.forward_edge_items, edge_mask)
        edge_count.index_add_(
            0,
            self.forward_edge_items,
            torch.ones_like(edge_mask)
        )
        item_mask_mean = mask_sum / edge_count.clamp_min(1.0)

        k = max(1, int(self.num_item * self.degree_overlap_ratio))
        k = min(k, self.num_item)

        degree_top = torch.topk(self.item_degree, k=k).indices
        retained_top = torch.topk(item_mask_mean, k=k).indices
        suppressed_top = torch.topk(1.0 - item_mask_mean, k=k).indices

        degree_bool = torch.zeros(
            self.num_item,
            dtype=torch.bool,
            device=edge_mask.device
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

        x = item_mask_mean
        y = torch.log1p(self.item_degree.to(x.dtype))
        x_center = x - x.mean()
        y_center = y - y.mean()
        denominator = torch.sqrt(
            x_center.pow(2).sum() * y_center.pow(2).sum()
        ).clamp_min(1e-12)
        pearson = (x_center * y_center).sum() / denominator

        stats = {
            'k': torch.tensor(k, device=edge_mask.device),
            'item_mask_mean': item_mask_mean.detach(),
            'retained_vs_degree_overlap_at_k': (
                retained_intersection.float() / float(k)
            ).detach(),
            'retained_vs_degree_jaccard': (
                retained_intersection.float() / retained_union.float()
            ).detach(),
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
        print('\n========== CF2BRANCH Mask vs Degree ==========')
        print('K items                           : {}'.format(
            int(stats['k'].item())
        ))
        print('Retained Top-K vs degree overlap : {:.4f}'.format(
            float(stats['retained_vs_degree_overlap_at_k'].item())
        ))
        print('Retained Top-K vs degree Jaccard : {:.4f}'.format(
            float(stats['retained_vs_degree_jaccard'].item())
        ))
        print('Suppressed Top-K vs degree overlap: {:.4f}'.format(
            float(stats['suppressed_vs_degree_overlap_at_k'].item())
        ))
        print('Suppressed Top-K vs degree Jaccard: {:.4f}'.format(
            float(stats['suppressed_vs_degree_jaccard'].item())
        ))
        print('Mask vs log-degree Pearson        : {:.4f}'.format(
            float(stats['mask_degree_pearson_log_degree'].item())
        ))
        print('===============================================\n')
        return stats

    @torch.no_grad()
    def get_parameter_statistics(self):
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            'reference_trainable_parameters': sum(
                p.numel()
                for module in (
                    self.reference_item_embedding,
                    self.reference_gcn,
                )
                for p in module.parameters()
                if p.requires_grad
            ),
            'branch1_trainable_parameters': sum(
                p.numel()
                for module in (
                    self.item_embedding_branch1,
                    self.branch1_gcn,
                )
                for p in module.parameters()
                if p.requires_grad
            ),
            'branch2_trainable_parameters': sum(
                p.numel()
                for module in (
                    self.item_embedding_branch2,
                    self.branch2_gcn,
                )
                for p in module.parameters()
                if p.requires_grad
            ),
            'mask_trainable_parameters': sum(
                p.numel() for p in self.mask_mlp.parameters()
                if p.requires_grad
            ),
        }

    def set_mask_temperature(self, temperature):
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError('temperature must be > 0.')
        self.mask_temperature = temperature

    def clear_reference_cache(self):
        self._cached_reference_observed = None


class GCN(nn.Module):
    """
    LightGCN-style encoder.

    `preference` is the trainable USER embedding table for this particular
    encoder.  Therefore reference_gcn, branch1_gcn, and branch2_gcn each own
    different user parameters.
    """

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
    """
    Weighted LightGCN propagation with branch-specific degree normalization.

    For an edge weight w_ij, the propagated normalized edge weight is

        w_ij / sqrt(d_i^w d_j^w)

    where weighted degree d_i^w is computed from the ACTIVE branch mask.
    Therefore A*M and A*(1-M) receive their own normalizations.
    """

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

        edge_index, edge_mask = remove_self_loops(edge_index, edge_mask)

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
