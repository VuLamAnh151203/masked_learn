# coding: utf-8
r"""
TOPKROUTERC -- Reduced Experiment C: pretrained/frozen full-graph router features.

This is the direct counterpart of TOPKROUTER (Experiment B), with ONE controlled
change: the full-graph representation used by the router is pretrained in Stage
1 and then frozen in Stage 2.

Stage 1
-------
Train TOPKPRETRAIN on the complete interaction graph with ordinary BPR and save
the best checkpoint.

Stage 2
-------
    frozen pretrained full-graph encoder
                    |
               stable item h_i
                    |
                Router MLP
                 D -> D -> 1
                    |
              score s_i
                    |
         soft 90/10-constrained mask
              /             \
         branch 1         branch 2
        weight 1-m_i       weight m_i
              |              |
            GCN-1          GCN-2
               \            /
                 -- concat --
                       |
                item-item GCN
                       |
                      BPR

Important experimental controls
-------------------------------
* The pretrained router encoder is frozen for ALL of Stage 2.
* Its full-graph representation can be cached because it never changes.
* The router MLP and both recommendation branches train jointly in Stage 2.
* By default, both recommendation branches are initialized from the Stage-1
  pretrained item/user embeddings, then become independent trainable branches.
* Degree is NEVER used as router input or training supervision.
* Degree is computed only after routing for overlap/Jaccard diagnostics.
* Training uses differentiable soft 90/10 routing.
* Validation/test uses EXACT hard Top-K routing.

Loss
----
    L = L_BPR
        + lambda_budget * (mean(m) - rho)^2
        + lambda_binary * mean(m * (1-m))
        + optional branch-embedding regularization

Required Stage-2 config
-----------------------
    router_pretrained_checkpoint: /path/to/TOPKPRETRAIN-checkpoint.pth

The loader supports common checkpoint layouts containing `state_dict` or
`model_state_dict`, as well as a raw PyTorch state_dict.
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
    """Read an optional config value without assuming config implements get()."""
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


class TOPKROUTERC(GeneralRecommender):
    """
    Reduced Experiment C: pretrained frozen router representation + learnable
    constrained Top-K graph partition.

    Place this file at:
        src/models/topkrouterc.py

    and use model name:
        TOPKROUTERC
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        print(
            '[TOPKROUTERC] number of users: {}, number of items: {}'.format(
                num_user, num_item
            )
        )

        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']

        self.feat_embed_dim = int(config['feat_embed_dim'])
        self.n_layers = int(config['n_mm_layers'])
        self.knn_k = int(config['knn_k'])

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = config['aggr_mode']
        self.num_layer = 1
        self.dataset = dataset
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.dim_latent = 64
        self.mm_adj = None
        self.config = config

        # =============================================================
        # Reduced-C defaults
        # =============================================================
        self.router_high_ratio = float(
            _cfg(config, 'router_high_ratio', 0.10)
        )
        if not (0.0 < self.router_high_ratio < 1.0):
            raise ValueError('router_high_ratio must be in (0, 1).')

        self.router_temperature = float(
            _cfg(config, 'router_temperature', 1.0)
        )
        if self.router_temperature <= 0.0:
            raise ValueError('router_temperature must be > 0.')

        self.router_hidden_dim = int(
            _cfg(config, 'router_hidden_dim', self.feat_embed_dim)
        )

        self.router_budget_weight = float(
            _cfg(config, 'router_budget_weight', 0.10)
        )
        self.router_binary_weight = float(
            _cfg(config, 'router_binary_weight', 0.001)
        )
        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        # The whole point of reduced C is that router features never move.
        self.cache_pretrained_router_representation = _as_bool(
            _cfg(config, 'router_cache_pretrained_representation', True)
        )

        # Start both trainable branches from the same useful pretrained state.
        self.initialize_branches_from_pretrained = _as_bool(
            _cfg(config, 'router_init_branches_from_pretrained', True)
        )

        self.threshold_trainable = _as_bool(
            _cfg(config, 'router_threshold_trainable', True)
        )

        self.print_degree_overlap_on_eval = _as_bool(
            _cfg(config, 'router_print_degree_overlap', True)
        )
        self._last_printed_hard_selection = None

        # checkpoint_path = _cfg(
        #     config,
        #     'router_pretrained_checkpoint',
        #     None
        # )
        checkpoint_path = "C:\\study\\Master_papers\\Robust MM\\CaMuRe\\saved\\TOPKPRETRAIN-book-seed999-Aug-11-2026-11-12-09.pth"
        
        if checkpoint_path is None or str(checkpoint_path).strip() == '':
            raise ValueError(
                'Reduced Experiment C requires `router_pretrained_checkpoint`. '
                'First train TOPKPRETRAIN, then set this config to its saved '
                'checkpoint path.'
            )
        self.router_pretrained_checkpoint = os.path.abspath(
            os.path.expanduser(str(checkpoint_path))
        )
        if not os.path.isfile(self.router_pretrained_checkpoint):
            raise FileNotFoundError(
                'Pretrained checkpoint not found: {}'.format(
                    self.router_pretrained_checkpoint
                )
            )

        rho = self.router_high_ratio
        initial_threshold = (
            self.router_temperature * math.log((1.0 - rho) / rho)
        )
        threshold_tensor = torch.tensor(
            float(initial_threshold), dtype=torch.float32
        )
        if self.threshold_trainable:
            self.router_threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer('router_threshold', threshold_tensor)

        self.num_high_items = max(
            1,
            int(self.num_item * self.router_high_ratio)
        )
        self.num_high_items = min(
            self.num_high_items,
            max(1, self.num_item - 1)
        )

        # =============================================================
        # Frozen Stage-1 full-graph encoder parameters.
        # =============================================================
        self.router_item_embedding = nn.Embedding(
            num_item, self.feat_embed_dim
        )

        # Trainable branch parameters.
        self.id_embedding_branch1 = nn.Embedding(
            num_item, self.feat_embed_dim
        )
        self.id_embedding_branch2 = nn.Embedding(
            num_item, self.feat_embed_dim
        )

        # Preserve GLORIA-side attributes expected elsewhere in the project.
        self.mlp_item = nn.Linear(
            self.t_feat.shape[-1], self.dim_latent, bias=False
        )
        self.mlp_user = nn.Linear(
            self.user_feat.shape[-1], self.dim_latent, bias=False
        )

        # Original item-item graph.
        _, text_adj = self.get_knn_adj_mat(self.t_feat)
        self.mm_adj = text_adj

        # =============================================================
        # Complete user-item graph + diagnostic degree baseline.
        # =============================================================
        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)

        forward_edges_np = self.pack_edge_index(train_interactions)
        self.num_interactions = int(forward_edges_np.shape[0])

        edge_item_ids = torch.tensor(
            train_interactions.col,
            dtype=torch.long
        )
        self.register_buffer('edge_item_ids', edge_item_ids)

        # Degree is DIAGNOSTIC ONLY.
        item_degree_np = np.bincount(
            train_interactions.col,
            minlength=self.num_item
        ).astype(np.float32)
        self.register_buffer(
            'item_degree',
            torch.from_numpy(item_degree_np)
        )

        degree_high_np = np.argsort(item_degree_np)[
            -self.num_high_items:
        ].astype(np.int64)

        degree_high_mask_np = np.zeros(
            self.num_item, dtype=np.bool_
        )
        degree_high_mask_np[degree_high_np] = True

        self.register_buffer(
            'degree_high_indices',
            torch.from_numpy(degree_high_np)
        )
        self.register_buffer(
            'degree_high_mask',
            torch.from_numpy(degree_high_mask_np)
        )

        forward_edges = torch.tensor(
            forward_edges_np,
            dtype=torch.long
        ).t().contiguous()
        reverse_edges = forward_edges[[1, 0], :]
        self.register_buffer(
            'edge_index',
            torch.cat([forward_edges, reverse_edges], dim=1)
        )

        # =============================================================
        # Frozen pretrained full-graph router encoder.
        # =============================================================
        self.router_gcn = GCN(
            self.dataset,
            batch_size,
            num_user,
            num_item,
            dim_x,
            self.aggr_mode,
            num_layer=self.num_layer,
            has_feature=False,
            dropout=self.drop_rate,
            dim_latent=self.dim_latent,
            device=self.device,
            features=self.router_item_embedding.weight
        )

        # =============================================================
        # Trainable scalar ranking function s_i = f_phi(h_i^pre).
        # =============================================================
        self.router_mlp = nn.Sequential(
            nn.Linear(
                self.feat_embed_dim,
                self.router_hidden_dim
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.router_hidden_dim, 1)
        )
        self._initialize_router_mlp()

        # =============================================================
        # Two independent trainable recommendation branches.
        # =============================================================
        self.branch1_gcn = GCN(
            self.dataset,
            batch_size,
            num_user,
            num_item,
            dim_x,
            self.aggr_mode,
            num_layer=self.num_layer,
            has_feature=False,
            dropout=self.drop_rate,
            dim_latent=self.dim_latent,
            device=self.device,
            features=self.id_embedding_branch1.weight
        )

        self.branch2_gcn = GCN(
            self.dataset,
            batch_size,
            num_user,
            num_item,
            dim_x,
            self.aggr_mode,
            num_layer=self.num_layer,
            has_feature=False,
            dropout=self.drop_rate,
            dim_latent=self.dim_latent,
            device=self.device,
            features=self.id_embedding_branch2.weight
        )

        # Load Stage-1 tensors before branch initialization/freeze.
        self._load_pretrained_router_checkpoint(
            self.router_pretrained_checkpoint
        )

        if self.initialize_branches_from_pretrained:
            self._initialize_branches_from_pretrained_router()
        else:
            nn.init.xavier_uniform_(self.id_embedding_branch1.weight)
            nn.init.xavier_uniform_(self.id_embedding_branch2.weight)

        self._freeze_router_encoder()

        self.result_embed = None
        self.last_router_rep = None
        self.last_router_scores = None
        self.last_soft_item_masks = None
        self.last_hard_item_masks = None
        self.last_degree_overlap = None
        self.last_training_diagnostics = None

        # Lazy cache: created only after the model has been moved to its final
        # device by the outer training framework.
        self._cached_router_rep = None

        print(
            '[TOPKROUTERC] loaded frozen router checkpoint: {}'.format(
                self.router_pretrained_checkpoint
            )
        )
        print(
            '[TOPKROUTERC] branches initialized from pretrained: {}'.format(
                self.initialize_branches_from_pretrained
            )
        )

    # =================================================================
    # Checkpoint loading / frozen Stage-1 encoder
    # =================================================================
    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict'):
                value = checkpoint.get(key, None)
                if isinstance(value, dict):
                    return value

            # Some training code stores the state dict under `model`.
            value = checkpoint.get('model', None)
            if isinstance(value, dict):
                return value

            # Raw state_dict case.
            if checkpoint and all(
                isinstance(k, str) for k in checkpoint.keys()
            ) and any(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint

        raise ValueError(
            'Could not find a PyTorch state_dict in the pretrained checkpoint.'
        )

    @staticmethod
    def _strip_module_prefix(state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _find_checkpoint_tensor(state_dict, candidate_keys, description):
        for key in candidate_keys:
            if key in state_dict and torch.is_tensor(state_dict[key]):
                return state_dict[key], key
        raise KeyError(
            'Could not find {} in checkpoint. Tried keys: {}'.format(
                description,
                candidate_keys
            )
        )

    def _load_pretrained_router_checkpoint(self, checkpoint_path):
        # This checkpoint is expected to be produced locally by TOPKPRETRAIN.
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu'
            )

        state_dict = self._strip_module_prefix(
            self._extract_state_dict(checkpoint)
        )

        item_tensor, item_key = self._find_checkpoint_tensor(
            state_dict,
            (
                'router_item_embedding.weight',
                'item_embedding.weight',
                'id_embedding.weight',
            ),
            'pretrained item embedding'
        )

        user_tensor, user_key = self._find_checkpoint_tensor(
            state_dict,
            (
                'router_gcn.preference',
                'gcn.preference',
                'full_gcn.preference',
            ),
            'pretrained user preference embedding'
        )

        expected_item_shape = tuple(self.router_item_embedding.weight.shape)
        expected_user_shape = tuple(self.router_gcn.preference.shape)

        if tuple(item_tensor.shape) != expected_item_shape:
            raise ValueError(
                'Pretrained item embedding shape {} from key `{}` does not '
                'match current dataset/model shape {}.'.format(
                    tuple(item_tensor.shape),
                    item_key,
                    expected_item_shape
                )
            )

        if tuple(user_tensor.shape) != expected_user_shape:
            raise ValueError(
                'Pretrained user preference shape {} from key `{}` does not '
                'match current dataset/model shape {}.'.format(
                    tuple(user_tensor.shape),
                    user_key,
                    expected_user_shape
                )
            )

        with torch.no_grad():
            self.router_item_embedding.weight.copy_(
                item_tensor.to(
                    dtype=self.router_item_embedding.weight.dtype,
                    device=self.router_item_embedding.weight.device
                )
            )
            self.router_gcn.preference.copy_(
                user_tensor.to(
                    dtype=self.router_gcn.preference.dtype,
                    device=self.router_gcn.preference.device
                )
            )

        self.pretrained_item_key = item_key
        self.pretrained_user_key = user_key

    def _initialize_branches_from_pretrained_router(self):
        with torch.no_grad():
            self.id_embedding_branch1.weight.copy_(
                self.router_item_embedding.weight
            )
            self.id_embedding_branch2.weight.copy_(
                self.router_item_embedding.weight
            )
            self.branch1_gcn.preference.copy_(
                self.router_gcn.preference
            )
            self.branch2_gcn.preference.copy_(
                self.router_gcn.preference
            )

    def _freeze_router_encoder(self):
        for parameter in self.router_item_embedding.parameters():
            parameter.requires_grad_(False)
        for parameter in self.router_gcn.parameters():
            parameter.requires_grad_(False)
        self.router_gcn.eval()

    def clear_pretrained_router_cache(self):
        """Clear the lazily cached fixed full-graph representation."""
        self._cached_router_rep = None

    # =================================================================
    # Initialization
    # =================================================================
    def _initialize_router_mlp(self):
        first = self.router_mlp[0]
        last = self.router_mlp[2]

        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)

        # Small final layer => scores initially near 0, making threshold
        # initialization control the initial ~90/10 soft mass.
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last.bias)

    # =================================================================
    # Graph utilities
    # =================================================================
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

        _, knn_ind = torch.topk(
            sim,
            self.knn_k,
            dim=-1
        )

        adj_size = sim.size()
        del sim

        indices0 = torch.arange(
            knn_ind.shape[0],
            device=mm_embeddings.device
        )
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)

        indices = torch.stack(
            (
                torch.flatten(indices0),
                torch.flatten(knn_ind)
            ),
            0
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

        row_sum = 1e-7 + torch.sparse.sum(
            adj,
            -1
        ).to_dense()

        r_inv_sqrt = torch.pow(
            row_sum,
            -0.5
        )

        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        norm_values = rows_inv_sqrt * cols_inv_sqrt

        return torch.sparse_coo_tensor(
            indices,
            norm_values,
            adj_size,
            device=indices.device
        ).coalesce()

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def item_item(self, rep):
        h = rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return rep + h

    @staticmethod
    def _to_bidirectional_mask(forward_mask):
        """Match [forward edges, reverse edges] edge_index ordering."""
        return torch.cat(
            [forward_mask, forward_mask],
            dim=0
        )

    # =================================================================
    # Full-graph router
    # =================================================================
    def compute_router_representation(self):
        """
        Return the FIXED Stage-1 full-graph representation.

        Because the pretrained encoder is frozen, this representation is
        detached and (by default) cached after its first computation.
        """
        if (
            self.cache_pretrained_router_representation
            and self._cached_router_rep is not None
        ):
            router_rep = self._cached_router_rep
            self.last_router_rep = router_rep
            return router_rep

        # Ensure the frozen feature extractor never tracks gradients.
        self.router_gcn.eval()
        with torch.no_grad():
            router_rep, router_preference = self.router_gcn(
                self.edge_index,
                self.router_item_embedding.weight,
                edge_mask=None
            )
            router_rep = router_rep.detach()

        self.last_router_rep = router_rep
        self.router_preference = router_preference.detach()

        if self.cache_pretrained_router_representation:
            self._cached_router_rep = router_rep

        return router_rep

    def compute_router_scores(self, router_rep=None):
        """
        Produce one scalar ranking score per item:

            s_i = MLP(normalize(h_i))

        Degree is not used.
        """
        if router_rep is None:
            router_rep = self.compute_router_representation()

        item_h = router_rep[self.num_user:]
        item_h = F.normalize(
            item_h,
            p=2,
            dim=1,
            eps=1e-8
        )

        scores = self.router_mlp(item_h).squeeze(-1)
        self.last_router_scores = scores
        return scores

    # =================================================================
    # Soft train mask / hard evaluation Top-K mask
    # =================================================================
    def get_soft_item_masks(self, scores=None):
        """
        Differentiable training masks.

            high_i = sigmoid((s_i - tau) / T)
            low_i  = 1 - high_i

        The budget loss pushes mean(high_i) toward router_high_ratio=0.10.

        Returns:
            masks: [num_item, 2], columns = [branch1, branch2]
        """
        if scores is None:
            scores = self.compute_router_scores()

        temperature = max(
            float(self.router_temperature),
            1e-6
        )

        high_mask = torch.sigmoid(
            (scores - self.router_threshold) / temperature
        )
        low_mask = 1.0 - high_mask

        masks = torch.stack(
            [low_mask, high_mask],
            dim=1
        )

        self.last_soft_item_masks = masks
        return masks

    def get_hard_item_masks(self, scores=None):
        """
        Exact hard Top-K masks used for validation/test.

        EXACTLY K items with the largest learned scores are routed to branch 2.
        The remaining items are routed to branch 1.
        """
        if scores is None:
            scores = self.compute_router_scores()

        topk_indices = torch.topk(
            scores,
            k=self.num_high_items,
            largest=True,
            sorted=False
        ).indices

        high_mask = torch.zeros_like(scores)
        high_mask[topk_indices] = 1.0
        low_mask = 1.0 - high_mask

        masks = torch.stack(
            [low_mask, high_mask],
            dim=1
        )

        self.last_hard_item_masks = masks
        return masks

    def get_forward_edge_masks(self, item_masks):
        """Convert item masks [N_item,2] to forward-edge masks [E,2]."""
        return item_masks[self.edge_item_ids]

    # =================================================================
    # Two recommendation branches
    # =================================================================
    def compute_branch_representations(self, hard=False):
        """
        hard=False (training): differentiable weighted graphs.
        hard=True  (eval)    : exact disjoint 90/10 item partition.
        """
        router_rep = self.compute_router_representation()
        scores = self.compute_router_scores(router_rep)

        if hard:
            item_masks = self.get_hard_item_masks(scores)
        else:
            item_masks = self.get_soft_item_masks(scores)

        edge_masks = self.get_forward_edge_masks(item_masks)

        branch1_edge_mask = self._to_bidirectional_mask(
            edge_masks[:, 0]
        )
        branch2_edge_mask = self._to_bidirectional_mask(
            edge_masks[:, 1]
        )

        # IMPORTANT: Base_gcn below recomputes WEIGHTED degrees separately for
        # each branch. This makes the soft training graph a faithful relaxation
        # of the hard graph split instead of normalizing with full-graph degree.
        branch1_rep, branch1_preference = self.branch1_gcn(
            self.edge_index,
            self.id_embedding_branch1.weight,
            edge_mask=branch1_edge_mask
        )

        branch2_rep, branch2_preference = self.branch2_gcn(
            self.edge_index,
            self.id_embedding_branch2.weight,
            edge_mask=branch2_edge_mask
        )

        self.branch1_rep = branch1_rep
        self.branch2_rep = branch2_rep
        self.branch1_preference = branch1_preference
        self.branch2_preference = branch2_preference
        self.last_item_masks = item_masks
        self.last_edge_masks = edge_masks

        return branch1_rep, branch2_rep, item_masks, scores

    def fuse_representations(self, branch1_rep, branch2_rep):
        """Preserve GLORIA's concat + item-item GCN fusion."""
        user_rep_1 = branch1_rep[:self.num_user]
        user_rep_2 = branch2_rep[:self.num_user]

        item_rep_1 = branch1_rep[self.num_user:]
        item_rep_2 = branch2_rep[self.num_user:]

        user_rep = torch.cat(
            [user_rep_1, user_rep_2],
            dim=1
        )

        item_rep = torch.cat(
            [item_rep_1, item_rep_2],
            dim=1
        )

        item_rep = self.item_item(item_rep)

        return torch.cat(
            [user_rep, item_rep],
            dim=0
        )

    # =================================================================
    # Recommendation scoring
    # =================================================================
    def pairwise_scores(self, representation, interaction):
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        user_tensor = representation[user_nodes]
        pos_item_tensor = representation[pos_item_nodes]
        neg_item_tensor = representation[neg_item_nodes]

        pos_scores = torch.sum(
            user_tensor * pos_item_tensor,
            dim=1
        )
        neg_scores = torch.sum(
            user_tensor * neg_item_tensor,
            dim=1
        )

        return pos_scores, neg_scores

    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        # Stable natural-log BPR. If your degree baseline uses log2 exactly,
        # the optimum is unchanged; only the constant loss scale differs.
        return -F.logsigmoid(
            pos_scores - neg_scores
        ).mean()

    # =================================================================
    # 90/10 regularization
    # =================================================================
    def mask_budget_loss(self, item_masks):
        """Force branch-2 soft item mass toward rho (=0.10 by default)."""
        high_mask = item_masks[:, 1]
        target = torch.as_tensor(
            self.router_high_ratio,
            device=high_mask.device,
            dtype=high_mask.dtype
        )
        return (high_mask.mean() - target).pow(2)

    @staticmethod
    def mask_binary_loss(item_masks):
        """
        Minimize m(1-m):
            m=0 or 1 -> 0
            m=0.5    -> 0.25

        This weakly encourages the soft relaxation to become more discrete.
        """
        high_mask = item_masks[:, 1]
        return (
            high_mask * (1.0 - high_mask)
        ).mean()

    def embedding_regularization_loss(self):
        # Only regularize trainable recommendation-branch embeddings here.
        if self.embedding_reg_weight <= 0.0:
            return self.router_threshold * 0.0

        reg = (
            self.id_embedding_branch1.weight.pow(2).mean()
            + self.id_embedding_branch2.weight.pow(2).mean()
            + self.branch1_gcn.preference.pow(2).mean()
            + self.branch2_gcn.preference.pow(2).mean()
        )
        return reg

    # =================================================================
    # Main forward / loss
    # =================================================================
    def forward(self, interaction, hard=None, return_aux=False):
        if hard is None:
            hard = not self.training

        (
            branch1_rep,
            branch2_rep,
            item_masks,
            router_scores
        ) = self.compute_branch_representations(hard=hard)

        self.result_embed = self.fuse_representations(
            branch1_rep,
            branch2_rep
        )

        pos_scores, neg_scores = self.pairwise_scores(
            self.result_embed,
            interaction
        )

        if not return_aux:
            return pos_scores, neg_scores

        aux = {
            'branch1_rep': branch1_rep,
            'branch2_rep': branch2_rep,
            'item_masks': item_masks,
            'router_scores': router_scores,
            'hard': hard,
        }
        return pos_scores, neg_scores, aux

    def calculate_loss(self, interaction):
        """
        REDUCED EXPERIMENT C: the full-graph router representation is frozen,
        while router MLP + branch1 + branch2 are trained jointly.

        Training ALWAYS uses soft masks so gradients reach the router MLP.
        """
        pos_scores, neg_scores, aux = self.forward(
            interaction,
            hard=False,
            return_aux=True
        )

        recommendation_loss = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        budget_loss = self.mask_budget_loss(
            aux['item_masks']
        )

        binary_loss = self.mask_binary_loss(
            aux['item_masks']
        )

        embedding_reg = self.embedding_regularization_loss()

        total_loss = (
            recommendation_loss
            + self.router_budget_weight * budget_loss
            + self.router_binary_weight * binary_loss
            + self.embedding_reg_weight * embedding_reg
        )

        # Save detached scalar diagnostics for easy inspection.
        with torch.no_grad():
            high_soft = aux['item_masks'][:, 1]
            self.last_training_diagnostics = {
                'bpr_loss': recommendation_loss.detach(),
                'budget_loss': budget_loss.detach(),
                'binary_loss': binary_loss.detach(),
                'soft_high_usage': high_soft.mean().detach(),
                'router_threshold': self.router_threshold.detach().clone(),
                'router_score_mean': aux['router_scores'].mean().detach(),
                'router_score_std': aux['router_scores'].std(
                    unbiased=False
                ).detach(),
            }

        return total_loss

    # =================================================================
    # Evaluation: exact hard Top-10%
    # =================================================================
    def full_sort_predict(self, interaction):
        # Explicit hard=True even if an external trainer forgets model.eval().
        (
            branch1_rep,
            branch2_rep,
            hard_masks,
            scores
        ) = self.compute_branch_representations(hard=True)

        self.result_embed = self.fuse_representations(
            branch1_rep,
            branch2_rep
        )

        # Cache overlap against the original degree split for inspection.
        self.last_degree_overlap = self._degree_overlap_from_scores(
            scores
        )
        self._maybe_print_eval_overlap(hard_masks)

        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[
            interaction[0],
            :
        ]

        score_matrix = torch.matmul(
            temp_user_tensor,
            item_tensor.t()
        )
        return score_matrix

    # =================================================================
    # Degree-overlap diagnostics (degree is NEVER used for routing)
    # =================================================================
    @torch.no_grad()
    def _maybe_print_eval_overlap(self, hard_masks):
        if not self.print_degree_overlap_on_eval:
            return

        current_selection = (
            hard_masks[:, 1] > 0.5
        ).detach().cpu()

        if (
            self._last_printed_hard_selection is not None
            and torch.equal(
                current_selection,
                self._last_printed_hard_selection
            )
        ):
            return

        self._last_printed_hard_selection = current_selection.clone()
        stats = self.last_degree_overlap

        print(
            '[TOPKROUTERC] learned-vs-degree Top-{}: '
            'intersection={} | overlap@K={:.4f} | jaccard={:.4f} | '
            'soft_high_usage={:.4f}'.format(
                self.num_high_items,
                int(stats['intersection_count'].item()),
                float(stats['overlap_ratio'].item()),
                float(stats['jaccard'].item()),
                float(stats['soft_high_usage'].item())
            )
        )

    @torch.no_grad()
    def _degree_overlap_from_scores(self, scores):
        learned_indices = torch.topk(
            scores,
            k=self.num_high_items,
            largest=True,
            sorted=False
        ).indices

        learned_mask = torch.zeros(
            self.num_item,
            device=scores.device,
            dtype=torch.bool
        )
        learned_mask[learned_indices] = True

        degree_mask = self.degree_high_mask.to(scores.device)

        intersection_mask = learned_mask & degree_mask
        union_mask = learned_mask | degree_mask

        intersection_count = intersection_mask.sum()
        union_count = union_mask.sum()

        k_tensor = torch.as_tensor(
            float(self.num_high_items),
            device=scores.device
        )

        overlap_ratio = (
            intersection_count.to(torch.float32) / k_tensor
        )

        jaccard = (
            intersection_count.to(torch.float32)
            / union_count.clamp_min(1).to(torch.float32)
        )

        item_degree = self.item_degree.to(scores.device)
        degree_indices = self.degree_high_indices.to(scores.device)

        learned_degrees = item_degree[learned_indices]
        degree_topk_degrees = item_degree[degree_indices]

        # Soft training mass is useful to see whether the 90/10 constraint is
        # actually being respected before hard Top-K evaluation.
        soft_masks = self.get_soft_item_masks(scores)
        soft_high_usage = soft_masks[:, 1].mean()

        stats = {
            'k': torch.tensor(
                self.num_high_items,
                device=scores.device,
                dtype=torch.long
            ),
            'target_high_ratio': torch.tensor(
                self.router_high_ratio,
                device=scores.device,
                dtype=torch.float32
            ),
            'learned_topk_indices': learned_indices.detach(),
            'degree_topk_indices': degree_indices.detach(),
            'intersection_indices': torch.nonzero(
                intersection_mask,
                as_tuple=False
            ).flatten().detach(),
            'intersection_count': intersection_count.detach(),
            'overlap_ratio': overlap_ratio.detach(),
            'jaccard': jaccard.detach(),
            'soft_high_usage': soft_high_usage.detach(),
            'hard_high_usage': torch.tensor(
                float(self.num_high_items) / float(self.num_item),
                device=scores.device
            ),
            'mean_degree_learned_topk': (
                learned_degrees.mean().detach()
                if learned_degrees.numel() > 0
                else torch.tensor(float('nan'), device=scores.device)
            ),
            'median_degree_learned_topk': (
                learned_degrees.median().detach()
                if learned_degrees.numel() > 0
                else torch.tensor(float('nan'), device=scores.device)
            ),
            'mean_degree_degree_topk': (
                degree_topk_degrees.mean().detach()
                if degree_topk_degrees.numel() > 0
                else torch.tensor(float('nan'), device=scores.device)
            ),
            'router_score_mean': scores.mean().detach(),
            'router_score_std': scores.std(
                unbiased=False
            ).detach(),
            'router_threshold': self.router_threshold.detach().clone(),
        }
        return stats

    @torch.no_grad()
    def get_degree_overlap_statistics(self):
        """
        Recompute the current hard learned Top-K and compare it with the fixed
        degree Top-K baseline.

        Example:

            stats = model.get_degree_overlap_statistics()
            print('overlap:', stats['overlap_ratio'].item())
            print('jaccard:', stats['jaccard'].item())
        """
        was_training = self.training
        self.eval()

        router_rep = self.compute_router_representation()
        scores = self.compute_router_scores(router_rep)
        stats = self._degree_overlap_from_scores(scores)
        self.last_degree_overlap = stats

        if was_training:
            self.train()

        return stats

    @torch.no_grad()
    def print_degree_overlap_statistics(self):
        """Convenience printer for the requested learned-vs-degree overlap."""
        stats = self.get_degree_overlap_statistics()

        print('\n========== Learned Top-K vs Degree Top-K ==========')
        print('K selected items       : {}'.format(
            int(stats['k'].item())
        ))
        print('Target branch-2 ratio  : {:.4f}'.format(
            float(stats['target_high_ratio'].item())
        ))
        print('Soft branch-2 usage    : {:.4f}'.format(
            float(stats['soft_high_usage'].item())
        ))
        print('Hard branch-2 usage    : {:.4f}'.format(
            float(stats['hard_high_usage'].item())
        ))
        print('Intersection count     : {}'.format(
            int(stats['intersection_count'].item())
        ))
        print('Overlap@K              : {:.4f}'.format(
            float(stats['overlap_ratio'].item())
        ))
        print('Jaccard                 : {:.4f}'.format(
            float(stats['jaccard'].item())
        ))
        print('Mean degree learned K  : {:.4f}'.format(
            float(stats['mean_degree_learned_topk'].item())
        ))
        print('Mean degree degree K   : {:.4f}'.format(
            float(stats['mean_degree_degree_topk'].item())
        ))
        print('Router score std       : {:.6f}'.format(
            float(stats['router_score_std'].item())
        ))
        print('===================================================\n')

        return stats

    # =================================================================
    # Other useful diagnostics
    # =================================================================
    @torch.no_grad()
    def get_router_statistics(self):
        """Return routing/budget diagnostics without using degree for routing."""
        was_training = self.training
        self.eval()

        router_rep = self.compute_router_representation()
        scores = self.compute_router_scores(router_rep)
        soft_masks = self.get_soft_item_masks(scores)
        hard_masks = self.get_hard_item_masks(scores)

        high_soft = soft_masks[:, 1]
        high_hard = hard_masks[:, 1]

        stats = {
            'soft_high_usage': high_soft.mean().detach(),
            'hard_high_usage': high_hard.mean().detach(),
            'soft_high_min': high_soft.min().detach(),
            'soft_high_max': high_soft.max().detach(),
            'soft_high_std': high_soft.std(unbiased=False).detach(),
            'router_score_mean': scores.mean().detach(),
            'router_score_std': scores.std(unbiased=False).detach(),
            'router_threshold': self.router_threshold.detach().clone(),
            'num_hard_branch1_items': (
                hard_masks[:, 0] > 0.5
            ).sum().detach(),
            'num_hard_branch2_items': (
                hard_masks[:, 1] > 0.5
            ).sum().detach(),
        }

        if was_training:
            self.train()
        return stats

    @torch.no_grad()
    def get_pretrained_router_statistics(self):
        """Verify that the Stage-1 encoder is frozen and report its source."""
        router_rep = self.compute_router_representation()
        frozen_params = list(self.router_item_embedding.parameters()) + list(
            self.router_gcn.parameters()
        )
        all_frozen = all(not p.requires_grad for p in frozen_params)

        return {
            'all_router_encoder_params_frozen': all_frozen,
            'cache_enabled': self.cache_pretrained_router_representation,
            'cache_populated': self._cached_router_rep is not None,
            'router_rep_norm_mean': router_rep.norm(dim=1).mean().detach(),
            'pretrained_item_key': self.pretrained_item_key,
            'pretrained_user_key': self.pretrained_user_key,
            'checkpoint_path': self.router_pretrained_checkpoint,
        }

    def set_router_temperature(self, temperature):
        """
        Optional manual temperature annealing hook.

        Example from the training script:
            model.set_router_temperature(0.5)
        """
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError('temperature must be > 0.')
        self.router_temperature = temperature


class GCN(nn.Module):
    """
    LightGCN-style encoder with its own trainable user preference embedding.

    For the router encoder, edge_mask=None => complete full graph.
    For the two recommendation branches, edge_mask gives soft/hard graph
    membership.
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

        if self.has_feature:
            preference_dim = self.dim_latent
        else:
            preference_dim = self.dim_feat

        self.preference = nn.Parameter(
            torch.empty(
                num_user,
                preference_dim
            )
        )
        nn.init.xavier_normal_(
            self.preference,
            gain=1.0
        )

        self.conv_embed_1 = Base_gcn(
            preference_dim,
            preference_dim,
            aggr=self.aggr_mode
        )

    def forward(self, edge_index, features, edge_mask=None):
        x = torch.cat(
            [self.preference, features],
            dim=0
        )

        x = F.normalize(
            x,
            p=2,
            dim=1
        )

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

        x_hat = x + h + h_1 + h_2
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    """
    LightGCN message passing with optional differentiable edge weights.

    IMPORTANT difference from the earlier soft-mask prototype:
    branch degrees are recomputed from the branch edge weights:

        d_v = sum_e w_e

    and each message uses:

        w_ij / sqrt(d_i d_j)

    Therefore as masks approach 0/1, the soft propagation approaches the
    normalization of the corresponding hard subgraph.
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
        super().__init__(
            aggr=aggr,
            **kwargs
        )
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

            # Weighted degree. Because edge_index contains both directions,
            # row-based outgoing weighted degree equals ordinary undirected
            # weighted degree for this symmetric graph.
            deg = torch.zeros(
                size[0],
                device=x.device,
                dtype=x.dtype
            )
            deg = deg.index_add(
                0,
                row,
                edge_mask
            )

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
