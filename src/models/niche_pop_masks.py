# coding: utf-8
"""
GLORIA variant with TWO INDEPENDENT learnable edge masks initialized from
the degrees of BOTH nodes connected by each user-item interaction edge.

Degree-based initialization
---------------------------
For every interaction edge e=(u,i):

    d_u = normalized log user degree
    d_i = normalized log item degree

where both d_u and d_i are scaled to [0, 1].

View 1 starts from a NICHE / PERSONALIZATION prior:

    niche_score = d_u * (1 - d_i)

This is high for:
    active user -> low-degree / tail item

View 2 starts from a POPULARITY / EXPOSURE prior:

    popularity_score = (1 - d_u) * d_i

This is high for:
    low-activity user -> high-degree / popular item

The scores are mapped into soft mask probabilities:

    mask = mask_min + (mask_max - mask_min) * score

and converted to logits so they remain fully learnable parameters.

Important:
    Degree is ONLY an initialization prior.
    After initialization, both masks are optimized freely by BPR and are
    NOT constrained to preserve these degree semantics.

Final representation:
    Z_final = [Z_niche_view || Z_popularity_view]

Training objective:
    L = BPR_fused

This keeps exactly two GCN branches and does not add a third full-graph view.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender


def _cfg(config, key, default):
    """Read an optional config value without assuming config implements .get()."""
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


class NICHE_POP_MASKS(GeneralRecommender):
    def __init__(self, config, dataset):
        super(NICHE_POP_MASKS, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        print('number of users: {}, number of items: {}'.format(num_user, num_item))

        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']

        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']

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

        # -------------------------------------------------------------
        # Degree-based mask initialization settings.
        #
        # Initial mask probabilities are constrained to:
        #     [degree_mask_min, degree_mask_max]
        #
        # Example defaults:
        #     score = 0 -> mask = 0.10
        #     score = 1 -> mask = 0.90
        #
        # init_logit_noise adds a tiny amount of symmetry-breaking noise
        # AFTER converting initial probabilities into logits.
        # -------------------------------------------------------------
        self.degree_mask_min = float(
            _cfg(config, 'degree_mask_min', 0.10)
        )
        self.degree_mask_max = float(
            _cfg(config, 'degree_mask_max', 0.90)
        )
        self.init_logit_noise = float(
            _cfg(config, 'init_logit_noise', 1e-2)
        )

        if not (0.0 < self.degree_mask_min < self.degree_mask_max < 1.0):
            raise ValueError(
                "Require 0 < degree_mask_min < degree_mask_max < 1."
            )

        # -------------------------------------------------------------
        # Separate item embeddings for the niche- and popularity-initialized views.
        # They start identically but become independent after optimization.
        # -------------------------------------------------------------
        self.id_embedding_view1 = nn.Embedding(num_item, self.feat_embed_dim)
        self.id_embedding_view2 = nn.Embedding(num_item, self.feat_embed_dim)

        nn.init.xavier_uniform_(self.id_embedding_view1.weight)
        with torch.no_grad():
            self.id_embedding_view2.weight.copy_(
                self.id_embedding_view1.weight
            )

        # These were present in your original model. Keep them if the rest of
        # your project uses them, even though the causal/non-causal UI branch
        # below does not directly use them.
        self.mlp_item = nn.Linear(self.t_feat.shape[-1], self.dim_latent, bias=False)
        self.mlp_user = nn.Linear(self.user_feat.shape[-1], self.dim_latent, bias=False)

        # -------------------------------------------------------------
        # Item-item graph from text features (same as original model).
        # -------------------------------------------------------------
        _, text_adj = self.get_knn_adj_mat(self.t_feat)
        self.mm_adj = text_adj

        # -------------------------------------------------------------
        # Build ONE user-item graph.
        # pack_edge_index() returns E unique user -> item edges.
        # We then append the reverse direction, so self.edge_index has 2E edges.
        # -------------------------------------------------------------
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        forward_edges = self.pack_edge_index(train_interactions)  # [E, 2]

        self.num_interactions = forward_edges.shape[0]

        forward_edges = torch.tensor(
            forward_edges,
            dtype=torch.long,
            device=self.device
        ).t().contiguous()  # [2, E]

        reverse_edges = forward_edges[[1, 0], :]

        # Ordering is exactly [all u->i edges, all i->u edges].
        # Therefore torch.cat([mask, mask]) matches edge_index correctly.
        self.edge_index = torch.cat(
            [forward_edges, reverse_edges],
            dim=1
        )  # [2, 2E]

        # -------------------------------------------------------------
        # TWO independent learnable masks initialized from endpoint degree.
        #
        # For an interaction edge (u, i):
        #
        #   niche_score      = d_u * (1 - d_i)
        #   popularity_score = (1 - d_u) * d_i
        #
        # d_u and d_i are log-degree normalized to [0, 1].
        #
        # These are INITIALIZATION priors only. Once wrapped in nn.Parameter,
        # every edge logit is optimized freely by the recommender loss.
        # -------------------------------------------------------------
        user_degree = np.bincount(
            train_interactions.row,
            minlength=num_user
        ).astype(np.float32)

        item_degree = np.bincount(
            train_interactions.col,
            minlength=num_item
        ).astype(np.float32)

        # Compress heavy-tailed degree distributions.
        user_degree_log = np.log1p(user_degree)
        item_degree_log = np.log1p(item_degree)

        def minmax_normalize(values):
            v_min = values.min()
            v_max = values.max()
            return (values - v_min) / (v_max - v_min + 1e-8)

        user_degree_norm = minmax_normalize(user_degree_log)
        item_degree_norm = minmax_normalize(item_degree_log)

        # Endpoint degrees for each UNIQUE user-item interaction edge.
        edge_user_degree = torch.tensor(
            user_degree_norm[train_interactions.row],
            dtype=torch.float32,
            device=self.device
        )

        edge_item_degree = torch.tensor(
            item_degree_norm[train_interactions.col],
            dtype=torch.float32,
            device=self.device
        )

        # View 1: active-user -> tail-item prior.
        niche_score = (
            edge_user_degree
            * (1.0 - edge_item_degree)
        )

        # View 2: low-activity-user -> popular-item prior.
        popularity_score = (
            (1.0 - edge_user_degree)
            * edge_item_degree
        )

        mask_range = self.degree_mask_max - self.degree_mask_min

        view1_init_mask = (
            self.degree_mask_min
            + mask_range * niche_score
        )

        view2_init_mask = (
            self.degree_mask_min
            + mask_range * popularity_score
        )

        def probability_to_logit(prob):
            prob = prob.clamp(
                min=1e-4,
                max=1.0 - 1e-4
            )
            return torch.log(
                prob / (1.0 - prob)
            )

        view1_init_logits = probability_to_logit(
            view1_init_mask
        )

        view2_init_logits = probability_to_logit(
            view2_init_mask
        )

        if self.init_logit_noise > 0.0:
            view1_init_logits = (
                view1_init_logits
                + self.init_logit_noise
                * torch.randn_like(view1_init_logits)
            )

            view2_init_logits = (
                view2_init_logits
                + self.init_logit_noise
                * torch.randn_like(view2_init_logits)
            )

        self.view1_mask_logits = nn.Parameter(
            view1_init_logits
        )

        self.view2_mask_logits = nn.Parameter(
            view2_init_logits
        )

        # Buffers are useful for analysis/debugging but are not optimized.
        self.register_buffer(
            'edge_user_degree_norm',
            edge_user_degree
        )
        self.register_buffer(
            'edge_item_degree_norm',
            edge_item_degree
        )
        self.register_buffer(
            'initial_niche_score',
            niche_score
        )
        self.register_buffer(
            'initial_popularity_score',
            popularity_score
        )

        print(
            "Initial degree masks | "
            "view1(niche) mean: {:.4f}, "
            "view2(popularity) mean: {:.4f}".format(
                view1_init_mask.mean().item(),
                view2_init_mask.mean().item()
            )
        )

        # -------------------------------------------------------------
        # Two separate GCNs.
        # Each branch has its own learnable user preference embedding.
        # -------------------------------------------------------------
        self.view1_gcn = GCN(
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
            features=self.id_embedding_view1.weight
        )

        self.view2_gcn = GCN(
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
            features=self.id_embedding_view2.weight
        )

        self.result_embed = None
        self.loss_components = {}

    # -----------------------------------------------------------------
    # Graph utilities
    # -----------------------------------------------------------------
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        )
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim

        indices0 = torch.arange(knn_ind.shape[0], device=self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)

        indices = torch.stack(
            (torch.flatten(indices0), torch.flatten(knn_ind)),
            0
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
        r_inv_sqrt = torch.pow(row_sum, -0.5)

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

    # -----------------------------------------------------------------
    # Two independent soft masks
    # -----------------------------------------------------------------
    def get_masks(self):
        """
        Returns two independent learnable soft masks.

        Shapes:
            view1_mask: [E]  # niche-initialized view
            view2_mask: [E]  # popularity-initialized view

        Degree only determines the initialization. These masks are free
        parameters during optimization and do not remain tied to degree.
        """
        view1_mask = torch.sigmoid(
            self.view1_mask_logits
        )
        view2_mask = torch.sigmoid(
            self.view2_mask_logits
        )

        return view1_mask, view2_mask

    @staticmethod
    def to_bidirectional_mask(mask):
        """[E] -> [2E], matching [u->i, i->u] edge ordering."""
        return torch.cat([mask, mask], dim=0)

    # -----------------------------------------------------------------
    # Representation construction
    # -----------------------------------------------------------------
    def compute_branch_representations(self):
        """
        Returns:
            view1_rep:  [num_user + num_item, D]
            view2_rep:  [num_user + num_item, D]
            view1_mask: [E]
            view2_mask: [E]
        """
        view1_mask, view2_mask = self.get_masks()

        view1_edge_mask = self.to_bidirectional_mask(view1_mask)
        view2_edge_mask = self.to_bidirectional_mask(view2_mask)

        view1_rep, view1_preference = self.view1_gcn(
            self.edge_index,
            self.id_embedding_view1.weight,
            edge_mask=view1_edge_mask
        )

        view2_rep, view2_preference = self.view2_gcn(
            self.edge_index,
            self.id_embedding_view2.weight,
            edge_mask=view2_edge_mask
        )

        # Useful for debugging / analysis.
        self.view1_preference = view1_preference
        self.view2_preference = view2_preference

        return view1_rep, view2_rep, view1_mask, view2_mask

    def fuse_representations(self, view1_rep, view2_rep):
        """
        Concatenate the two learned graph views.

        View 1 starts from a niche/personalization degree prior.
        View 2 starts from a popularity/exposure degree prior.

        Users:
            [view1_user || view2_user]

        Items:
            [view1_item || view2_item]
            followed by the original text-based item-item propagation.
        """
        user_view1 = view1_rep[:self.num_user]
        user_view2 = view2_rep[:self.num_user]

        item_view1 = view1_rep[self.num_user:]
        item_view2 = view2_rep[self.num_user:]

        user_rep = torch.cat(
            [user_view1, user_view2],
            dim=1
        )

        item_rep = torch.cat(
            [item_view1, item_view2],
            dim=1
        )

        item_rep = self.item_item(item_rep)

        return torch.cat(
            [user_rep, item_rep],
            dim=0
        )

    # -----------------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------------
    def pairwise_scores(self, representation, interaction):
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        user_tensor = representation[user_nodes]
        pos_item_tensor = representation[pos_item_nodes]
        neg_item_tensor = representation[neg_item_nodes]

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        return pos_scores, neg_scores

    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        # Numerically stable equivalent of -log(sigmoid(pos-neg)).
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    # -----------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------
    def forward(self, interaction, return_aux=False):
        view1_rep, view2_rep, view1_mask, view2_mask = (
            self.compute_branch_representations()
        )

        self.view1_rep = view1_rep
        self.view2_rep = view2_rep

        self.result_embed = self.fuse_representations(
            view1_rep,
            view2_rep
        )

        pos_scores, neg_scores = self.pairwise_scores(
            self.result_embed,
            interaction
        )

        if not return_aux:
            return pos_scores, neg_scores

        aux = {
            'view1_mask': view1_mask,
            'view2_mask': view2_mask,
            'view1_rep': view1_rep,
            'view2_rep': view2_rep,
        }

        return pos_scores, neg_scores, aux

    # -----------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------
    def calculate_loss(self, interaction):
        """
        Train only with fused BPR.

        Degree is used only to initialize the two masks; no degree-preserving
        regularizer is imposed afterward.
        """
        pos_scores, neg_scores, aux = self.forward(
            interaction,
            return_aux=True
        )

        bpr_loss = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        view1_mask = aux['view1_mask']
        view2_mask = aux['view2_mask']

        # Diagnostics only. They do NOT contribute to the objective.
        with torch.no_grad():
            self.loss_components = {
                'bpr': float(
                    bpr_loss.detach().cpu()
                ),
                'view1_mask_mean': float(
                    view1_mask.mean().detach().cpu()
                ),
                'view2_mask_mean': float(
                    view2_mask.mean().detach().cpu()
                ),
                'mask_mean_abs_diff': float(
                    (view1_mask - view2_mask)
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                ),
                'view1_niche_alignment': float(
                    (
                        view1_mask
                        * self.initial_niche_score
                    ).mean().detach().cpu()
                ),
                'view2_popularity_alignment': float(
                    (
                        view2_mask
                        * self.initial_popularity_score
                    ).mean().detach().cpu()
                ),
            }

        return bpr_loss

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def full_sort_predict(self, interaction):
        view1_rep, view2_rep, _, _ = (
            self.compute_branch_representations()
        )

        self.result_embed = self.fuse_representations(
            view1_rep,
            view2_rep
        )

        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(
            temp_user_tensor,
            item_tensor.t()
        )

        return score_matrix


class GCN(torch.nn.Module):
    """
    LightGCN-style propagation module.

    Each branch owns a separate GCN instance, therefore each branch also owns
    a separate user preference embedding. Item embeddings are passed from the
    parent model and are separate between the two degree-initialized graph views.
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
        super(GCN, self).__init__()

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
            torch.empty(num_user, preference_dim)
        )
        nn.init.xavier_normal_(self.preference, gain=1.0)

        # Base_gcn has no learnable linear transform; dimensions are retained
        # for compatibility with your original class.
        self.conv_embed_1 = Base_gcn(
            preference_dim,
            preference_dim,
            aggr=self.aggr_mode
        )

    def forward(self, edge_index, features, edge_mask=None):
        temp_features = features
        temp_profile = self.preference

        x = torch.cat(
            (temp_profile, temp_features),
            dim=0
        )
        x = F.normalize(x, p=2, dim=1)

        # Preserve your original three propagation steps and residual sum.
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
    LightGCN-style normalized message passing with an optional scalar edge mask.

    If edge_mask is None, every edge receives weight 1 and this reduces to the
    original full-graph propagation.

    Message on edge j -> i:
        (1 / sqrt(d_j d_i)) * mask_e * x_j

    Degrees are computed from the original graph topology. The mask changes the
    message strength but does not re-normalize graph degrees.
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
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)

        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_mask=None, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        if size is None:
            size = (x.size(0), x.size(0))

        # Keep edge attributes aligned if self-loops ever appear.
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

            deg = degree(
                row,
                size[0],
                dtype=x.dtype
            )

            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(
                torch.isinf(deg_inv_sqrt),
                0.0
            )

            norm = (
                deg_inv_sqrt[row]
                * deg_inv_sqrt[col]
            )

            edge_weight = norm * edge_mask
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