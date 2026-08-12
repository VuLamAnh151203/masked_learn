# coding: utf-8
"""
GLORIA variant with TWO INDEPENDENT learnable edge-mask views.

Goal
----
Keep exactly two GCN branches, but replace the strict CaGE complement
construction M and (1 - M) with:

    view1_mask = sigmoid(view1_mask_logits)
    view2_mask = sigmoid(view2_mask_logits)

View 1 starts close to the FULL graph:
    view1_mask_logits ~= +3  ->  sigmoid(3) ~= 0.953

View 2 starts as a neutral soft mask:
    view2_mask_logits ~= 0   ->  sigmoid(0) = 0.5

Both masks are trainable. Therefore, training starts close to the strong
"full graph + learned mask" architecture, while still allowing the first
branch to deviate slightly from the full graph if this helps recommendation.

Final representation:
    Z_final = [Z_view1 || Z_view2]

Training objective:
    L = BPR_fused
        + view1_preserve_weight
          * (mean(view1_mask) - view1_target_ratio)^2

The preservation term is intentionally tiny and can be disabled by setting:
    view1_preserve_weight: 0.0

The implementation keeps:
  * one bidirectional user-item graph,
  * one scalar mask per UNIQUE user-item interaction per view,
  * separate item embeddings for the two branches,
  * separate GCNs / user preference embeddings,
  * the original text-based item-item propagation after concatenation.

This version intentionally removes the old causal/non-causal intervention,
mask entropy, mask balance, and causal-only BPR losses so that the experiment
isolates the effect of learning TWO graph views and concatenating them.
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


class TWO_LEARNABLE_MASKS(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TWO_LEARNABLE_MASKS, self).__init__(config, dataset)

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
        # Two-view mask settings.
        #
        # View 1 starts close to the full graph.
        # sigmoid(3.0) ~= 0.953.
        #
        # View 2 starts around 0.5 and is free to specialize.
        # -------------------------------------------------------------
        self.view1_init_logit = float(
            _cfg(config, 'view1_init_logit', 3.0)
        )
        self.view2_init_std = float(
            _cfg(config, 'view2_init_std', 1e-2)
        )

        # Optional weak regularizer that keeps view 1 broadly connected.
        # Set to 0.0 for pure BPR-only training.
        self.view1_preserve_weight = float(
            _cfg(config, 'view1_preserve_weight', 1e-3)
        )
        self.view1_target_ratio = float(
            _cfg(config, 'view1_target_ratio', 0.95)
        )

        # -------------------------------------------------------------
        # Separate item embeddings for the two learned graph views.
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
        # TWO independent learnable masks.
        #
        # View 1:
        #   starts close to the full graph.
        #
        # View 2:
        #   starts around 0.5 and is free to specialize.
        #
        # They are NOT complements. An edge may be important to both views,
        # one view, or neither view.
        # -------------------------------------------------------------
        self.view1_mask_logits = nn.Parameter(
            torch.full(
                (self.num_interactions,),
                self.view1_init_logit,
                device=self.device,
                dtype=torch.float32
            )
        )

        self.view2_mask_logits = nn.Parameter(
            self.view2_init_std * torch.randn(
                self.num_interactions,
                device=self.device
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
        Returns two independent soft masks, one for each UNIQUE interaction.

        Shapes:
            view1_mask: [E]
            view2_mask: [E]

        View 1 starts near 1.0 (almost-full graph).
        View 2 starts near 0.5 (free learned view).
        """
        view1_mask = torch.sigmoid(self.view1_mask_logits)
        view2_mask = torch.sigmoid(self.view2_mask_logits)

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
        Two-view recommendation objective.

        Main objective:
            fused BPR on [view1 || view2].

        Optional regularizer:
            keep the average view-1 mask near view1_target_ratio.

        Set:
            view1_preserve_weight: 0.0
        if you want completely unconstrained two-mask learning.
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

        view1_preserve_loss = (
            view1_mask.mean() - self.view1_target_ratio
        ).pow(2)

        total_loss = (
            bpr_loss
            + self.view1_preserve_weight * view1_preserve_loss
        )

        # Diagnostics only; these do NOT add extra loss terms.
        with torch.no_grad():
            self.loss_components = {
                'bpr': float(bpr_loss.detach().cpu()),
                'view1_preserve': float(
                    view1_preserve_loss.detach().cpu()
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
            }

        return total_loss

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
    parent model and are separate between the two learned graph views.
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