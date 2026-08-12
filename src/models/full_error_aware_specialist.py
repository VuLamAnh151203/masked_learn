# coding: utf-8
"""
GLORIA variant: FULL GRAPH + ERROR-AWARE SPECIALIST MASK.

Architecture
------------
Branch 1 (generalist):
    Full user-item graph.
    edge_mask = None

Branch 2 (specialist):
    Same graph topology, but each unique user-item interaction has one
    learnable soft edge mask:

        specialist_mask = sigmoid(mask_logits)

The two branch representations are concatenated:

    Z_final = [Z_full || Z_specialist]

Because scoring is a dot product, concatenation gives an additive score:

    s_final(u,i)
        = s_full(u,i)
        + s_specialist(u,i)

Error-aware specialization
--------------------------
The full branch defines how difficult each BPR training tuple is:

    full_margin
        = s_full(u, i_pos)
        - s_full(u, i_neg)

Hardness weight:

    w = sigmoid(
            (hardness_margin - full_margin.detach())
            / hardness_temperature
        )

So:
    * easy samples for the full branch -> small w
    * uncertain / wrongly ranked samples -> large w

The specialist branch receives an additional weighted BPR objective:

    L_specialist
        = sum_b w_b * [-log sigmoid(mask_margin_b)]
          / (sum_b w_b + eps)

Final training objective:

    L
        = L_fused_BPR
        + specialist_weight * L_specialist

Important:
    hardness is detached from the full branch, so the full branch cannot make
    examples artificially easy/hard to manipulate the specialist weights.

This model still uses exactly TWO GCN forward passes.
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


class FULL_ERROR_AWARE_SPECIALIST(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FULL_ERROR_AWARE_SPECIALIST, self).__init__(config, dataset)

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
        # Error-aware specialist settings.
        # -------------------------------------------------------------

        # Strength of the error-aware specialist objective.
        self.specialist_weight = float(
            _cfg(config, 'specialist_weight', 0.10)
        )

        # Samples whose full-branch margin is below this value receive
        # larger specialist weights.
        self.hardness_margin = float(
            _cfg(config, 'hardness_margin', 1.0)
        )

        # Smaller temperature -> sharper easy/hard separation.
        self.hardness_temperature = float(
            _cfg(config, 'hardness_temperature', 1.0)
        )

        if self.specialist_weight < 0.0:
            raise ValueError(
                "specialist_weight must be >= 0."
            )

        if self.hardness_temperature <= 0.0:
            raise ValueError(
                "hardness_temperature must be > 0."
            )

        # -------------------------------------------------------------
        # Separate item embeddings for the full/generalist and
        # masked/specialist branches.
        # -------------------------------------------------------------
        self.id_embedding_full = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )
        self.id_embedding_specialist = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )

        # Match the strong Full + Mask baseline:
        # keep the two embedding tables independently initialized.
        # nn.Embedding initializes them independently by default.

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
        # One learnable soft mask for the specialist branch.
        #
        # Match the strong Full + Mask baseline exactly:
        #     mask_logits = 0  ->  sigmoid(mask_logits) = 0.5
        # -------------------------------------------------------------
        self.specialist_mask_logits = nn.Parameter(
            torch.zeros(
                self.num_interactions,
                device=self.device
            )
        )

        self.register_buffer(
            'initial_specialist_mask',
            torch.full(
                (self.num_interactions,),
                0.5,
                dtype=torch.float32,
                device=self.device
            )
        )

        # -------------------------------------------------------------
        # Two separate GCNs.
        #
        # full_gcn:
        #   always runs on the complete graph.
        #
        # specialist_gcn:
        #   runs on the learnable masked graph.
        # -------------------------------------------------------------
        self.full_gcn = GCN(
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
            features=self.id_embedding_full.weight
        )

        self.specialist_gcn = GCN(
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
            features=self.id_embedding_specialist.weight
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
    # Specialist soft mask
    # -----------------------------------------------------------------
    def get_specialist_mask(self):
        """
        Returns:
            specialist_mask: [E]
        """
        return torch.sigmoid(
            self.specialist_mask_logits
        )

    @staticmethod
    def to_bidirectional_mask(mask):
        """[E] -> [2E], matching [u->i, i->u] edge ordering."""
        return torch.cat(
            [mask, mask],
            dim=0
        )

    # -----------------------------------------------------------------
    # Representation construction
    # -----------------------------------------------------------------
    def compute_branch_representations(self):
        """
        Returns:
            full_rep:
                full/generalist GCN representation

            specialist_rep:
                masked/specialist GCN representation

            specialist_mask:
                [E]
        """
        specialist_mask = self.get_specialist_mask()

        specialist_edge_mask = self.to_bidirectional_mask(
            specialist_mask
        )

        # Full graph: no edge mask.
        full_rep, full_preference = self.full_gcn(
            self.edge_index,
            self.id_embedding_full.weight,
            edge_mask=None
        )

        # Learnable specialist graph.
        specialist_rep, specialist_preference = (
            self.specialist_gcn(
                self.edge_index,
                self.id_embedding_specialist.weight,
                edge_mask=specialist_edge_mask
            )
        )

        self.full_preference = full_preference
        self.specialist_preference = specialist_preference

        return (
            full_rep,
            specialist_rep,
            specialist_mask
        )

    def prepare_single_branch_representation(self, rep):
        """
        Apply the same item-item propagation used by the fused model.

        This makes branch-only scores directly comparable with the fused score.

        Because item_item() is linear, applying it before concatenation is
        equivalent to applying it after concatenation branch-wise.
        """
        user_rep = rep[:self.num_user]
        item_rep = rep[self.num_user:]

        item_rep = self.item_item(
            item_rep
        )

        return torch.cat(
            [user_rep, item_rep],
            dim=0
        )

    def fuse_representations(self, full_rep, specialist_rep):
        """
        Final representation:

            [full/generalist || masked/specialist]
        """
        user_full = full_rep[:self.num_user]
        user_specialist = specialist_rep[:self.num_user]

        item_full = full_rep[self.num_user:]
        item_specialist = specialist_rep[self.num_user:]

        user_rep = torch.cat(
            [
                user_full,
                user_specialist
            ],
            dim=1
        )

        item_rep = torch.cat(
            [
                item_full,
                item_specialist
            ],
            dim=1
        )

        item_rep = self.item_item(
            item_rep
        )

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
        # Match the original strong Full + Mask objective exactly.
        return -torch.mean(
            torch.log2(
                torch.sigmoid(
                    pos_scores - neg_scores
                )
            )
        )

    # -----------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------
    def forward(self, interaction, return_aux=False):
        (
            full_rep,
            specialist_rep,
            specialist_mask
        ) = self.compute_branch_representations()

        self.full_rep = full_rep
        self.specialist_rep = specialist_rep

        # Fused [full || specialist] representation.
        self.result_embed = self.fuse_representations(
            full_rep,
            specialist_rep
        )

        pos_scores, neg_scores = self.pairwise_scores(
            self.result_embed,
            interaction
        )

        if not return_aux:
            return pos_scores, neg_scores

        # Branch-only representations including the same item-item propagation.
        full_scoring_rep = self.prepare_single_branch_representation(
            full_rep
        )

        specialist_scoring_rep = (
            self.prepare_single_branch_representation(
                specialist_rep
            )
        )

        full_pos_scores, full_neg_scores = self.pairwise_scores(
            full_scoring_rep,
            interaction
        )

        specialist_pos_scores, specialist_neg_scores = (
            self.pairwise_scores(
                specialist_scoring_rep,
                interaction
            )
        )

        aux = {
            'specialist_mask': specialist_mask,

            'full_rep': full_rep,
            'specialist_rep': specialist_rep,

            'full_pos_scores': full_pos_scores,
            'full_neg_scores': full_neg_scores,

            'specialist_pos_scores': specialist_pos_scores,
            'specialist_neg_scores': specialist_neg_scores,
        }

        return pos_scores, neg_scores, aux

    # -----------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------
    def calculate_loss(self, interaction):
        """
        Clean error-aware specialist experiment.

        specialist_weight == 0:
            exactly the original Full + Mask BPR objective.

        specialist_weight > 0:
            fused BPR + error-aware weighted specialist BPR.
        """

        # Exact Full + Mask baseline path.
        if self.specialist_weight == 0.0:
            pos_scores, neg_scores = self.forward(
                interaction,
                return_aux=False
            )

            return self.bpr_loss(
                pos_scores,
                neg_scores
            )

        # Error-aware specialist path.
        pos_scores, neg_scores, aux = self.forward(
            interaction,
            return_aux=True
        )

        main_bpr = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        full_margin = (
            aux['full_pos_scores']
            - aux['full_neg_scores']
        )

        # Stop gradients through hardness.
        with torch.no_grad():
            hardness = torch.sigmoid(
                (
                    self.hardness_margin
                    - full_margin
                )
                / self.hardness_temperature
            )

        specialist_margin = (
            aux['specialist_pos_scores']
            - aux['specialist_neg_scores']
        )

        # Same log2 BPR scale as the baseline.
        specialist_per_sample_loss = -torch.log2(
            torch.sigmoid(
                specialist_margin
            )
        )

        specialist_loss = (
            hardness
            * specialist_per_sample_loss
        ).sum() / (
            hardness.sum() + 1e-8
        )

        total_loss = (
            main_bpr
            + self.specialist_weight
            * specialist_loss
        )

        with torch.no_grad():
            specialist_mask = aux['specialist_mask']
            fused_margin = pos_scores - neg_scores

            hard_fraction = (
                hardness > 0.5
            ).float().mean()

            self.loss_components = {
                'total': float(total_loss.detach().cpu()),
                'main_bpr': float(main_bpr.detach().cpu()),
                'specialist_loss': float(specialist_loss.detach().cpu()),
                'hardness_mean': float(hardness.mean().cpu()),
                'hard_fraction': float(hard_fraction.cpu()),
                'full_margin_mean': float(full_margin.mean().detach().cpu()),
                'specialist_margin_mean': float(
                    specialist_margin.mean().detach().cpu()
                ),
                'fused_margin_mean': float(
                    fused_margin.mean().detach().cpu()
                ),
                'specialist_mask_mean': float(
                    specialist_mask.mean().detach().cpu()
                ),
                'specialist_mask_change_from_init': float(
                    (
                        specialist_mask
                        - self.initial_specialist_mask
                    )
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


class GCN(torch.nn.Module):
    """
    LightGCN-style propagation module.

    The full/generalist and masked/specialist branches own separate GCN
    instances, user preference embeddings, and item embeddings.
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
        if edge_mask is None:
            edge_mask = torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype
            )

        if size is None:
            edge_index, _ = remove_self_loops(edge_index)

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(
            edge_index,
            size=(x.size(0), x.size(0)),
            x=x,
            edge_mask=edge_mask
        )

    def message(self, x_j, edge_index, size, edge_mask):
        if self.aggr == 'add':
            row, col = edge_index

            deg = degree(
                row,
                size[0],
                dtype=x_j.dtype
            )

            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[
                torch.isinf(deg_inv_sqrt)
            ] = 0

            norm = (
                deg_inv_sqrt[row]
                * deg_inv_sqrt[col]
            )

            return (
                norm.view(-1, 1)
                * edge_mask.view(-1, 1)
                * x_j
            )

        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels
        )