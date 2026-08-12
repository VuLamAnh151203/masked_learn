# coding: utf-8
"""
GLORIA variant: FULL GRAPH + DETERMINISTIC MASK + WEAK INFORMATION BOTTLENECK.

This version is designed as a clean ablation from the strong Full + Mask
pipeline.

Architecture
------------
Branch 1:
    FULL user-item graph

        Z_full = GCN(G)

Branch 2:
    deterministic learnable soft mask

        M_e = sigmoid(mask_logits[e])

        Z_mask = GCN(M * G)

Final representation:

        Z_final = [Z_full || Z_mask]

Training objective
------------------
The recommendation loss remains the fused BPR loss:

        L_BPR

The only added term is a WEAK Bernoulli information-bottleneck regularizer:

        KL_e =
            M_e * log(M_e / rho)
            + (1-M_e) * log((1-M_e)/(1-rho))

        L_IB = mean_e KL_e

Final objective:

        L =
            L_BPR
            + ib_beta * L_IB

Important
---------
1) There is NO stochastic Concrete / Gumbel sampling.
2) There is NO hard edge dropping during training.
3) The full branch is always unchanged.
4) The mask branch behaves exactly like a normal deterministic soft-mask GCN.
5) The IB term is deliberately weak so that it nudges the mask rather than
   forcing strong graph compression.

Recommended first setting:

    mask_init_prob       = 0.80
    ib_prior_retention   = 0.70
    ib_beta              = 1e-4

This keeps the model close to the strong Full + Mask baseline while testing
whether a mild compression prior is useful.
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


class FULL_DETERMINISTIC_WEAK_IB_MASK(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FULL_DETERMINISTIC_WEAK_IB_MASK, self).__init__(config, dataset)

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
        # Deterministic mask + weak information-bottleneck settings.
        # -------------------------------------------------------------

        # Initial soft-mask probability.
        self.mask_init_prob = float(
            _cfg(config, 'mask_init_prob', 0.50)
        )

        # Bernoulli prior retention probability rho.
        # Start high so the IB term is only a mild compression pressure.
        self.ib_prior_retention = float(
            _cfg(config, 'ib_prior_retention', 0.70)
        )

        # Weak IB regularization strength.
        self.ib_beta = float(
            _cfg(config, 'ib_beta', 0.0)
        )

        # Small logit-space initialization noise.
        self.mask_init_noise = float(
            _cfg(config, 'mask_init_noise', 1e-2)
        )

        # Numerical stability.
        self.ib_eps = float(
            _cfg(config, 'ib_eps', 1e-8)
        )

        if not (0.0 < self.mask_init_prob < 1.0):
            raise ValueError(
                "Require 0 < mask_init_prob < 1."
            )

        if not (0.0 < self.ib_prior_retention < 1.0):
            raise ValueError(
                "Require 0 < ib_prior_retention < 1."
            )

        if self.ib_beta < 0.0:
            raise ValueError(
                "ib_beta must be >= 0."
            )

        # -------------------------------------------------------------
        # Separate item embeddings for:
        #   1) full/general branch
        #   2) deterministic masked branch
        #
        # They start identically, then become independent.
        # -------------------------------------------------------------
        self.id_embedding_full = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )

        self.id_embedding_ib = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )

        nn.init.xavier_uniform_(
            self.id_embedding_full.weight
        )

        with torch.no_grad():
            self.id_embedding_ib.weight.copy_(
                self.id_embedding_full.weight
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
        # ONE deterministic learnable soft mask.
        #
        # Full branch:
        #   edge_mask = None
        #
        # Mask branch:
        #   M = sigmoid(mask_logits)
        # -------------------------------------------------------------

        def probability_to_logit(prob):
            prob = prob.clamp(
                min=1e-4,
                max=1.0 - 1e-4
            )

            return torch.log(
                prob / (1.0 - prob)
            )

        initial_mask_prob = torch.full(
            (self.num_interactions,),
            self.mask_init_prob,
            dtype=torch.float32,
            device=self.device
        )

        initial_mask_logits = probability_to_logit(
            initial_mask_prob
        )

        if self.mask_init_noise > 0.0:
            initial_mask_logits = (
                initial_mask_logits
                + self.mask_init_noise
                * torch.randn_like(initial_mask_logits)
            )

        self.mask_logits = nn.Parameter(
            initial_mask_logits
        )

        self.register_buffer(
            'initial_mask_prob',
            initial_mask_prob
        )

        # -------------------------------------------------------------
        # Two separate GCNs.
        #
        # full_gcn:
        #   complete interaction graph
        #
        # ib_gcn:
        #   deterministic soft-masked graph
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

        self.ib_gcn = GCN(
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
            features=self.id_embedding_ib.weight
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
    # Deterministic soft mask + weak information bottleneck
    # -----------------------------------------------------------------
    def get_mask(self):
        """
        Deterministic soft edge-retention weights:

            M_e = sigmoid(mask_logits[e])

        Shape:
            [E]
        """
        return torch.sigmoid(
            self.mask_logits
        )

    def ib_kl_loss(self):
        """
        Weak Bernoulli KL regularizer:

            KL(
                Bern(M_e)
                ||
                Bern(rho)
            )

        averaged over interaction edges.
        """
        mask = self.get_mask().clamp(
            min=self.ib_eps,
            max=1.0 - self.ib_eps
        )

        rho = torch.as_tensor(
            self.ib_prior_retention,
            dtype=mask.dtype,
            device=mask.device
        ).clamp(
            min=self.ib_eps,
            max=1.0 - self.ib_eps
        )

        kl = (
            mask
            * (
                torch.log(mask)
                - torch.log(rho)
            )
            + (1.0 - mask)
            * (
                torch.log1p(-mask)
                - torch.log1p(-rho)
            )
        )

        return kl.mean()

    @staticmethod
    def to_bidirectional_mask(mask):
        """[E] -> [2E], matching [u->i, i->u] ordering."""
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
            full_rep: [num_user + num_item, D]
            mask_rep: [num_user + num_item, D]
            mask:     [E]
        """
        mask = self.get_mask()

        edge_mask = self.to_bidirectional_mask(
            mask
        )

        # Stable full-graph anchor.
        full_rep, full_preference = self.full_gcn(
            self.edge_index,
            self.id_embedding_full.weight,
            edge_mask=None
        )

        # Deterministic learnable masked graph.
        mask_rep, mask_preference = self.ib_gcn(
            self.edge_index,
            self.id_embedding_ib.weight,
            edge_mask=edge_mask
        )

        self.full_preference = full_preference
        self.mask_preference = mask_preference

        return (
            full_rep,
            mask_rep,
            mask
        )

    def fuse_representations(self, full_rep, mask_rep):
        """
        Concatenate:

            users: [full || masked]
            items: [full || masked]

        followed by the original text-based item-item propagation.
        """
        user_full = full_rep[:self.num_user]
        user_mask = mask_rep[:self.num_user]

        item_full = full_rep[self.num_user:]
        item_mask = mask_rep[self.num_user:]

        user_rep = torch.cat(
            [user_full, user_mask],
            dim=1
        )

        item_rep = torch.cat(
            [item_full, item_mask],
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
        # Numerically stable equivalent of -log(sigmoid(pos-neg)).
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    # -----------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------
    def forward(
        self,
        interaction,
        return_aux=False
    ):
        (
            full_rep,
            mask_rep,
            mask
        ) = self.compute_branch_representations()

        self.full_rep = full_rep
        self.mask_rep = mask_rep

        self.result_embed = self.fuse_representations(
            full_rep,
            mask_rep
        )

        pos_scores, neg_scores = self.pairwise_scores(
            self.result_embed,
            interaction
        )

        if not return_aux:
            return pos_scores, neg_scores

        aux = {
            'full_rep': full_rep,
            'mask_rep': mask_rep,
            'mask': mask,
        }

        return (
            pos_scores,
            neg_scores,
            aux
        )

    # -----------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------
    def calculate_loss(self, interaction):
        """
        Full graph + deterministic mask + weak IB regularization.

        Objective:

            L =
                fused_BPR
                + ib_beta * KL(
                    Bern(mask)
                    ||
                    Bern(rho)
                )
        """
        (
            pos_scores,
            neg_scores,
            aux
        ) = self.forward(
            interaction,
            return_aux=True
        )

        bpr_loss = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        ib_kl = self.ib_kl_loss()

        total_loss = (
            bpr_loss
            + self.ib_beta * ib_kl
        )

        # -------------------------------------------------------------
        # Diagnostics only.
        # -------------------------------------------------------------
        with torch.no_grad():
            mask = aux['mask']

            mask_mean = mask.mean()
            mask_drop_ratio = 1.0 - mask_mean

            mask_entropy = -(
                mask
                * torch.log(
                    mask.clamp_min(
                        self.ib_eps
                    )
                )
                + (1.0 - mask)
                * torch.log(
                    (1.0 - mask).clamp_min(
                        self.ib_eps
                    )
                )
            ).mean()

            self.loss_components = {
                'total': float(
                    total_loss.detach().cpu()
                ),

                'bpr': float(
                    bpr_loss.detach().cpu()
                ),

                'ib_kl': float(
                    ib_kl.detach().cpu()
                ),

                'weighted_ib_kl': float(
                    (
                        self.ib_beta
                        * ib_kl
                    )
                    .detach()
                    .cpu()
                ),

                'mask_mean': float(
                    mask_mean.detach().cpu()
                ),

                'mask_drop_ratio': float(
                    mask_drop_ratio.detach().cpu()
                ),

                'mask_entropy': float(
                    mask_entropy.detach().cpu()
                ),

                'mask_change_from_init': float(
                    (
                        mask
                        - self.initial_mask_prob
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
        (
            full_rep,
            mask_rep,
            _
        ) = self.compute_branch_representations()

        self.result_embed = self.fuse_representations(
            full_rep,
            mask_rep
        )

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

    The full and deterministic masked branches own separate GCN instances,
    user preference embeddings, and item embeddings.
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