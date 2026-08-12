# coding: utf-8
"""
GLORIA with TWO GCN views using:

    1) one learnable SHARED backbone S
    2) one COMPETITIVE two-way private router Q=[q1, q2]

For every user-item interaction edge e=(u,i):

    S_e = sigmoid(shared_mask_logits[e])

    [q1_e, q2_e] =
        softmax(private_router_logits[e] / router_temperature)

Therefore:

    q1_e + q2_e = 1

The two final graph masks are:

    M1_e = S_e + (1 - S_e) * q1_e
    M2_e = S_e + (1 - S_e) * q2_e

Interpretation
--------------
S:
    information shared by BOTH GCN branches.

q1:
    private routing weight for View 1.
    It is initialized toward a NICHE prior:

        niche_score = d_u * (1 - d_i)

q2:
    private routing weight for View 2.
    It is initialized toward a POPULARITY prior:

        popularity_score = (1 - d_u) * d_i

where d_u and d_i are log-degree values normalized to [0,1].

The degree priors ONLY initialize the router.
After initialization, the shared mask and router logits are fully learnable.

Why competitive routing?
------------------------
The common part is allowed to overlap through S, while the private part must
allocate its mass between the two branches because q1 + q2 = 1. This reduces
private-view collapse without forcing the entire graph masks to be complements.

There are still only TWO expensive GCN forward passes:

    GCN1(G, M1)
    GCN2(G, M2)

Final representation:

    Z_final = [Z_view1 || Z_view2]

Training objective:

    L = BPR_fused

No additional diversity or mask-preservation loss is used by default.
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


class SHARED_COMPETITIVE_PRIVATE_MASKS(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SHARED_COMPETITIVE_PRIVATE_MASKS, self).__init__(config, dataset)

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
        # Shared-backbone + competitive-private-router settings.
        # -------------------------------------------------------------

        # Moderate shared value preserves common collaborative structure while
        # leaving enough room for private routing:
        #
        #   dM/dq = 1 - S
        #
        # so S should not start too close to 1.
        self.shared_init_mask = float(
            _cfg(config, 'shared_init_mask', 0.70)
        )

        # Temperature of the two-way softmax private router.
        #   > 1.0 : softer / more balanced routing
        #   = 1.0 : standard softmax
        #   < 1.0 : sharper specialization
        self.router_temperature = float(
            _cfg(config, 'router_temperature', 1.0)
        )

        # Controls how strongly the degree priors separate the initial router.
        # 0 -> every edge begins q1=q2=0.5.
        # Larger values create stronger niche-vs-popularity specialization.
        self.router_prior_scale = float(
            _cfg(config, 'router_prior_scale', 2.0)
        )

        # Tiny noise in router/shared logit space for symmetry breaking.
        self.init_logit_noise = float(
            _cfg(config, 'init_logit_noise', 1e-2)
        )

        if not (0.0 < self.shared_init_mask < 1.0):
            raise ValueError(
                "Require 0 < shared_init_mask < 1."
            )

        if self.router_temperature <= 0.0:
            raise ValueError(
                "router_temperature must be > 0."
            )

        if self.router_prior_scale < 0.0:
            raise ValueError(
                "router_prior_scale must be >= 0."
            )

        # -------------------------------------------------------------
        # Separate item embeddings for the two shared+competitive-private graph views.
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
        # SHARED BACKBONE + COMPETITIVE PRIVATE ROUTER
        #
        # Final masks:
        #
        #   [q1, q2] = softmax(router_logits / temperature)
        #
        #   M1 = S + (1-S) * q1
        #   M2 = S + (1-S) * q2
        #
        # The shared component S may be strong in BOTH branches.
        # Only the PRIVATE routing mass is competitive.
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # Compute endpoint degree statistics from training interactions.
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

            return (
                (values - v_min)
                / (v_max - v_min + 1e-8)
            )

        user_degree_norm = minmax_normalize(
            user_degree_log
        )
        item_degree_norm = minmax_normalize(
            item_degree_log
        )

        # Endpoint degree for each UNIQUE interaction edge.
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

        # -------------------------------------------------------------
        # Degree priors used ONLY to initialize the private router.
        # -------------------------------------------------------------

        # Active-user -> tail-item preference.
        niche_score = (
            edge_user_degree
            * (1.0 - edge_item_degree)
        )

        # Low-activity-user -> popular-item preference.
        popularity_score = (
            (1.0 - edge_user_degree)
            * edge_item_degree
        )

        # Shared component starts constant across edges.
        shared_init_mask_tensor = torch.full(
            (self.num_interactions,),
            self.shared_init_mask,
            dtype=torch.float32,
            device=self.device
        )

        def probability_to_logit(prob):
            prob = prob.clamp(
                min=1e-4,
                max=1.0 - 1e-4
            )
            return torch.log(
                prob / (1.0 - prob)
            )

        shared_init_logits = probability_to_logit(
            shared_init_mask_tensor
        )

        # -------------------------------------------------------------
        # Initialize the two-way competitive router.
        #
        # Important:
        # get_private_routing() later computes:
        #
        #   softmax(private_router_logits / router_temperature)
        #
        # Therefore multiplying the initial logits by temperature keeps the
        # INITIAL routing distribution independent of temperature.
        #
        # The difference niche_score - popularity_score controls which branch
        # initially receives more private mass.
        # -------------------------------------------------------------
        prior_logits = self.router_prior_scale * torch.stack(
            [
                niche_score,
                popularity_score
            ],
            dim=1
        )  # [E, 2]

        private_router_init_logits = (
            self.router_temperature
            * prior_logits
        )

        if self.init_logit_noise > 0.0:
            shared_init_logits = (
                shared_init_logits
                + self.init_logit_noise
                * torch.randn_like(shared_init_logits)
            )

            private_router_init_logits = (
                private_router_init_logits
                + self.init_logit_noise
                * torch.randn_like(private_router_init_logits)
            )

        # Fully learnable parameters.
        self.shared_mask_logits = nn.Parameter(
            shared_init_logits
        )

        self.private_router_logits = nn.Parameter(
            private_router_init_logits
        )

        # Initial private routing probabilities and final graph masks.
        with torch.no_grad():
            initial_private_routing = F.softmax(
                private_router_init_logits
                / self.router_temperature,
                dim=1
            )

            initial_q1 = initial_private_routing[:, 0]
            initial_q2 = initial_private_routing[:, 1]

            initial_view1_mask = (
                shared_init_mask_tensor
                + (1.0 - shared_init_mask_tensor)
                * initial_q1
            )

            initial_view2_mask = (
                shared_init_mask_tensor
                + (1.0 - shared_init_mask_tensor)
                * initial_q2
            )

        # Non-trainable analysis buffers.
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
        self.register_buffer(
            'initial_shared_mask',
            shared_init_mask_tensor
        )
        self.register_buffer(
            'initial_private_q1',
            initial_q1
        )
        self.register_buffer(
            'initial_private_q2',
            initial_q2
        )
        self.register_buffer(
            'initial_view1_mask',
            initial_view1_mask
        )
        self.register_buffer(
            'initial_view2_mask',
            initial_view2_mask
        )

        print(
            "Initial competitive routing | "
            "shared: {:.4f}, "
            "q1(niche): {:.4f}, "
            "q2(popularity): {:.4f}".format(
                shared_init_mask_tensor.mean().item(),
                initial_q1.mean().item(),
                initial_q2.mean().item()
            )
        )

        print(
            "Initial final masks | "
            "view1(shared+niche-router): {:.4f}, "
            "view2(shared+pop-router): {:.4f}".format(
                initial_view1_mask.mean().item(),
                initial_view2_mask.mean().item()
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
    # Shared backbone + competitive private routing
    # -----------------------------------------------------------------
    def get_private_routing(self):
        """
        Returns private two-way routing probabilities.

        Shape:
            routing: [E, 2]

        For every edge:
            routing[e, 0] + routing[e, 1] = 1
        """
        return F.softmax(
            self.private_router_logits
            / self.router_temperature,
            dim=1
        )

    def get_mask_components(self):
        """
        Returns:
            shared_mask: [E]
            private_q1:  [E]
            private_q2:  [E]
        """
        shared_mask = torch.sigmoid(
            self.shared_mask_logits
        )

        routing = self.get_private_routing()

        private_q1 = routing[:, 0]
        private_q2 = routing[:, 1]

        return (
            shared_mask,
            private_q1,
            private_q2
        )

    def get_masks(self):
        """
        Construct the two final graph masks:

            M1 = S + (1-S) * q1
            M2 = S + (1-S) * q2

        The shared backbone may overlap freely.
        Only the private routing mass is competitive.
        """
        (
            shared_mask,
            private_q1,
            private_q2
        ) = self.get_mask_components()

        view1_mask = (
            shared_mask
            + (1.0 - shared_mask)
            * private_q1
        )

        view2_mask = (
            shared_mask
            + (1.0 - shared_mask)
            * private_q2
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
            view1_rep:  shared + niche-routed GCN representation
            view2_rep:  shared + popularity-routed GCN representation
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
        Concatenate the two shared+competitive-private graph views.

        View 1:
            shared backbone + niche-routed private structure.

        View 2:
            shared backbone + popularity-routed private structure.

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
        Fused BPR only.

        The architecture itself creates private specialization through
        two-way softmax routing, so no explicit diversity loss is used.

        Diagnostics monitor:
            - shared backbone collapse,
            - routing sharpness / entropy,
            - final view separation,
            - movement away from degree initialization.
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

        (
            shared_mask,
            private_q1,
            private_q2
        ) = self.get_mask_components()

        routing = torch.stack(
            [private_q1, private_q2],
            dim=1
        )

        routing_entropy = -(
            routing
            * torch.log(routing.clamp_min(1e-8))
        ).sum(dim=1).mean()

        routing_confidence = torch.max(
            routing,
            dim=1
        ).values.mean()

        with torch.no_grad():
            self.loss_components = {
                'bpr': float(
                    bpr_loss.detach().cpu()
                ),

                'shared_mask_mean': float(
                    shared_mask.mean().detach().cpu()
                ),

                'private_q1_mean': float(
                    private_q1.mean().detach().cpu()
                ),

                'private_q2_mean': float(
                    private_q2.mean().detach().cpu()
                ),

                'routing_entropy': float(
                    routing_entropy.detach().cpu()
                ),

                'routing_confidence': float(
                    routing_confidence.detach().cpu()
                ),

                'view1_mask_mean': float(
                    view1_mask.mean().detach().cpu()
                ),

                'view2_mask_mean': float(
                    view2_mask.mean().detach().cpu()
                ),

                'final_mask_mean_abs_diff': float(
                    (view1_mask - view2_mask)
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                ),

                'shared_change_from_init': float(
                    (
                        shared_mask
                        - self.initial_shared_mask
                    )
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                ),

                'q1_change_from_init': float(
                    (
                        private_q1
                        - self.initial_private_q1
                    )
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                ),

                'q2_change_from_init': float(
                    (
                        private_q2
                        - self.initial_private_q2
                    )
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                ),

                'view1_change_from_init': float(
                    (
                        view1_mask
                        - self.initial_view1_mask
                    )
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                ),

                'view2_change_from_init': float(
                    (
                        view2_mask
                        - self.initial_view2_mask
                    )
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
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
    parent model and are separate between the two shared+competitive-private graph views.
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