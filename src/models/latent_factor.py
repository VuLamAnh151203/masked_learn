# coding: utf-8
r"""
GLORIA with a deterministic LATENT-FACTOR COMPATIBILITY edge mask.

This version does NOT use item degree for routing.

Main idea
---------
1. Encode the complete user-item graph with a shared LightGCN-style encoder:

       H = GCN_router(G)

   giving a representation h_u for each user and h_i for each item.

2. Map the shared representations to positive latent factor affiliations:

       z_u = softplus(f_user(h_u))
       z_i = softplus(f_item(h_i))

   With two factors, z_u and z_i have shape [*, 2].

3. For every observed interaction (u, i), compute factor compatibility:

       s_ui^k = gamma_k * z_uk * z_ik

   and convert it to a two-way edge-routing distribution:

       m_ui = softmax(s_ui / temperature)

4. Build two weighted interaction graphs:

       A_1[u, i] = A[u, i] * m_ui^1
       A_2[u, i] = A[u, i] * m_ui^2

   The SAME mask value is used for u->i and i->u so each branch stays
   symmetric.

5. Run the two GLORIA GCN branches, concatenate their representations,
   apply the original item-item GCN, and optimize recommendation with BPR.

Architecture
------------
                        full graph G
                            |
                      shared router GCN
                            |
                      h_user / h_item
                       /           \
              user factor net    item factor net
                       |           |
                      z_u         z_i
                        \         /
                  factor compatibility
                  s_ui^k=gamma_k*z_uk*z_ik
                            |
                     softmax over k
                      /           \
                 edge mask 1   edge mask 2
                    |             |
                  GCN-1         GCN-2
                    |             |
                    +--- concat --+
                            |
                       item-item GCN
                            |
                           BPR

Important distinction from the previous free item mask
------------------------------------------------------
Previous:
    item ID -> free parameter theta_i -> mask_i

This model:
    graph -> h_u,h_i -> z_u,z_i -> compatibility(u,i) -> edge mask_ui

Therefore the same item can route differently for different users.

This is inspired by the factor-derived graph weighting idea in DiGGR, but it
is intentionally a simpler deterministic adaptation for recommendation. It
DOES NOT implement DiGGR's Gamma prior / Weibull posterior / ELBO.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from common.abstract_recommender import GeneralRecommender


def _cfg(config, key, default):
    """Read an optional config value without assuming config implements .get()."""
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


class LATENT_FACTOR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LATENT_FACTOR, self).__init__(config, dataset)

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
        # Latent-factor router hyperparameters.
        # -------------------------------------------------------------
        # The current implementation has exactly two graph branches.
        self.num_factors = int(_cfg(config, 'latent_factor_num', 2))
        if self.num_factors != 2:
            raise ValueError(
                'This GLORIA variant currently implements exactly 2 latent '
                'factors/graph branches. Set latent_factor_num: 2.'
            )

        self.factor_temperature = float(
            _cfg(config, 'factor_temperature', 1.0)
        )

        self.factor_hidden_dim = int(
            _cfg(config, 'factor_hidden_dim', self.feat_embed_dim)
        )

        # Optional regularizers. These do not use degree.
        # Small entropy weight encourages more decisive edge routing.
        self.factor_entropy_weight = float(
            _cfg(config, 'factor_entropy_weight', 0.0)
        )

        # Prevent one factor graph from receiving virtually all edge mass.
        # This is a floor, NOT a 50/50 constraint.
        self.factor_min_branch_usage = float(
            _cfg(config, 'factor_min_branch_usage', 0.05)
        )
        self.factor_balance_weight = float(
            _cfg(config, 'factor_balance_weight', 0.05)
        )

        # Encourage the two learned factor columns to be less correlated.
        self.factor_independence_weight = float(
            _cfg(config, 'factor_independence_weight', 1e-3)
        )

        # Optional direct structural supervision for the latent factors.
        # It uses the same observed positive / sampled negative triplets that
        # BPR already receives, but scores them ONLY with z_u,z_i.
        # This is a pragmatic deterministic substitute for DiGGR's graph
        # likelihood objective, not an implementation of its ELBO.
        self.factor_structure_weight = float(
            _cfg(config, 'factor_structure_weight', 0.05)
        )

        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        # -------------------------------------------------------------
        # Branch item embeddings used by the two downstream graph experts.
        # -------------------------------------------------------------
        self.id_embedding_branch1 = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )
        self.id_embedding_branch2 = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )

        nn.init.xavier_uniform_(self.id_embedding_branch1.weight)
        with torch.no_grad():
            self.id_embedding_branch2.weight.copy_(
                self.id_embedding_branch1.weight
            )

        # -------------------------------------------------------------
        # Separate shared-router item embeddings.
        # These are used only to infer h_i -> z_i for routing.
        # -------------------------------------------------------------
        self.router_item_embedding = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.router_item_embedding.weight)

        # Keep the original GLORIA MLPs because the surrounding project may
        # expect them to exist.
        self.mlp_item = nn.Linear(
            self.t_feat.shape[-1],
            self.dim_latent,
            bias=False
        )
        self.mlp_user = nn.Linear(
            self.user_feat.shape[-1],
            self.dim_latent,
            bias=False
        )

        # -------------------------------------------------------------
        # Original item-item graph from text features.
        # -------------------------------------------------------------
        _, text_adj = self.get_knn_adj_mat(self.t_feat)
        self.mm_adj = text_adj

        # -------------------------------------------------------------
        # Build ONE complete user-item graph.
        # Degree is not computed or used for routing.
        # -------------------------------------------------------------
        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)

        forward_edges_np = self.pack_edge_index(train_interactions)
        self.num_interactions = forward_edges_np.shape[0]

        # One-to-one with the unique forward u -> i edges.
        self.edge_user_ids = torch.tensor(
            train_interactions.row,
            dtype=torch.long,
            device=self.device
        )
        self.edge_item_ids = torch.tensor(
            train_interactions.col,
            dtype=torch.long,
            device=self.device
        )

        forward_edges = torch.tensor(
            forward_edges_np,
            dtype=torch.long,
            device=self.device
        ).t().contiguous()

        reverse_edges = forward_edges[[1, 0], :]

        # Exact ordering:
        #   [all u -> i edges, then exact reverse i -> u edges]
        self.edge_index = torch.cat(
            [forward_edges, reverse_edges],
            dim=1
        )

        # -------------------------------------------------------------
        # Shared full-graph router encoder.
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Deterministic factor networks:
        #     h_u -> z_u >= 0
        #     h_i -> z_i >= 0
        # -------------------------------------------------------------
        self.user_factor_net = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.factor_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.factor_hidden_dim, self.num_factors)
        )

        self.item_factor_net = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.factor_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.factor_hidden_dim, self.num_factors)
        )

        self._init_factor_net(self.user_factor_net)
        self._init_factor_net(self.item_factor_net)

        # Positive factor activation gamma_k = softplus(raw_gamma_k).
        # raw ~= 0.5413 gives softplus(raw) ~= 1.0.
        self.factor_gamma_raw = nn.Parameter(
            torch.full(
                (self.num_factors,),
                0.54132485,
                device=self.device
            )
        )

        # -------------------------------------------------------------
        # Two downstream GLORIA GCN branches.
        # -------------------------------------------------------------
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

        self.result_embed = None

    @staticmethod
    def _init_factor_net(module):
        """Small, stable initialization for factor-routing MLPs."""
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    # -----------------------------------------------------------------
    # Graph utilities
    # -----------------------------------------------------------------
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
            device=self.device
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
            h = torch.sparse.mm(
                self.mm_adj,
                h
            )
        return rep + h

    @staticmethod
    def to_bidirectional_mask(mask):
        """
        Convert [E] -> [2E] matching edge ordering:
            [u->i edges, i->u edges].
        """
        return torch.cat([mask, mask], dim=0)

    # -----------------------------------------------------------------
    # Latent-factor compatibility routing
    # -----------------------------------------------------------------
    def get_factor_gamma(self):
        """Positive factor activation vector gamma, shape [2]."""
        return F.softplus(self.factor_gamma_raw) + 1e-8

    def compute_latent_factors(self):
        """
        Encode the full graph, then infer positive user/item latent factors.

        Returns:
            router_rep: [U + I, D]
            z_user:     [U, 2], positive
            z_item:     [I, 2], positive
        """
        # No edge mask: the router sees the complete interaction graph.
        router_rep, router_preference = self.router_gcn(
            self.edge_index,
            self.router_item_embedding.weight,
            edge_mask=None
        )

        h_user = router_rep[:self.num_user]
        h_item = router_rep[self.num_user:]

        z_user = F.softplus(
            self.user_factor_net(h_user)
        ) + 1e-8

        z_item = F.softplus(
            self.item_factor_net(h_item)
        ) + 1e-8

        self.router_rep = router_rep
        self.router_preference = router_preference
        self.last_z_user = z_user
        self.last_z_item = z_item

        return router_rep, z_user, z_item

    def compute_forward_edge_masks(self, z_user, z_item):
        """
        Derive a 2-way routing distribution for every observed interaction.

        For forward interaction e=(u,i):

            s_e,k = gamma_k * z_u,k * z_i,k
            m_e   = softmax(s_e / T)

        Returns:
            edge_factor_scores: [E, 2]
            edge_masks:         [E, 2], each row sums to 1
        """
        z_u_edge = z_user[self.edge_user_ids]
        z_i_edge = z_item[self.edge_item_ids]

        gamma = self.get_factor_gamma().view(1, -1)

        edge_factor_scores = (
            gamma
            * z_u_edge
            * z_i_edge
        )

        temperature = max(
            self.factor_temperature,
            1e-6
        )

        edge_masks = torch.softmax(
            edge_factor_scores / temperature,
            dim=1
        )

        return edge_factor_scores, edge_masks

    # -----------------------------------------------------------------
    # Two factor-weighted graph branches
    # -----------------------------------------------------------------
    def compute_branch_representations(self):
        """
        1. Infer latent factors from the full graph.
        2. Derive edge-level factor masks.
        3. Run two separately parameterized GCN branches.
        """
        router_rep, z_user, z_item = self.compute_latent_factors()

        edge_factor_scores, edge_masks = self.compute_forward_edge_masks(
            z_user,
            z_item
        )

        branch1_edge_mask = self.to_bidirectional_mask(
            edge_masks[:, 0]
        )
        branch2_edge_mask = self.to_bidirectional_mask(
            edge_masks[:, 1]
        )

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

        self.branch1_preference = branch1_preference
        self.branch2_preference = branch2_preference
        self.branch1_rep = branch1_rep
        self.branch2_rep = branch2_rep
        self.last_edge_factor_scores = edge_factor_scores
        self.last_edge_masks = edge_masks

        return (
            branch1_rep,
            branch2_rep,
            router_rep,
            z_user,
            z_item,
            edge_factor_scores,
            edge_masks
        )

    def fuse_representations(self, branch1_rep, branch2_rep):
        """
        Preserve original GLORIA fusion:
          - concatenate user outputs from both branches
          - concatenate item outputs from both branches
          - then apply the existing item-item graph to item representations
        """
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

    # -----------------------------------------------------------------
    # Recommendation scoring
    # -----------------------------------------------------------------
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
        return -F.logsigmoid(
            pos_scores - neg_scores
        ).mean()

    # -----------------------------------------------------------------
    # Factor/routing regularizers
    # -----------------------------------------------------------------
    def routing_entropy_loss(self, edge_masks):
        """
        Optional: lower entropy -> more decisive factor assignment per edge.
        """
        if edge_masks.numel() == 0:
            return edge_masks.sum() * 0.0

        eps = 1e-8
        entropy = -(
            edge_masks
            * torch.log(edge_masks + eps)
        ).sum(dim=1)

        return entropy.mean()

    def routing_collapse_loss(self, edge_masks):
        """
        Prevent either factor graph from disappearing.

        This only enforces a minimum average EDGE mass per branch and does not
        force 50/50 routing.
        """
        if edge_masks.numel() == 0:
            return edge_masks.sum() * 0.0

        branch_usage = edge_masks.mean(dim=0)

        minimum = torch.tensor(
            self.factor_min_branch_usage,
            device=edge_masks.device,
            dtype=edge_masks.dtype
        )

        return F.relu(
            minimum - branch_usage
        ).pow(2).sum()

    @staticmethod
    def factor_independence_loss(z_user, z_item):
        """
        Penalize correlation between the two factor columns.

        We center and standardize each factor column, then penalize only the
        off-diagonal correlation. For K=2 this is simply corr(z[:,0],z[:,1])^2.
        """
        z = torch.cat(
            [z_user, z_item],
            dim=0
        )

        if z.size(0) <= 1:
            return z.sum() * 0.0

        z = z - z.mean(dim=0, keepdim=True)
        z = z / (
            z.std(dim=0, unbiased=False, keepdim=True)
            + 1e-6
        )

        corr = torch.matmul(
            z.transpose(0, 1),
            z
        ) / float(z.size(0))

        eye = torch.eye(
            corr.size(0),
            device=corr.device,
            dtype=corr.dtype
        )

        off_diag = corr * (1.0 - eye)
        return off_diag.pow(2).sum() / max(
            corr.numel() - corr.size(0),
            1
        )

    def factor_structure_bpr_loss(self, interaction, z_user, z_item):
        """
        Optional direct supervision that teaches latent factors to explain
        observed interactions better than sampled negatives.

        score_factor(u,i) = sum_k gamma_k * z_uk * z_ik

        This is NOT DiGGR's Bernoulli-Poisson ELBO. It is a lightweight
        recommendation-specific structural objective using the training BPR
        triplets that already exist in GLORIA.
        """
        user_ids = interaction[0]
        pos_item_ids = interaction[1]
        neg_item_ids = interaction[2]

        z_u = z_user[user_ids]
        z_pos = z_item[pos_item_ids]
        z_neg = z_item[neg_item_ids]

        gamma = self.get_factor_gamma().view(1, -1)

        pos_factor_score = (
            gamma * z_u * z_pos
        ).sum(dim=1)

        neg_factor_score = (
            gamma * z_u * z_neg
        ).sum(dim=1)

        return -F.logsigmoid(
            pos_factor_score - neg_factor_score
        ).mean()

    def embedding_regularization_loss(self):
        if self.embedding_reg_weight <= 0:
            return self.factor_gamma_raw.sum() * 0.0

        reg = (
            self.id_embedding_branch1.weight.pow(2).mean()
            + self.id_embedding_branch2.weight.pow(2).mean()
            + self.router_item_embedding.weight.pow(2).mean()
            + self.branch1_gcn.preference.pow(2).mean()
            + self.branch2_gcn.preference.pow(2).mean()
            + self.router_gcn.preference.pow(2).mean()
        )

        return reg

    # -----------------------------------------------------------------
    # Main forward / loss
    # -----------------------------------------------------------------
    def forward(self, interaction, return_aux=False):
        (
            branch1_rep,
            branch2_rep,
            router_rep,
            z_user,
            z_item,
            edge_factor_scores,
            edge_masks
        ) = self.compute_branch_representations()

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
            'router_rep': router_rep,
            'z_user': z_user,
            'z_item': z_item,
            'factor_gamma': self.get_factor_gamma(),
            'edge_factor_scores': edge_factor_scores,
            'edge_masks': edge_masks,
        }

        return pos_scores, neg_scores, aux

    def calculate_loss(self, interaction):
        """
        Total objective:

            L = L_BPR
              + lambda_struct * L_factor_struct
              + lambda_balance * L_edge_collapse
              + lambda_indep * L_factor_independence
              + lambda_entropy * L_edge_entropy
              + lambda_reg * L_embedding

        Main BPR gradients reach the router through:

          recommendation score
            -> branch GCN
            -> edge mask
            -> factor compatibility
            -> z_user,z_item
            -> factor nets
            -> shared router GCN
        """
        pos_scores, neg_scores, aux = self.forward(
            interaction,
            return_aux=True
        )

        recommendation_loss = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        structure_loss = self.factor_structure_bpr_loss(
            interaction,
            aux['z_user'],
            aux['z_item']
        )

        balance_loss = self.routing_collapse_loss(
            aux['edge_masks']
        )

        independence_loss = self.factor_independence_loss(
            aux['z_user'],
            aux['z_item']
        )

        entropy_loss = self.routing_entropy_loss(
            aux['edge_masks']
        )

        embedding_reg = self.embedding_regularization_loss()

        total_loss = (
            recommendation_loss
            + self.factor_structure_weight * structure_loss
            + self.factor_balance_weight * balance_loss
            + self.factor_independence_weight * independence_loss
            + self.factor_entropy_weight * entropy_loss
            + self.embedding_reg_weight * embedding_reg
        )

        return total_loss

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def full_sort_predict(self, interaction):
        (
            branch1_rep,
            branch2_rep,
            _,
            _,
            _,
            _,
            _
        ) = self.compute_branch_representations()

        self.result_embed = self.fuse_representations(
            branch1_rep,
            branch2_rep
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

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    @torch.no_grad()
    def get_factor_routing_statistics(self):
        """
        Inspect what the latent-factor router learned without using degree.

        Useful quantities:
          - average edge mass assigned to each branch
          - number of interactions whose hard argmax is branch 1 / branch 2
          - average routing entropy
          - mean user/item factor affiliation
          - learned positive gamma values
          - factor-column correlation
        """
        _, z_user, z_item = self.compute_latent_factors()
        edge_factor_scores, edge_masks = self.compute_forward_edge_masks(
            z_user,
            z_item
        )

        hard_branch = torch.argmax(
            edge_masks,
            dim=1
        )

        branch_usage = edge_masks.mean(dim=0)

        eps = 1e-8
        entropy = -(
            edge_masks
            * torch.log(edge_masks + eps)
        ).sum(dim=1)

        z_all = torch.cat([z_user, z_item], dim=0)
        z_centered = z_all - z_all.mean(dim=0, keepdim=True)
        z_std = z_centered.std(
            dim=0,
            unbiased=False,
            keepdim=True
        ) + 1e-6
        z_norm = z_centered / z_std
        corr = torch.matmul(
            z_norm.transpose(0, 1),
            z_norm
        ) / float(max(z_norm.size(0), 1))

        stats = {
            'factor_gamma': self.get_factor_gamma().detach(),
            'mean_user_factors': z_user.mean(dim=0).detach(),
            'mean_item_factors': z_item.mean(dim=0).detach(),
            'edge_branch_usage': branch_usage.detach(),
            'mean_edge_entropy': entropy.mean().detach(),
            'num_branch1_interactions': (hard_branch == 0).sum().detach(),
            'num_branch2_interactions': (hard_branch == 1).sum().detach(),
            'hard_branch_per_interaction': hard_branch.detach(),
            'edge_masks': edge_masks.detach(),
            'edge_factor_scores': edge_factor_scores.detach(),
            'factor_correlation_matrix': corr.detach(),
        }

        return stats


class GCN(torch.nn.Module):
    """
    LightGCN-style propagation module.

    The router GCN and the two downstream branch GCNs each own their own user
    preference embedding. Branch GCNs receive soft factor-derived edge weights.
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
            (
                self.preference,
                features
            ),
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
    LightGCN-style message passing with optional weighted edges.

    IMPORTANT:
    When an edge mask is supplied, normalization is recomputed from the
    factor-weighted graph itself:

        d_v^(k) = sum_{e incident from v} m_e^(k)

        message_e = m_e^(k) / sqrt(d_src^(k) d_dst^(k)) * x_src

    This makes each branch behave as its own weighted adjacency matrix A^(k),
    rather than normalizing every branch with degrees from the original graph.
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
        super(Base_gcn, self).__init__(
            aggr=aggr,
            **kwargs
        )

        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_mask=None, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        if size is None:
            size = (
                x.size(0),
                x.size(0)
            )

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

            # Weighted branch-specific degrees. This operation remains
            # differentiable with respect to edge_mask.
            deg = torch.zeros(
                size[0],
                device=x.device,
                dtype=x.dtype
            ).scatter_add(
                0,
                row,
                edge_mask
            )

            deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)

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
        return (
            edge_weight.view(-1, 1)
            * x_j
        )

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels
        )
