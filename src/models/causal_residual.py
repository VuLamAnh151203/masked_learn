# coding: utf-8
"""
GLORIA variant with CaGE-inspired causal / non-causal soft edge masks.

Design:
  1) One user-item interaction graph G (stored bidirectionally).
  2) One learnable scalar logit per UNIQUE user-item interaction.
  3) causal_mask = sigmoid(mask_logits)
     noncausal_mask = 1 - causal_mask
  4) Separate item embeddings and separate GCNs for the causal/complementary branches.
  5) Remove from the complementary representation the component parallel to
     the causal representation:
         Z_res = Z_comp - Proj_Zcausal(Z_comp)
  6) Concatenate [Z_causal || Z_res] for recommendation.
  7) Train the fused representation with BPR only.

Important:
  - The randomization rule follows the user-provided CaGE description:
        m <= 0.5 -> Uniform(0, 0.5)
        m >  0.5 -> Uniform(0.5 + eps, 1)
  - The training loss below is a practical LightGCN adaptation, NOT an exact
    reproduction of CaGE's full Equation (7), because that equation/objective
    was not provided in the source code.
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


class CAUSAL_RESIDUAL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CAUSAL_RESIDUAL, self).__init__(config, dataset)
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
        # Loss weights for this LightGCN adaptation.
        # Add these keys to your yaml if you want to change them.
        # -------------------------------------------------------------
        self.causal_bpr_weight = float(_cfg(config, 'causal_bpr_weight', 1.0))
        self.intervention_weight = float(_cfg(config, 'intervention_weight', 0.1))
        self.mask_entropy_weight = float(_cfg(config, 'mask_entropy_weight', 1e-3))
        self.mask_balance_weight = float(_cfg(config, 'mask_balance_weight', 0.1))
        self.causal_ratio = float(_cfg(config, 'causal_ratio', 0.5))
        self.mask_random_eps = float(_cfg(config, 'mask_random_eps', 1e-4))
        # self.alpha = 0.5
        # self.beta = 0.5

        self.residual_alpha_logit = nn.Parameter(
                        torch.tensor(
                            0.0,
                            dtype=torch.float32,
                            device=self.device
                        )
                    )

        self.residual_beta_logit = nn.Parameter(
                                torch.tensor(
                                    0.0,
                                    dtype=torch.float32,
                                    device=self.device
                                )
                            )

        # -------------------------------------------------------------
        # Separate item embeddings for causal and non-causal branches.
        # -------------------------------------------------------------
        self.id_embedding_causal = nn.Embedding(num_item, self.feat_embed_dim)
        self.id_embedding_noncausal = nn.Embedding(num_item, self.feat_embed_dim)

        nn.init.xavier_uniform_(self.id_embedding_causal.weight)
        with torch.no_grad():
            # Same starting point, but parameters are independent afterward.
            self.id_embedding_noncausal.weight.copy_(
                self.id_embedding_causal.weight
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
        # One learnable scalar logit per UNIQUE user-item interaction.
        # causal = sigmoid(logit), noncausal = 1 - causal.
        # Small noise avoids every edge starting exactly at the 0.5 threshold.
        # -------------------------------------------------------------
        self.mask_logits = nn.Parameter(
            1e-2 * torch.randn(
                self.num_interactions,
                device=self.device
            )
        )

        # -------------------------------------------------------------
        # Two separate GCNs.
        # Each GCN also has its own learnable user preference embedding.
        # -------------------------------------------------------------
        self.causal_gcn = GCN(
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
            features=self.id_embedding_causal.weight
        )

        self.noncausal_gcn = GCN(
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
            features=self.id_embedding_noncausal.weight
        )

        # Cached normal (non-randomized) fused embedding for prediction.
        self.result_embed = None

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
    # Soft masks
    # -----------------------------------------------------------------
    def get_masks(self):
        """
        Returns one soft causal and one soft non-causal mask for each UNIQUE
        user-item interaction.

        Shapes:
            causal_mask:    [E]
            noncausal_mask: [E]
        """
        causal_mask = torch.sigmoid(self.mask_logits)
        noncausal_mask = 1.0 - causal_mask
        return causal_mask, noncausal_mask

    @staticmethod
    def to_bidirectional_mask(mask):
        """[E] -> [2E], matching [u->i, i->u] edge ordering."""
        return torch.cat([mask, mask], dim=0)

    def randomize_noncausal_mask(self, noncausal_mask):
        """
        CaGE-style randomization from the user-provided description:

            m <= 0.5 : sample Uniform(0, 0.5)
            m >  0.5 : sample Uniform(0.5 + eps, 1)

        The threshold side is preserved, so the coarse topology induced by
        threshold 0.5 does not switch during the intervention.
        """
        eps = self.mask_random_eps

        low = torch.rand_like(noncausal_mask) * 0.5
        high = (
            0.5 + eps
            + torch.rand_like(noncausal_mask) * (0.5 - eps)
        )

        return torch.where(
            noncausal_mask <= 0.5,
            low,
            high
        )

    # -----------------------------------------------------------------
    # Representation construction
    # -----------------------------------------------------------------
    def compute_branch_representations(self, randomize_noncausal=False):
        """
        Returns:
            causal_rep:      [num_user + num_item, D]
            noncausal_rep:   [num_user + num_item, D]
            causal_mask:     [E]
            noncausal_mask:  [E] (normal mask, before randomization)
        """
        causal_mask, noncausal_mask = self.get_masks()

        causal_edge_mask = self.to_bidirectional_mask(causal_mask)

        if randomize_noncausal:
            used_noncausal_mask = self.randomize_noncausal_mask(noncausal_mask)
        else:
            used_noncausal_mask = noncausal_mask

        noncausal_edge_mask = self.to_bidirectional_mask(used_noncausal_mask)

        causal_rep, causal_preference = self.causal_gcn(
            self.edge_index,
            self.id_embedding_causal.weight,
            edge_mask=causal_edge_mask
        )

        # IMPORTANT: randomized intervention reuses the SAME noncausal_gcn and
        # SAME noncausal item embeddings. Only the graph mask changes.
        noncausal_rep, noncausal_preference = self.noncausal_gcn(
            self.edge_index,
            self.id_embedding_noncausal.weight,
            edge_mask=noncausal_edge_mask
        )

        # Keep references available for debugging/analysis.
        self.causal_preference = causal_preference
        self.noncausal_preference = noncausal_preference

        return causal_rep, noncausal_rep, causal_mask, noncausal_mask

    def compute_random_noncausal_representation(self, noncausal_mask):
        """Run the SAME non-causal GCN under a randomized non-causal mask."""
        random_noncausal_mask = self.randomize_noncausal_mask(noncausal_mask)
        random_noncausal_edge_mask = self.to_bidirectional_mask(
            random_noncausal_mask
        )

        random_noncausal_rep, _ = self.noncausal_gcn(
            self.edge_index,
            self.id_embedding_noncausal.weight,
            edge_mask=random_noncausal_edge_mask
        )

        return random_noncausal_rep

    @staticmethod
    def remove_causal_projection(causal_rep, complementary_rep, eps=1e-8):
        """
        Remove from the complementary representation the component that is
        parallel to the causal representation, independently for every node.

        For each node:
            projection = <z_r, z_c> / (||z_c||^2 + eps) * z_c
            z_residual = z_r - projection

        Shapes:
            causal_rep:        [N, D]
            complementary_rep: [N, D]
            residual_rep:      [N, D]
        """
        projection_scale = (
            (complementary_rep * causal_rep).sum(dim=1, keepdim=True)
            / causal_rep.pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
        )

        alpha = torch.sigmoid(
                        self.residual_alpha_logit
                    )
        causal_direction = causal_rep.detach()
        projection = projection_scale * causal_direction
        # residual_rep = complementary_rep - projection
        residual_rep = complementary_rep - alpha * projection


        return residual_rep

    def fuse_representations(self, causal_rep, noncausal_rep):
        """
        Two-view residual fusion.

        View 1:
            causal/core representation z_c.

        View 2:
            complementary representation z_r, constructed from (1 - M).

        Before concatenation, remove the component of z_r that lies along z_c:

            z_res = z_r - Proj_zc(z_r)

        Final representation:
            z_final = [z_c || z_res]

        This keeps exactly two GCN views while encouraging the second half to
        contribute information that is different from the causal/core half.
        """
        residual_rep = self.remove_causal_projection(
            causal_rep,
            noncausal_rep
        )

        self.residual_rep = residual_rep

        user_causal = causal_rep[:self.num_user]
        user_residual = residual_rep[:self.num_user]

        item_causal = causal_rep[self.num_user:]
        item_residual = residual_rep[self.num_user:]

        # user_rep = torch.cat(
        #     [user_causal, user_residual],
        #     dim=1
        # )

        # item_rep = torch.cat(
        #     [item_causal, item_residual],
        #     dim=1
        # )

        beta = torch.sigmoid(
                        self.residual_beta_logit)
        
        user_rep = torch.cat(
                [user_causal, beta * user_residual],
                dim=1
            )
    
        item_rep = torch.cat(
                [item_causal, beta * item_residual],
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
        # Normal causal + normal non-causal graph.
        causal_rep, noncausal_rep, causal_mask, noncausal_mask = (
            self.compute_branch_representations(randomize_noncausal=False)
        )

        self.causal_rep = causal_rep
        self.noncausal_rep = noncausal_rep

        self.result_embed = self.fuse_representations(
            causal_rep,
            noncausal_rep
        )

        pos_scores, neg_scores = self.pairwise_scores(
            self.result_embed,
            interaction
        )

        if not return_aux:
            return pos_scores, neg_scores

        # Intervention: causal graph is unchanged; only non-causal mask is randomized.
        random_noncausal_rep = self.compute_random_noncausal_representation(
            noncausal_mask
        )

        self.random_noncausal_rep = random_noncausal_rep

        random_result_embed = self.fuse_representations(
            causal_rep,
            random_noncausal_rep
        )

        random_pos_scores, random_neg_scores = self.pairwise_scores(
            random_result_embed,
            interaction
        )

        # Causal-only scores are useful for explicitly training the causal branch.
        causal_pos_scores, causal_neg_scores = self.pairwise_scores(
            causal_rep,
            interaction
        )

        aux = {
            'causal_mask': causal_mask,
            'noncausal_mask': noncausal_mask,
            'causal_rep': causal_rep,
            'noncausal_rep': noncausal_rep,
            'random_noncausal_rep': random_noncausal_rep,
            'random_pos_scores': random_pos_scores,
            'random_neg_scores': random_neg_scores,
            'causal_pos_scores': causal_pos_scores,
            'causal_neg_scores': causal_neg_scores,
        }

        return pos_scores, neg_scores, aux

    # def compute_causal_representation(self):

    #     causal_mask = torch.sigmoid(
    #         self.mask_logits
    #     )

    #     causal_edge_mask = self.to_bidirectional_mask(
    #         causal_mask
    #     )

    #     causal_rep, causal_preference = self.causal_gcn(
    #         self.edge_index,
    #         self.id_embedding_causal.weight,
    #         edge_mask=causal_edge_mask
    #     )

    #     self.causal_preference = causal_preference

    #     return causal_rep, causal_mask

    # def forward(self, interaction, return_aux=False):

    #     causal_rep, causal_mask = (
    #         self.compute_causal_representation()
    #     )

    #     self.causal_rep = causal_rep

    #     self.result_embed = self.fuse_representations(
    #         causal_rep
    #     )

    #     pos_scores, neg_scores = self.pairwise_scores(
    #         self.result_embed,
    #         interaction
    #     )

    #     if not return_aux:
    #         return pos_scores, neg_scores

    #     aux = {
    #         'causal_mask': causal_mask,
    #         'causal_rep': causal_rep
    #     }

    #     return pos_scores, neg_scores, aux

    # -----------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------
    # def calculate_loss(self, interaction):

    #     pos_scores, neg_scores, aux = self.forward(
    #         interaction,
    #         return_aux=True
    #     )

    #     # Recommendation loss
    #     bpr_loss = self.bpr_loss(
    #         pos_scores,
    #         neg_scores
    #     )

    #     causal_mask = aux['causal_mask']

    #     eps = 1e-8

    #     # Encourage decisive masks
    #     mask_entropy_loss = -(
    #         causal_mask * torch.log(causal_mask + eps)
    #         +
    #         (1.0 - causal_mask)
    #         * torch.log(1.0 - causal_mask + eps)
    #     ).mean()

    #     # Control how much graph is retained
    #     mask_balance_loss = (
    #         causal_mask.mean()
    #         - self.causal_ratio
    #     ).pow(2)

    #     total_loss = (
    #         bpr_loss
    #         + self.mask_entropy_weight * mask_entropy_loss
    #         + self.mask_balance_weight * mask_balance_loss
    #     )

    #     return total_loss
    def calculate_loss(self, interaction):
        """
        Recommendation objective for the residual two-view model.

        The final representation is:
            [causal/core || residual-complementary]

        Start with BPR only so the experiment isolates whether residual
        complementary fusion itself improves recommendation performance.
        """
        pos_scores, neg_scores = self.forward(
            interaction,
            return_aux=False
        )

        return self.bpr_loss(
            pos_scores,
            neg_scores
        )

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def full_sort_predict(self, interaction):
        # Use normal causal + complementary graphs, then residual fusion.
        causal_rep, noncausal_rep, _, _ = self.compute_branch_representations(
            randomize_noncausal=False
        )

        self.result_embed = self.fuse_representations(
            causal_rep,
            noncausal_rep
        )

        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

        return score_matrix


class GCN(torch.nn.Module):
    """
    LightGCN-style propagation module.

    Each branch owns a separate GCN instance, therefore each branch also owns
    a separate user preference embedding. Item embeddings are passed from the
    parent model and are also separate between causal/non-causal branches.
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