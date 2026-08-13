# coding: utf-8
"""
GLORIA with learnable ITEM-LEVEL Gumbel hard routing.

Main idea
---------
Each item i owns two trainable routing logits. During training, straight-through
Gumbel-Softmax converts those logits into a one-hot route:

    [r_i^(1), r_i^(2)] in {[1, 0], [0, 1]}

so every item is present in exactly one user-item graph branch in the forward
pass. The straight-through estimator keeps gradients flowing back to the item
routing logits.

A separate deterministic soft distribution is also kept:

    p_i = softmax(item_mask_logits[i] / temperature)

and is used for collapse regularization and diagnostics.

Architecture
------------
    user-item graph G
           |
      item router logits
           |
   straight-through Gumbel
       /             \
   hard branch 1   hard branch 2
       |               |
     GCN-1           GCN-2
       |               |
       +---- concat ----+
              |
         item-item GCN
              |
             BPR

Important implementation detail
-------------------------------
The LightGCN normalization is recomputed from each branch's masked edges, so a
zero-routed edge does not contribute to that branch's degree normalization.
This makes the hard-routed branches behave much more like separately built
interaction graphs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from Experiment.CaMuRe.src.common.abstract_recommender import GeneralRecommender


def _cfg(config, key, default):
    """Read an optional config value without assuming config implements .get()."""
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


class HARD_ROUTING_GUMBEL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HARD_ROUTING_GUMBEL, self).__init__(config, dataset)

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
        # Learnable Gumbel item-router hyperparameters.
        # -------------------------------------------------------------
        # Temperature controls the underlying soft Gumbel sample.
        # hard=True still makes forward routes one-hot.
        self.mask_temperature = float(
            _cfg(config, 'mask_temperature', 1.0)
        )

        # Gumbel hard routing is already one-hot in the forward pass, so
        # entropy regularization is optional. Default is disabled.
        self.mask_entropy_weight = float(
            _cfg(config, 'mask_entropy_weight', 0.0)
        )

        # Collapse prevention. We do NOT force exactly 50/50 branch usage.
        # Instead each branch must receive at least this average item mass.
        self.mask_min_branch_usage = float(
            _cfg(config, 'mask_min_branch_usage', 0.05)
        )
        self.mask_balance_weight = float(
            _cfg(config, 'mask_balance_weight', 0.1)
        )

        # Optional L2 regularization on the two branch item/user ID embeddings.
        # Keep 0.0 if your training framework already applies weight decay.
        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        # -------------------------------------------------------------
        # Two separate item embedding tables, mirroring the old two-branch
        # low-degree / high-degree architecture.
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
            # Same starting point for a fair branch comparison.
            # They become independent after optimization begins.
            self.id_embedding_branch2.weight.copy_(
                self.id_embedding_branch1.weight
            )

        # Keep these from the original GLORIA implementation because other
        # parts of the project may expect them.
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
        # Existing item-item graph built from text features.
        # -------------------------------------------------------------
        _, text_adj = self.get_knn_adj_mat(self.t_feat)
        self.mm_adj = text_adj

        # -------------------------------------------------------------
        # Build ONE full user-item interaction graph.
        # -------------------------------------------------------------
        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)

        # Unique forward user -> item edges, shape [E, 2].
        forward_edges_np = self.pack_edge_index(train_interactions)
        self.num_interactions = forward_edges_np.shape[0]

        # IMPORTANT:
        # train_interactions.col contains item IDs WITHOUT the user offset.
        # This aligns one-to-one with forward_edges_np.
        self.edge_item_ids = torch.tensor(
            train_interactions.col,
            dtype=torch.long,
            device=self.device
        )  # [E]

        # Item degree is used ONLY for analysis/diagnostics, NOT for routing.
        item_degree_np = np.bincount(
            train_interactions.col,
            minlength=self.num_item
        )
        self.item_degree = torch.tensor(
            item_degree_np,
            dtype=torch.float32,
            device=self.device
        )
        self.active_item_mask = self.item_degree > 0

        forward_edges = torch.tensor(
            forward_edges_np,
            dtype=torch.long,
            device=self.device
        ).t().contiguous()  # [2, E]

        reverse_edges = forward_edges[[1, 0], :]

        # Exact ordering:
        #   [all u -> i edges, then the exact reverse i -> u edges]
        self.edge_index = torch.cat(
            [forward_edges, reverse_edges],
            dim=1
        )  # [2, 2E]

        # -------------------------------------------------------------
        # ITEM-LEVEL learnable two-way routing logits.
        # Shape [num_items, 2], not [num_edges].
        # -------------------------------------------------------------
        self.item_mask_logits = nn.Parameter(
            1e-2 * torch.randn(
                self.num_item,
                2,
                device=self.device
            )
        )

        # -------------------------------------------------------------
        # Two separate GCN branches.
        # Each GCN has its own user preference embedding, matching the old
        # separate low/high GCN design.
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

    # -----------------------------------------------------------------
    # Learnable ITEM routing
    # -----------------------------------------------------------------
    def get_item_masks(self):
        """
        Return both hard routing masks and soft routing probabilities.

        During training:
            hard_masks are sampled with straight-through Gumbel-Softmax.
            Forward values are one-hot, but gradients flow through the soft
            Gumbel sample.

        During evaluation:
            hard_masks are deterministic argmax one-hot assignments.

        Returns:
            hard_masks: [num_item, 2], one-hot in the forward pass
            soft_masks: [num_item, 2], deterministic probabilities
        """
        temperature = max(
            self.mask_temperature,
            1e-6
        )

        # Deterministic probabilities are useful for regularization, logging,
        # and stable diagnostics.
        soft_masks = torch.softmax(
            self.item_mask_logits / temperature,
            dim=1
        )

        if self.training:
            # Straight-through Gumbel-Softmax:
            #   forward  -> exact [1, 0] or [0, 1]
            #   backward -> differentiable soft gradient
            hard_masks = F.gumbel_softmax(
                self.item_mask_logits,
                tau=temperature,
                hard=True,
                dim=1
            )
        else:
            # No Gumbel noise at validation/test time.
            assignment = torch.argmax(
                self.item_mask_logits,
                dim=1
            )
            hard_masks = F.one_hot(
                assignment,
                num_classes=2
            ).to(dtype=self.item_mask_logits.dtype)

        return hard_masks, soft_masks

    def get_forward_edge_masks(self, item_masks):
        """
        Convert item masks [num_item, 2] to forward-edge masks [E, 2].

        Because routing is item-level, all interactions involving the same item
        receive the same route.
        """
        return item_masks[self.edge_item_ids]

    @staticmethod
    def to_bidirectional_mask(mask):
        """
        Convert mask [E] -> [2E], matching self.edge_index ordering:
            [u->i edges, i->u edges].
        """
        return torch.cat(
            [mask, mask],
            dim=0
        )

    # -----------------------------------------------------------------
    # Two-branch representations
    # -----------------------------------------------------------------
    def compute_branch_representations(self):
        """
        Build two hard-routed graph branches.

        Returns:
            branch1_rep:    [num_user + num_item, D]
            branch2_rep:    [num_user + num_item, D]
            hard_item_masks:[num_item, 2]
            soft_item_masks:[num_item, 2]
            hard_edge_masks:[E, 2]
        """
        hard_item_masks, soft_item_masks = self.get_item_masks()

        # Message passing uses the HARD one-hot routes.
        hard_edge_masks = self.get_forward_edge_masks(
            hard_item_masks
        )

        branch1_edge_mask = self.to_bidirectional_mask(
            hard_edge_masks[:, 0]
        )
        branch2_edge_mask = self.to_bidirectional_mask(
            hard_edge_masks[:, 1]
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

        # Keep for analysis/debugging.
        self.branch1_preference = branch1_preference
        self.branch2_preference = branch2_preference
        self.branch1_rep = branch1_rep
        self.branch2_rep = branch2_rep
        self.last_hard_item_masks = hard_item_masks
        self.last_soft_item_masks = soft_item_masks
        self.last_hard_edge_masks = hard_edge_masks

        return (
            branch1_rep,
            branch2_rep,
            hard_item_masks,
            soft_item_masks,
            hard_edge_masks
        )

    def fuse_representations(self, branch1_rep, branch2_rep):
        """
        Preserve the successful original GLORIA two-branch fusion:

          users: concat branch-1 and branch-2 outputs
          items: concat branch-1 and branch-2 outputs, then item-item GCN
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

        # Existing item-item graph propagation after branch fusion.
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
    # Mask regularizers
    # -----------------------------------------------------------------
    def mask_entropy_loss(self, item_masks):
        """
        Minimize entropy to make item routing less ambiguous.

        A row [0.5, 0.5] has high entropy.
        A row [0.99, 0.01] has low entropy.

        Only items appearing in the training graph are regularized.
        """
        active_masks = item_masks[self.active_item_mask]

        if active_masks.numel() == 0:
            return item_masks.sum() * 0.0

        eps = 1e-8
        entropy = -(
            active_masks
            * torch.log(active_masks + eps)
        ).sum(dim=1)

        return entropy.mean()

    def mask_collapse_loss(self, item_masks):
        """
        Prevent either branch from disappearing without forcing a 50/50 split.

        Example with min usage = 0.05:
            usage [0.90, 0.10] -> no penalty
            usage [0.99, 0.01] -> branch 2 is penalized

        This is more appropriate than forcing exact 50/50 when the useful
        decomposition may naturally be unbalanced, as in low/high-degree splits.
        """
        active_masks = item_masks[self.active_item_mask]

        if active_masks.numel() == 0:
            return item_masks.sum() * 0.0

        branch_usage = active_masks.mean(dim=0)

        minimum = torch.tensor(
            self.mask_min_branch_usage,
            device=item_masks.device,
            dtype=item_masks.dtype
        )

        return F.relu(
            minimum - branch_usage
        ).pow(2).sum()

    def embedding_regularization_loss(self):
        """Optional small L2 regularization for branch ID embeddings."""
        if self.embedding_reg_weight <= 0:
            return self.item_mask_logits.sum() * 0.0

        reg = (
            self.id_embedding_branch1.weight.pow(2).mean()
            + self.id_embedding_branch2.weight.pow(2).mean()
            + self.branch1_gcn.preference.pow(2).mean()
            + self.branch2_gcn.preference.pow(2).mean()
        )

        return reg

    # -----------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------
    def forward(self, interaction, return_aux=False):
        (
            branch1_rep,
            branch2_rep,
            hard_item_masks,
            soft_item_masks,
            hard_edge_masks
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
            'hard_item_masks': hard_item_masks,
            'soft_item_masks': soft_item_masks,
            'hard_edge_masks': hard_edge_masks,
        }

        return pos_scores, neg_scores, aux

    # -----------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------
    def calculate_loss(self, interaction):
        """
        Objective:

            L = BPR
                + optional lambda_entropy * soft-mask entropy
                + lambda_balance * collapse prevention
                + optional embedding L2

        GCN message passing uses hard Gumbel routes. The straight-through
        estimator lets BPR gradients update item_mask_logits. Regularizers use
        the deterministic soft probabilities.
        """
        pos_scores, neg_scores, aux = self.forward(
            interaction,
            return_aux=True
        )

        recommendation_loss = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        entropy_loss = self.mask_entropy_loss(
            aux['soft_item_masks']
        )

        collapse_loss = self.mask_collapse_loss(
            aux['soft_item_masks']
        )

        embedding_reg = self.embedding_regularization_loss()

        total_loss = (
            recommendation_loss
            + self.mask_entropy_weight * entropy_loss
            + self.mask_balance_weight * collapse_loss
            + self.embedding_reg_weight * embedding_reg
        )

        return total_loss

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def full_sort_predict(self, interaction):
        branch1_rep, branch2_rep, _, _, _ = (
            self.compute_branch_representations()
        )

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
    # Diagnostics / analysis helpers
    # -----------------------------------------------------------------
    @torch.no_grad()
    def get_item_routing_statistics(self):
        """
        Return useful tensors for analyzing what the learned split discovered.

        Useful checks after training:
          - compare branch assignment with item degree
          - count hard branch assignments
          - inspect mean branch usage

        This function does not affect training.
        """
        # Diagnostics are deterministic even if the model is currently in
        # training mode: use logits -> soft probabilities -> argmax.
        temperature = max(self.mask_temperature, 1e-6)
        soft_masks = torch.softmax(
            self.item_mask_logits / temperature,
            dim=1
        )
        hard_branch_all = torch.argmax(
            self.item_mask_logits,
            dim=1
        )
        hard_masks = F.one_hot(
            hard_branch_all,
            num_classes=2
        ).to(dtype=soft_masks.dtype)

        active = self.active_item_mask
        active_soft_masks = soft_masks[active]
        active_hard_masks = hard_masks[active]
        active_degree = self.item_degree[active]
        hard_branch = hard_branch_all[active]

        soft_branch_usage = active_soft_masks.mean(dim=0)
        hard_branch_usage = active_hard_masks.mean(dim=0)

        branch1_degree = active_degree[hard_branch == 0]
        branch2_degree = active_degree[hard_branch == 1]

        stats = {
            'soft_item_masks': soft_masks.detach(),
            'hard_item_masks': hard_masks.detach(),
            'active_soft_item_masks': active_soft_masks.detach(),
            'active_hard_item_masks': active_hard_masks.detach(),
            'active_item_degree': active_degree.detach(),
            'hard_branch': hard_branch.detach(),
            'soft_branch_usage': soft_branch_usage.detach(),
            'hard_branch_usage': hard_branch_usage.detach(),
            'num_branch1_items': (hard_branch == 0).sum().detach(),
            'num_branch2_items': (hard_branch == 1).sum().detach(),
            'mean_degree_branch1': (
                branch1_degree.mean().detach()
                if branch1_degree.numel() > 0
                else torch.tensor(float('nan'), device=self.device)
            ),
            'mean_degree_branch2': (
                branch2_degree.mean().detach()
                if branch2_degree.numel() > 0
                else torch.tensor(float('nan'), device=self.device)
            ),
        }

        return stats


class GCN(torch.nn.Module):
    """
    LightGCN-style propagation module.

    Each branch owns a separate GCN instance and therefore a separate learnable
    user preference embedding. The parent model also provides separate branch
    item embeddings.
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

        # LightGCN message passing: no feature transformation matrix.
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

        # Preserve the original GLORIA three propagation steps and residual sum.
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

    For edge j -> i:

        message = 1 / sqrt(d_j * d_i) * mask_e * x_j

    The degree normalization is recomputed from the masked branch graph.
    Therefore zero-routed edges do not contribute to branch degrees.
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

        # Keep edge masks aligned if self-loops ever appear.
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

            # Branch-specific weighted degree. For hard Gumbel routing the
            # forward values of edge_mask are exactly 0/1, so this is the same
            # degree that would be obtained from the explicitly routed branch
            # graph. Because edge_mask is straight-through, gradients can still
            # flow to the router during training.
            deg = torch.zeros(
                size[0],
                device=x.device,
                dtype=x.dtype
            ).scatter_add(
                0,
                row,
                edge_mask
            )

            positive = deg > 0
            safe_deg = deg.clamp_min(1e-12)
            deg_inv_sqrt = torch.where(
                positive,
                safe_deg.pow(-0.5),
                torch.zeros_like(deg)
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