"""
GLORIA with DEGREE-GUIDED learnable boundary correction.

Main idea
---------
Start from the exact fixed high/low-degree item split that already works well:

    high items = top `high_ratio` items by training interaction degree
    low items  = all remaining items

Then make ONLY a small band of items around the degree cutoff learnable.
Items far from the cutoff are permanently fixed to their original degree branch.
Boundary items receive a trainable scalar correction that can move them across
that cutoff if the recommendation loss consistently supports the change.

For item i:

    base_score_i = +prior_strength   if degree prior says HIGH
                 = -prior_strength   if degree prior says LOW

    score_i = base_score_i + correction_i

Only boundary items have a trainable correction_i. All other items have
correction_i = 0 forever.

Routing uses a deterministic straight-through hard threshold (NO Gumbel noise):

    forward : score_i >= 0 -> HIGH, else LOW       (exact hard graph split)
    backward: gradients flow through sigmoid(score_i / temperature)

Therefore training starts from the exact fixed high/low topology and can only
make local assignment corrections near the boundary.

Architecture
------------
    training user-item graph G
              |
        fixed degree prior
              |
      only boundary corrections
              |
      deterministic hard router
          /             \
      low graph       high graph
          |               |
        GCN-low         GCN-high
          |               |
          +---- concat ----+
                 |
            item-item GCN
                 |
                BPR

Important implementation detail
-------------------------------
The LightGCN normalization is recomputed from each branch's active hard edges,
so zero-routed edges do not contribute to that branch's degree normalization.
"""

import os
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


class BOUNDARY_CORRECTION(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BOUNDARY_CORRECTION, self).__init__(config, dataset)

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
        # Degree-guided boundary-correction hyperparameters.
        # -------------------------------------------------------------
        # Match the original GLORIA experiment: top 10% highest-degree items
        # form the HIGH branch by default.
        self.high_ratio = float(
            _cfg(config, 'high_ratio', 0.10)
        )

        # Only this fraction of all items, centered around the degree cutoff,
        # is allowed to change assignment. Example: 0.02 means about 1% just
        # below + 1% just above the cutoff are learnable.
        self.boundary_fraction = float(
            _cfg(config, 'boundary_fraction', 0.02)
        )

        # Initial fixed prior score. Positive means HIGH, negative means LOW.
        # With correction=0, the hard route exactly matches the degree split.
        self.degree_prior_strength = float(
            _cfg(config, 'degree_prior_strength', 2.0)
        )

        # Maximum magnitude of a boundary item's learnable correction.
        # It must be > degree_prior_strength if items are allowed to cross.
        self.max_assignment_correction = float(
            _cfg(config, 'max_assignment_correction', 4.0)
        )

        # Used only for the straight-through backward surrogate and soft
        # diagnostics; the forward graph remains hard 0/1.
        self.correction_temperature = float(
            _cfg(config, 'correction_temperature', 1.0)
        )

        # Encourage boundary items to stay close to the known-good degree
        # prior unless BPR gives enough evidence to move them.
        self.prior_preserve_weight = float(
            _cfg(config, 'prior_preserve_weight', 1e-2)
        )
        self.correction_l2_weight = float(
            _cfg(config, 'correction_l2_weight', 1e-3)
        )

        # Optional safety term. Usually unnecessary because most assignments
        # are fixed by the degree prior, so default it to zero.
        self.mask_min_branch_usage = float(
            _cfg(config, 'mask_min_branch_usage', 0.05)
        )
        self.mask_balance_weight = float(
            _cfg(config, 'mask_balance_weight', 0.0)
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

        # -------------------------------------------------------------
        # Degree prior: reproduce the original fixed top-high-ratio split.
        # -------------------------------------------------------------
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

        # Original code used:
        #     high_items = np.argsort(item_degree)[-num_high:]
        # Keep exactly that top-k definition here.
        num_high = int(self.num_item * self.high_ratio)
        num_high = max(1, min(self.num_item, num_high))
        self.num_degree_prior_high = num_high

        degree_order = np.argsort(item_degree_np)  # ascending degree
        high_items_np = degree_order[-num_high:]

        prior_high_np = np.zeros(
            self.num_item,
            dtype=np.float32
        )
        prior_high_np[high_items_np] = 1.0

        self.degree_prior_high = torch.tensor(
            prior_high_np,
            dtype=torch.float32,
            device=self.device
        )

        # A signed base score used by the deterministic router:
        #   LOW  -> -prior_strength
        #   HIGH -> +prior_strength
        self.degree_prior_score = (
            2.0 * self.degree_prior_high - 1.0
        ) * self.degree_prior_strength

        # -------------------------------------------------------------
        # Select ONLY items around the top-k cutoff as learnable.
        # -------------------------------------------------------------
        # In ascending order the fixed split boundary is:
        #   [... LOW ... | HIGH ...]
        cutoff_pos = self.num_item - num_high

        boundary_count = int(
            round(self.num_item * self.boundary_fraction)
        )
        boundary_count = max(0, min(self.num_item, boundary_count))

        if boundary_count > 0:
            num_below = boundary_count // 2
            num_above = boundary_count - num_below

            boundary_start = max(
                0,
                cutoff_pos - num_below
            )
            boundary_end = min(
                self.num_item,
                cutoff_pos + num_above
            )

            boundary_items_np = degree_order[
                boundary_start:boundary_end
            ].astype(np.int64)
        else:
            boundary_items_np = np.empty(
                0,
                dtype=np.int64
            )

        self.boundary_item_ids = torch.tensor(
            boundary_items_np,
            dtype=torch.long,
            device=self.device
        )
        self.num_boundary_items = int(
            boundary_items_np.shape[0]
        )

        boundary_mask_np = np.zeros(
            self.num_item,
            dtype=np.bool_
        )
        boundary_mask_np[boundary_items_np] = True
        self.boundary_item_mask = torch.tensor(
            boundary_mask_np,
            dtype=torch.bool,
            device=self.device
        )

        # IMPORTANT: only boundary items own trainable assignment corrections.
        # Non-boundary items have no router parameter at all.
        self.boundary_assignment_delta = nn.Parameter(
            torch.zeros(
                self.num_boundary_items,
                dtype=torch.float32,
                device=self.device
            )
        )

        # Useful cutoff diagnostics.
        if num_high < self.num_item:
            self.max_low_degree = float(
                item_degree_np[degree_order[cutoff_pos - 1]]
            ) if cutoff_pos > 0 else float('nan')
            self.min_high_degree = float(
                item_degree_np[degree_order[cutoff_pos]]
            )
        else:
            self.max_low_degree = float('nan')
            self.min_high_degree = float(
                item_degree_np[degree_order[0]]
            )

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

        print(
            'degree prior: high_ratio={:.4f}, high_items={}, '
            'boundary_items={}, max_low_degree={}, min_high_degree={}'.format(
                self.high_ratio,
                self.num_degree_prior_high,
                self.num_boundary_items,
                self.max_low_degree,
                self.min_high_degree
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
    # Degree-guided boundary routing
    # -----------------------------------------------------------------
    def get_full_assignment_correction(self):
        """
        Build a full [num_item] correction vector.

        Only boundary items have trainable parameters. Non-boundary entries are
        exactly zero and therefore can never change their degree-prior branch.
        """
        full_correction = torch.zeros(
            self.num_item,
            device=self.device,
            dtype=self.degree_prior_score.dtype
        )

        if self.num_boundary_items == 0:
            return full_correction

        # Bounded correction keeps the learned edit local/stable.
        boundary_correction = (
            self.max_assignment_correction
            * torch.tanh(self.boundary_assignment_delta)
        )

        return full_correction.index_copy(
            0,
            self.boundary_item_ids,
            boundary_correction
        )

    def get_item_masks(self):
        """
        Degree prior + learnable boundary correction.

        Forward pass:
            exact hard [LOW, HIGH] one-hot routing with NO random Gumbel noise.

        Backward pass during training:
            straight-through gradients come from sigmoid(score / temperature).

        Returns:
            hard_masks:       [num_item, 2], hard in forward
            soft_masks:       [num_item, 2], differentiable probabilities
            full_correction:  [num_item], nonzero only for boundary items
            routing_score:    [num_item], prior score + correction
        """
        temperature = max(
            self.correction_temperature,
            1e-6
        )

        full_correction = self.get_full_assignment_correction()
        routing_score = (
            self.degree_prior_score
            + full_correction
        )

        # Probability of HIGH. LOW is its complement.
        soft_high = torch.sigmoid(
            routing_score / temperature
        )
        soft_masks = torch.stack(
            [1.0 - soft_high, soft_high],
            dim=1
        )

        # Deterministic hard threshold. At initialization this exactly equals
        # the fixed high/low degree split because correction == 0.
        hard_high = (
            routing_score >= 0.0
        ).to(dtype=soft_high.dtype)

        if self.training:
            # Straight-through estimator WITHOUT sampling/noise:
            # forward  -> hard 0/1
            # backward -> sigmoid gradient
            routed_high = (
                hard_high.detach()
                - soft_high.detach()
                + soft_high
            )
        else:
            routed_high = hard_high

        hard_masks = torch.stack(
            [1.0 - routed_high, routed_high],
            dim=1
        )

        return (
            hard_masks,
            soft_masks,
            full_correction,
            routing_score
        )

    def get_forward_edge_masks(self, item_masks):
        """
        Convert item masks [num_item, 2] to forward-edge masks [E, 2].

        Because routing is item-level, all interactions involving the same item
        receive the same branch assignment.
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
        Build the two hard graph branches from the corrected degree prior.

        Branch 1 = LOW branch.
        Branch 2 = HIGH branch.
        """
        (
            hard_item_masks,
            soft_item_masks,
            full_correction,
            routing_score
        ) = self.get_item_masks()

        hard_edge_masks = self.get_forward_edge_masks(
            hard_item_masks
        )

        low_edge_mask = self.to_bidirectional_mask(
            hard_edge_masks[:, 0]
        )
        high_edge_mask = self.to_bidirectional_mask(
            hard_edge_masks[:, 1]
        )

        low_rep, low_preference = self.branch1_gcn(
            self.edge_index,
            self.id_embedding_branch1.weight,
            edge_mask=low_edge_mask
        )

        high_rep, high_preference = self.branch2_gcn(
            self.edge_index,
            self.id_embedding_branch2.weight,
            edge_mask=high_edge_mask
        )

        # Keep for analysis/debugging.
        self.branch1_preference = low_preference
        self.branch2_preference = high_preference
        self.branch1_rep = low_rep
        self.branch2_rep = high_rep
        self.last_hard_item_masks = hard_item_masks
        self.last_soft_item_masks = soft_item_masks
        self.last_hard_edge_masks = hard_edge_masks
        self.last_assignment_correction = full_correction
        self.last_routing_score = routing_score

        return (
            low_rep,
            high_rep,
            hard_item_masks,
            soft_item_masks,
            hard_edge_masks,
            full_correction,
            routing_score
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
    # Degree-prior / correction regularizers
    # -----------------------------------------------------------------
    def prior_preservation_loss(self, soft_item_masks):
        """
        Keep learnable boundary items close to the known-good degree prior.

        This does not forbid flips; it simply makes a flip require enough BPR
        improvement to overcome a small prior penalty.
        """
        if self.num_boundary_items == 0:
            return soft_item_masks.sum() * 0.0

        high_prob = soft_item_masks[
            self.boundary_item_ids,
            1
        ].clamp(1e-6, 1.0 - 1e-6)

        target_high = self.degree_prior_high[
            self.boundary_item_ids
        ]

        return F.binary_cross_entropy(
            high_prob,
            target_high
        )

    def correction_l2_loss(self):
        """Penalize unnecessarily large boundary edits."""
        if self.num_boundary_items == 0:
            return self.degree_prior_score.sum() * 0.0

        actual_correction = (
            self.max_assignment_correction
            * torch.tanh(self.boundary_assignment_delta)
        )

        return actual_correction.pow(2).mean()

    def mask_collapse_loss(self, item_masks):
        """
        Optional safety penalty if a branch falls below a minimum item mass.

        Most items are fixed, so this should normally be left at weight 0.
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
            return self.degree_prior_score.sum() * 0.0

        return (
            self.id_embedding_branch1.weight.pow(2).mean()
            + self.id_embedding_branch2.weight.pow(2).mean()
            + self.branch1_gcn.preference.pow(2).mean()
            + self.branch2_gcn.preference.pow(2).mean()
        )

    # -----------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------
    def forward(self, interaction, return_aux=False):
        (
            low_rep,
            high_rep,
            hard_item_masks,
            soft_item_masks,
            hard_edge_masks,
            full_correction,
            routing_score
        ) = self.compute_branch_representations()

        self.result_embed = self.fuse_representations(
            low_rep,
            high_rep
        )

        pos_scores, neg_scores = self.pairwise_scores(
            self.result_embed,
            interaction
        )

        if not return_aux:
            return pos_scores, neg_scores

        aux = {
            'low_rep': low_rep,
            'high_rep': high_rep,
            'hard_item_masks': hard_item_masks,
            'soft_item_masks': soft_item_masks,
            'hard_edge_masks': hard_edge_masks,
            'assignment_correction': full_correction,
            'routing_score': routing_score,
        }

        return pos_scores, neg_scores, aux

    # -----------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------
    def calculate_loss(self, interaction):
        """
        Objective:

            L = BPR
                + lambda_prior * degree-prior preservation
                + lambda_corr  * correction magnitude penalty
                + optional branch-collapse safety
                + optional embedding L2

        The forward graph is always hard. Gradients reach ONLY the boundary
        assignment corrections through a deterministic sigmoid straight-through
        surrogate; there is no Gumbel sampling.
        """
        pos_scores, neg_scores, aux = self.forward(
            interaction,
            return_aux=True
        )

        recommendation_loss = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        prior_loss = self.prior_preservation_loss(
            aux['soft_item_masks']
        )
        correction_loss = self.correction_l2_loss()
        collapse_loss = self.mask_collapse_loss(
            aux['soft_item_masks']
        )
        embedding_reg = self.embedding_regularization_loss()

        total_loss = (
            recommendation_loss
            + self.prior_preserve_weight * prior_loss
            + self.correction_l2_weight * correction_loss
            + self.mask_balance_weight * collapse_loss
            + self.embedding_reg_weight * embedding_reg
        )

        return total_loss

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def full_sort_predict(self, interaction):
        low_rep, high_rep, _, _, _, _, _ = (
            self.compute_branch_representations()
        )

        self.result_embed = self.fuse_representations(
            low_rep,
            high_rep
        )

        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[
            interaction[0],
            :
        ]

        return torch.matmul(
            temp_user_tensor,
            item_tensor.t()
        )

    # -----------------------------------------------------------------
    # Diagnostics / analysis helpers
    # -----------------------------------------------------------------
    @torch.no_grad()
    def get_item_routing_statistics(self):
        """
        Analyze how much the model changed the original degree split.

        The most important outputs are:
            num_flipped_items
            boundary_flip_rate
            flipped_item_ids
            prior vs learned high-branch counts
            mean degree of learned LOW/HIGH branches
        """
        temperature = max(
            self.correction_temperature,
            1e-6
        )

        full_correction = self.get_full_assignment_correction()
        routing_score = (
            self.degree_prior_score
            + full_correction
        )

        soft_high = torch.sigmoid(
            routing_score / temperature
        )
        soft_masks = torch.stack(
            [1.0 - soft_high, soft_high],
            dim=1
        )

        learned_high = routing_score >= 0.0
        prior_high = self.degree_prior_high > 0.5
        flipped = learned_high != prior_high

        # By construction, flips should only be possible inside boundary items.
        boundary_flipped = flipped & self.boundary_item_mask

        active = self.active_item_mask
        active_degree = self.item_degree[active]
        active_learned_high = learned_high[active]

        low_degree = active_degree[~active_learned_high]
        high_degree = active_degree[active_learned_high]

        active_soft_masks = soft_masks[active]
        hard_masks = torch.stack(
            [
                (~learned_high).to(dtype=soft_masks.dtype),
                learned_high.to(dtype=soft_masks.dtype)
            ],
            dim=1
        )

        if self.num_boundary_items > 0:
            boundary_actual_correction = full_correction[
                self.boundary_item_ids
            ]
            boundary_flip_rate = boundary_flipped[
                self.boundary_item_ids
            ].float().mean()
        else:
            boundary_actual_correction = torch.empty(
                0,
                device=self.device
            )
            boundary_flip_rate = torch.tensor(
                0.0,
                device=self.device
            )

        return {
            'soft_item_masks': soft_masks.detach(),
            'hard_item_masks': hard_masks.detach(),
            'degree_prior_high': prior_high.detach(),
            'learned_high': learned_high.detach(),
            'boundary_item_ids': self.boundary_item_ids.detach(),
            'boundary_item_degree': self.item_degree[
                self.boundary_item_ids
            ].detach(),
            'assignment_correction': full_correction.detach(),
            'routing_score': routing_score.detach(),
            'flipped_item_mask': flipped.detach(),
            'flipped_item_ids': torch.where(flipped)[0].detach(),
            'num_flipped_items': flipped.sum().detach(),
            'num_boundary_flipped_items': boundary_flipped.sum().detach(),
            'boundary_flip_rate': boundary_flip_rate.detach(),
            'num_prior_low_items': (~prior_high).sum().detach(),
            'num_prior_high_items': prior_high.sum().detach(),
            'num_learned_low_items': (~learned_high).sum().detach(),
            'num_learned_high_items': learned_high.sum().detach(),
            'soft_branch_usage_active': active_soft_masks.mean(dim=0).detach(),
            'mean_degree_low': (
                low_degree.mean().detach()
                if low_degree.numel() > 0
                else torch.tensor(float('nan'), device=self.device)
            ),
            'mean_degree_high': (
                high_degree.mean().detach()
                if high_degree.numel() > 0
                else torch.tensor(float('nan'), device=self.device)
            ),
            'mean_abs_boundary_correction': (
                boundary_actual_correction.abs().mean().detach()
                if boundary_actual_correction.numel() > 0
                else torch.tensor(0.0, device=self.device)
            ),
            'max_abs_boundary_correction': (
                boundary_actual_correction.abs().max().detach()
                if boundary_actual_correction.numel() > 0
                else torch.tensor(0.0, device=self.device)
            ),
            'max_low_degree_prior': torch.tensor(
                self.max_low_degree,
                device=self.device
            ),
            'min_high_degree_prior': torch.tensor(
                self.min_high_degree,
                device=self.device
            ),
        }


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

            # Branch-specific weighted degree. The forward edge_mask is
            # exactly 0/1, so this is the same degree obtained from explicitly
            # constructing the routed branch graph. Because the boundary mask
            # uses a straight-through surrogate during training, gradients can
            # still reach the learnable boundary corrections.
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