# coding: utf-8
r"""
ELCREC: ELCRec-inspired LEARNABLE PROTOTYPE INTENT MASKING.

This variant deliberately does NOT learn an edge mask and does NOT modify the
user-item graph topology.

Motivation
----------
Previous learnable routing variants changed / reweighted the interaction graph
before graph propagation. If that hurts recommendation, a safer alternative is:

    full graph -> collaborative representation -> learnable intent mask
               -> intent-specific representation experts -> fusion -> BPR

The graph used by the collaborative encoder is always the complete training
interaction graph.

Core idea
---------
1. Encode the full user-item graph once with a shared LightGCN-style encoder:

       H = GCN_full(G)

   H contains h_u for every user and h_i for every item.

2. Learn K trainable intent prototypes:

       C = [c_1, ..., c_K],  c_k in R^D

3. Assign every node softly to the prototypes using cosine similarity:

       s_vk = cos(h_v, c_k)
       q_vk = softmax(s_vk / T)

4. Each intent has its own residual expert f_k. The expert output is gated by
   the soft prototype assignment:

       e_vk = q_vk * f_k(h_v)

   In this implementation f_k(h) = h + Delta_k(h), which preserves the useful
   full-graph collaborative signal while allowing intent-specific corrections.

5. Concatenate all intent representations:

       z_v = e_v1 || ... || e_vK

   Then preserve GLORIA's item-item propagation on the concatenated item
   representation and train with BPR.

Important distinction
---------------------
NO degree routing:
    degree(i) is never computed for the intent mechanism.

NO edge routing:
    A is never replaced by A * M.

The learnable mask is representation-level:
    h_v -> similarities to prototypes -> q_v -> gated intent experts.

Losses
------
Main recommendation:
    L_bpr

Prototype separation:
    discourage c_j and c_k from pointing in the same direction.

Soft clustering:
    pull node representations toward the prototypes to which they currently
    have high responsibility.

Anti-collapse:
    require each intent to receive at least a small minimum usage, separately
    for users and items. This is a floor, not a 50/50 constraint.

Expert specialization:
    decorrelate the learned residual corrections Delta_k(H), rather than
    decorrelating the full expert outputs. This lets all experts retain the
    common collaborative backbone H while specializing their corrections.

This is an ELCRec-inspired recommender adaptation, not a reproduction of the full
ELCRec training pipeline.
"""

import math
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


class IntentExpert(nn.Module):
    """
    Small residual expert:

        output = h + delta(h)     if residual=True
        output = delta(h)         otherwise

    The final layer is initialized with a very small weight so the model starts
    close to the stable full-graph representation rather than immediately
    replacing it with a randomly transformed embedding.
    """

    def __init__(self, dim, hidden_dim, residual=True):
        super(IntentExpert, self).__init__()
        self.residual = residual

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        # Near-identity initialization when residual=True.
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h):
        delta = self.fc2(
            F.leaky_relu(
                self.fc1(h),
                negative_slope=0.2
            )
        )

        if self.residual:
            out = h + delta
        else:
            out = delta

        return out, delta


class ELCREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

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
        # Prototype / intent hyperparameters.
        # -------------------------------------------------------------
        self.num_intents = int(
            _cfg(config, 'prototype_intent_num', 2)
        )
        if self.num_intents < 2:
            raise ValueError('prototype_intent_num must be >= 2.')

        self.prototype_temperature = float(
            _cfg(config, 'prototype_temperature', 1.0)
        )

        self.prototype_mask_strength = float(
            _cfg(config, 'prototype_mask_strength', 1.0)
        )
        if not (0.0 <= self.prototype_mask_strength <= 1.0):
            raise ValueError('prototype_mask_strength must be in [0, 1].')

        self.expert_hidden_dim = int(
            _cfg(config, 'prototype_expert_hidden_dim', 64)
        )

        self.expert_residual = _as_bool(
            _cfg(config, 'prototype_expert_residual', True)
        )

        # sqrt(K) preserves the approximate score scale when all assignments
        # start near uniform and all residual experts are near identity.
        self.prototype_gate_scale = float(
            _cfg(
                config,
                'prototype_gate_scale',
                math.sqrt(float(self.num_intents))
            )
        )

        # -------------------------------------------------------------
        # Auxiliary objectives.
        # -------------------------------------------------------------
        self.prototype_cluster_weight = float(
            _cfg(config, 'prototype_cluster_weight', 0.01)
        )

        self.prototype_separation_weight = float(
            _cfg(config, 'prototype_separation_weight', 0.01)
        )

        self.intent_min_usage = float(
            _cfg(config, 'prototype_min_intent_usage', 0.05)
        )

        self.intent_balance_weight = float(
            _cfg(config, 'prototype_balance_weight', 0.01)
        )

        # Positive weight minimizes assignment entropy -> sharper assignments.
        # Keep 0 initially; prototype clustering and BPR often provide enough
        # pressure without forcing early over-confidence.
        self.assignment_entropy_weight = float(
            _cfg(config, 'prototype_entropy_weight', 0.0)
        )

        self.expert_independence_weight = float(
            _cfg(config, 'prototype_expert_independence_weight', 0.001)
        )

        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        # Treat current soft assignments as responsibilities in the clustering
        # loss. Detaching them gives a stable soft-k-means-like update while BPR
        # still trains the assignment path end-to-end through the gated experts.
        self.cluster_detach_assignment = _as_bool(
            _cfg(config, 'prototype_cluster_detach_assignment', True)
        )

        # -------------------------------------------------------------
        # One item embedding table + one shared full-graph GCN.
        # -------------------------------------------------------------
        self.id_embedding = nn.Embedding(
            num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.id_embedding.weight)

        # Keep the original architecture MLPs because the surrounding project may
        # expect these attributes to exist.
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
        # Build the COMPLETE user-item graph.
        # No degree and no learned edge masks are used here.
        # -------------------------------------------------------------
        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)

        forward_edges_np = self.pack_edge_index(train_interactions)
        self.num_interactions = forward_edges_np.shape[0]

        # Stored only for diagnostics of prototype alignment on observed edges.
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
        self.edge_index = torch.cat(
            [forward_edges, reverse_edges],
            dim=1
        )

        # -------------------------------------------------------------
        # Shared collaborative encoder: ALWAYS sees the full graph.
        # -------------------------------------------------------------
        self.shared_gcn = GCN(
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
            features=self.id_embedding.weight
        )

        # -------------------------------------------------------------
        # Learnable intent prototypes C in the SAME space as H.
        # -------------------------------------------------------------
        self.intent_prototypes = nn.Parameter(
            torch.empty(
                self.num_intents,
                self.feat_embed_dim,
                device=self.device
            )
        )
        nn.init.xavier_uniform_(self.intent_prototypes)

        # -------------------------------------------------------------
        # One representation expert per intent.
        # -------------------------------------------------------------
        self.intent_experts = nn.ModuleList([
            IntentExpert(
                self.feat_embed_dim,
                self.expert_hidden_dim,
                residual=self.expert_residual
            )
            for _ in range(self.num_intents)
        ])

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
    # Shared full-graph representation
    # -----------------------------------------------------------------
    def compute_shared_representation(self):
        """
        Encode the complete interaction graph. No learnable graph mask is
        involved anywhere in this step.
        """
        shared_rep, shared_preference = self.shared_gcn(
            self.edge_index,
            self.id_embedding.weight,
            edge_mask=None
        )

        self.shared_rep = shared_rep
        self.shared_preference = shared_preference
        return shared_rep

    # -----------------------------------------------------------------
    # Prototype intent assignments
    # -----------------------------------------------------------------
    def compute_intent_assignments(self, shared_rep):
        """
        For each user/item node v:

            similarity[v,k] = cosine(h_v, c_k)
            q[v,k]          = softmax(similarity / temperature)

        prototype_mask_strength optionally interpolates between a uniform mask
        and the learned mask without changing the graph:

            q_eff = (1-alpha)/K + alpha*q

        alpha=1 gives the pure learned mask.
        """
        node_norm = F.normalize(
            shared_rep,
            p=2,
            dim=1,
            eps=1e-8
        )

        prototype_norm = F.normalize(
            self.intent_prototypes,
            p=2,
            dim=1,
            eps=1e-8
        )

        similarities = torch.matmul(
            node_norm,
            prototype_norm.transpose(0, 1)
        )

        temperature = max(
            self.prototype_temperature,
            1e-6
        )

        assignments = torch.softmax(
            similarities / temperature,
            dim=1
        )

        alpha = self.prototype_mask_strength
        uniform = 1.0 / float(self.num_intents)
        effective_assignments = (
            (1.0 - alpha) * uniform
            + alpha * assignments
        )

        self.last_intent_similarities = similarities
        self.last_intent_assignments = assignments
        self.last_effective_assignments = effective_assignments

        return similarities, assignments, effective_assignments

    # -----------------------------------------------------------------
    # Intent-specific representation experts
    # -----------------------------------------------------------------
    def compute_intent_representations(
        self,
        shared_rep,
        effective_assignments
    ):
        """
        Produce one representation branch per intent WITHOUT changing A.

            raw_k   = Expert_k(H)
            gated_k = gate_scale * q_k * raw_k

        Returns:
            gated_reps: list of K tensors [U+I, D]
            raw_reps:   list of K tensors [U+I, D]
            deltas:     list of K tensors [U+I, D]
        """
        gated_reps = []
        raw_reps = []
        deltas = []

        for k, expert in enumerate(self.intent_experts):
            raw_k, delta_k = expert(shared_rep)

            gate_k = effective_assignments[:, k:k + 1]
            gated_k = (
                self.prototype_gate_scale
                * gate_k
                * raw_k
            )

            raw_reps.append(raw_k)
            deltas.append(delta_k)
            gated_reps.append(gated_k)

        self.last_raw_intent_reps = raw_reps
        self.last_intent_deltas = deltas
        self.last_gated_intent_reps = gated_reps

        return gated_reps, raw_reps, deltas

    def fuse_representations(self, gated_reps):
        """
        ELCREC fusion generalized to K intent experts:
          - concatenate all user intent representations
          - concatenate all item intent representations
          - apply the original item-item graph only to item representation
        """
        user_parts = [
            rep[:self.num_user]
            for rep in gated_reps
        ]

        item_parts = [
            rep[self.num_user:]
            for rep in gated_reps
        ]

        user_rep = torch.cat(
            user_parts,
            dim=1
        )

        item_rep = torch.cat(
            item_parts,
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
        # Match the original GLORIA loss scale, which uses log base 2.
        return (
            -F.logsigmoid(pos_scores - neg_scores).mean()
            / math.log(2.0)
        )

    # -----------------------------------------------------------------
    # Prototype / expert objectives
    # -----------------------------------------------------------------
    def prototype_separation_loss(self):
        """
        Penalize pairwise cosine similarity between different prototypes.

        K=2 reduces to cos(c1,c2)^2.
        """
        p = F.normalize(
            self.intent_prototypes,
            p=2,
            dim=1,
            eps=1e-8
        )

        gram = torch.matmul(
            p,
            p.transpose(0, 1)
        )

        eye = torch.eye(
            self.num_intents,
            device=gram.device,
            dtype=gram.dtype
        )

        off_diag = gram * (1.0 - eye)
        denom = max(
            self.num_intents * (self.num_intents - 1),
            1
        )

        return off_diag.pow(2).sum() / float(denom)

    def prototype_cluster_loss(self, similarities, assignments):
        """
        Soft prototype clustering objective:

            L_cluster = E_v sum_k r_vk * (1 - cos(h_v, c_k))

        By default r is detached from the softmax path and acts as the current
        soft responsibility. BPR still trains q end-to-end through expert gates.
        """
        responsibility = assignments
        if self.cluster_detach_assignment:
            responsibility = responsibility.detach()

        # Because responsibilities sum to one, this has the same gradient as
        # -sum r*s but remains non-negative and easier to inspect in logs.
        return (
            responsibility * (1.0 - similarities)
        ).sum(dim=1).mean()

    def intent_balance_loss(self, assignments):
        """
        Prevent an intent from disappearing. The floor is applied separately
        to users and items and does NOT force equal 1/K usage.
        """
        user_assign = assignments[:self.num_user]
        item_assign = assignments[self.num_user:]

        minimum = torch.tensor(
            self.intent_min_usage,
            device=assignments.device,
            dtype=assignments.dtype
        )

        losses = []
        for part in (user_assign, item_assign):
            if part.numel() == 0:
                continue
            usage = part.mean(dim=0)
            losses.append(
                F.relu(minimum - usage).pow(2).sum()
            )

        if not losses:
            return assignments.sum() * 0.0

        return torch.stack(losses).mean()

    @staticmethod
    def assignment_entropy_loss(assignments):
        if assignments.numel() == 0:
            return assignments.sum() * 0.0

        eps = 1e-8
        entropy = -(
            assignments
            * torch.log(assignments + eps)
        ).sum(dim=1)
        return entropy.mean()

    @staticmethod
    def expert_independence_loss(deltas):
        """
        Encourage intent-specific residual corrections to encode different
        directions while allowing all experts to retain the same shared H.

        For each pair (a,b):
            mean_v cos(delta_a[v], delta_b[v])^2
        """
        if len(deltas) <= 1:
            return deltas[0].sum() * 0.0

        pair_losses = []

        for a in range(len(deltas)):
            za = F.normalize(
                deltas[a],
                p=2,
                dim=1,
                eps=1e-8
            )

            for b in range(a + 1, len(deltas)):
                zb = F.normalize(
                    deltas[b],
                    p=2,
                    dim=1,
                    eps=1e-8
                )

                cosine = (
                    za * zb
                ).sum(dim=1)

                pair_losses.append(
                    cosine.pow(2).mean()
                )

        return torch.stack(pair_losses).mean()

    def embedding_regularization_loss(self):
        if self.embedding_reg_weight <= 0:
            return self.intent_prototypes.sum() * 0.0

        reg_terms = [
            self.id_embedding.weight.pow(2).mean(),
            self.shared_gcn.preference.pow(2).mean(),
            self.intent_prototypes.pow(2).mean(),
        ]

        for expert in self.intent_experts:
            for param in expert.parameters():
                reg_terms.append(param.pow(2).mean())

        return torch.stack(reg_terms).sum()

    # -----------------------------------------------------------------
    # Main forward / loss
    # -----------------------------------------------------------------
    def compute_model_representations(self):
        shared_rep = self.compute_shared_representation()

        (
            similarities,
            assignments,
            effective_assignments
        ) = self.compute_intent_assignments(shared_rep)

        (
            gated_reps,
            raw_reps,
            deltas
        ) = self.compute_intent_representations(
            shared_rep,
            effective_assignments
        )

        result_embed = self.fuse_representations(
            gated_reps
        )

        self.result_embed = result_embed

        return {
            'shared_rep': shared_rep,
            'similarities': similarities,
            'assignments': assignments,
            'effective_assignments': effective_assignments,
            'gated_reps': gated_reps,
            'raw_reps': raw_reps,
            'deltas': deltas,
            'result_embed': result_embed,
        }

    def forward(self, interaction, return_aux=False):
        aux = self.compute_model_representations()

        pos_scores, neg_scores = self.pairwise_scores(
            aux['result_embed'],
            interaction
        )

        if return_aux:
            return pos_scores, neg_scores, aux

        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        """
        Total objective:

          L = L_BPR
            + lambda_cluster * L_cluster
            + lambda_sep * L_prototype_separation
            + lambda_balance * L_intent_balance
            + lambda_entropy * L_assignment_entropy
            + lambda_expert * L_expert_independence
            + lambda_reg * L_embedding

        Most importantly, NO loss term changes the interaction adjacency used by
        the shared GCN. Prototype learning happens after full-graph propagation.
        """
        pos_scores, neg_scores, aux = self.forward(
            interaction,
            return_aux=True
        )

        recommendation_loss = self.bpr_loss(
            pos_scores,
            neg_scores
        )

        cluster_loss = self.prototype_cluster_loss(
            aux['similarities'],
            aux['assignments']
        )

        separation_loss = self.prototype_separation_loss()

        balance_loss = self.intent_balance_loss(
            aux['assignments']
        )

        entropy_loss = self.assignment_entropy_loss(
            aux['assignments']
        )

        expert_independence = self.expert_independence_loss(
            aux['deltas']
        )

        embedding_reg = self.embedding_regularization_loss()

        total_loss = (
            recommendation_loss
            + self.prototype_cluster_weight * cluster_loss
            + self.prototype_separation_weight * separation_loss
            + self.intent_balance_weight * balance_loss
            + self.assignment_entropy_weight * entropy_loss
            + self.expert_independence_weight * expert_independence
            + self.embedding_reg_weight * embedding_reg
        )

        return total_loss

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def full_sort_predict(self, interaction):
        aux = self.compute_model_representations()
        result_embed = aux['result_embed']

        user_tensor = result_embed[:self.n_users]
        item_tensor = result_embed[self.n_users:]

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
    def get_prototype_intent_statistics(self):
        """
        Diagnostics for checking whether the learned intents are meaningful.

        Watch especially:
          - prototype_cosine_matrix: off-diagonal should not stay near 1
          - user/item_intent_usage: no intent should disappear
          - mean_assignment_entropy: very high means assignments remain uniform
          - hard_edge_same_intent_ratio: whether observed user-item pairs tend
            to align to the same learned prototype
          - expert_delta_cosine_matrix: residual experts should not be identical
        """
        shared_rep = self.compute_shared_representation()
        similarities, assignments, effective_assignments = (
            self.compute_intent_assignments(shared_rep)
        )

        gated_reps, raw_reps, deltas = self.compute_intent_representations(
            shared_rep,
            effective_assignments
        )

        user_assign = assignments[:self.num_user]
        item_assign = assignments[self.num_user:]

        user_usage = user_assign.mean(dim=0)
        item_usage = item_assign.mean(dim=0)
        all_usage = assignments.mean(dim=0)

        eps = 1e-8
        entropy = -(
            assignments
            * torch.log(assignments + eps)
        ).sum(dim=1)

        hard_all = torch.argmax(assignments, dim=1)
        hard_user = hard_all[:self.num_user]
        hard_item = hard_all[self.num_user:]

        hard_user_counts = torch.stack([
            (hard_user == k).sum()
            for k in range(self.num_intents)
        ])

        hard_item_counts = torch.stack([
            (hard_item == k).sum()
            for k in range(self.num_intents)
        ])

        p = F.normalize(
            self.intent_prototypes,
            p=2,
            dim=1,
            eps=1e-8
        )
        prototype_cosine = torch.matmul(
            p,
            p.transpose(0, 1)
        )

        # Similarity matrix between expert residual corrections.
        expert_delta_cosine = torch.eye(
            self.num_intents,
            device=shared_rep.device,
            dtype=shared_rep.dtype
        )

        for a in range(self.num_intents):
            za = F.normalize(
                deltas[a],
                p=2,
                dim=1,
                eps=1e-8
            )
            for b in range(a + 1, self.num_intents):
                zb = F.normalize(
                    deltas[b],
                    p=2,
                    dim=1,
                    eps=1e-8
                )
                c = (
                    za * zb
                ).sum(dim=1).mean()
                expert_delta_cosine[a, b] = c
                expert_delta_cosine[b, a] = c

        # Alignment of observed u-i pairs in prototype space.
        edge_user_assign = user_assign[self.edge_user_ids]
        edge_item_assign = item_assign[self.edge_item_ids]

        soft_edge_same_intent = (
            edge_user_assign * edge_item_assign
        ).sum(dim=1)

        hard_edge_same = (
            torch.argmax(edge_user_assign, dim=1)
            == torch.argmax(edge_item_assign, dim=1)
        ).float()

        mean_delta_norm = torch.stack([
            delta.norm(p=2, dim=1).mean()
            for delta in deltas
        ])

        stats = {
            'prototypes': self.intent_prototypes.detach(),
            'prototype_cosine_matrix': prototype_cosine.detach(),
            'node_intent_usage': all_usage.detach(),
            'user_intent_usage': user_usage.detach(),
            'item_intent_usage': item_usage.detach(),
            'mean_assignment_entropy': entropy.mean().detach(),
            'mean_max_assignment_probability': assignments.max(dim=1).values.mean().detach(),
            'hard_user_counts': hard_user_counts.detach(),
            'hard_item_counts': hard_item_counts.detach(),
            'assignments': assignments.detach(),
            'effective_assignments': effective_assignments.detach(),
            'similarities': similarities.detach(),
            'soft_edge_same_intent_probability': soft_edge_same_intent.mean().detach(),
            'hard_edge_same_intent_ratio': hard_edge_same.mean().detach(),
            'expert_delta_cosine_matrix': expert_delta_cosine.detach(),
            'mean_expert_delta_norm': mean_delta_norm.detach(),
        }

        return stats


class GCN(nn.Module):
    """LightGCN-style full-graph propagation module."""

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
    LightGCN-style message passing.

    This model always calls it with edge_mask=None, so the complete graph is
    preserved. Optional edge_mask support is retained only for compatibility.
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
