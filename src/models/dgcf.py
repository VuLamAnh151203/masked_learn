# coding: utf-8
r"""
ELCREC: ELCRec-inspired prototype intent masking on TWO full-graph branches.

This version is designed for a fairer comparison with the original two-branch
GLORIA architecture while avoiding the unstable learned edge-routing behavior
seen in previous experiments.

Core design
-----------
Both branches see the SAME complete user-item interaction graph:

    full graph G ---------------------> Branch-1 LightGCN -> H1
         |
         +----------------------------> Branch-2 LightGCN -> H2

Each branch has its OWN trainable item embedding table and its OWN trainable
user preference matrix. Therefore the dominant collaborative parameters are
approximately doubled compared with the previous single-shared-GCN ELCREC.

The ELCREC-inspired router does NOT change graph topology. It first forms a
stable routing representation

    H_router = (H1 + H2) / 2

then learns two trainable prototypes c1 and c2:

    s_vk = cosine(H_router[v], c_k)
    q_vk = softmax(s_vk / T)

The assignments gate the branch outputs AFTER graph propagation:

    Z1[v] = gate_scale * q_v1 * H1[v]
    Z2[v] = gate_scale * q_v2 * H2[v]

and the final representation is

    Z[v] = Z1[v] || Z2[v]

followed by the original item-item graph propagation on item representations.

Important
---------
- NO degree-based routing.
- NO learned edge masks.
- NO graph splitting.
- Two independent collaborative branches, both using the full graph.
- Prototype masking is representation-level only.

Loss
----
    L = L_BPR
      + lambda_cluster * L_cluster
      + lambda_sep * L_prototype_separation
      + lambda_balance * L_balance
      + lambda_entropy * L_entropy
      + lambda_branch * L_branch_independence
      + lambda_reg * L_embedding

The defaults match the previous ELCREC configuration:
    prototype_intent_num: 2
    prototype_temperature: 1.0
    prototype_mask_strength: 1.0
    prototype_expert_hidden_dim: 64   # retained for config compatibility
    prototype_expert_residual: true   # retained for config compatibility
    prototype_cluster_weight: 0.01
    prototype_separation_weight: 0.01
    prototype_balance_weight: 0.01
    prototype_min_intent_usage: 0.05
    prototype_expert_independence_weight: 0.001
    prototype_entropy_weight: 0.0
    prototype_cluster_detach_assignment: true
    embedding_reg_weight: 0.0

This is an ELCRec-inspired GLORIA adaptation, not a reproduction of the full
ELCRec method.
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
    """Read an optional config value without requiring config.get()."""
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


class ELCREC(GeneralRecommender):
    """
    ELCRec-inspired prototype gating with two independent full-graph branches.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        print(
            'number of users: {}, number of items: {}'.format(
                num_user,
                num_item
            )
        )

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
        # Prototype hyperparameters.
        # This implementation is intentionally exactly TWO branches.
        # -------------------------------------------------------------
        self.num_intents = int(
            _cfg(config, 'prototype_intent_num', 2)
        )
        if self.num_intents != 2:
            raise ValueError(
                'elcrec_two_branch.py requires prototype_intent_num == 2.'
            )

        self.prototype_temperature = float(
            _cfg(config, 'prototype_temperature', 1.0)
        )

        self.prototype_mask_strength = float(
            _cfg(config, 'prototype_mask_strength', 1.0)
        )
        if not (0.0 <= self.prototype_mask_strength <= 1.0):
            raise ValueError('prototype_mask_strength must be in [0, 1].')

        # Kept so the same YAML can be reused without edits.
        self.expert_hidden_dim = int(
            _cfg(config, 'prototype_expert_hidden_dim', 64)
        )
        self.expert_residual = _as_bool(
            _cfg(config, 'prototype_expert_residual', True)
        )

        # With uniform q=[0.5, 0.5], sqrt(2) makes the total dot-product
        # scale of concatenated identical branches roughly match one D branch.
        self.prototype_gate_scale = float(
            _cfg(config, 'prototype_gate_scale', math.sqrt(2.0))
        )

        self.prototype_cluster_weight = float(
            _cfg(config, 'prototype_cluster_weight', 0.01)
        )
        self.prototype_separation_weight = float(
            _cfg(config, 'prototype_separation_weight', 0.01)
        )
        self.intent_balance_weight = float(
            _cfg(config, 'prototype_balance_weight', 0.01)
        )
        self.intent_min_usage = float(
            _cfg(config, 'prototype_min_intent_usage', 0.05)
        )
        self.assignment_entropy_weight = float(
            _cfg(config, 'prototype_entropy_weight', 0.0)
        )

        # Reuse the old config name for backwards compatibility. Here this
        # regularizes the TWO GCN branch representations instead of MLP experts.
        self.branch_independence_weight = float(
            _cfg(
                config,
                'prototype_branch_independence_weight',
                _cfg(config, 'prototype_expert_independence_weight', 0.001)
            )
        )

        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        self.cluster_detach_assignment = _as_bool(
            _cfg(config, 'prototype_cluster_detach_assignment', True)
        )

        # Stable initialization option: branch 2 starts exactly from branch 1.
        # The prototype gates immediately create different gradients, so the
        # two branches can then specialize during learning.
        self.identical_branch_init = _as_bool(
            _cfg(config, 'prototype_identical_branch_init', True)
        )

        # -------------------------------------------------------------
        # TWO independent item embedding tables.
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
        nn.init.xavier_uniform_(self.id_embedding_branch2.weight)

        if self.identical_branch_init:
            with torch.no_grad():
                self.id_embedding_branch2.weight.copy_(
                    self.id_embedding_branch1.weight
                )

        # Keep these original GLORIA attributes for compatibility with the
        # surrounding project, even though the prototype mechanism itself does
        # not use the raw multimodal MLP outputs.
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
        # ONE complete user-item graph shared structurally by both branches.
        # -------------------------------------------------------------
        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)

        forward_edges_np = self.pack_edge_index(train_interactions)
        self.num_interactions = forward_edges_np.shape[0]

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
        # TWO independent collaborative encoders.
        # Both see self.edge_index (the entire graph), but each has its own
        # trainable user preference matrix and item embedding table.
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

        if self.identical_branch_init:
            with torch.no_grad():
                self.branch2_gcn.preference.copy_(
                    self.branch1_gcn.preference
                )

        # -------------------------------------------------------------
        # TWO learnable prototype centers in the same D-dimensional space.
        # -------------------------------------------------------------
        self.intent_prototypes = nn.Parameter(
            torch.empty(
                2,
                self.feat_embed_dim,
                device=self.device
            )
        )
        nn.init.xavier_uniform_(self.intent_prototypes)

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
    # TWO branch full-graph representations
    # -----------------------------------------------------------------
    def compute_branch_representations(self):
        """
        Both branches propagate on the SAME complete interaction graph.

        No assignment is used inside either GCN. Prototype masking happens only
        after H1/H2 have been computed.
        """
        branch1_rep, branch1_preference = self.branch1_gcn(
            self.edge_index,
            self.id_embedding_branch1.weight,
            edge_mask=None
        )

        branch2_rep, branch2_preference = self.branch2_gcn(
            self.edge_index,
            self.id_embedding_branch2.weight,
            edge_mask=None
        )

        self.branch1_rep = branch1_rep
        self.branch2_rep = branch2_rep
        self.branch1_preference = branch1_preference
        self.branch2_preference = branch2_preference

        return branch1_rep, branch2_rep

    def compute_router_representation(self, branch1_rep, branch2_rep):
        """
        Use the mean of the two full-graph branches to decide prototype
        responsibilities. This avoids privileging either branch as the router.
        """
        return 0.5 * (branch1_rep + branch2_rep)

    # -----------------------------------------------------------------
    # Prototype assignments
    # -----------------------------------------------------------------
    def compute_intent_assignments(self, router_rep):
        """
        q[v,k] = softmax(cos(router_rep[v], prototype[k]) / T)

        The effective assignment can be interpolated with uniform routing:

            q_eff = (1-alpha)/2 + alpha*q

        alpha=1.0 uses the fully learned prototype assignment.
        alpha=0.0 is a useful ablation: both branches receive uniform 0.5 gates.
        """
        node_norm = F.normalize(
            router_rep,
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
        effective_assignments = (
            (1.0 - alpha) * 0.5
            + alpha * assignments
        )

        self.last_intent_similarities = similarities
        self.last_intent_assignments = assignments
        self.last_effective_assignments = effective_assignments

        return similarities, assignments, effective_assignments

    # -----------------------------------------------------------------
    # Representation-level gating of the TWO branches
    # -----------------------------------------------------------------
    def apply_prototype_gates(
        self,
        branch1_rep,
        branch2_rep,
        effective_assignments
    ):
        """
        Important: q does NOT enter the graph convolution.

            Z1 = scale * q[:,0] * H1
            Z2 = scale * q[:,1] * H2
        """
        gate1 = effective_assignments[:, 0:1]
        gate2 = effective_assignments[:, 1:2]

        gated1 = self.prototype_gate_scale * gate1 * branch1_rep
        gated2 = self.prototype_gate_scale * gate2 * branch2_rep

        self.last_gated_branch1 = gated1
        self.last_gated_branch2 = gated2

        return gated1, gated2

    def fuse_representations(self, gated1, gated2):
        """
        Match the original two-branch GLORIA fusion:
          user = branch1 || branch2
          item = branch1 || branch2 -> item-item GCN
        """
        user_rep1 = gated1[:self.num_user]
        user_rep2 = gated2[:self.num_user]
        item_rep1 = gated1[self.num_user:]
        item_rep2 = gated2[self.num_user:]

        user_rep = torch.cat(
            [user_rep1, user_rep2],
            dim=1
        )

        item_rep = torch.cat(
            [item_rep1, item_rep2],
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
        # Same scale as original GLORIA's -mean(log2(sigmoid(diff))).
        return (
            -F.logsigmoid(pos_scores - neg_scores).mean()
            / math.log(2.0)
        )

    # -----------------------------------------------------------------
    # Prototype objectives
    # -----------------------------------------------------------------
    def prototype_separation_loss(self):
        """For K=2, minimize cos(c1, c2)^2."""
        p = F.normalize(
            self.intent_prototypes,
            p=2,
            dim=1,
            eps=1e-8
        )

        cosine = torch.sum(p[0] * p[1])
        return cosine.pow(2)

    def prototype_cluster_loss(self, similarities, assignments):
        """
        Soft clustering objective:

            E_v sum_k responsibility[v,k] * (1 - similarity[v,k])
        """
        responsibility = assignments
        if self.cluster_detach_assignment:
            responsibility = responsibility.detach()

        return (
            responsibility * (1.0 - similarities)
        ).sum(dim=1).mean()

    def intent_balance_loss(self, assignments):
        """
        Prevent either prototype from disappearing. This only enforces a small
        minimum usage; it does NOT force 50/50 utilization.
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
    def branch_independence_loss(branch1_rep, branch2_rep):
        """
        Cross-covariance decorrelation between the two raw GCN branch outputs.

        This is more suitable than node-wise cosine alone because both branches
        intentionally retain collaborative information from the same full graph.
        We center each feature dimension across nodes, normalize columns, and
        penalize squared cross-correlation.
        """
        z1 = branch1_rep - branch1_rep.mean(dim=0, keepdim=True)
        z2 = branch2_rep - branch2_rep.mean(dim=0, keepdim=True)

        z1 = F.normalize(z1, p=2, dim=0, eps=1e-8)
        z2 = F.normalize(z2, p=2, dim=0, eps=1e-8)

        cross = torch.matmul(
            z1.transpose(0, 1),
            z2
        )

        return cross.pow(2).mean()

    def embedding_regularization_loss(self):
        if self.embedding_reg_weight <= 0:
            return self.intent_prototypes.sum() * 0.0

        reg_terms = [
            self.id_embedding_branch1.weight.pow(2).mean(),
            self.id_embedding_branch2.weight.pow(2).mean(),
            self.branch1_gcn.preference.pow(2).mean(),
            self.branch2_gcn.preference.pow(2).mean(),
            self.intent_prototypes.pow(2).mean(),
        ]

        return torch.stack(reg_terms).sum()

    # -----------------------------------------------------------------
    # Main representation pipeline
    # -----------------------------------------------------------------
    def compute_model_representations(self):
        branch1_rep, branch2_rep = self.compute_branch_representations()

        router_rep = self.compute_router_representation(
            branch1_rep,
            branch2_rep
        )

        (
            similarities,
            assignments,
            effective_assignments
        ) = self.compute_intent_assignments(router_rep)

        gated1, gated2 = self.apply_prototype_gates(
            branch1_rep,
            branch2_rep,
            effective_assignments
        )

        result_embed = self.fuse_representations(
            gated1,
            gated2
        )

        self.result_embed = result_embed

        return {
            'branch1_rep': branch1_rep,
            'branch2_rep': branch2_rep,
            'router_rep': router_rep,
            'similarities': similarities,
            'assignments': assignments,
            'effective_assignments': effective_assignments,
            'gated_branch1': gated1,
            'gated_branch2': gated2,
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

        branch_independence = self.branch_independence_loss(
            aux['branch1_rep'],
            aux['branch2_rep']
        )

        embedding_reg = self.embedding_regularization_loss()

        total_loss = (
            recommendation_loss
            + self.prototype_cluster_weight * cluster_loss
            + self.prototype_separation_weight * separation_loss
            + self.intent_balance_weight * balance_loss
            + self.assignment_entropy_weight * entropy_loss
            + self.branch_independence_weight * branch_independence
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
        aux = self.compute_model_representations()

        assignments = aux['assignments']
        effective_assignments = aux['effective_assignments']
        branch1_rep = aux['branch1_rep']
        branch2_rep = aux['branch2_rep']

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

        b1 = F.normalize(
            branch1_rep,
            p=2,
            dim=1,
            eps=1e-8
        )
        b2 = F.normalize(
            branch2_rep,
            p=2,
            dim=1,
            eps=1e-8
        )
        node_branch_cosine = (
            b1 * b2
        ).sum(dim=1)

        hard_all = torch.argmax(assignments, dim=1)
        hard_user = hard_all[:self.num_user]
        hard_item = hard_all[self.num_user:]

        hard_user_counts = torch.stack([
            (hard_user == k).sum()
            for k in range(2)
        ])
        hard_item_counts = torch.stack([
            (hard_item == k).sum()
            for k in range(2)
        ])

        edge_user_assign = user_assign[self.edge_user_ids]
        edge_item_assign = item_assign[self.edge_item_ids]

        soft_edge_same_intent = (
            edge_user_assign * edge_item_assign
        ).sum(dim=1)

        hard_edge_same = (
            torch.argmax(edge_user_assign, dim=1)
            == torch.argmax(edge_item_assign, dim=1)
        ).float()

        return {
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
            'similarities': aux['similarities'].detach(),
            'mean_branch_cosine': node_branch_cosine.mean().detach(),
            'mean_abs_branch_cosine': node_branch_cosine.abs().mean().detach(),
            'branch_independence_loss': self.branch_independence_loss(
                branch1_rep,
                branch2_rep
            ).detach(),
            'soft_edge_same_intent_probability': soft_edge_same_intent.mean().detach(),
            'hard_edge_same_intent_ratio': hard_edge_same.mean().detach(),
        }

    def get_trainable_parameter_statistics(self):
        """Return a simple parameter-count breakdown for capacity comparisons."""
        def count(module):
            return sum(
                p.numel()
                for p in module.parameters()
                if p.requires_grad
            )

        branch1_item = self.id_embedding_branch1.weight.numel()
        branch2_item = self.id_embedding_branch2.weight.numel()
        branch1_user = self.branch1_gcn.preference.numel()
        branch2_user = self.branch2_gcn.preference.numel()
        prototypes = self.intent_prototypes.numel()
        compatibility_mlps = count(self.mlp_item) + count(self.mlp_user)
        total = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            'total_trainable': total,
            'branch1_item_embedding': branch1_item,
            'branch2_item_embedding': branch2_item,
            'branch1_user_preference': branch1_user,
            'branch2_user_preference': branch2_user,
            'prototypes': prototypes,
            'compatibility_mlps': compatibility_mlps,
        }


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
    """LightGCN-style symmetric normalized propagation."""

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
