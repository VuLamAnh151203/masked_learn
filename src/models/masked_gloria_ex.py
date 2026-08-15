# coding: utf-8
"""
MASKED_GLORIA with an Adversarial Counterfactual Mask.

Core idea
---------
Factual branch:
    G_full -> GCN_full -> z_full

Counterfactual branch:
    G_cf = G_full * M,  M = sigmoid(mask_logits)
    G_cf -> GCN_mask -> z_cf

The recommender minimizes recommendation loss on the factual/counterfactual
representations, while the mask receives the *reversed* gradient from the
counterfactual recommendation loss. Therefore, with a normal single optimizer:

    recommender: min L_cf
    mask:        max L_cf

The mask is simultaneously regularized to make the perturbation small and to
respect a maximum drop budget rho.

This Gradient Reversal implementation means the existing trainer can keep
calling model.calculate_loss(interaction) with ONE optimizer; no alternating
optimizer/trainer modification is required for the first experiment.
"""

import math
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from torch.nn import MultiheadAttention
# from .transformer import TransformerEncoder


def _cfg(config, key, default):
    """Read an optional config value without requiring it to exist."""
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


class GradientReverse(torch.autograd.Function):
    """
    Identity in the forward pass, sign reversal in the backward pass.

    Forward:
        y = x

    Backward:
        dL/dx = -lambda_adv * dL/dy

    This lets the GCN minimize the counterfactual recommendation loss while
    mask_logits maximize that same loss using a standard gradient-descent
    optimizer.
    """

    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = float(lambda_adv)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x, lambda_adv=1.0):
    return GradientReverse.apply(x, lambda_adv)


class MASKED_GLORIA_EX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MASKED_GLORIA_EX, self).__init__(config, dataset)

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
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.dataset = dataset
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.t_rep = None
        self.t_preference = None
        self.dim_latent = 64
        self.mm_adj = None
        self.config = config

        # ================================================================
        # Counterfactual / adversarial hyperparameters
        # ================================================================
        # rho: maximum fraction of interaction edges that the learned mask
        # is encouraged to suppress.
        self.cf_mask_budget = float(_cfg(config, 'cf_mask_budget', 0.10))

        # Strength of the reversed recommendation gradient seen by mask_logits.
        self.cf_adv_lambda = float(_cfg(config, 'cf_adv_lambda', 1.0))

        # Recommender auxiliary losses.
        # L = L_fused + full_weight * L_full + cf_alpha * L_cf + mask regs
        self.cf_alpha = float(_cfg(config, 'cf_alpha', 0.5))
        self.cf_full_weight = float(_cfg(config, 'cf_full_weight', 0.5))

        # Masker objective is approximately:
        # max L_cf - sparse_weight * mean(1-M)
        self.cf_sparse_weight = float(_cfg(config, 'cf_sparse_weight', 1e-3))

        # Strong penalty only when drop ratio exceeds rho.
        self.cf_budget_weight = float(_cfg(config, 'cf_budget_weight', 10.0))

        # Encourage a near-binary edge selection rather than the trivial
        # solution M_e ~= 1-rho for every edge. Minimize M(1-M), whose minima
        # are at 0 and 1. Start small; this term can saturate sigmoid if large.
        self.cf_binary_weight = float(_cfg(config, 'cf_binary_weight', 1e-3))

        # Optional sigmoid temperature. 1.0 is the safest starting point.
        self.cf_mask_temperature = float(_cfg(config, 'cf_mask_temperature', 1.0))

        # Small epsilon for numerical safety.
        self.eps = 1e-8

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        # Keep your original two independent item embedding tables.
        # For the adversarial-mask experiment we do NOT need to subtract the
        # two latent spaces, so independent tables are fine.
        self.id_embedding_full = nn.Embedding(num_item, self.feat_embed_dim)
        self.id_embedding_masked = nn.Embedding(num_item, self.feat_embed_dim)

        self.mlp_item = nn.Linear(self.t_feat.shape[-1], self.dim_latent, bias=False)
        self.mlp_user = nn.Linear(self.user_feat.shape[-1], self.dim_latent, bias=False)

        indices, text_adj = self.get_knn_adj_mat(self.t_feat)
        self.mm_adj = text_adj

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)

        edge_index_np = self.pack_edge_index(train_interactions)
        self.num_interactions = edge_index_np.shape[0]

        forward_edges = torch.tensor(
            edge_index_np,
            dtype=torch.long,
            device=self.device
        ).t().contiguous()

        # Undirected user-item interaction graph.
        self.edge_index = torch.cat(
            [forward_edges, forward_edges[[1, 0]]],
            dim=1
        )

        # ================================================================
        # Learnable KEEP mask M in [0, 1]
        # ================================================================
        # IMPORTANT: zeros would give sigmoid(0)=0.5, i.e. a 50% attenuation
        # at initialization. Instead initialize around keep_prob = 1-rho.
        rho = min(max(self.cf_mask_budget, 1e-4), 1.0 - 1e-4)
        initial_keep_prob = 1.0 - rho
        initial_logit = math.log(initial_keep_prob / (1.0 - initial_keep_prob))

        self.mask_logits = nn.Parameter(
            torch.full(
                (self.num_interactions,),
                fill_value=initial_logit,
                dtype=torch.float32,
                device=self.device
            )
        )

        # ================================================================
        # Keep the original degree split attributes (currently unused by the
        # forward path) so this file remains compatible with your experiments.
        # ================================================================
        item_ids = edge_index_np[:, 1] - self.num_user
        item_degree = np.bincount(item_ids, minlength=self.num_item)

        high_ratio = 0.10
        num_high = max(1, int(self.num_item * high_ratio))
        high_items = set(np.argsort(item_degree)[-num_high:].tolist())

        low_edges = []
        high_edges = []

        for edge in edge_index_np:
            item_id = edge[1] - self.num_user
            if item_id in high_items:
                high_edges.append(edge)
            else:
                low_edges.append(edge)

        self.edge_index_low = self._make_undirected_edge_tensor(low_edges)
        self.edge_index_high = self._make_undirected_edge_tensor(high_edges)

        # ================================================================
        # Two GCN branches
        # ================================================================
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
            dim_latent=64,
            device=self.device,
            features=self.id_embedding_full.weight
        )

        self.mask_gcn = GCN(
            self.dataset,
            batch_size,
            num_user,
            num_item,
            dim_x,
            self.aggr_mode,
            num_layer=self.num_layer,
            has_feature=False,
            dropout=self.drop_rate,
            dim_latent=64,
            device=self.device,
            features=self.id_embedding_masked.weight
        )

        if config['fusion'] in ['add', 'pool']:
            pass
        elif config['fusion'] == 'Multi-Head Attention':
            self.multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        elif config['fusion'] == 'Transformer':
            self.transformer = TransformerEncoder(64, num_heads=4, layers=2)
        else:
            raise NotImplementedError

        # Cached inference representation.
        self.result_embed = None

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def _make_undirected_edge_tensor(self, edges):
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        edges = np.asarray(edges, dtype=np.int64)
        edge_tensor = torch.tensor(
            edges,
            dtype=torch.long,
            device=self.device
        ).t().contiguous()

        return torch.cat(
            [edge_tensor, edge_tensor[[1, 0]]],
            dim=1
        )

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        )
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim

        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)

        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(
            indices,
            torch.ones_like(indices[0], dtype=torch.float32),
            adj_size,
            device=indices.device
        )
        row_sum = self.eps + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt

        return torch.sparse_coo_tensor(
            indices,
            values,
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

    # ------------------------------------------------------------------
    # Counterfactual mask utilities
    # ------------------------------------------------------------------
    def get_soft_mask(self):
        """Return KEEP probabilities M for the original (non-reversed) edges."""
        temperature = max(self.cf_mask_temperature, 1e-6)
        return torch.sigmoid(self.mask_logits / temperature)

    def _double_mask(self, mask):
        """Same keep weight for u->i and its reverse i->u edge."""
        return torch.cat([mask, mask], dim=0)

    def mask_regularization(self, raw_mask):
        """
        Small-perturbation regularization and hard-budget relaxation.

        M_e = keep probability
        1-M_e = dropped/suppressed amount

        sparse_loss:
            minimize mean(1-M), matching max L_cf - lambda ||1-M||_1

        budget_loss:
            penalize only the part above rho:
            relu(mean(1-M)-rho)^2
        """
        drop_ratio = (1.0 - raw_mask).mean()
        sparse_loss = drop_ratio

        budget_violation = F.relu(drop_ratio - self.cf_mask_budget)
        budget_loss = budget_violation.pow(2)

        # Soft-binary regularizer: 0 at M=0 or M=1, maximum at M=0.5.
        binary_loss = (raw_mask * (1.0 - raw_mask)).mean()

        return sparse_loss, budget_loss, binary_loss, drop_ratio

    @torch.no_grad()
    def get_mask_statistics(self):
        mask = self.get_soft_mask()
        return {
            'mean_keep': mask.mean().item(),
            'mean_drop': (1.0 - mask).mean().item(),
            'min_keep': mask.min().item(),
            'max_keep': mask.max().item(),
            'fraction_keep_lt_0.5': (mask < 0.5).float().mean().item(),
            'budget': self.cf_mask_budget,
        }

    # ------------------------------------------------------------------
    # Representation / scoring helpers
    # ------------------------------------------------------------------
    def _split_and_item_propagate(self, rep):
        """
        Preserve your original item-item propagation, but apply it per branch.
        Since item_item() is linear, this is equivalent to applying it after
        concatenating the two branches.
        """
        user_rep = rep[:self.num_user]
        item_rep = rep[self.num_user:]
        item_rep = self.item_item(item_rep)
        return user_rep, item_rep

    def _run_full_branch(self):
        self.full_rep, self.full_preference = self.full_gcn(
            self.edge_index,
            self.id_embedding_full.weight
        )
        return self._split_and_item_propagate(self.full_rep)

    def _run_mask_branch(self, one_direction_mask):
        edge_mask = self._double_mask(one_direction_mask)

        mask_rep, mask_preference = self.mask_gcn(
            self.edge_index,
            self.id_embedding_masked.weight,
            edge_mask=edge_mask
        )

        return mask_rep, mask_preference, self._split_and_item_propagate(mask_rep)

    @staticmethod
    def _score_triplet(user_rep, item_rep, user_nodes, pos_item_nodes, neg_item_nodes):
        user_tensor = user_rep[user_nodes]
        pos_item_tensor = item_rep[pos_item_nodes]
        neg_item_tensor = item_rep[neg_item_nodes]

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        return pos_scores, neg_scores

    @staticmethod
    def _bpr_loss(pos_scores, neg_scores):
        # Same objective as -log2(sigmoid(pos-neg)), but numerically stable.
        return F.softplus(-(pos_scores - neg_scores)).mean() / math.log(2.0)

    def _cache_fused_representation(self, full_user, full_item, cf_user, cf_item):
        user_rep = torch.cat([full_user, cf_user], dim=1)
        item_rep = torch.cat([full_item, cf_item], dim=1)
        self.result_embed = torch.cat([user_rep, item_rep], dim=0)
        return user_rep, item_rep

    # ------------------------------------------------------------------
    # Standard forward: factual + learned counterfactual view
    # ------------------------------------------------------------------
    def forward(self, interaction):
        # IMPORTANT: do not modify interaction tensors in-place.
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1]
        neg_item_nodes = interaction[2]

        full_user, full_item = self._run_full_branch()

        raw_mask = self.get_soft_mask()
        self.mask_rep, self.mask_preference, (cf_user, cf_item) = self._run_mask_branch(raw_mask)

        fused_user, fused_item = self._cache_fused_representation(
            full_user,
            full_item,
            cf_user,
            cf_item
        )

        return self._score_triplet(
            fused_user,
            fused_item,
            user_nodes,
            pos_item_nodes,
            neg_item_nodes
        )

    # ------------------------------------------------------------------
    # Min-max loss with ONE ordinary optimizer
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        """
        Joint objective:

        Recommender parameters receive ordinary minimizing gradients from:
            L_fused + w_full * L_full + alpha_cf * L_cf

        mask_logits receive:
            -alpha_cf * lambda_adv * grad(L_cf)
            + sparse/budget regularization gradients

        because L_cf uses grad_reverse(mask).

        Thus, at the gradient level:
            recommender: min L_cf
            mask:        max L_cf - regularization

        Fused training uses mask.detach() so the mask is adversarially driven
        specifically by the counterfactual branch rather than by the fusion
        objective. This closely follows the requested min-max formulation.
        """
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1]
        neg_item_nodes = interaction[2]

        # --------------------------------------------------------------
        # 1) Factual branch
        # --------------------------------------------------------------
        full_user, full_item = self._run_full_branch()

        full_pos, full_neg = self._score_triplet(
            full_user,
            full_item,
            user_nodes,
            pos_item_nodes,
            neg_item_nodes
        )
        full_loss = self._bpr_loss(full_pos, full_neg)

        # --------------------------------------------------------------
        # 2) Raw learned mask
        # --------------------------------------------------------------
        raw_mask = self.get_soft_mask()

        # --------------------------------------------------------------
        # 3) Counterfactual view used by the fused recommender
        # --------------------------------------------------------------
        # Detach ONLY the mask values. The masked GCN / embedding parameters
        # still receive gradients from L_fused, while mask_logits do not.
        fused_mask = raw_mask.detach()

        self.mask_rep, self.mask_preference, (cf_user_fused, cf_item_fused) = \
            self._run_mask_branch(fused_mask)

        fused_user, fused_item = self._cache_fused_representation(
            full_user,
            full_item,
            cf_user_fused,
            cf_item_fused
        )

        fused_pos, fused_neg = self._score_triplet(
            fused_user,
            fused_item,
            user_nodes,
            pos_item_nodes,
            neg_item_nodes
        )
        fused_loss = self._bpr_loss(fused_pos, fused_neg)

        # --------------------------------------------------------------
        # 4) Adversarial counterfactual branch
        # --------------------------------------------------------------
        # Same forward mask values, but the mask gradient is multiplied by
        # -cf_adv_lambda. Other GCN parameters still minimize L_cf normally.
        adversarial_mask = grad_reverse(raw_mask, self.cf_adv_lambda)

        _, _, (cf_user_adv, cf_item_adv) = self._run_mask_branch(adversarial_mask)

        cf_pos, cf_neg = self._score_triplet(
            cf_user_adv,
            cf_item_adv,
            user_nodes,
            pos_item_nodes,
            neg_item_nodes
        )
        cf_loss = self._bpr_loss(cf_pos, cf_neg)

        # --------------------------------------------------------------
        # 5) Minimal-perturbation + budget constraints
        # --------------------------------------------------------------
        sparse_loss, budget_loss, binary_loss, drop_ratio = self.mask_regularization(raw_mask)

        # --------------------------------------------------------------
        # 6) Total loss
        # --------------------------------------------------------------
        loss = (
            fused_loss
            + self.cf_full_weight * full_loss
            + self.cf_alpha * cf_loss
            + self.cf_sparse_weight * sparse_loss
            + self.cf_budget_weight * budget_loss
            + self.cf_binary_weight * binary_loss
        )

        # Cache detached diagnostics for logging if desired.
        self.latest_loss_dict = {
            'total': loss.detach(),
            'fused_bpr': fused_loss.detach(),
            'full_bpr': full_loss.detach(),
            'cf_bpr': cf_loss.detach(),
            'mask_sparse': sparse_loss.detach(),
            'mask_budget': budget_loss.detach(),
            'mask_binary': binary_loss.detach(),
            'drop_ratio': drop_ratio.detach(),
            'keep_ratio': (1.0 - drop_ratio).detach(),
        }

        return loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def full_sort_predict(self, interaction):
        # Recompute current embeddings so evaluation does not depend on the
        # final minibatch cached by calculate_loss(). No gradient reversal is
        # needed at inference; it would have identical forward values anyway.
        full_user, full_item = self._run_full_branch()

        raw_mask = self.get_soft_mask()
        self.mask_rep, self.mask_preference, (cf_user, cf_item) = self._run_mask_branch(raw_mask)

        fused_user, fused_item = self._cache_fused_representation(
            full_user,
            full_item,
            cf_user,
            cf_item
        )

        temp_user_tensor = fused_user[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, fused_item.t())
        return score_matrix


class GCN(torch.nn.Module):
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
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.tensor(
                        np.random.randn(num_user, self.dim_latent),
                        dtype=torch.float32,
                        requires_grad=True
                    ),
                    gain=1
                )
            )
            self.conv_embed_1 = Base_gcn(
                self.dim_latent,
                self.dim_latent,
                aggr=self.aggr_mode
            )
        else:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.tensor(
                        np.random.randn(num_user, self.dim_feat),
                        dtype=torch.float32,
                        requires_grad=True
                    ),
                    gain=1
                )
            )
            self.conv_embed_1 = Base_gcn(
                self.dim_latent,
                self.dim_latent,
                aggr=self.aggr_mode
            )

    def forward(self, edge_index, features, edge_mask=None):
        temp_features = features
        temp_profile = self.preference

        x = torch.cat((temp_profile, temp_features), dim=0)
        x = F.normalize(x, dim=1)

        h = self.conv_embed_1(x, edge_index, edge_mask)
        h_1 = self.conv_embed_1(h, edge_index, edge_mask)
        h_2 = self.conv_embed_1(h_1, edge_index, edge_mask)

        x_hat = h + x + h_1 + h_2
        return x_hat, self.preference


class Base_gcn(MessagePassing):
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
        else:
            edge_mask = edge_mask.to(device=x.device, dtype=x.dtype)

        # Keep edge weights aligned if self-loops are removed.
        edge_index, edge_mask = remove_self_loops(edge_index, edge_mask)

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

            # ----------------------------------------------------------
            # Weighted degree normalization
            # ----------------------------------------------------------
            # Original code used unweighted degree(row), meaning a masked
            # edge still contributed fully to degree normalization. For a
            # counterfactual weighted graph, degree should depend on M_e.
            weighted_deg = torch.zeros(
                size[0],
                device=x_j.device,
                dtype=x_j.dtype
            )
            weighted_deg.index_add_(0, row, edge_mask)

            deg_inv_sqrt = weighted_deg.clamp_min(1e-12).pow(-0.5)

            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            return (
                norm.view(-1, 1)
                * edge_mask.view(-1, 1)
                * x_j
            )

        return edge_mask.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels
        )
