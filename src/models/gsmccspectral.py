# coding: utf-8
r"""
GSMCCSPECTRAL
=============
GS-MCC-inspired high-/low-frequency graph decomposition adapted to the
GLORIA/CaMuRe recommendation pipeline.

IMPORTANT ADAPTATION
--------------------
The GS-MCC paper defines low/high graph filters

    L_low  = I + D^{-1/2} A D^{-1/2}
    L_high = I - D^{-1/2} A D^{-1/2}

and its public implementation additionally applies FFT machinery along an
ordered conversational sequence.  A recommendation user-item graph has no
meaningful ordering of node IDs, so applying a vanilla FFT across node IDs
would be arbitrary.  This implementation therefore keeps the graph-spectral
part that transfers naturally to recommendation:

    S = D^{-1/2} A D^{-1/2}
    P_low  = scale * (I + S)
    P_high = scale * (I - S)

and applies these sparse operators directly to the bipartite graph.

Use:
    gsmcc_filter_scale: 0.5   # stable normalized adaptation (default)

Set:
    gsmcc_filter_scale: 1.0

to use the literal I+S / I-S amplitude from the paper.

ARCHITECTURE
------------

                    full user-item graph A
                             |
                    S = D^-1/2 A D^-1/2
                             |
               +-------------+-------------+
               |                           |
               v                           v
        LOW-FREQUENCY                  HIGH-FREQUENCY
        separate U_l / I_l             separate U_h / I_h
               |                           |
        scale * (I + S)                scale * (I - S)
               |                           |
         Spectral GCN-L                  Spectral GCN-H
               |                           |
              Z_l                         Z_h
               |                           |
               +--------- concat ---------+
                             |
                      item-item GCN
                             |
                            BPR

The recommendation branches have completely separate trainable user and item
embedding tables, matching the capacity pattern of the original two-branch
low-/high-degree GLORIA architecture.

DEFAULTS
--------
    gsmcc_num_ui_layers: 3
    gsmcc_filter_scale: 0.5
    gsmcc_layer_aggregation: sum       # sum | mean | last
    gsmcc_use_item_item: true
    gsmcc_identical_initialization: false

    # Optional GS-MCC-inspired cross-frequency contrastive separation.
    # Keep OFF for the first clean spectral-split experiment.
    gsmcc_contrastive_weight: 0.0
    gsmcc_contrastive_temperature: 0.2
    gsmcc_contrastive_max_nodes: 1024

    embedding_reg_weight: 0.0
    gsmcc_analysis_top_ratio: 0.10
    gsmcc_analysis_num_degree_bins: 5

ANALYSIS INCLUDED
-----------------
After training/evaluation, the model can report:

    model.print_spectral_statistics()
    stats = model.get_spectral_statistics()

    model.print_degree_frequency_statistics()
    stats = model.get_degree_frequency_statistics()

    model.export_item_spectral_analysis("item_spectral_analysis.csv")

Important diagnostics:
  * mean low/high representation energy
  * per-item high-frequency energy ratio
  * user/item low-vs-high branch cosine similarity
  * edge smoothness / Dirichlet-style ratio for each branch
  * Pearson and Spearman correlation between log(item degree) and high-freq ratio
  * Top-K overlap between high-frequency items and high-degree items
  * equal-count degree-bin statistics

Place at:
    src/models/gsmccspectral.py

Use model name:
    GSMCCSPECTRAL
"""

import csv
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from common.abstract_recommender import GeneralRecommender


def _cfg(config, key, default):
    """Read optional config values without requiring config.get()."""
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


def _pearson_torch(x, y, eps=1e-12):
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt(x.pow(2).sum() * y.pow(2).sum()).clamp_min(eps)
    return (x * y).sum() / den


def _rankdata_average_ties(values):
    """Numpy rankdata with average ranks for ties, avoiding scipy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]

    start = 0
    n = len(values)
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        # 0-based average rank is enough for Pearson/Spearman.
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman_numpy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2:
        return float('nan')
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if den <= 1e-12:
        return 0.0
    return float((rx * ry).sum() / den)


class GSMCCSPECTRAL(GeneralRecommender):
    """GS-MCC-inspired low/high graph-frequency recommender."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset = dataset
        self.num_user = self.n_users
        self.num_item = self.n_items

        print(
            '[GSMCCSPECTRAL] users={}, items={}'.format(
                self.num_user, self.num_item
            )
        )

        # ---------------------------------------------------------------
        # Base GLORIA/CaMuRe settings.
        # ---------------------------------------------------------------
        self.batch_size = int(config['train_batch_size'])
        self.feat_embed_dim = int(config['feat_embed_dim'])
        self.n_mm_layers = int(config['n_mm_layers'])
        self.knn_k = int(config['knn_k'])
        self.aggr_mode = config['aggr_mode']
        self.dim_latent = 64
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1

        # ---------------------------------------------------------------
        # Spectral settings.
        # ---------------------------------------------------------------
        self.num_ui_layers = int(
            _cfg(config, 'gsmcc_num_ui_layers', 3)
        )
        if self.num_ui_layers < 1:
            raise ValueError('gsmcc_num_ui_layers must be >= 1.')

        self.filter_scale = float(
            _cfg(config, 'gsmcc_filter_scale', 0.5)
        )
        if self.filter_scale <= 0.0:
            raise ValueError('gsmcc_filter_scale must be > 0.')

        self.layer_aggregation = str(
            _cfg(config, 'gsmcc_layer_aggregation', 'sum')
        ).strip().lower()
        if self.layer_aggregation not in {'sum', 'mean', 'last'}:
            raise ValueError(
                "gsmcc_layer_aggregation must be 'sum', 'mean', or 'last'."
            )

        self.use_item_item = _as_bool(
            _cfg(config, 'gsmcc_use_item_item', True)
        )
        self.identical_initialization = _as_bool(
            _cfg(config, 'gsmcc_identical_initialization', False)
        )

        # Optional GS-MCC-inspired cross-frequency contrastive separation.
        self.contrastive_weight = float(
            _cfg(config, 'gsmcc_contrastive_weight', 0.0)
        )
        self.contrastive_temperature = float(
            _cfg(config, 'gsmcc_contrastive_temperature', 0.2)
        )
        if self.contrastive_temperature <= 0.0:
            raise ValueError('gsmcc_contrastive_temperature must be > 0.')
        self.contrastive_max_nodes = int(
            _cfg(config, 'gsmcc_contrastive_max_nodes', 1024)
        )
        if self.contrastive_max_nodes < 2:
            raise ValueError('gsmcc_contrastive_max_nodes must be >= 2.')

        self.embedding_reg_weight = float(
            _cfg(config, 'embedding_reg_weight', 0.0)
        )

        # Analysis settings.
        self.analysis_top_ratio = float(
            _cfg(config, 'gsmcc_analysis_top_ratio', 0.10)
        )
        if not (0.0 < self.analysis_top_ratio <= 1.0):
            raise ValueError('gsmcc_analysis_top_ratio must be in (0, 1].')
        self.analysis_num_degree_bins = int(
            _cfg(config, 'gsmcc_analysis_num_degree_bins', 5)
        )
        if self.analysis_num_degree_bins < 2:
            raise ValueError('gsmcc_analysis_num_degree_bins must be >= 2.')

        # ---------------------------------------------------------------
        # Full user-item graph. No degree split is used for training.
        # ---------------------------------------------------------------
        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)
        self.num_interactions = int(train_interactions.nnz)

        forward_edges_np = self.pack_edge_index(train_interactions)
        forward_edges = torch.tensor(
            forward_edges_np,
            dtype=torch.long
        ).t().contiguous()
        reverse_edges = forward_edges[[1, 0], :]

        self.register_buffer('forward_edge_index', forward_edges)
        self.register_buffer(
            'edge_index',
            torch.cat([forward_edges, reverse_edges], dim=1)
        )
        self.register_buffer(
            'forward_edge_users',
            torch.tensor(train_interactions.row, dtype=torch.long)
        )
        self.register_buffer(
            'forward_edge_items',
            torch.tensor(train_interactions.col, dtype=torch.long)
        )

        # Degree is ANALYSIS ONLY; it never affects the filters or training.
        item_degree_np = np.bincount(
            train_interactions.col,
            minlength=self.num_item
        ).astype(np.float32)
        user_degree_np = np.bincount(
            train_interactions.row,
            minlength=self.num_user
        ).astype(np.float32)
        self.register_buffer('item_degree', torch.from_numpy(item_degree_np))
        self.register_buffer('user_degree', torch.from_numpy(user_degree_np))

        # ---------------------------------------------------------------
        # Independent LOW-frequency branch: U_l / I_l.
        # ---------------------------------------------------------------
        self.item_embedding_low = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.item_embedding_low.weight)

        self.low_gcn = SpectralGCN(
            num_user=self.num_user,
            num_item=self.num_item,
            features=self.item_embedding_low.weight,
            aggr_mode=self.aggr_mode,
            mode='low',
            num_layers=self.num_ui_layers,
            filter_scale=self.filter_scale,
            layer_aggregation=self.layer_aggregation,
        )

        # ---------------------------------------------------------------
        # Independent HIGH-frequency branch: U_h / I_h.
        # ---------------------------------------------------------------
        self.item_embedding_high = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.item_embedding_high.weight)

        self.high_gcn = SpectralGCN(
            num_user=self.num_user,
            num_item=self.num_item,
            features=self.item_embedding_high.weight,
            aggr_mode=self.aggr_mode,
            mode='high',
            num_layers=self.num_ui_layers,
            filter_scale=self.filter_scale,
            layer_aggregation=self.layer_aggregation,
        )

        # Optional identical initialization is useful for a clean ablation:
        # differences then arise from P_low vs P_high rather than initialization.
        if self.identical_initialization:
            with torch.no_grad():
                self.item_embedding_high.weight.copy_(
                    self.item_embedding_low.weight
                )
                self.high_gcn.preference.copy_(
                    self.low_gcn.preference
                )

        # ---------------------------------------------------------------
        # GLORIA item-item graph after low/high concatenation.
        # ---------------------------------------------------------------
        self.mm_adj = None
        if self.use_item_item:
            t_feat = getattr(self, 't_feat', None)
            if t_feat is None:
                print(
                    '[GSMCCSPECTRAL] t_feat unavailable; disabling '
                    'gsmcc_use_item_item.'
                )
                self.use_item_item = False
            else:
                _, self.mm_adj = self.get_knn_adj_mat(t_feat)

        # Caches for prediction/analysis.
        self.result_embed = None
        self.last_low_rep = None
        self.last_high_rep = None
        self.last_low_states = None
        self.last_high_states = None
        self.last_loss_components = None

        self._print_parameter_summary()

    # ==================================================================
    # Graph utilities / GLORIA item-item graph
    # ==================================================================
    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(
                mm_embeddings,
                p=2,
                dim=-1,
                keepdim=True
            ).clamp_min(1e-12)
        )
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim

        indices0 = torch.arange(
            knn_ind.shape[0],
            device=mm_embeddings.device
        ).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack(
            [indices0.reshape(-1), knn_ind.reshape(-1)],
            dim=0
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
        inv_sqrt = row_sum.pow(-0.5)
        norm_values = inv_sqrt[indices[0]] * inv_sqrt[indices[1]]

        return torch.sparse_coo_tensor(
            indices,
            norm_values,
            adj_size,
            device=indices.device
        ).coalesce()

    def item_item(self, item_rep):
        if not self.use_item_item or self.mm_adj is None:
            return item_rep

        h = item_rep
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return item_rep + h

    # ==================================================================
    # Spectral branches
    # ==================================================================
    def compute_spectral_branches(self, return_states=False):
        low_rep, low_states = self.low_gcn(
            self.edge_index,
            self.item_embedding_low.weight,
            return_states=True
        )
        high_rep, high_states = self.high_gcn(
            self.edge_index,
            self.item_embedding_high.weight,
            return_states=True
        )

        self.last_low_rep = low_rep
        self.last_high_rep = high_rep
        self.last_low_states = low_states
        self.last_high_states = high_states

        if return_states:
            return low_rep, high_rep, low_states, high_states
        return low_rep, high_rep

    def fuse_branches(self, low_rep, high_rep):
        low_user = low_rep[:self.num_user]
        high_user = high_rep[:self.num_user]
        low_item = low_rep[self.num_user:]
        high_item = high_rep[self.num_user:]

        user_rep = torch.cat([low_user, high_user], dim=1)
        item_rep = torch.cat([low_item, high_item], dim=1)

        # Match original GLORIA: concatenate first, then item-item propagation.
        item_rep = self.item_item(item_rep)

        result = torch.cat([user_rep, item_rep], dim=0)
        self.result_embed = result
        return result

    def compute_model_representations(self):
        low_rep, high_rep = self.compute_spectral_branches()
        result = self.fuse_branches(low_rep, high_rep)
        return result, low_rep, high_rep

    # ==================================================================
    # Recommendation objective
    # ==================================================================
    def pairwise_scores(self, representation, interaction):
        users = interaction[0]
        pos_items = interaction[1] + self.num_user
        neg_items = interaction[2] + self.num_user

        user_tensor = representation[users]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]

        pos = (user_tensor * pos_tensor).sum(dim=1)
        neg = (user_tensor * neg_tensor).sum(dim=1)
        return pos, neg

    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def _sample_contrastive_nodes(self, interaction):
        users = interaction[0]
        pos = interaction[1] + self.num_user
        neg = interaction[2] + self.num_user
        nodes = torch.unique(torch.cat([users, pos, neg], dim=0))

        if nodes.numel() > self.contrastive_max_nodes:
            # Random subsampling is only used for the optional O(B^2)
            # contrastive regularizer, never for BPR or graph propagation.
            perm = torch.randperm(nodes.numel(), device=nodes.device)
            nodes = nodes[perm[:self.contrastive_max_nodes]]
        return nodes

    def cross_frequency_contrastive_loss(self, low_rep, high_rep, interaction):
        """
        GS-MCC-inspired cross-frequency separation.

        The paper treats low-frequency anchors against high-frequency negatives
        and symmetrically high-frequency anchors against low-frequency negatives.
        Here the fixed positive logit is 1/tau and all sampled opposite-frequency
        node embeddings act as negatives. This is optional and OFF by default.
        """
        nodes = self._sample_contrastive_nodes(interaction)
        if nodes.numel() < 2:
            return low_rep.sum() * 0.0

        low = F.normalize(low_rep[nodes], p=2, dim=1)
        high = F.normalize(high_rep[nodes], p=2, dim=1)
        tau = self.contrastive_temperature

        logits_lh = torch.matmul(low, high.t()) / tau
        logits_hl = logits_lh.t()

        positive = torch.full(
            (nodes.numel(), 1),
            1.0 / tau,
            device=low.device,
            dtype=low.dtype
        )

        loss_l = (
            torch.logsumexp(
                torch.cat([positive, logits_lh], dim=1),
                dim=1
            ) - positive.squeeze(1)
        ).mean()

        loss_h = (
            torch.logsumexp(
                torch.cat([positive, logits_hl], dim=1),
                dim=1
            ) - positive.squeeze(1)
        ).mean()

        return loss_l + loss_h

    def embedding_regularization_loss(self):
        if self.embedding_reg_weight <= 0.0:
            return self.item_embedding_low.weight.sum() * 0.0

        return (
            self.item_embedding_low.weight.pow(2).mean()
            + self.item_embedding_high.weight.pow(2).mean()
            + self.low_gcn.preference.pow(2).mean()
            + self.high_gcn.preference.pow(2).mean()
        )

    def forward(self, interaction, return_aux=False):
        result, low_rep, high_rep = self.compute_model_representations()
        pos, neg = self.pairwise_scores(result, interaction)

        if not return_aux:
            return pos, neg
        return pos, neg, {
            'result': result,
            'low_rep': low_rep,
            'high_rep': high_rep,
        }

    def calculate_loss(self, interaction):
        pos, neg, aux = self.forward(interaction, return_aux=True)
        rec_loss = self.bpr_loss(pos, neg)

        contrastive_loss = rec_loss * 0.0
        if self.contrastive_weight > 0.0:
            contrastive_loss = self.cross_frequency_contrastive_loss(
                aux['low_rep'],
                aux['high_rep'],
                interaction
            )

        emb_reg = self.embedding_regularization_loss()

        total = (
            rec_loss
            + self.contrastive_weight * contrastive_loss
            + self.embedding_reg_weight * emb_reg
        )

        self.last_loss_components = {
            'total': total.detach(),
            'bpr': rec_loss.detach(),
            'cross_frequency_contrastive': contrastive_loss.detach(),
            'embedding_reg': emb_reg.detach(),
        }
        return total

    def full_sort_predict(self, interaction):
        self.result_embed, _, _ = self.compute_model_representations()
        user_tensor = self.result_embed[:self.num_user]
        item_tensor = self.result_embed[self.num_user:]
        temp_user_tensor = user_tensor[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_tensor.t())

    # ==================================================================
    # Analysis helpers
    # ==================================================================
    @staticmethod
    def _row_cosine(a, b, eps=1e-12):
        a_n = F.normalize(a, p=2, dim=1, eps=eps)
        b_n = F.normalize(b, p=2, dim=1, eps=eps)
        return (a_n * b_n).sum(dim=1)

    def _edge_difference_metrics(self, rep):
        """
        Smoothness / Dirichlet-style diagnostics on observed user-item edges.

        edge_diff_mean = mean ||z_u - z_i||^2
        dirichlet_ratio = sum ||z_u-z_i||^2 / (sum_v ||z_v||^2 + eps)
        """
        u = self.forward_edge_users
        i = self.num_user + self.forward_edge_items
        diff_sq = (rep[u] - rep[i]).pow(2).sum(dim=1)
        numerator = diff_sq.sum()
        denominator = rep.pow(2).sum().clamp_min(1e-12)
        return diff_sq.mean(), numerator / denominator

    @torch.no_grad()
    def get_item_spectral_analysis(self):
        was_training = self.training
        self.eval()
        low_rep, high_rep = self.compute_spectral_branches()

        low_item = low_rep[self.num_user:]
        high_item = high_rep[self.num_user:]

        low_energy = low_item.pow(2).sum(dim=1)
        high_energy = high_item.pow(2).sum(dim=1)
        high_ratio = high_energy / (
            low_energy + high_energy
        ).clamp_min(1e-12)
        cosine = self._row_cosine(low_item, high_item)

        output = {
            'item_id': torch.arange(
                self.num_item,
                device=low_item.device,
                dtype=torch.long
            ),
            'degree': self.item_degree.to(low_item.device).detach(),
            'log_degree': torch.log1p(
                self.item_degree.to(low_item.device)
            ).detach(),
            'low_energy': low_energy.detach(),
            'high_energy': high_energy.detach(),
            'high_frequency_ratio': high_ratio.detach(),
            'low_norm': low_item.norm(dim=1).detach(),
            'high_norm': high_item.norm(dim=1).detach(),
            'low_high_cosine': cosine.detach(),
        }

        if was_training:
            self.train()
        return output

    @torch.no_grad()
    def get_spectral_statistics(self):
        was_training = self.training
        self.eval()
        low_rep, high_rep = self.compute_spectral_branches()

        low_user = low_rep[:self.num_user]
        high_user = high_rep[:self.num_user]
        low_item = low_rep[self.num_user:]
        high_item = high_rep[self.num_user:]

        low_item_energy = low_item.pow(2).sum(dim=1)
        high_item_energy = high_item.pow(2).sum(dim=1)
        item_high_ratio = high_item_energy / (
            low_item_energy + high_item_energy
        ).clamp_min(1e-12)

        low_user_energy = low_user.pow(2).sum(dim=1)
        high_user_energy = high_user.pow(2).sum(dim=1)
        user_high_ratio = high_user_energy / (
            low_user_energy + high_user_energy
        ).clamp_min(1e-12)

        low_edge_diff, low_dirichlet = self._edge_difference_metrics(low_rep)
        high_edge_diff, high_dirichlet = self._edge_difference_metrics(high_rep)

        stats = {
            'filter_scale': torch.tensor(
                self.filter_scale, device=low_rep.device
            ),
            'num_ui_layers': torch.tensor(
                self.num_ui_layers, device=low_rep.device
            ),
            'low_item_energy_mean': low_item_energy.mean().detach(),
            'high_item_energy_mean': high_item_energy.mean().detach(),
            'item_high_frequency_ratio_mean': item_high_ratio.mean().detach(),
            'item_high_frequency_ratio_std': item_high_ratio.std(
                unbiased=False
            ).detach(),
            'low_user_energy_mean': low_user_energy.mean().detach(),
            'high_user_energy_mean': high_user_energy.mean().detach(),
            'user_high_frequency_ratio_mean': user_high_ratio.mean().detach(),
            'user_high_frequency_ratio_std': user_high_ratio.std(
                unbiased=False
            ).detach(),
            'item_low_high_cosine_mean': self._row_cosine(
                low_item, high_item
            ).mean().detach(),
            'user_low_high_cosine_mean': self._row_cosine(
                low_user, high_user
            ).mean().detach(),
            'low_edge_difference_mean': low_edge_diff.detach(),
            'high_edge_difference_mean': high_edge_diff.detach(),
            'low_dirichlet_ratio': low_dirichlet.detach(),
            'high_dirichlet_ratio': high_dirichlet.detach(),
        }

        if was_training:
            self.train()
        return stats

    @torch.no_grad()
    def get_degree_frequency_statistics(self, num_bins=None, top_ratio=None):
        analysis = self.get_item_spectral_analysis()

        degree = analysis['degree'].float()
        log_degree = analysis['log_degree'].float()
        ratio = analysis['high_frequency_ratio'].float()
        low_energy = analysis['low_energy'].float()
        high_energy = analysis['high_energy'].float()
        cosine = analysis['low_high_cosine'].float()

        pearson = _pearson_torch(log_degree, ratio)
        spearman = _spearman_numpy(
            degree.detach().cpu().numpy(),
            ratio.detach().cpu().numpy()
        )

        if top_ratio is None:
            top_ratio = self.analysis_top_ratio
        top_ratio = float(top_ratio)
        k = max(1, int(round(self.num_item * top_ratio)))
        k = min(k, self.num_item)

        degree_top = torch.topk(degree, k=k).indices
        hf_top = torch.topk(ratio, k=k).indices
        lf_top = torch.topk(1.0 - ratio, k=k).indices

        degree_bool = torch.zeros(
            self.num_item,
            dtype=torch.bool,
            device=degree.device
        )
        hf_bool = torch.zeros_like(degree_bool)
        lf_bool = torch.zeros_like(degree_bool)
        degree_bool[degree_top] = True
        hf_bool[hf_top] = True
        lf_bool[lf_top] = True

        hf_inter = (degree_bool & hf_bool).sum()
        lf_inter = (degree_bool & lf_bool).sum()
        hf_union = (degree_bool | hf_bool).sum().clamp_min(1)
        lf_union = (degree_bool | lf_bool).sum().clamp_min(1)

        # Equal-count bins after sorting by degree.  This remains informative
        # even when many items share the exact same integer degree.
        if num_bins is None:
            num_bins = self.analysis_num_degree_bins
        num_bins = max(2, min(int(num_bins), self.num_item))
        order = torch.argsort(degree)
        chunks = torch.tensor_split(order, num_bins)
        degree_bins = []
        for bin_id, ids in enumerate(chunks):
            if ids.numel() == 0:
                continue
            degree_bins.append({
                'bin': bin_id,
                'count': int(ids.numel()),
                'degree_min': float(degree[ids].min().item()),
                'degree_max': float(degree[ids].max().item()),
                'degree_mean': float(degree[ids].mean().item()),
                'high_frequency_ratio_mean': float(ratio[ids].mean().item()),
                'low_energy_mean': float(low_energy[ids].mean().item()),
                'high_energy_mean': float(high_energy[ids].mean().item()),
                'low_high_cosine_mean': float(cosine[ids].mean().item()),
            })

        return {
            'top_k': k,
            'top_ratio': top_ratio,
            'pearson_log_degree_vs_high_ratio': pearson.detach(),
            'spearman_degree_vs_high_ratio': spearman,
            'high_frequency_topk_vs_degree_overlap': (
                hf_inter.float() / float(k)
            ).detach(),
            'high_frequency_topk_vs_degree_jaccard': (
                hf_inter.float() / hf_union.float()
            ).detach(),
            'low_frequency_topk_vs_degree_overlap': (
                lf_inter.float() / float(k)
            ).detach(),
            'low_frequency_topk_vs_degree_jaccard': (
                lf_inter.float() / lf_union.float()
            ).detach(),
            'degree_topk_high_ratio_mean': ratio[degree_top].mean().detach(),
            'all_items_high_ratio_mean': ratio.mean().detach(),
            'degree_bins': degree_bins,
        }

    @torch.no_grad()
    def print_spectral_statistics(self):
        s = self.get_spectral_statistics()
        print('\n========== GSMCCSPECTRAL Spectral Statistics ==========')
        print('filter scale                         : {:.4f}'.format(
            float(s['filter_scale'].item())
        ))
        print('UI spectral layers                   : {}'.format(
            int(s['num_ui_layers'].item())
        ))
        print('mean item LOW energy                 : {:.6f}'.format(
            float(s['low_item_energy_mean'].item())
        ))
        print('mean item HIGH energy                : {:.6f}'.format(
            float(s['high_item_energy_mean'].item())
        ))
        print('mean item HIGH-frequency ratio       : {:.6f}'.format(
            float(s['item_high_frequency_ratio_mean'].item())
        ))
        print('std item HIGH-frequency ratio        : {:.6f}'.format(
            float(s['item_high_frequency_ratio_std'].item())
        ))
        print('mean user HIGH-frequency ratio       : {:.6f}'.format(
            float(s['user_high_frequency_ratio_mean'].item())
        ))
        print('item LOW/HIGH cosine                 : {:.6f}'.format(
            float(s['item_low_high_cosine_mean'].item())
        ))
        print('user LOW/HIGH cosine                 : {:.6f}'.format(
            float(s['user_low_high_cosine_mean'].item())
        ))
        print('LOW edge difference                  : {:.6f}'.format(
            float(s['low_edge_difference_mean'].item())
        ))
        print('HIGH edge difference                 : {:.6f}'.format(
            float(s['high_edge_difference_mean'].item())
        ))
        print('LOW Dirichlet-style ratio            : {:.6f}'.format(
            float(s['low_dirichlet_ratio'].item())
        ))
        print('HIGH Dirichlet-style ratio           : {:.6f}'.format(
            float(s['high_dirichlet_ratio'].item())
        ))
        print('========================================================\n')
        return s

    @torch.no_grad()
    def print_degree_frequency_statistics(self, num_bins=None, top_ratio=None):
        s = self.get_degree_frequency_statistics(
            num_bins=num_bins,
            top_ratio=top_ratio
        )
        print('\n========== GSMCCSPECTRAL Degree vs Frequency ==========')
        print('Top-K items                          : {}'.format(s['top_k']))
        print('Top ratio                            : {:.4f}'.format(s['top_ratio']))
        print('Pearson log-degree vs HIGH ratio     : {:.6f}'.format(
            float(s['pearson_log_degree_vs_high_ratio'].item())
        ))
        print('Spearman degree vs HIGH ratio        : {:.6f}'.format(
            float(s['spearman_degree_vs_high_ratio'])
        ))
        print('HIGH-freq Top-K vs degree overlap    : {:.6f}'.format(
            float(s['high_frequency_topk_vs_degree_overlap'].item())
        ))
        print('HIGH-freq Top-K vs degree Jaccard    : {:.6f}'.format(
            float(s['high_frequency_topk_vs_degree_jaccard'].item())
        ))
        print('LOW-freq Top-K vs degree overlap     : {:.6f}'.format(
            float(s['low_frequency_topk_vs_degree_overlap'].item())
        ))
        print('Degree Top-K mean HIGH ratio         : {:.6f}'.format(
            float(s['degree_topk_high_ratio_mean'].item())
        ))
        print('All items mean HIGH ratio            : {:.6f}'.format(
            float(s['all_items_high_ratio_mean'].item())
        ))
        print('\nEqual-count degree bins (low -> high degree):')
        for row in s['degree_bins']:
            print(
                '  bin {bin}: n={count}, degree=[{degree_min:.0f},{degree_max:.0f}], '
                'mean_degree={degree_mean:.3f}, high_ratio={high_frequency_ratio_mean:.4f}, '
                'low_E={low_energy_mean:.4f}, high_E={high_energy_mean:.4f}, '
                'cos={low_high_cosine_mean:.4f}'.format(**row)
            )
        print('========================================================\n')
        return s

    @torch.no_grad()
    def export_item_spectral_analysis(self, path):
        """Export one row per item for plotting/post-hoc analysis."""
        analysis = self.get_item_spectral_analysis()
        path = os.path.abspath(os.path.expanduser(str(path)))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        columns = [
            'item_id',
            'degree',
            'log_degree',
            'low_energy',
            'high_energy',
            'high_frequency_ratio',
            'low_norm',
            'high_norm',
            'low_high_cosine',
        ]
        arrays = {
            key: analysis[key].detach().cpu().numpy()
            for key in columns
        }

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for idx in range(self.num_item):
                writer.writerow([
                    int(arrays['item_id'][idx]),
                    float(arrays['degree'][idx]),
                    float(arrays['log_degree'][idx]),
                    float(arrays['low_energy'][idx]),
                    float(arrays['high_energy'][idx]),
                    float(arrays['high_frequency_ratio'][idx]),
                    float(arrays['low_norm'][idx]),
                    float(arrays['high_norm'][idx]),
                    float(arrays['low_high_cosine'][idx]),
                ])

        print('[GSMCCSPECTRAL] exported item spectral analysis to: {}'.format(path))
        return path

    @torch.no_grad()
    def get_parameter_statistics(self):
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            'low_branch_trainable_parameters': sum(
                p.numel()
                for module in (self.item_embedding_low, self.low_gcn)
                for p in module.parameters()
                if p.requires_grad
            ),
            'high_branch_trainable_parameters': sum(
                p.numel()
                for module in (self.item_embedding_high, self.high_gcn)
                for p in module.parameters()
                if p.requires_grad
            ),
        }

    def _print_parameter_summary(self):
        s = self.get_parameter_statistics()
        print(
            '[GSMCCSPECTRAL] filter_scale={} | ui_layers={} | '
            'aggregation={} | contrastive_weight={}'.format(
                self.filter_scale,
                self.num_ui_layers,
                self.layer_aggregation,
                self.contrastive_weight
            )
        )
        print(
            '[GSMCCSPECTRAL] parameters total={} | trainable={} | '
            'low_branch={} | high_branch={}'.format(
                s['total_parameters'],
                s['trainable_parameters'],
                s['low_branch_trainable_parameters'],
                s['high_branch_trainable_parameters'],
            )
        )


class SpectralGCN(nn.Module):
    """
    Sparse GS-MCC-inspired graph-frequency branch.

    Let S = D^{-1/2} A D^{-1/2}.

    LOW branch:
        h_{l+1} = scale * (h_l + S h_l)

    HIGH branch:
        h_{l+1} = scale * (h_l - S h_l)

    With scale=1.0 these are the paper's I+S and I-S amplitudes.
    scale=0.5 normalizes their spectral response to [0,1] on eigenvalues
    of S in [-1,1], which is safer for weight-free LightGCN propagation.
    """

    def __init__(
        self,
        num_user,
        num_item,
        features,
        aggr_mode='add',
        mode='low',
        num_layers=3,
        filter_scale=0.5,
        layer_aggregation='sum',
    ):
        super().__init__()

        if mode not in {'low', 'high'}:
            raise ValueError("mode must be 'low' or 'high'.")

        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.aggr_mode = aggr_mode
        self.mode = mode
        self.num_layers = int(num_layers)
        self.filter_scale = float(filter_scale)
        self.layer_aggregation = layer_aggregation

        # Branch-specific trainable USER table.
        self.preference = nn.Parameter(
            torch.empty(num_user, self.dim_feat)
        )
        nn.init.xavier_normal_(self.preference, gain=1.0)

        # Sx. This module contains no trainable transform, matching LightGCN.
        self.propagation = NormalizedGraphPropagation(
            aggr=self.aggr_mode
        )

    def _aggregate_states(self, states):
        if self.layer_aggregation == 'last':
            return states[-1]
        stacked = torch.stack(states, dim=0)
        if self.layer_aggregation == 'mean':
            return stacked.mean(dim=0)
        return stacked.sum(dim=0)

    def forward(self, edge_index, features, return_states=False):
        x = torch.cat([self.preference, features], dim=0)
        x = F.normalize(x, p=2, dim=1)

        states = [x]
        h = x
        for _ in range(self.num_layers):
            neighbor = self.propagation(h, edge_index)
            if self.mode == 'low':
                h = self.filter_scale * (h + neighbor)
            else:
                h = self.filter_scale * (h - neighbor)
            states.append(h)

        output = self._aggregate_states(states)
        if return_states:
            return output, states
        return output, self.preference


class NormalizedGraphPropagation(MessagePassing):
    """Compute Sx where S = D^{-1/2} A D^{-1/2} on the full graph."""

    def __init__(self, aggr='add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        if size is None:
            size = (x.size(0), x.size(0))

        edge_index, _ = remove_self_loops(edge_index)

        if self.aggr == 'add':
            row, col = edge_index
            one = torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype
            )
            deg = torch.zeros(
                size[0],
                device=x.device,
                dtype=x.dtype
            )
            deg = deg.index_add(0, row, one)

            inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
            inv_sqrt = torch.where(
                deg > 0,
                inv_sqrt,
                torch.zeros_like(inv_sqrt)
            )
            edge_weight = inv_sqrt[row] * inv_sqrt[col]
        else:
            edge_weight = torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype
            )

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
