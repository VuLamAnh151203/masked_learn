# coding: utf-8
r"""
TOPKPRETRAIN -- Stage 1 for reduced Experiment C.

Purpose
-------
Train ONE normal full-user-item-graph LightGCN encoder with BPR.  The saved
checkpoint is then used by TOPKROUTERC as a frozen feature generator for the
learnable 90/10 router.

The parameter names are intentionally:
    router_item_embedding.weight
    router_gcn.preference
so TOPKROUTERC can load them directly from this checkpoint.

Architecture
------------
    full user-item graph
            |
            v
       LightGCN
            |
        user/item H
            |
           BPR

There is NO degree split and NO router in this stage.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from common.abstract_recommender import GeneralRecommender


class TOPKPRETRAIN(GeneralRecommender):
    """Stage-1 full-graph pretraining model for reduced Experiment C."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.num_user = self.n_users
        self.num_item = self.n_items
        self.dataset = dataset

        print(
            '[TOPKPRETRAIN] number of users: {}, number of items: {}'.format(
                self.num_user, self.num_item
            )
        )

        self.batch_size = config['train_batch_size']
        self.feat_embed_dim = int(config['feat_embed_dim'])
        self.aggr_mode = config['aggr_mode']
        self.num_layer = 1
        self.dim_latent = 64

        # These exact names are shared with TOPKROUTERC.
        self.router_item_embedding = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.router_item_embedding.weight)

        train_interactions = dataset.inter_matrix(
            form='coo'
        ).astype(np.float32)

        forward_edges_np = self.pack_edge_index(train_interactions)
        forward_edges = torch.tensor(
            forward_edges_np,
            dtype=torch.long
        ).t().contiguous()
        reverse_edges = forward_edges[[1, 0], :]

        self.register_buffer(
            'edge_index',
            torch.cat([forward_edges, reverse_edges], dim=1)
        )

        self.router_gcn = GCN(
            datasets=self.dataset,
            batch_size=self.batch_size,
            num_user=self.num_user,
            num_item=self.num_item,
            dim_id=int(config['embedding_size']),
            aggr_mode=self.aggr_mode,
            num_layer=self.num_layer,
            has_feature=False,
            dropout=0.1,
            dim_latent=self.dim_latent,
            device=self.device,
            features=self.router_item_embedding.weight
        )

        self.result_embed = None

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def compute_full_graph_representation(self):
        rep, _ = self.router_gcn(
            self.edge_index,
            self.router_item_embedding.weight,
            edge_mask=None
        )
        return rep

    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        return -F.logsigmoid(pos_scores - neg_scores).mean()

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

    def forward(self, interaction):
        self.result_embed = self.compute_full_graph_representation()
        return self.pairwise_scores(self.result_embed, interaction)

    def calculate_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        return self.bpr_loss(pos_scores, neg_scores)

    def full_sort_predict(self, interaction):
        # Recompute so evaluation never depends on a stale training batch.
        self.result_embed = self.compute_full_graph_representation()

        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_tensor.t())

    @torch.no_grad()
    def get_pretrain_export_statistics(self):
        """Small sanity check before using the saved checkpoint in stage 2."""
        rep = self.compute_full_graph_representation()
        return {
            'num_users': torch.tensor(self.num_user),
            'num_items': torch.tensor(self.num_item),
            'embedding_dim': torch.tensor(self.feat_embed_dim),
            'representation_norm_mean': rep.norm(dim=1).mean().detach(),
            'item_embedding_std': self.router_item_embedding.weight.std(
                unbiased=False
            ).detach(),
            'user_preference_std': self.router_gcn.preference.std(
                unbiased=False
            ).detach(),
        }


class GCN(nn.Module):
    """LightGCN-style encoder with a trainable user preference table."""

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
        super().__init__()

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

        preference_dim = self.dim_latent if self.has_feature else self.dim_feat

        self.preference = nn.Parameter(
            torch.empty(num_user, preference_dim)
        )
        nn.init.xavier_normal_(self.preference, gain=1.0)

        self.conv_embed_1 = Base_gcn(
            preference_dim,
            preference_dim,
            aggr=self.aggr_mode
        )

    def forward(self, edge_index, features, edge_mask=None):
        x = torch.cat([self.preference, features], dim=0)
        x = F.normalize(x, p=2, dim=1)

        h = self.conv_embed_1(x, edge_index, edge_mask=edge_mask)
        h_1 = self.conv_embed_1(h, edge_index, edge_mask=edge_mask)
        h_2 = self.conv_embed_1(h_1, edge_index, edge_mask=edge_mask)

        return x + h + h_1 + h_2, self.preference


class Base_gcn(MessagePassing):
    """LightGCN propagation with optional weighted-edge normalization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True,
        bias=True,
        aggr='add',
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_mask=None, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        if size is None:
            size = (x.size(0), x.size(0))

        edge_index, edge_mask = remove_self_loops(edge_index, edge_mask)

        if edge_mask is None:
            edge_mask = torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype
            )
        else:
            edge_mask = edge_mask.to(device=x.device, dtype=x.dtype)

        if self.aggr == 'add':
            row, col = edge_index
            deg = torch.zeros(size[0], device=x.device, dtype=x.dtype)
            deg = deg.index_add(0, row, edge_mask)

            deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
            deg_inv_sqrt = torch.where(
                deg > 0,
                deg_inv_sqrt,
                torch.zeros_like(deg_inv_sqrt)
            )

            edge_weight = (
                edge_mask
                * deg_inv_sqrt[row]
                * deg_inv_sqrt[col]
            )
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
