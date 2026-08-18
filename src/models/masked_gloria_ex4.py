import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from torch.nn import MultiheadAttention
# from .transformer import TransformerEncoder

class MASKED_GLORIA_EX4(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MASKED_GLORIA_EX4, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        print('number of users: {}, number of items: {}'.format(num_user, num_item))
        
        batch_size = config['train_batch_size']         # not used
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

        # ============================================================
        # Continuous counterfactual regularization
        # ============================================================
        # Keep the learned user mask fully continuous.
        # No STE, no hard routing, no balance regularizer.
        #
        # Factual masked graph:
        #     A_f(u, i) = m_u * A(u, i)
        #
        # Counterfactual graph:
        #     A_cf(u, i) = 1 * A(u, i)
        #
        # In words: "what if this user were NOT attenuated?"
        #
        # lambda_cf controls how strongly the model is encouraged to make
        # learned attenuation useful relative to the no-attenuation baseline.
        try:
            _lambda_cf = config['lambda_cf']
        except Exception:
            _lambda_cf = None
        self.lambda_cf = 0.1 if _lambda_cf is None else float(_lambda_cf)

        # If True, the no-attenuation counterfactual acts as a stop-gradient
        # baseline. This avoids directly training the shared encoder to make
        # the counterfactual branch artificially worse.
        try:
            _detach_cf = config['detach_cf_baseline']
        except Exception:
            _detach_cf = None
        self.detach_cf_baseline = True if _detach_cf is None else bool(_detach_cf)

        self.last_loss_dict = {}
        self.drop_rate = 0.1
        self.t_rep = None
        self.t_preference = None
        self.dim_latent = 64
        self.mm_adj = None
        self.config = config
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        # self.id_embedding_low = nn.Embedding(num_item, self.feat_embed_dim)
        # self.id_embedding_high = nn.Embedding(num_item, self.feat_embed_dim)
        self.id_embedding_full = nn.Embedding(num_item, self.feat_embed_dim)
        self.id_embedding_masked = nn.Embedding(num_item, self.feat_embed_dim)

        self.mlp_item = nn.Linear(self.t_feat.shape[-1], self.dim_latent, bias=False)
        self.mlp_user = nn.Linear(self.user_feat.shape[-1], self.dim_latent, bias=False)

        indices, text_adj = self.get_knn_adj_mat(self.t_feat)
        self.mm_adj = text_adj

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)

        # ============================================================
        # Original user-item edges (one direction only)
        # ============================================================
        edge_index = self.pack_edge_index(train_interactions)
        self.num_interactions = edge_index.shape[0]

        # edge_user_ids[e] tells us which user owns original edge e.
        # Shape: [num_interactions]
        #
        # User-level scalar mask: 
        #     M_(u,i) = sigmoid(user_mask_logits[u])
        self.register_buffer(
            'edge_user_ids',
            torch.tensor(
                edge_index[:, 0],
                dtype=torch.long,
                device=self.device
            )
        )

        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long,
            device=self.device
        ).t().contiguous()

        # Keep the same undirected interaction graph as the original code.
        self.edge_index = torch.cat(
            [edge_index, edge_index[[1, 0]]],
            dim=1
        )

        # ============================================================
        # USER-LEVEL LEARNABLE SCALAR MASK
        # ============================================================
        # Original:
        #     one learnable logit per interaction edge -> O(|E|)
        #
        # Here:
        #     one learnable logit per user -> O(|U|)
        #
        # All interaction edges of user u receive the same scalar:
        #     m_u = sigmoid(user_mask_logits[u])
        #
        # Initialize at zero so sigmoid(0)=0.5, matching the original
        # edge-level mask initialization for a clean ablation.
        self.user_mask_logits = nn.Parameter(
            torch.zeros(
                self.num_user,
                device=self.device
            )
        )

        edge_index = self.pack_edge_index(train_interactions)

        item_ids = edge_index[:, 1] - self.num_user
        item_degree = np.bincount(item_ids, minlength=self.num_item)

        high_ratio = 0.10
        num_high = int(self.num_item * high_ratio)

        high_items = np.argsort(item_degree)[-num_high:]
        high_items = set(high_items.tolist())

        low_edges = []
        high_edges = []

        for edge in edge_index:
            item_id = edge[1] - self.num_user

            if item_id in high_items:
                high_edges.append(edge)
            else:
                low_edges.append(edge)

        low_edges = np.array(low_edges, dtype=np.int64)
        high_edges = np.array(high_edges, dtype=np.int64)

        self.edge_index_low = torch.tensor(low_edges, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index_high = torch.tensor(high_edges, dtype=torch.long).t().contiguous().to(self.device)

        self.edge_index_low = torch.cat(
            (self.edge_index_low, self.edge_index_low[[1, 0]]),
            dim=1
        )

        self.edge_index_high = torch.cat(
            (self.edge_index_high, self.edge_index_high[[1, 0]]),
            dim=1
        )
        # self.edge = concat 2 edge_index to make the graph undirected
        # self.edge_index = torch.cat((self.edge_index_low, self.edge_index_high), dim=1)

        # self.idl_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
        #                 num_layer=self.num_layer, has_feature=False, dropout=self.drop_rate, dim_latent=64,
        #                 device=self.device, features=self.id_embedding.weight)
        # self.idh_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
        #                 num_layer=self.num_layer, has_feature=False, dropout=self.drop_rate, dim_latent=64,
        #                 device=self.device, features=self.id_embedding.weight)

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
        fusion = config['fusion'] or 'add'
        if fusion in ['add', 'pool']:
            pass
        elif fusion == 'Multi-Head Attention':
            self.multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        elif fusion == 'Transformer':
            self.transformer = TransformerEncoder(64, num_heads= 4, layers=2)
        else:
            raise NotImplementedError
        


    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def item_item(self, rep):
        h = rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return rep + h

    def get_user_mask(self):
        """Return one learned scalar mask value for each user."""
        return torch.sigmoid(self.user_mask_logits)

    def get_original_edge_mask(self):
        """
        Expand user-level scalar masks to the original user-item edges.

        For original edge e=(u,i):
            edge_mask[e] = sigmoid(user_mask_logits[u])
        """
        user_mask = self.get_user_mask()
        return user_mask[self.edge_user_ids]

    @torch.no_grad()
    def get_user_mask_statistics(self):
        """Diagnostics for the learned user-level mask."""
        user_mask = self.get_user_mask().detach()
        return {
            'mean_keep': user_mask.mean().item(),
            'mean_attenuation': (1.0 - user_mask).mean().item(),
            'min_keep': user_mask.min().item(),
            'max_keep': user_mask.max().item(),
            'std_keep': user_mask.std(unbiased=False).item(),
        }

    @torch.no_grad()
    def get_counterfactual_training_statistics(self):
        """Return the most recent batch-level CF diagnostics."""
        return dict(self.last_loss_dict)

    def forward(self, interaction):
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        # ============================================================
        # 1) Full branch: unchanged from the original model
        # ============================================================
        self.full_rep, self.full_preference = self.full_gcn(
            self.edge_index,
            self.id_embedding_full.weight
        )

        # ============================================================
        # 2) FACTUAL learnable branch: continuous user mask m_u
        # ============================================================
        # user_mask: [num_users]
        # original_mask: [num_original_interactions]
        #
        # All edges belonging to the same user receive the same m_u.
        user_mask = self.get_user_mask()
        original_mask = user_mask[self.edge_user_ids]

        # edge_index stores original edges followed by their reverse edges.
        # Use the same user-level weight in both directions.
        factual_edge_mask = torch.cat(
            [original_mask, original_mask],
            dim=0
        )

        self.mask_rep, self.mask_preference = self.mask_gcn(
            self.edge_index,
            self.id_embedding_masked.weight,
            edge_mask=factual_edge_mask
        )

        # ============================================================
        # 3) COUNTERFACTUAL learnable branch: no attenuation
        # ============================================================
        # Counterfactual intervention:
        #     do(m_u = 1) for every user.
        #
        # IMPORTANT:
        #   - SAME mask_gcn parameters as the factual branch.
        #   - SAME id_embedding_masked parameters.
        #   - Only the graph edge weights change.
        #
        # This asks:
        #   "What would the ranking be if the learned branch did not
        #    attenuate user-edge influence at all?"
        cf_edge_mask = torch.ones_like(factual_edge_mask)

        self.cf_mask_rep, _ = self.mask_gcn(
            self.edge_index,
            self.id_embedding_masked.weight,
            edge_mask=cf_edge_mask
        )

        # ============================================================
        # 4) Build FACTUAL representation
        # ============================================================
        item_rep_full = self.full_rep[self.num_user:]
        user_rep_full = self.full_rep[:self.num_user]

        item_rep_mask = self.mask_rep[self.num_user:]
        user_rep_mask = self.mask_rep[:self.num_user]

        factual_item_rep = torch.cat(
            (item_rep_full, item_rep_mask),
            dim=1
        )
        factual_item_rep = self.item_item(factual_item_rep)

        factual_user_rep = torch.cat(
            (user_rep_full, user_rep_mask),
            dim=1
        )

        # Cache factual embeddings for normal full-sort evaluation.
        self.result_embed = torch.cat(
            (factual_user_rep, factual_item_rep),
            dim=0
        )

        factual_user_tensor = self.result_embed[user_nodes]
        factual_pos_item_tensor = self.result_embed[pos_item_nodes]
        factual_neg_item_tensor = self.result_embed[neg_item_nodes]

        pos_scores = torch.sum(
            factual_user_tensor * factual_pos_item_tensor,
            dim=1
        )
        neg_scores = torch.sum(
            factual_user_tensor * factual_neg_item_tensor,
            dim=1
        )

        # ============================================================
        # 5) Build COUNTERFACTUAL representation
        # ============================================================
        item_rep_cf = self.cf_mask_rep[self.num_user:]
        user_rep_cf = self.cf_mask_rep[:self.num_user]

        cf_item_rep = torch.cat(
            (item_rep_full, item_rep_cf),
            dim=1
        )
        cf_item_rep = self.item_item(cf_item_rep)

        cf_user_rep = torch.cat(
            (user_rep_full, user_rep_cf),
            dim=1
        )

        cf_result_embed = torch.cat(
            (cf_user_rep, cf_item_rep),
            dim=0
        )

        cf_user_tensor = cf_result_embed[user_nodes]
        cf_pos_item_tensor = cf_result_embed[pos_item_nodes]
        cf_neg_item_tensor = cf_result_embed[neg_item_nodes]

        cf_pos_scores = torch.sum(
            cf_user_tensor * cf_pos_item_tensor,
            dim=1
        )
        cf_neg_scores = torch.sum(
            cf_user_tensor * cf_neg_item_tensor,
            dim=1
        )

        # Mask values for users appearing in this training batch.
        # These stay continuous in [0, 1].
        batch_user_mask = user_mask[user_nodes]

        return (
            pos_scores,
            neg_scores,
            cf_pos_scores,
            cf_neg_scores,
            batch_user_mask
        )

    def calculate_loss(self, interaction):
        (
            pos_scores,
            neg_scores,
            cf_pos_scores,
            cf_neg_scores,
            batch_user_mask
        ) = self.forward(interaction)

        # ============================================================
        # Main recommendation objective: unchanged factual BPR ranking
        # ============================================================
        factual_margin = pos_scores - neg_scores

        rec_loss = -torch.mean(
            torch.log2(
                torch.sigmoid(factual_margin) + 1e-8
            )
        )

        # ============================================================
        # Continuous counterfactual objective
        # ============================================================
        # Factual:
        #     learned m_u
        # Counterfactual:
        #     do(m_u = 1), i.e. no attenuation
        #
        # We want learned attenuation to be useful whenever the model
        # actually chooses a non-trivial attenuation.
        cf_margin = cf_pos_scores - cf_neg_scores

        # Optional stop-gradient baseline. The counterfactual still gets
        # recomputed every step using the current shared encoder, but the
        # CF loss does not directly optimize it to become worse.
        if self.detach_cf_baseline:
            cf_reference = cf_margin.detach()
        else:
            cf_reference = cf_margin

        # How much intervention did the model choose for this user?
        # detach() is intentional: the model cannot reduce this CF loss
        # merely by pushing m_u toward 1.
        attenuation = (1.0 - batch_user_mask).detach()

        # softplus(cf - factual):
        #   small when factual ranking margin >= counterfactual margin,
        #   larger when removing attenuation gives a better ranking.
        cf_per_sample = F.softplus(
            cf_reference - factual_margin
        )

        cf_loss = torch.mean(
            attenuation * cf_per_sample
        )

        loss_value = (
            rec_loss
            + self.lambda_cf * cf_loss
        )

        # Lightweight diagnostics for training logs / analysis.
        with torch.no_grad():
            cf_gain = factual_margin - cf_margin
            active = attenuation > 1e-6
            if active.any():
                weighted_cf_gain = cf_gain[active].mean().item()
                cf_win_rate = (cf_gain[active] > 0).float().mean().item()
            else:
                weighted_cf_gain = 0.0
                cf_win_rate = 0.0

            self.last_loss_dict = {
                'rec_loss': rec_loss.item(),
                'cf_loss': cf_loss.item(),
                'total_loss': loss_value.item(),
                'mean_mask_batch': batch_user_mask.mean().item(),
                'mean_attenuation_batch': (1.0 - batch_user_mask).mean().item(),
                'factual_margin': factual_margin.mean().item(),
                'cf_margin': cf_margin.mean().item(),
                'cf_gain_factual_minus_no_attenuation': cf_gain.mean().item(),
                'cf_win_rate': cf_win_rate,
            }

        return loss_value

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

class GCN(torch.nn.Module):
    def __init__(self,datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_feature, dropout,
                 dim_latent=None,device = None,features=None, user_profile=None):
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
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self,edge_index,features, edge_mask = None):
        temp_features = features
        temp_profile = self.preference
        x = torch.cat((temp_profile, temp_features), dim=0)
        x = F.normalize(x)
        h = self.conv_embed_1(x, edge_index,edge_mask)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index,edge_mask)  # equation 1
        h_2 = self.conv_embed_1(h_1, edge_index,edge_mask)

        x_hat =h + x + h_1 + h_2
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index,edge_mask=None, size=None):
        # pdb.set_trace()

        if edge_mask is None:
            edge_mask = torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype
            )

        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_mask = edge_mask)

    def message(self, x_j, edge_index, size,edge_mask):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[
            torch.isinf(deg_inv_sqrt)
            ] = 0

            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            # return norm.view(-1, 1) * x_j
            return (
                norm.view(-1, 1)
                * edge_mask.view(-1, 1)
                * x_j
            )
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
