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
    """
    Counterfactual user-routing V1.

    Main idea
    ---------
    1) Keep the original full branch unchanged:
           full_rep = full_gcn(G)

    2) Learn one scalar per user:
           m_soft[u] = sigmoid(theta_u)

    3) Convert the soft mask into a hard 0/1 routing decision with STE:
           m_route[u] in {0, 1}

       All interaction edges of the same user receive the same route.

    4) Build two complementary graph views:
           G_A : edge weight = m_route[u]
           G_B : edge weight = 1 - m_route[u]

       IMPORTANT: both views use the SAME mask_gcn parameters.

    5) For a user routed to A, A is factual and B is counterfactual.
       For a user routed to B, B is factual and A is counterfactual.

    6) Optimize:
           L = L_rec + lambda_route * L_route
                     + lambda_balance * L_balance

       where L_route encourages the factual route to have a better
       BPR margin than the counterfactual route.
    """

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
        self.drop_rate = 0.1
        self.t_rep = None
        self.t_preference = None
        self.dim_latent = 64
        self.mm_adj = None
        self.config = config

        # ------------------------------------------------------------
        # Counterfactual-routing hyperparameters.
        # Defaults are used when the keys are not present in config.
        # ------------------------------------------------------------
        self.route_margin = self._config_value(config, 'route_margin', 0.5, float)
        self.lambda_route = self._config_value(config, 'lambda_route', 0.1, float)
        self.lambda_balance = self._config_value(config, 'lambda_balance', 1e-3, float)
        self.mask_threshold = self._config_value(config, 'mask_threshold', 0.5, float)
        self.mask_init_std = self._config_value(config, 'mask_init_std', 1e-2, float)

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
        # For every original interaction e=(u,i):
        #     M_e = M_u
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

        # Undirected interaction graph:
        # [original user->item edges, reverse item->user edges]
        self.edge_index = torch.cat(
            [edge_index, edge_index[[1, 0]]],
            dim=1
        )

        # ============================================================
        # USER-LEVEL LEARNABLE ROUTING MASK
        # ============================================================
        # One learnable logit per user.
        #
        # Do NOT initialize every logit to exactly 0 when using a hard
        # threshold, because every user would make the same route decision
        # at initialization. Small random noise breaks this symmetry while
        # keeping sigmoid(theta) close to 0.5.
        self.user_mask_logits = nn.Parameter(
            torch.empty(
                self.num_user,
                device=self.device
            )
        )
        nn.init.normal_(
            self.user_mask_logits,
            mean=0.0,
            std=self.mask_init_std
        )

        # ============================================================
        # Existing fixed high/low degree split code (kept unchanged)
        # ============================================================
        edge_index_np = self.pack_edge_index(train_interactions)

        item_ids = edge_index_np[:, 1] - self.num_user
        item_degree = np.bincount(item_ids, minlength=self.num_item)

        high_ratio = 0.10
        num_high = int(self.num_item * high_ratio)

        high_items = np.argsort(item_degree)[-num_high:]
        high_items = set(high_items.tolist())

        low_edges = []
        high_edges = []

        for edge in edge_index_np:
            item_id = edge[1] - self.num_user

            if item_id in high_items:
                high_edges.append(edge)
            else:
                low_edges.append(edge)

        low_edges = np.array(low_edges, dtype=np.int64)
        high_edges = np.array(high_edges, dtype=np.int64)

        self.edge_index_low = torch.tensor(
            low_edges,
            dtype=torch.long
        ).t().contiguous().to(self.device)

        self.edge_index_high = torch.tensor(
            high_edges,
            dtype=torch.long
        ).t().contiguous().to(self.device)

        self.edge_index_low = torch.cat(
            (self.edge_index_low, self.edge_index_low[[1, 0]]),
            dim=1
        )

        self.edge_index_high = torch.cat(
            (self.edge_index_high, self.edge_index_high[[1, 0]]),
            dim=1
        )

        # ============================================================
        # GCNs
        # ============================================================
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

        # One shared mask GCN. It is called twice: once on G_A and once on G_B.
        # Using one parameter set makes the counterfactual comparison cleaner:
        # the intervention is the graph routing, not a different encoder.
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

        # Optional caches/diagnostics.
        self.result_embed_a = None
        self.result_embed_b = None
        self.last_m_soft = None
        self.last_m_route = None
        self.last_loss_dict = {}

    @staticmethod
    def _config_value(config, key, default, cast_fn):
        """Safely read an optional config value."""
        try:
            value = config[key]
        except Exception:
            value = None

        if value is None:
            return default
        return cast_fn(value)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True)
        )
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim

        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack(
            (torch.flatten(indices0), torch.flatten(knn_ind)),
            0
        )

        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(
            indices,
            torch.ones_like(indices[0]),
            adj_size
        )
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
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return rep + h

    # ================================================================
    # User mask / routing helpers
    # ================================================================
    def get_user_mask(self):
        """
        Return the soft user mask.

        Shape: [num_users]
        Range: (0, 1)
        """
        return torch.sigmoid(self.user_mask_logits)

    def get_user_route_mask(self):
        """
        Return both soft masks and hard STE routing decisions.

        Forward:
            m_route == m_hard in {0, 1}

        Backward:
            gradient is approximated through m_soft.
        """
        m_soft = self.get_user_mask()
        m_hard = (m_soft > self.mask_threshold).to(m_soft.dtype)

        # Straight-Through Estimator:
        # forward value = m_hard
        # backward gradient ~= gradient through m_soft
        m_route = m_hard - m_soft.detach() + m_soft

        return m_soft, m_route

    def get_original_edge_mask(self, hard=False, complement=False):
        """
        Expand the user-level mask to original user-item edges.

        Parameters
        ----------
        hard : bool
            If False, return sigmoid(theta_u) per edge.
            If True, return STE hard routing per edge.

        complement : bool
            If True, return 1-mask.
        """
        if hard:
            _, user_mask = self.get_user_route_mask()
        else:
            user_mask = self.get_user_mask()

        if complement:
            user_mask = 1.0 - user_mask

        return user_mask[self.edge_user_ids]

    @torch.no_grad()
    def get_user_mask_statistics(self):
        """Diagnostics for the learned user-level routing mask."""
        m_soft = self.get_user_mask().detach()
        m_hard = (m_soft > self.mask_threshold).float()

        return {
            'mean_keep': m_soft.mean().item(),
            'mean_attenuation': (1.0 - m_soft).mean().item(),
            'min_keep': m_soft.min().item(),
            'max_keep': m_soft.max().item(),
            'std_keep': m_soft.std(unbiased=False).item(),
            'fraction_route_A': m_hard.mean().item(),
            'fraction_route_B': (1.0 - m_hard).mean().item(),
        }

    # ================================================================
    # Graph-view construction
    # ================================================================
    def _build_result_embedding(self, full_rep, masked_rep):
        """
        Reproduce the original fusion:
            user = concat(full_user, masked_user)
            item = item_item(concat(full_item, masked_item))
        """
        item_full = full_rep[self.num_user:]
        item_masked = masked_rep[self.num_user:]

        item_rep = torch.cat(
            (item_full, item_masked),
            dim=1
        )
        item_rep = self.item_item(item_rep)

        user_full = full_rep[:self.num_user]
        user_masked = masked_rep[:self.num_user]

        user_rep = torch.cat(
            (user_full, user_masked),
            dim=1
        )

        return torch.cat(
            (user_rep, item_rep),
            dim=0
        )

    def _compute_graph_views(self):
        """
        Compute the full branch and the two complementary routed graph views.

        Returns
        -------
        result_embed_a : Tensor [num_users + num_items, 2D]
        result_embed_b : Tensor [num_users + num_items, 2D]
        m_soft         : Tensor [num_users]
        m_route        : Tensor [num_users], forward values are 0/1
        """

        # ------------------------------------------------------------
        # 1) Full graph branch: unchanged from the original model.
        # ------------------------------------------------------------
        self.full_rep, self.full_preference = self.full_gcn(
            self.edge_index,
            self.id_embedding_full.weight
        )

        # ------------------------------------------------------------
        # 2) User-level routing mask.
        # ------------------------------------------------------------
        m_soft, m_route = self.get_user_route_mask()

        # One value per ORIGINAL user-item interaction.
        edge_mask_a_original = m_route[self.edge_user_ids]
        edge_mask_b_original = (1.0 - m_route)[self.edge_user_ids]

        # self.edge_index contains original + reverse edges.
        # Use exactly the same route value for both directions.
        edge_mask_a = torch.cat(
            [edge_mask_a_original, edge_mask_a_original],
            dim=0
        )
        edge_mask_b = torch.cat(
            [edge_mask_b_original, edge_mask_b_original],
            dim=0
        )

        # ------------------------------------------------------------
        # 3) Complementary graph views with SHARED mask_gcn params.
        # ------------------------------------------------------------
        self.mask_rep_a, self.mask_preference_a = self.mask_gcn(
            self.edge_index,
            self.id_embedding_masked.weight,
            edge_mask=edge_mask_a
        )

        self.mask_rep_b, self.mask_preference_b = self.mask_gcn(
            self.edge_index,
            self.id_embedding_masked.weight,
            edge_mask=edge_mask_b
        )

        # ------------------------------------------------------------
        # 4) Fuse full + routed representation exactly as original code.
        # ------------------------------------------------------------
        result_embed_a = self._build_result_embedding(
            self.full_rep,
            self.mask_rep_a
        )
        result_embed_b = self._build_result_embedding(
            self.full_rep,
            self.mask_rep_b
        )

        # Cache for diagnostics / compatibility.
        self.result_embed_a = result_embed_a
        self.result_embed_b = result_embed_b
        self.last_m_soft = m_soft
        self.last_m_route = m_route

        return result_embed_a, result_embed_b, m_soft, m_route

    @staticmethod
    def _pair_scores(result_embed, user_nodes, pos_item_nodes, neg_item_nodes):
        user_tensor = result_embed[user_nodes]
        pos_item_tensor = result_embed[pos_item_nodes]
        neg_item_tensor = result_embed[neg_item_nodes]

        pos_scores = torch.sum(
            user_tensor * pos_item_tensor,
            dim=1
        )
        neg_scores = torch.sum(
            user_tensor * neg_item_tensor,
            dim=1
        )

        return pos_scores, neg_scores

    # ================================================================
    # Forward / loss
    # ================================================================
    def forward(self, interaction):
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        result_embed_a, result_embed_b, m_soft, m_route = \
            self._compute_graph_views()

        # Score the SAME triplets on both complementary graph views.
        pos_a, neg_a = self._pair_scores(
            result_embed_a,
            user_nodes,
            pos_item_nodes,
            neg_item_nodes
        )

        pos_b, neg_b = self._pair_scores(
            result_embed_b,
            user_nodes,
            pos_item_nodes,
            neg_item_nodes
        )

        # User-specific route in this training batch.
        route = m_route[user_nodes]

        # ------------------------------------------------------------
        # Factual view
        # ------------------------------------------------------------
        # route=1 -> A factual
        # route=0 -> B factual
        pos_factual = (
            route * pos_a
            + (1.0 - route) * pos_b
        )
        neg_factual = (
            route * neg_a
            + (1.0 - route) * neg_b
        )

        # ------------------------------------------------------------
        # Counterfactual view = force the SAME user to the other graph.
        # ------------------------------------------------------------
        pos_cf = (
            (1.0 - route) * pos_a
            + route * pos_b
        )
        neg_cf = (
            (1.0 - route) * neg_a
            + route * neg_b
        )

        return pos_factual, neg_factual, pos_cf, neg_cf, m_soft

    def calculate_loss(self, interaction):
        (
            pos_factual,
            neg_factual,
            pos_cf,
            neg_cf,
            m_soft
        ) = self.forward(interaction)

        # ------------------------------------------------------------
        # 1) Standard recommendation objective on the factual route.
        # ------------------------------------------------------------
        factual_margin = pos_factual - neg_factual

        # Equivalent to the original:
        #   -mean(log2(sigmoid(pos-neg)))
        # but numerically more stable.
        rec_loss = -F.logsigmoid(factual_margin).mean() / np.log(2.0)

        # ------------------------------------------------------------
        # 2) Counterfactual routing loss.
        # ------------------------------------------------------------
        cf_margin = pos_cf - neg_cf

        # We want:
        #     factual_margin >= cf_margin + route_margin
        #
        # L_route = max(0, gamma - d_F + d_CF)
        route_loss = F.relu(
            self.route_margin
            - factual_margin
            + cf_margin
        ).mean()

        # ------------------------------------------------------------
        # 3) Weak population-level anti-collapse regularization.
        # ------------------------------------------------------------
        # This is intentionally weak. It only discourages the trivial
        # all-A or all-B solution; it should not dominate user-specific
        # routing learned from recommendation/counterfactual signals.
        balance_loss = (
            m_soft.mean() - 0.5
        ).pow(2)

        total_loss = (
            rec_loss
            + self.lambda_route * route_loss
            + self.lambda_balance * balance_loss
        )

        # Handy diagnostics for logging during training.
        with torch.no_grad():
            self.last_loss_dict = {
                'total': float(total_loss.detach().cpu()),
                'rec': float(rec_loss.detach().cpu()),
                'route': float(route_loss.detach().cpu()),
                'balance': float(balance_loss.detach().cpu()),
                'factual_margin': float(factual_margin.mean().detach().cpu()),
                'cf_margin': float(cf_margin.mean().detach().cpu()),
                'mean_mask': float(m_soft.mean().detach().cpu()),
            }

        return total_loss

    # ================================================================
    # Full-sort evaluation
    # ================================================================
    def full_sort_predict(self, interaction):
        """
        Full-sort prediction respecting each user's learned route.

        We cannot use one global `result_embed` anymore because graph A and
        graph B produce different item embeddings. For a query user u:

            if route[u] == 1:
                score with (user_A[u], items_A)
            else:
                score with (user_B[u], items_B)

        The STE route has hard 0/1 forward values, so this selection is exact
        in the forward pass while still remaining differentiable in training.
        """
        user_ids = interaction[0]

        result_embed_a, result_embed_b, _, m_route = \
            self._compute_graph_views()

        user_a = result_embed_a[:self.n_users]
        item_a = result_embed_a[self.n_users:]

        user_b = result_embed_b[:self.n_users]
        item_b = result_embed_b[self.n_users:]

        score_matrix_a = torch.matmul(
            user_a[user_ids],
            item_a.t()
        )
        score_matrix_b = torch.matmul(
            user_b[user_ids],
            item_b.t()
        )

        route = m_route[user_ids].unsqueeze(1)

        score_matrix = (
            route * score_matrix_a
            + (1.0 - route) * score_matrix_b
        )

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

        x = torch.cat(
            (temp_profile, temp_features),
            dim=0
        )
        x = F.normalize(x)

        h = self.conv_embed_1(
            x,
            edge_index,
            edge_mask
        )
        h_1 = self.conv_embed_1(
            h,
            edge_index,
            edge_mask
        )
        h_2 = self.conv_embed_1(
            h_1,
            edge_index,
            edge_mask
        )

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
            edge_mask = edge_mask.to(
                device=x.device,
                dtype=x.dtype
            )

        # Keep edge_mask aligned if self-loops ever appear.
        edge_index, edge_mask = remove_self_loops(
            edge_index,
            edge_mask
        )

        if size is None:
            size = (x.size(0), x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(
            edge_index,
            size=size,
            x=x,
            edge_mask=edge_mask
        )

    def message(self, x_j, edge_index, size, edge_mask):
        if self.aggr == 'add':
            row, col = edge_index

            # ========================================================
            # IMPORTANT FIX:
            # degree must respect the edge mask.
            #
            # Old code counted masked-out edges in the degree even when
            # edge_mask[e] == 0. That means a supposedly removed edge still
            # changed graph normalization.
            # ========================================================
            deg = torch.zeros(
                size[0],
                device=x_j.device,
                dtype=x_j.dtype
            )
            deg.index_add_(
                0,
                row,
                edge_mask
            )

            # Numerically safe inverse sqrt degree.
            positive = deg > 0
            safe_deg = torch.where(
                positive,
                deg,
                torch.ones_like(deg)
            )
            deg_inv_sqrt = safe_deg.pow(-0.5)
            deg_inv_sqrt = deg_inv_sqrt * positive.to(deg_inv_sqrt.dtype)

            norm = (
                deg_inv_sqrt[row]
                * deg_inv_sqrt[col]
            )

            return (
                norm.view(-1, 1)
                * edge_mask.view(-1, 1)
                * x_j
            )

        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels
        )