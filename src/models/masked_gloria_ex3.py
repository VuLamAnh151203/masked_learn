"""Counterfactual-calibrated user modulation for MASKED_GLORIA_EX."""

import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, remove_self_loops

from common.abstract_recommender import GeneralRecommender


def _cfg(config, key, default):
    """Read an optional config value without requiring ``dict.get``."""
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


def _safe_torch_load(path):
    """Load normal PyTorch checkpoints across old and new torch versions."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = checkpoint.get(key, None)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            if any(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
    raise ValueError('Could not locate a state_dict in the base checkpoint.')


def _strip_module_prefix(state_dict):
    output = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        output[key] = value
    return output


def _find_checkpoint_tensor(state_dict, aliases):
    for alias in aliases:
        value = state_dict.get(alias, None)
        if torch.is_tensor(value):
            return value, alias

    for alias in aliases:
        matches = [
            (key, value)
            for key, value in state_dict.items()
            if key.endswith(alias) and torch.is_tensor(value)
        ]
        if len(matches) == 1:
            return matches[0][1], matches[0][0]
    return None, None


def _parse_gamma_grid(value):
    if isinstance(value, str):
        values = [part.strip() for part in value.split(',') if part.strip()]
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
    else:
        raise TypeError('cf_gamma_grid must be a sequence or comma-separated string.')

    grid = [float(item) for item in values]
    if len(grid) < 2:
        raise ValueError('cf_gamma_grid must contain at least two strengths.')
    if any(not 0.0 <= item <= 1.0 for item in grid):
        raise ValueError('Every counterfactual gamma must be in [0, 1].')
    if grid != sorted(set(grid)):
        raise ValueError('cf_gamma_grid must be strictly increasing and unique.')
    return grid


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class MASKED_GLORIA_EX3(GeneralRecommender):
    """Frozen MASKED_GLORIA_EX plus a learned per-user strength calibrator."""

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_EX3, self).__init__(config, dataset)

        self.config = config
        self.dataset = dataset
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.batch_size = int(config['train_batch_size'])
        self.feat_embed_dim = int(config['feat_embed_dim'])
        self.n_layers = int(config['n_mm_layers'])
        self.knn_k = int(config['knn_k'])
        self.aggr_mode = config['aggr_mode']
        self.num_layer = 1
        self.dim_latent = 64

        print(
            '[MASKED_GLORIA_EX3] users={}, items={}'.format(
                self.num_user,
                self.num_item
            )
        )

        self.gamma_grid_values = _parse_gamma_grid(
            _cfg(config, 'cf_gamma_grid', [0.0, 0.25, 0.5, 0.75, 1.0])
        )
        self.register_buffer(
            'gamma_grid',
            torch.tensor(
                self.gamma_grid_values,
                dtype=torch.float32,
                device=self.device
            )
        )

        self.cf_negatives_per_positive = int(
            _cfg(config, 'cf_negatives_per_positive', 20)
        )
        self.cf_negative_seed = int(_cfg(config, 'seed', 999))
        self.cf_label_tolerance = float(
            _cfg(config, 'cf_label_tolerance', 1e-6)
        )
        self.cf_calibration_train_ratio = float(
            _cfg(config, 'cf_calibration_train_ratio', 0.8)
        )
        self.cf_score_batch_size = int(
            _cfg(config, 'cf_score_batch_size', 262144)
        )
        self.cf_label_cache_dir = os.path.abspath(
            os.path.expanduser(
                str(
                    _cfg(
                        config,
                        'cf_label_cache_dir',
                        os.path.join('saved', 'calibration_labels')
                    )
                )
            )
        )

        if self.cf_negatives_per_positive <= 0:
            raise ValueError('cf_negatives_per_positive must be positive.')
        if self.cf_label_tolerance < 0.0:
            raise ValueError('cf_label_tolerance must be non-negative.')
        if not 0.0 < self.cf_calibration_train_ratio < 1.0:
            raise ValueError('cf_calibration_train_ratio must be in (0, 1).')
        if self.cf_score_batch_size <= 0:
            raise ValueError('cf_score_batch_size must be positive.')

        checkpoint_path = _cfg(config, 'cf_base_checkpoint', None)
        if checkpoint_path is None or not str(checkpoint_path).strip():
            raise ValueError(
                'MASKED_GLORIA_EX3 requires `cf_base_checkpoint` pointing '
                'to a trained MASKED_GLORIA_EX checkpoint.'
            )
        self.cf_base_checkpoint = os.path.abspath(
            os.path.expanduser(str(checkpoint_path))
        )
        if not os.path.isfile(self.cf_base_checkpoint):
            raise FileNotFoundError(self.cf_base_checkpoint)
        self.cf_base_checkpoint_sha256 = _sha256_file(self.cf_base_checkpoint)

        self.id_embedding_full = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
        self.id_embedding_masked = nn.Embedding(
            self.num_item,
            self.feat_embed_dim
        )
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

        _, self.mm_adj = self.get_knn_adj_mat(self.t_feat)

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        packed_edge_index = self.pack_edge_index(train_interactions)
        self.num_interactions = packed_edge_index.shape[0]

        self.register_buffer(
            'edge_user_ids',
            torch.tensor(
                packed_edge_index[:, 0],
                dtype=torch.long,
                device=self.device
            )
        )

        directed_edges = torch.tensor(
            packed_edge_index,
            dtype=torch.long,
            device=self.device
        ).t().contiguous()
        self.edge_index = torch.cat(
            [directed_edges, directed_edges[[1, 0]]],
            dim=1
        )

        train_degree = np.bincount(
            packed_edge_index[:, 0],
            minlength=self.num_user
        ).astype(np.float32)
        log_degree = np.log1p(train_degree)
        degree_mean = float(log_degree.mean())
        degree_std = float(log_degree.std())
        if degree_std <= 0.0:
            degree_std = 1.0
        normalized_log_degree = (log_degree - degree_mean) / degree_std
        self.register_buffer(
            'normalized_log_train_degree',
            torch.tensor(
                normalized_log_degree,
                dtype=torch.float32,
                device=self.device
            )
        )

        self.user_mask_logits = nn.Parameter(
            torch.zeros(self.num_user, device=self.device)
        )
        self.full_gcn = GCN(
            self.num_user,
            self.num_item,
            self.aggr_mode,
            self.id_embedding_full.weight
        )
        self.mask_gcn = GCN(
            self.num_user,
            self.num_item,
            self.aggr_mode,
            self.id_embedding_masked.weight
        )

        calibrator_hidden_dim = int(
            _cfg(config, 'cf_calibrator_hidden_dim', 64)
        )
        if calibrator_hidden_dim <= 0:
            raise ValueError('cf_calibrator_hidden_dim must be positive.')

        calibrator_input_dim = int(self.user_feat.shape[-1]) + 2
        self.calibrator = nn.Sequential(
            nn.Linear(calibrator_input_dim, calibrator_hidden_dim),
            nn.ReLU(),
            nn.Linear(calibrator_hidden_dim, len(self.gamma_grid_values))
        )
        self._initialize_calibrator()

        self._load_and_freeze_base_checkpoint()

        self.result_embed = None
        self._inference_cache_valid = False
        self.calibration_metadata = {
            'label_cache_path': None,
            'num_labeled_users': 0,
            'class_counts': [0] * len(self.gamma_grid_values),
            'best_calibration_epoch': None,
            'best_calibration_ce': None,
        }

    def _initialize_calibrator(self):
        for module in self.calibrator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _load_and_freeze_base_checkpoint(self):
        checkpoint = _safe_torch_load(self.cf_base_checkpoint)
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        target_parameters = dict(self.named_parameters())

        aliases = {
            'user_mask_logits': ['user_mask_logits'],
            'full_gcn.preference': [
                'full_gcn.preference',
                'full_preference'
            ],
            'mask_gcn.preference': [
                'mask_gcn.preference',
                'mask_preference'
            ],
            'id_embedding_full.weight': ['id_embedding_full.weight'],
            'id_embedding_masked.weight': ['id_embedding_masked.weight'],
            'mlp_item.weight': ['mlp_item.weight'],
            'mlp_user.weight': ['mlp_user.weight'],
        }

        loaded_keys = {}
        with torch.no_grad():
            for target_name, candidate_keys in aliases.items():
                source, source_name = _find_checkpoint_tensor(
                    state_dict,
                    candidate_keys
                )
                if source is None:
                    raise KeyError(
                        'Missing base parameter {}. Checkpoint keys include: {}'.format(
                            target_name,
                            list(state_dict.keys())[:25]
                        )
                    )

                target = target_parameters[target_name]
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        'Shape mismatch for {}: checkpoint {} vs model {}.'.format(
                            target_name,
                            tuple(source.shape),
                            tuple(target.shape)
                        )
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded_keys[target_name] = source_name

        for name, parameter in self.named_parameters():
            if not name.startswith('calibrator.'):
                parameter.requires_grad_(False)
                parameter.grad = None

        print(
            '[MASKED_GLORIA_EX3] loaded and froze base checkpoint:\n'
            '  path: {}\n'
            '  sha256: {}\n'
            '  mapped keys: {}'.format(
                self.cf_base_checkpoint,
                self.cf_base_checkpoint_sha256,
                loaded_keys
            )
        )

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True)
        )
        similarity = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_indices = torch.topk(similarity, self.knn_k, dim=-1)
        adjacency_size = similarity.size()
        del similarity

        row_indices = torch.arange(knn_indices.shape[0], device=self.device)
        row_indices = row_indices.unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack(
            (torch.flatten(row_indices), torch.flatten(knn_indices)),
            dim=0
        )
        return indices, self.compute_normalized_laplacian(
            indices,
            adjacency_size
        )

    @staticmethod
    def compute_normalized_laplacian(indices, adjacency_size):
        adjacency = torch.sparse_coo_tensor(
            indices,
            torch.ones(
                indices.size(1),
                dtype=torch.float32,
                device=indices.device
            ),
            adjacency_size
        )
        row_sum = 1e-7 + torch.sparse.sum(adjacency, -1).to_dense()
        inverse_sqrt = torch.pow(row_sum, -0.5)
        values = inverse_sqrt[indices[0]] * inverse_sqrt[indices[1]]
        return torch.sparse_coo_tensor(indices, values, adjacency_size)

    def pack_edge_index(self, interaction_matrix):
        rows = interaction_matrix.row
        columns = interaction_matrix.col + self.n_users
        return np.column_stack((rows, columns))

    def item_item(self, representation):
        propagated = representation
        for _ in range(self.n_layers):
            propagated = torch.sparse.mm(self.mm_adj, propagated)
        return representation + propagated

    def get_user_mask(self):
        """Return the frozen base personalization mask ``m_u``."""
        return torch.sigmoid(self.user_mask_logits)

    def get_calibrator_features(self, user_ids=None):
        content = F.normalize(self.user_feat, p=2, dim=1)
        attenuation = (1.0 - self.get_user_mask()).unsqueeze(1)
        degree_feature = self.normalized_log_train_degree.unsqueeze(1)
        features = torch.cat(
            [content, attenuation, degree_feature],
            dim=1
        ).detach()
        if user_ids is not None:
            features = features[user_ids]
        return features

    def get_calibrator_logits(self, user_ids=None):
        return self.calibrator(self.get_calibrator_features(user_ids))

    def get_gamma_probabilities(self, user_ids=None):
        return torch.softmax(self.get_calibrator_logits(user_ids), dim=-1)

    def get_user_gamma(self, user_ids=None):
        probabilities = self.get_gamma_probabilities(user_ids)
        return torch.sum(probabilities * self.gamma_grid, dim=-1)

    def get_effective_user_mask(self, gamma=None):
        base_mask = self.get_user_mask()
        if gamma is None:
            gamma = self.get_user_gamma()

        if not torch.is_tensor(gamma):
            gamma = torch.tensor(
                gamma,
                dtype=base_mask.dtype,
                device=base_mask.device
            )
        else:
            gamma = gamma.to(device=base_mask.device, dtype=base_mask.dtype)

        if gamma.ndim == 0:
            scalar_gamma = float(gamma.item())
            if scalar_gamma == 0.0:
                return torch.ones_like(base_mask)
            if scalar_gamma == 1.0:
                return base_mask
            gamma = gamma.expand_as(base_mask)
        else:
            gamma = gamma.reshape(-1)
            if gamma.numel() != self.num_user:
                raise ValueError(
                    'Per-user gamma must have shape [{}], received {}.'.format(
                        self.num_user,
                        tuple(gamma.shape)
                    )
                )

        if torch.any(gamma < 0.0) or torch.any(gamma > 1.0):
            raise ValueError('Every gamma value must be in [0, 1].')
        return 1.0 - gamma * (1.0 - base_mask)

    def compute_result_embedding(self, gamma):
        effective_user_mask = self.get_effective_user_mask(gamma)
        original_edge_mask = effective_user_mask[self.edge_user_ids]
        edge_mask = torch.cat(
            [original_edge_mask, original_edge_mask],
            dim=0
        )

        self.full_rep, _ = self.full_gcn(
            self.edge_index,
            self.id_embedding_full.weight
        )
        self.mask_rep, _ = self.mask_gcn(
            self.edge_index,
            self.id_embedding_masked.weight,
            edge_mask=edge_mask
        )

        user_representation = torch.cat(
            [
                self.full_rep[:self.num_user],
                self.mask_rep[:self.num_user]
            ],
            dim=1
        )
        item_representation = torch.cat(
            [
                self.full_rep[self.num_user:],
                self.mask_rep[self.num_user:]
            ],
            dim=1
        )
        item_representation = self.item_item(item_representation)

        self.result_embed = torch.cat(
            [user_representation, item_representation],
            dim=0
        )
        return self.result_embed

    def invalidate_inference_cache(self):
        self._inference_cache_valid = False

    def forward(self, interaction):
        self.invalidate_inference_cache()
        representation = self.compute_result_embedding(self.get_user_gamma())
        users = interaction[0]
        positive_items = interaction[1] + self.num_user
        negative_items = interaction[2] + self.num_user
        positive_scores = torch.sum(
            representation[users] * representation[positive_items],
            dim=1
        )
        negative_scores = torch.sum(
            representation[users] * representation[negative_items],
            dim=1
        )
        return positive_scores, negative_scores

    def calculate_loss(self, interaction):
        raise RuntimeError(
            'MASKED_GLORIA_EX3 must be trained with '
            'CounterfactualCalibrationTrainer.'
        )

    @torch.no_grad()
    def recompute_inference_result_embedding(self):
        self.eval()
        representation = self.compute_result_embedding(self.get_user_gamma())
        self._inference_cache_valid = True
        return representation

    def full_sort_predict(self, interaction):
        if not self._inference_cache_valid:
            self.recompute_inference_result_embedding()
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        return torch.matmul(user_tensor[interaction[0]], item_tensor.t())

    @staticmethod
    def _sample_fixed_negatives(
        train_dataframe,
        validation_dataframe,
        user_field,
        item_field,
        num_items,
        negatives_per_positive,
        seed
    ):
        rng = np.random.RandomState(seed)
        train_seen = {
            int(user): set(group[item_field].astype(np.int64).tolist())
            for user, group in train_dataframe.groupby(user_field)
        }

        pair_users = []
        pair_positive_items = []
        pair_negative_items = []

        for user, group in validation_dataframe.groupby(user_field, sort=True):
            user = int(user)
            positive_items = group[item_field].astype(np.int64).tolist()
            forbidden = set(train_seen.get(user, set()))
            forbidden.update(positive_items)
            available_count = num_items - len(forbidden)
            if available_count < negatives_per_positive:
                raise ValueError(
                    'User {} has only {} eligible counterfactual negatives.'.format(
                        user,
                        available_count
                    )
                )

            for positive_item in positive_items:
                sampled = []
                sampled_set = set()
                while len(sampled) < negatives_per_positive:
                    candidate = int(rng.randint(0, num_items))
                    if candidate in forbidden or candidate in sampled_set:
                        continue
                    sampled.append(candidate)
                    sampled_set.add(candidate)

                pair_users.extend([user] * negatives_per_positive)
                pair_positive_items.extend(
                    [int(positive_item)] * negatives_per_positive
                )
                pair_negative_items.extend(sampled)

        return (
            np.asarray(pair_users, dtype=np.int64),
            np.asarray(pair_positive_items, dtype=np.int64),
            np.asarray(pair_negative_items, dtype=np.int64)
        )

    @torch.no_grad()
    def _counterfactual_user_losses(
        self,
        pair_users,
        pair_positive_items,
        pair_negative_items,
        labeled_users
    ):
        device = self.user_mask_logits.device
        pair_count = int(pair_users.shape[0])
        all_strength_losses = []

        pair_user_counts = torch.bincount(
            torch.from_numpy(pair_users),
            minlength=self.num_user
        ).to(dtype=torch.float32)
        labeled_user_tensor_cpu = torch.from_numpy(labeled_users).long()

        self.eval()
        for gamma in self.gamma_grid_values:
            representation = self.compute_result_embedding(float(gamma))
            loss_sums = torch.zeros(
                self.num_user,
                dtype=torch.float32,
                device=device
            )

            for start in range(0, pair_count, self.cf_score_batch_size):
                end = min(start + self.cf_score_batch_size, pair_count)
                users = torch.from_numpy(pair_users[start:end]).to(device)
                positive_items = torch.from_numpy(
                    pair_positive_items[start:end]
                ).to(device) + self.num_user
                negative_items = torch.from_numpy(
                    pair_negative_items[start:end]
                ).to(device) + self.num_user

                positive_scores = torch.sum(
                    representation[users] * representation[positive_items],
                    dim=1
                )
                negative_scores = torch.sum(
                    representation[users] * representation[negative_items],
                    dim=1
                )
                pair_losses = F.softplus(negative_scores - positive_scores)
                loss_sums.index_add_(0, users, pair_losses)

            labeled_loss_sums = loss_sums.cpu()[labeled_user_tensor_cpu]
            labeled_counts = pair_user_counts[labeled_user_tensor_cpu]
            if torch.any(labeled_counts <= 0):
                raise RuntimeError('Every labeled user must own validation pairs.')
            all_strength_losses.append(labeled_loss_sums / labeled_counts)

        self.invalidate_inference_cache()
        return torch.stack(all_strength_losses, dim=1)

    def _select_counterfactual_labels(self, loss_matrix):
        minimum_loss = loss_matrix.min(dim=1, keepdim=True).values
        eligible = loss_matrix <= (minimum_loss + self.cf_label_tolerance)
        # argmax returns the first True; gamma_grid is ascending, therefore
        # near-ties conservatively choose the smallest intervention strength.
        return eligible.to(dtype=torch.int64).argmax(dim=1)

    def _stratified_user_split(self, user_ids, labels):
        rng = np.random.RandomState(self.cf_negative_seed)
        training_rows = []
        validation_rows = []

        labels_numpy = labels.cpu().numpy()
        for class_index in range(len(self.gamma_grid_values)):
            class_rows = np.flatnonzero(labels_numpy == class_index)
            if class_rows.size == 0:
                continue
            rng.shuffle(class_rows)

            if class_rows.size == 1:
                training_rows.extend(class_rows.tolist())
                continue

            training_count = int(
                round(class_rows.size * self.cf_calibration_train_ratio)
            )
            training_count = min(
                max(training_count, 1),
                class_rows.size - 1
            )
            training_rows.extend(class_rows[:training_count].tolist())
            validation_rows.extend(class_rows[training_count:].tolist())

        if not validation_rows:
            raise RuntimeError(
                'The stratified calibration split produced no validation users.'
            )

        rng.shuffle(training_rows)
        rng.shuffle(validation_rows)
        training_rows = torch.tensor(training_rows, dtype=torch.long)
        validation_rows = torch.tensor(validation_rows, dtype=torch.long)

        return {
            'train_user_ids': user_ids[training_rows],
            'train_labels': labels[training_rows],
            'validation_user_ids': user_ids[validation_rows],
            'validation_labels': labels[validation_rows],
        }

    def _label_cache_specification(self):
        return {
            'format_version': 2,
            'dataset': str(self.config['dataset']),
            'base_checkpoint_sha256': self.cf_base_checkpoint_sha256,
            'gamma_grid': list(self.gamma_grid_values),
            'negatives_per_positive': self.cf_negatives_per_positive,
            'negative_seed': self.cf_negative_seed,
            'split_seed': self.cf_negative_seed,
            'label_tolerance': self.cf_label_tolerance,
            'calibration_train_ratio': self.cf_calibration_train_ratio,
        }

    def _label_cache_path(self):
        specification = self._label_cache_specification()
        payload = json.dumps(
            specification,
            sort_keys=True,
            separators=(',', ':')
        ).encode('utf-8')
        cache_key = hashlib.sha256(payload).hexdigest()[:20]
        filename = '{}-{}.pt'.format(self.config['dataset'], cache_key)
        return os.path.join(self.cf_label_cache_dir, filename)

    def prepare_counterfactual_calibration(self, train_data, validation_data):
        """Create or load deterministic per-user counterfactual labels."""
        cache_path = self._label_cache_path()
        expected_specification = self._label_cache_specification()

        if os.path.isfile(cache_path):
            artifact = _safe_torch_load(cache_path)
            if artifact.get('specification') != expected_specification:
                raise ValueError(
                    'Calibration cache metadata does not match the current run.'
                )
        else:
            train_dataframe = train_data.dataset.df
            validation_dataframe = validation_data.dataset.df
            user_field = train_data.dataset.uid_field
            item_field = train_data.dataset.iid_field

            pair_users, pair_positive_items, pair_negative_items = (
                self._sample_fixed_negatives(
                    train_dataframe=train_dataframe,
                    validation_dataframe=validation_dataframe,
                    user_field=user_field,
                    item_field=item_field,
                    num_items=self.num_item,
                    negatives_per_positive=self.cf_negatives_per_positive,
                    seed=self.cf_negative_seed
                )
            )
            labeled_users = np.sort(
                validation_dataframe[user_field].unique().astype(np.int64)
            )
            loss_matrix = self._counterfactual_user_losses(
                pair_users,
                pair_positive_items,
                pair_negative_items,
                labeled_users
            )
            labels = self._select_counterfactual_labels(loss_matrix)
            user_ids = torch.from_numpy(labeled_users).long()
            split = self._stratified_user_split(user_ids, labels)
            class_counts = torch.bincount(
                labels,
                minlength=len(self.gamma_grid_values)
            )

            artifact = {
                'specification': expected_specification,
                'base_checkpoint_sha256': self.cf_base_checkpoint_sha256,
                'gamma_grid': torch.tensor(
                    self.gamma_grid_values,
                    dtype=torch.float32
                ),
                'negative_seed': self.cf_negative_seed,
                'split_seed': self.cf_negative_seed,
                'user_ids': user_ids,
                'loss_matrix': loss_matrix.cpu(),
                'labels': labels.cpu(),
                'class_counts': class_counts.cpu(),
                **split,
            }
            os.makedirs(self.cf_label_cache_dir, exist_ok=True)
            temporary_path = cache_path + '.tmp'
            torch.save(artifact, temporary_path)
            os.replace(temporary_path, cache_path)

        required_keys = {
            'base_checkpoint_sha256',
            'gamma_grid',
            'negative_seed',
            'split_seed',
            'user_ids',
            'loss_matrix',
            'labels',
            'class_counts',
            'train_user_ids',
            'train_labels',
            'validation_user_ids',
            'validation_labels',
        }
        missing_keys = required_keys.difference(artifact.keys())
        if missing_keys:
            raise KeyError(
                'Calibration artifact is missing keys: {}'.format(
                    sorted(missing_keys)
                )
            )

        class_counts = artifact['class_counts'].tolist()
        self.calibration_metadata.update({
            'label_cache_path': cache_path,
            'num_labeled_users': int(artifact['user_ids'].numel()),
            'class_counts': [int(value) for value in class_counts],
            'num_calibration_train_users': int(
                artifact['train_user_ids'].numel()
            ),
            'num_calibration_validation_users': int(
                artifact['validation_user_ids'].numel()
            ),
        })
        return artifact

    @torch.no_grad()
    def get_calibration_statistics(self):
        self.eval()
        gamma = self.get_user_gamma()
        base_mask = self.get_user_mask()
        effective_mask = self.get_effective_user_mask(gamma)
        return {
            'gamma_mean': gamma.mean().item(),
            'gamma_std': gamma.std(unbiased=False).item(),
            'gamma_min': gamma.min().item(),
            'gamma_max': gamma.max().item(),
            'base_attenuation_mean': (1.0 - base_mask).mean().item(),
            'effective_attenuation_mean': (
                1.0 - effective_mask
            ).mean().item(),
        }

    def get_checkpoint_metadata(self):
        return {
            'method': 'counterfactual_calibrated_user_modulation',
            'base_checkpoint': self.cf_base_checkpoint,
            'base_checkpoint_sha256': self.cf_base_checkpoint_sha256,
            'gamma_grid': list(self.gamma_grid_values),
            'negative_seed': self.cf_negative_seed,
            'split_seed': self.cf_negative_seed,
            'negatives_per_positive': self.cf_negatives_per_positive,
            'calibration_train_ratio': self.cf_calibration_train_ratio,
            **self.calibration_metadata,
            **self.get_calibration_statistics(),
        }


class GCN(nn.Module):
    """LightGCN-style propagation branch with its own user preference."""

    def __init__(self, num_user, num_item, aggregation, item_features):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggregation = aggregation
        feature_dimension = item_features.size(1)
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(
                torch.tensor(
                    np.random.randn(num_user, feature_dimension),
                    dtype=torch.float32,
                    requires_grad=True
                ),
                gain=1
            )
        )
        self.convolution = BaseGCN(aggregation=self.aggregation)

    def forward(self, edge_index, item_features, edge_mask=None):
        representation = torch.cat(
            [self.preference, item_features],
            dim=0
        )
        representation = F.normalize(representation)
        layer_1 = self.convolution(
            representation,
            edge_index,
            edge_mask
        )
        layer_2 = self.convolution(layer_1, edge_index, edge_mask)
        layer_3 = self.convolution(layer_2, edge_index, edge_mask)
        return representation + layer_1 + layer_2 + layer_3, self.preference


class BaseGCN(MessagePassing):
    def __init__(self, aggregation='add'):
        super(BaseGCN, self).__init__(aggr=aggregation)
        self.aggregation = aggregation

    def forward(self, x, edge_index, edge_mask=None, size=None):
        if edge_mask is None:
            edge_mask = torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype
            )
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.propagate(
            edge_index,
            size=(x.size(0), x.size(0)),
            x=x,
            edge_mask=edge_mask
        )

    def message(self, x_j, edge_index, size, edge_mask):
        if self.aggregation != 'add':
            return x_j
        row, column = edge_index
        node_degree = degree(row, size[0], dtype=x_j.dtype)
        inverse_sqrt = node_degree.pow(-0.5)
        inverse_sqrt[torch.isinf(inverse_sqrt)] = 0
        normalization = inverse_sqrt[row] * inverse_sqrt[column]
        return (
            normalization.view(-1, 1)
            * edge_mask.view(-1, 1)
            * x_j
        )
