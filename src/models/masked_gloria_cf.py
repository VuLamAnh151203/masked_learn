# coding: utf-8

import math
import random

import torch
import torch.nn.functional as F

from models.masked_gloria import MASKED_GLORIA


def _cfg(config, key, default):
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


def _cfg_bool(config, key, default):
    value = _cfg(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(value)


class MASKED_GLORIA_CF(MASKED_GLORIA):
    """MASKED_GLORIA with boundary-aware mask regularization.

    This variant does not intervene on, remove, or rank training edges.  It
    samples a pseudo-positive item, measures its margin against a fixed Top-K
    boundary competitor in the current masked graph, and gives samples close
    to that boundary a larger margin-regularization weight.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_CF, self).__init__(config, dataset)

        # ``cf_lambda`` is retained as the public weight name used by the
        # previous implementation; it now means lambda_b in the boundary loss.
        self.cf_lambda = float(_cfg(config, 'cf_lambda', 0.1))
        self.cf_temperature = float(
            _cfg(config, 'cf_temperature', _cfg(config, 'cf_tau', 1.0))
        )
        self.cf_gamma = float(
            _cfg(
                config,
                'cf_gamma',
                _cfg(config, 'cf_target_margin', 0.1)
            )
        )
        self.cf_warmup_ratio = float(_cfg(config, 'cf_warmup_ratio', 0.10))
        configured_warmup_epochs = int(_cfg(config, 'cf_warmup_epochs', 0))
        self.cf_user_ratio = float(_cfg(config, 'cf_user_ratio', 0.10))
        self.cf_batch_size = int(_cfg(config, 'cf_batch_size', 8))
        self.cf_k = int(_cfg(config, 'cf_k', 20))
        self.cf_min_history = int(_cfg(config, 'cf_min_history', 2))
        self.cf_seed_offset = int(_cfg(config, 'cf_seed_offset', 10000))
        self.cf_log_stats = _cfg_bool(config, 'cf_log_stats', True)
        self.cf_detach_boundary_weight = _cfg_bool(
            config,
            'cf_detach_boundary_weight',
            True
        )

        max_epochs = int(_cfg(config, 'epochs', 1000))
        if configured_warmup_epochs >= 0:
            self.cf_warmup_epochs = configured_warmup_epochs
        else:
            self.cf_warmup_epochs = int(
                math.ceil(max_epochs * self.cf_warmup_ratio)
            )

        self._validate_cf_config()
        self.current_epoch = 0
        self._cf_rng = random.Random(self.cf_seed_offset)
        self.user_seen_items = self._build_cf_history()
        self.cf_stats = self._new_cf_stats()

    def _validate_cf_config(self):
        if self.cf_lambda < 0.0:
            raise ValueError('cf_lambda must be non-negative.')
        if self.cf_temperature <= 0.0:
            raise ValueError('cf_temperature/cf_tau must be positive.')
        if self.cf_gamma < 0.0:
            raise ValueError('cf_gamma must be non-negative.')
        if not 0.0 <= self.cf_user_ratio <= 1.0:
            raise ValueError('cf_user_ratio must be in [0, 1].')
        if self.cf_batch_size <= 0:
            raise ValueError('cf_batch_size must be positive.')
        if self.cf_k <= 0:
            raise ValueError('cf_k must be positive.')
        if self.cf_min_history < 2:
            raise ValueError('cf_min_history must be at least 2.')
        if self.cf_warmup_epochs < 0:
            raise ValueError('cf_warmup_epochs must be non-negative.')

    def _build_cf_history(self):
        user_seen_items = [[] for _ in range(self.num_user)]
        edge_users = self.forward_edge_users.detach().cpu().tolist()
        edge_items = self.forward_edge_items.detach().cpu().tolist()
        for user_id, item_id in zip(edge_users, edge_items):
            user_seen_items[int(user_id)].append(int(item_id))
        return tuple(
            tuple(sorted(set(items))) for items in user_seen_items
        )

    @staticmethod
    def _new_cf_stats():
        return {
            'samples': 0,
            'eligible': 0,
            'used': 0,
            'loss_sum': 0.0,
            'margin_sum': 0.0,
            'weight_sum': 0.0,
        }

    def set_training_epoch(self, epoch_idx):
        epoch_idx = int(epoch_idx)
        if epoch_idx < 0:
            raise ValueError('epoch_idx must be non-negative.')
        self.current_epoch = epoch_idx
        self._cf_rng.seed(self.cf_seed_offset + epoch_idx)

    def pre_epoch_processing(self):
        self.cf_stats = self._new_cf_stats()

    def post_epoch_processing(self):
        if not self.cf_log_stats:
            return None

        used = max(self.cf_stats['used'], 1)
        return (
            'boundary regularization: epoch={epoch}, warmup_epochs={warmup}, '
            'lambda_b={lambda_b:.6f}, tau={tau:.6f}, gamma={gamma:.6f}, '
            'samples={samples}, eligible={eligible}, used={used_count}, '
            'boundary_loss={loss:.6f}, margin={margin:.6f}, weight={weight:.6f}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf_warmup_epochs),
            lambda_b=float(self.cf_lambda),
            tau=float(self.cf_temperature),
            gamma=float(self.cf_gamma),
            samples=int(self.cf_stats['samples']),
            eligible=int(self.cf_stats['eligible']),
            used_count=int(self.cf_stats['used']),
            loss=float(self.cf_stats['loss_sum'] / used),
            margin=float(self.cf_stats['margin_sum'] / used),
            weight=float(self.cf_stats['weight_sum'] / used),
        )

    def _is_cf_active(self):
        return (
            self.training
            and self.cf_lambda > 0.0
            and self.cf_user_ratio > 0.0
            and self.current_epoch >= self.cf_warmup_epochs
        )

    def _calculate_rec_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        return -torch.mean(
            torch.log2(torch.sigmoid(pos_scores - neg_scores))
        )

    def calculate_loss(self, interaction):
        loss_rec = self._calculate_rec_loss(interaction)
        if not self._is_cf_active():
            self.result_embed = None
            return loss_rec

        loss_boundary = self._calculate_boundary_loss(
            interaction,
            loss_rec
        )
        weighted_boundary = self.cf_lambda * loss_boundary
        self.result_embed = None
        return loss_rec, weighted_boundary

    def _sample_cf_users(self, interaction):
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.cf_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(sample_count, self.cf_batch_size, len(users))
        return self._cf_rng.sample(users, sample_count)

    def _calculate_boundary_loss(self, interaction, reference_loss):
        """Regularize pseudo-positive margins in the current graph only."""
        sampled_users = self._sample_cf_users(interaction)
        self.cf_stats['samples'] += len(sampled_users)
        if not sampled_users:
            return reference_loss * 0.0

        if self.result_embed is None:
            return reference_loss * 0.0

        boundary_losses = []
        for user_id in sampled_users:
            seen_items = self.user_seen_items[int(user_id)]
            if len(seen_items) < self.cf_min_history:
                continue

            self.cf_stats['eligible'] += 1
            pseudo_item_id = self._cf_rng.choice(seen_items)
            scores = self._score_user_items(
                self.result_embed,
                user_id
            ).clone()
            self._mask_remaining_history(
                scores,
                user_id,
                pseudo_item_id
            )
            boundary_item_id = self._select_boundary_competitor(
                scores,
                pseudo_item_id
            )
            if boundary_item_id is None:
                continue

            pseudo_score = scores[int(pseudo_item_id)]
            boundary_score = scores[int(boundary_item_id)]
            margin = pseudo_score - boundary_score
            weight = torch.exp(-self.cf_temperature * torch.abs(margin))
            if self.cf_detach_boundary_weight:
                weight = weight.detach()
            loss_u = weight * F.softplus(
                boundary_score - pseudo_score + self.cf_gamma
            )
            boundary_losses.append(loss_u)

            with torch.no_grad():
                self.cf_stats['used'] += 1
                self.cf_stats['loss_sum'] += float(loss_u.detach().cpu())
                self.cf_stats['margin_sum'] += float(margin.detach().cpu())
                self.cf_stats['weight_sum'] += float(weight.detach().cpu())

        if not boundary_losses:
            return reference_loss * 0.0
        return torch.stack(boundary_losses).mean()

    def _score_user_items(self, embedding, user_id):
        user_vector = embedding[int(user_id)]
        item_matrix = embedding[self.num_user:]
        return torch.matmul(item_matrix, user_vector)

    def _mask_remaining_history(self, scores, user_id, pseudo_item_id):
        seen_items = [
            item_id for item_id in self.user_seen_items[int(user_id)]
            if item_id != int(pseudo_item_id)
        ]
        if not seen_items:
            return

        seen_tensor = torch.tensor(
            seen_items,
            dtype=torch.long,
            device=scores.device
        )
        scores[seen_tensor] = torch.finfo(scores.dtype).min

    def _select_boundary_competitor(self, scores, pseudo_item_id):
        """Select the fixed Top-K competitor, excluding the pseudo-positive."""
        top_count = min(self.num_item, self.cf_k + 1)
        if top_count <= 0 or self.cf_k > top_count:
            return None

        _, ranked_items = torch.topk(scores.detach(), k=top_count, dim=0)
        valid_floor = torch.finfo(scores.dtype).min / 2.0
        for index in range(self.cf_k - 1, top_count):
            item_id = int(ranked_items[index].item())
            if item_id == int(pseudo_item_id):
                continue
            if float(scores[item_id].detach().cpu()) <= valid_floor:
                continue
            return item_id
        return None
