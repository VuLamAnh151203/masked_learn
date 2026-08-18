# coding: utf-8

import itertools
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


class MASKED_GLORIA_CF2(MASKED_GLORIA):
    """MASKED_GLORIA with representation-guided mask regularization.

    The regularizer does not drop or intervene on graph edges. It computes a
    user's relevant-item prototype from the current masked branch, detaches
    edge-relevance similarities, and softly orders edge-mask weights in the
    same direction as those similarities.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_CF2, self).__init__(config, dataset)

        self.cf2_lambda = float(_cfg(config, 'cf2_lambda', 0.1))
        self.cf2_temperature = float(
            _cfg(config, 'cf2_temperature', 1.0)
        )
        self.cf2_warmup_ratio = float(
            _cfg(config, 'cf2_warmup_ratio', 0.10)
        )
        configured_warmup_epochs = int(
            _cfg(config, 'cf2_warmup_epochs', -1)
        )
        self.cf2_user_ratio = float(_cfg(config, 'cf2_user_ratio', 0.10))
        self.cf2_batch_size = int(_cfg(config, 'cf2_batch_size', 8))
        self.cf2_pair_count = int(_cfg(config, 'cf2_pair_count', 32))
        self.cf2_min_history = int(_cfg(config, 'cf2_min_history', 2))
        self.cf2_similarity_eps = float(
            _cfg(config, 'cf2_similarity_eps', 1e-6)
        )
        self.cf2_seed_offset = int(
            _cfg(config, 'cf2_seed_offset', 20000)
        )
        self.cf2_log_stats = _cfg_bool(config, 'cf2_log_stats', True)

        max_epochs = int(_cfg(config, 'epochs', 1000))
        if configured_warmup_epochs >= 0:
            self.cf2_warmup_epochs = configured_warmup_epochs
        else:
            self.cf2_warmup_epochs = int(
                math.ceil(max_epochs * self.cf2_warmup_ratio)
            )

        self._validate_cf2_config()
        self.current_epoch = 0
        self._cf2_rng = random.Random(self.cf2_seed_offset)
        self.user_to_edge_ids = self._build_cf2_history()
        self.cf2_stats = self._new_cf2_stats()

    def _validate_cf2_config(self):
        if self.cf2_lambda < 0.0:
            raise ValueError('cf2_lambda must be non-negative.')
        if self.cf2_temperature <= 0.0:
            raise ValueError('cf2_temperature must be positive.')
        if not 0.0 <= self.cf2_user_ratio <= 1.0:
            raise ValueError('cf2_user_ratio must be in [0, 1].')
        if self.cf2_batch_size <= 0:
            raise ValueError('cf2_batch_size must be positive.')
        if self.cf2_pair_count <= 0:
            raise ValueError('cf2_pair_count must be positive.')
        if self.cf2_min_history < 2:
            raise ValueError('cf2_min_history must be at least 2.')
        if self.cf2_similarity_eps < 0.0:
            raise ValueError('cf2_similarity_eps must be non-negative.')
        if self.cf2_warmup_epochs < 0:
            raise ValueError('cf2_warmup_epochs must be non-negative.')

    def _build_cf2_history(self):
        user_to_edge_ids = [[] for _ in range(self.num_user)]
        edge_users = self.forward_edge_users.detach().cpu().tolist()
        for edge_id, user_id in enumerate(edge_users):
            user_to_edge_ids[int(user_id)].append(int(edge_id))
        return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

    @staticmethod
    def _new_cf2_stats():
        return {
            'samples': 0,
            'eligible': 0,
            'pairs': 0,
            'used': 0,
            'loss_sum': 0.0,
            'similarity_gap_sum': 0.0,
        }

    def set_training_epoch(self, epoch_idx):
        epoch_idx = int(epoch_idx)
        if epoch_idx < 0:
            raise ValueError('epoch_idx must be non-negative.')
        self.current_epoch = epoch_idx
        self._cf2_rng.seed(self.cf2_seed_offset + epoch_idx)

    def pre_epoch_processing(self):
        self.cf2_stats = self._new_cf2_stats()

    def post_epoch_processing(self):
        if not self.cf2_log_stats:
            return None

        used = max(self.cf2_stats['used'], 1)
        return (
            'mask-representation regularization: epoch={epoch}, '
            'warmup_epochs={warmup}, lambda={lambda_cf2:.6f}, '
            'temperature={temperature:.6f}, samples={samples}, '
            'eligible={eligible}, pairs={pairs}, used={used_count}, '
            'loss={loss:.6f}, similarity_gap={gap:.6f}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf2_warmup_epochs),
            lambda_cf2=float(self.cf2_lambda),
            temperature=float(self.cf2_temperature),
            samples=int(self.cf2_stats['samples']),
            eligible=int(self.cf2_stats['eligible']),
            pairs=int(self.cf2_stats['pairs']),
            used_count=int(self.cf2_stats['used']),
            loss=float(self.cf2_stats['loss_sum'] / used),
            gap=float(self.cf2_stats['similarity_gap_sum'] / used),
        )

    def _is_cf2_active(self):
        return (
            self.training
            and self.cf2_lambda > 0.0
            and self.cf2_user_ratio > 0.0
            and self.current_epoch >= self.cf2_warmup_epochs
        )

    def _calculate_rec_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        return -torch.mean(
            torch.log2(torch.sigmoid(pos_scores - neg_scores))
        )

    def calculate_loss(self, interaction):
        loss_rec = self._calculate_rec_loss(interaction)
        if not self._is_cf2_active():
            self.result_embed = None
            return loss_rec

        loss_mask_relation = self._calculate_mask_relation_loss(
            interaction,
            loss_rec
        )
        weighted_relation = self.cf2_lambda * loss_mask_relation
        self.result_embed = None
        return loss_rec, weighted_relation

    def _sample_cf2_users(self, interaction):
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.cf2_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(sample_count, self.cf2_batch_size, len(users))
        return self._cf2_rng.sample(users, sample_count)

    def _sample_cf2_pairs(self, history_size):
        total_pairs = history_size * (history_size - 1) // 2
        if total_pairs <= self.cf2_pair_count:
            return list(itertools.combinations(range(history_size), 2))

        pairs = set()
        while len(pairs) < self.cf2_pair_count:
            left = self._cf2_rng.randrange(history_size)
            right = self._cf2_rng.randrange(history_size)
            if left == right:
                continue
            pairs.add(tuple(sorted((left, right))))
        return list(pairs)

    def _calculate_mask_relation_loss(self, interaction, reference_loss):
        """Apply pairwise similarity ordering to current mask weights."""
        sampled_users = self._sample_cf2_users(interaction)
        self.cf2_stats['samples'] += len(sampled_users)
        if not sampled_users or self.result_embed is None:
            return reference_loss * 0.0

        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()
        losses = []

        for user_id in sampled_users:
            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf2_min_history:
                continue

            self.cf2_stats['eligible'] += 1
            edge_tensor = torch.tensor(
                edge_ids,
                dtype=torch.long,
                device=self.forward_edge_users.device
            )
            item_ids = self.forward_edge_items[edge_tensor]
            prototype = item_rep[item_ids].mean(dim=0)
            relevance = F.cosine_similarity(
                item_rep[item_ids],
                prototype.unsqueeze(0),
                dim=1
            ).detach()

            pair_positions = self._sample_cf2_pairs(len(edge_ids))

            for left_pos, right_pos in pair_positions:
                relevance_gap = (
                    relevance[left_pos] - relevance[right_pos]
                )
                if abs(float(relevance_gap.detach().cpu())) <= self.cf2_similarity_eps:
                    continue

                left_edge = int(edge_ids[left_pos])
                right_edge = int(edge_ids[right_pos])
                mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
                direction = torch.sign(relevance_gap)
                pair_loss = F.softplus(
                    -self.cf2_temperature * direction * mask_gap
                )
                losses.append(pair_loss)

                with torch.no_grad():
                    self.cf2_stats['pairs'] += 1
                    self.cf2_stats['used'] += 1
                    self.cf2_stats['loss_sum'] += float(
                        pair_loss.detach().cpu()
                    )
                    self.cf2_stats['similarity_gap_sum'] += float(
                        relevance_gap.abs().detach().cpu()
                    )

        if not losses:
            return reference_loss * 0.0
        return torch.stack(losses).mean()
