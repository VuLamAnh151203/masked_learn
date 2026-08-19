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


class MASKED_GLORIA_CF5(MASKED_GLORIA):
    """MASKED_GLORIA with broad mask loss + target-aware counterfactual loss.

    Broad loss:
        preserve the existing history-prototype-guided ordering of edge masks.

    Counterfactual loss:
        for each sampled (user, positive, negative) training triple, construct
        a factual target-aware history readout and a counterfactual readout.
        Both use the same history and the same static mask prior; only the
        target-specific relevance is intervened from q(e,p) to -q(e,p).

        The loss encourages the factual-vs-counterfactual effect to be larger
        for the positive target than for the negative target.

    No graph-edge dropping and no full-branch teacher are required.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_CF5, self).__init__(config, dataset)

        self.cf5_lambda = float(_cfg(config, 'cf5_lambda', 0.1))
        self.cf5_temperature = float(
            _cfg(config, 'cf5_temperature', 1.0)
        )
        self.cf5_warmup_ratio = float(
            _cfg(config, 'cf5_warmup_ratio', 0.10)
        )
        configured_warmup_epochs = int(
            _cfg(config, 'cf5_warmup_epochs', 50)
        )
        self.cf5_user_ratio = float(_cfg(config, 'cf5_user_ratio', 0.10))
        self.cf5_batch_size = int(_cfg(config, 'cf5_batch_size', 8))
        self.cf5_pair_count = int(_cfg(config, 'cf5_pair_count', 32))
        self.cf5_min_history = int(_cfg(config, 'cf5_min_history', 2))
        self.cf5_similarity_eps = float(
            _cfg(config, 'cf5_similarity_eps', 1e-6)
        )
        self.cf5_seed_offset = int(
            _cfg(config, 'cf5_seed_offset', 20000)
        )
        self.cf5_log_stats = _cfg_bool(config, 'cf5_log_stats', True)

        # Target-aware counterfactual readout.
        self.cf5_cf_lambda = float(
            _cfg(config, 'cf5_cf_lambda', 0.005)
        )
        self.cf5_target_temperature = float(
            _cfg(config, 'cf5_target_temperature', 1.0)
        )
        self.cf5_cf_temperature = float(
            _cfg(config, 'cf5_cf_temperature', 1.0)
        )
        self.cf5_cf_margin = float(
            _cfg(config, 'cf5_cf_margin', 0.0)
        )
        self.cf5_cf_user_ratio = float(
            _cfg(config, 'cf5_cf_user_ratio', self.cf5_user_ratio)
        )
        self.cf5_cf_batch_size = int(
            _cfg(config, 'cf5_cf_batch_size', self.cf5_batch_size)
        )
        self.cf5_cf_use_mask_prior = _cfg_bool(
            config, 'cf5_cf_use_mask_prior', True
        )
        self.cf5_cf_detach_mask_prior = _cfg_bool(
            config, 'cf5_cf_detach_mask_prior', True
        )
        self.cf5_cf_mask_eps = float(
            _cfg(config, 'cf5_cf_mask_eps', 1e-8)
        )

        max_epochs = int(_cfg(config, 'epochs', 1000))
        if configured_warmup_epochs >= 0:
            self.cf5_warmup_epochs = configured_warmup_epochs
        else:
            self.cf5_warmup_epochs = int(
                math.ceil(max_epochs * self.cf5_warmup_ratio)
            )

        self._validate_cf5_config()
        self.current_epoch = 0
        self._cf5_rng = random.Random(self.cf5_seed_offset)
        self.user_to_edge_ids = self._build_cf5_history()
        self.cf5_stats = self._new_cf5_stats()

    def _validate_cf5_config(self):
        if self.cf5_lambda < 0.0:
            raise ValueError('cf5_lambda must be non-negative.')
        if self.cf5_temperature <= 0.0:
            raise ValueError('cf5_temperature must be positive.')
        if not 0.0 <= self.cf5_user_ratio <= 1.0:
            raise ValueError('cf5_user_ratio must be in [0, 1].')
        if self.cf5_batch_size <= 0:
            raise ValueError('cf5_batch_size must be positive.')
        if self.cf5_pair_count <= 0:
            raise ValueError('cf5_pair_count must be positive.')
        if self.cf5_min_history < 2:
            raise ValueError('cf5_min_history must be at least 2.')
        if self.cf5_similarity_eps < 0.0:
            raise ValueError('cf5_similarity_eps must be non-negative.')
        if self.cf5_warmup_epochs < 0:
            raise ValueError('cf5_warmup_epochs must be non-negative.')
        if self.cf5_cf_lambda < 0.0:
            raise ValueError('cf5_cf_lambda must be non-negative.')
        if self.cf5_target_temperature <= 0.0:
            raise ValueError('cf5_target_temperature must be positive.')
        if self.cf5_cf_temperature <= 0.0:
            raise ValueError('cf5_cf_temperature must be positive.')
        if self.cf5_cf_margin < 0.0:
            raise ValueError('cf5_cf_margin must be non-negative.')
        if not 0.0 <= self.cf5_cf_user_ratio <= 1.0:
            raise ValueError('cf5_cf_user_ratio must be in [0, 1].')
        if self.cf5_cf_batch_size <= 0:
            raise ValueError('cf5_cf_batch_size must be positive.')
        if self.cf5_cf_mask_eps <= 0.0:
            raise ValueError('cf5_cf_mask_eps must be positive.')

    def _build_cf5_history(self):
        user_to_edge_ids = [[] for _ in range(self.num_user)]
        edge_users = self.forward_edge_users.detach().cpu().tolist()
        for edge_id, user_id in enumerate(edge_users):
            user_to_edge_ids[int(user_id)].append(int(edge_id))
        return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

    @staticmethod
    def _new_cf5_stats():
        return {
            'samples': 0,
            'eligible': 0,
            'pairs': 0,
            'used': 0,
            'loss_sum': 0.0,
            'similarity_gap_sum': 0.0,
            'cf_samples': 0,
            'cf_eligible': 0,
            'cf_used': 0,
            'cf_loss_sum': 0.0,
            'cf_pos_effect_sum': 0.0,
            'cf_neg_effect_sum': 0.0,
            'cf_effect_gap_sum': 0.0,
            'cf_pos_fact_sum': 0.0,
            'cf_pos_counter_sum': 0.0,
            'cf_neg_fact_sum': 0.0,
            'cf_neg_counter_sum': 0.0,
        }

    def set_training_epoch(self, epoch_idx):
        epoch_idx = int(epoch_idx)
        if epoch_idx < 0:
            raise ValueError('epoch_idx must be non-negative.')
        self.current_epoch = epoch_idx
        self._cf5_rng.seed(self.cf5_seed_offset + epoch_idx)

    def pre_epoch_processing(self):
        self.cf5_stats = self._new_cf5_stats()

    def post_epoch_processing(self):
        if not self.cf5_log_stats:
            return None

        used = max(self.cf5_stats['used'], 1)
        cf_used = max(self.cf5_stats['cf_used'], 1)
        return (
            'broad-mask + target-aware counterfactual: '
            'epoch={epoch}, warmup_epochs={warmup}, '
            'broad_lambda={lambda_cf5:.6f}, '
            'broad_temperature={temperature:.6f}, '
            'broad_samples={samples}, broad_eligible={eligible}, '
            'broad_pairs={pairs}, broad_used={used_count}, '
            'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
            'cf_lambda={cf_lambda:.6f}, '
            'target_temperature={target_temperature:.6f}, '
            'cf_temperature={cf_temperature:.6f}, '
            'cf_margin={cf_margin:.6f}, '
            'cf_samples={cf_samples}, cf_eligible={cf_eligible}, '
            'cf_used={cf_used_count}, cf_loss={cf_loss:.6f}, '
            'cf_pos_effect={cf_pos_effect:.6f}, '
            'cf_neg_effect={cf_neg_effect:.6f}, '
            'cf_effect_gap={cf_effect_gap:.6f}, '
            'pos_fact={pos_fact:.6f}, pos_counter={pos_counter:.6f}, '
            'neg_fact={neg_fact:.6f}, neg_counter={neg_counter:.6f}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf5_warmup_epochs),
            lambda_cf5=float(self.cf5_lambda),
            temperature=float(self.cf5_temperature),
            samples=int(self.cf5_stats['samples']),
            eligible=int(self.cf5_stats['eligible']),
            pairs=int(self.cf5_stats['pairs']),
            used_count=int(self.cf5_stats['used']),
            loss=float(self.cf5_stats['loss_sum'] / used),
            gap=float(self.cf5_stats['similarity_gap_sum'] / used),
            cf_lambda=float(self.cf5_cf_lambda),
            target_temperature=float(self.cf5_target_temperature),
            cf_temperature=float(self.cf5_cf_temperature),
            cf_margin=float(self.cf5_cf_margin),
            cf_samples=int(self.cf5_stats['cf_samples']),
            cf_eligible=int(self.cf5_stats['cf_eligible']),
            cf_used_count=int(self.cf5_stats['cf_used']),
            cf_loss=float(self.cf5_stats['cf_loss_sum'] / cf_used),
            cf_pos_effect=float(self.cf5_stats['cf_pos_effect_sum'] / cf_used),
            cf_neg_effect=float(self.cf5_stats['cf_neg_effect_sum'] / cf_used),
            cf_effect_gap=float(self.cf5_stats['cf_effect_gap_sum'] / cf_used),
            pos_fact=float(self.cf5_stats['cf_pos_fact_sum'] / cf_used),
            pos_counter=float(self.cf5_stats['cf_pos_counter_sum'] / cf_used),
            neg_fact=float(self.cf5_stats['cf_neg_fact_sum'] / cf_used),
            neg_counter=float(self.cf5_stats['cf_neg_counter_sum'] / cf_used),
        )

    def _is_cf5_active(self):
        return (
            self.training
            and (self.cf5_lambda > 0.0 or self.cf5_cf_lambda > 0.0)
            and (self.cf5_user_ratio > 0.0 or self.cf5_cf_user_ratio > 0.0)
            and self.current_epoch >= self.cf5_warmup_epochs
        )

    def _calculate_rec_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        return -torch.mean(
            torch.log2(torch.sigmoid(pos_scores - neg_scores))
        )

    def calculate_loss(self, interaction):
        loss_rec = self._calculate_rec_loss(interaction)
        if not self._is_cf5_active():
            self.result_embed = None
            return loss_rec

        if self.cf5_lambda > 0.0:
            loss_mask_relation = self._calculate_mask_relation_loss(
                interaction,
                loss_rec
            )
        else:
            loss_mask_relation = loss_rec * 0.0

        if self.cf5_cf_lambda > 0.0:
            loss_counterfactual = self._calculate_counterfactual_loss(
                interaction,
                loss_rec
            )
        else:
            loss_counterfactual = loss_rec * 0.0

        weighted_auxiliary = (
            self.cf5_lambda * loss_mask_relation
            + self.cf5_cf_lambda * loss_counterfactual
        )
        self.result_embed = None
        return loss_rec, weighted_auxiliary

    def _sample_cf5_users(self, interaction):
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.cf5_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(sample_count, self.cf5_batch_size, len(users))
        return self._cf5_rng.sample(users, sample_count)

    def _sample_cf5_pairs(self, history_size):
        total_pairs = history_size * (history_size - 1) // 2
        if total_pairs <= self.cf5_pair_count:
            return list(itertools.combinations(range(history_size), 2))

        pairs = set()
        while len(pairs) < self.cf5_pair_count:
            left = self._cf5_rng.randrange(history_size)
            right = self._cf5_rng.randrange(history_size)
            if left == right:
                continue
            pairs.add(tuple(sorted((left, right))))
        return list(pairs)

    def _calculate_mask_relation_loss(self, interaction, reference_loss):
        """Apply pairwise similarity ordering to current mask weights."""
        sampled_users = self._sample_cf5_users(interaction)
        self.cf5_stats['samples'] += len(sampled_users)
        if not sampled_users or self.result_embed is None:
            return reference_loss * 0.0

        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()
        losses = []

        for user_id in sampled_users:
            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf5_min_history:
                continue

            self.cf5_stats['eligible'] += 1
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

            pair_positions = self._sample_cf5_pairs(len(edge_ids))

            for left_pos, right_pos in pair_positions:
                relevance_gap = (
                    relevance[left_pos] - relevance[right_pos]
                )
                if abs(float(relevance_gap.detach().cpu())) <= self.cf5_similarity_eps:
                    continue

                left_edge = int(edge_ids[left_pos])
                right_edge = int(edge_ids[right_pos])
                mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
                direction = torch.sign(relevance_gap)
                pair_loss = F.softplus(
                    -self.cf5_temperature * direction * mask_gap
                )
                losses.append(pair_loss)

                with torch.no_grad():
                    self.cf5_stats['pairs'] += 1
                    self.cf5_stats['used'] += 1
                    self.cf5_stats['loss_sum'] += float(
                        pair_loss.detach().cpu()
                    )
                    self.cf5_stats['similarity_gap_sum'] += float(
                        relevance_gap.abs().detach().cpu()
                    )

        if not losses:
            return reference_loss * 0.0
        return torch.stack(losses).mean()

    def _sample_cf_users(self, interaction):
        """Sample users for target-aware counterfactual supervision."""
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.cf5_cf_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(sample_count, self.cf5_cf_batch_size, len(users))
        return self._cf5_rng.sample(users, sample_count)

    @staticmethod
    def _build_batch_pos_neg_targets(interaction):
        """Map each user to its (positive, negative) pairs in the batch.

        Assumes the standard triplet layout:
            interaction[0] -> user ids
            interaction[1] -> positive item ids
            interaction[2] -> negative item ids
        """
        if interaction is None or len(interaction) < 3:
            return {}

        users = interaction[0].detach().view(-1).cpu().tolist()
        positives = interaction[1].detach().view(-1).cpu().tolist()
        negatives = interaction[2].detach().view(-1).cpu().tolist()

        if not (len(users) == len(positives) == len(negatives)):
            return {}

        mapping = {}
        for user_id, pos_item, neg_item in zip(users, positives, negatives):
            mapping.setdefault(int(user_id), []).append(
                (int(pos_item), int(neg_item))
            )
        return mapping

    def _target_aware_effect(
        self,
        history_rep,
        target_rep,
        history_mask_weights=None
    ):
        """Compute factual score, counterfactual score, and their effect.

        The factual view emphasizes history items similar to the target.
        The counterfactual view reverses only target-specific relevance q -> -q.
        Static mask prior is identical in both views.
        """
        target_similarity = F.cosine_similarity(
            history_rep,
            target_rep.unsqueeze(0),
            dim=1
        )

        factual_logits = target_similarity / self.cf5_target_temperature
        counter_logits = -target_similarity / self.cf5_target_temperature

        if self.cf5_cf_use_mask_prior and history_mask_weights is not None:
            mask_prior = history_mask_weights.clamp_min(self.cf5_cf_mask_eps)
            if self.cf5_cf_detach_mask_prior:
                mask_prior = mask_prior.detach()
            log_mask_prior = torch.log(mask_prior)
            factual_logits = factual_logits + log_mask_prior
            counter_logits = counter_logits + log_mask_prior

        factual_attention = torch.softmax(factual_logits, dim=0)
        counter_attention = torch.softmax(counter_logits, dim=0)

        factual_history = torch.sum(
            factual_attention.unsqueeze(-1) * history_rep,
            dim=0
        )
        counter_history = torch.sum(
            counter_attention.unsqueeze(-1) * history_rep,
            dim=0
        )

        factual_score = F.cosine_similarity(
            factual_history.unsqueeze(0),
            target_rep.unsqueeze(0),
            dim=1
        ).squeeze(0)
        counter_score = F.cosine_similarity(
            counter_history.unsqueeze(0),
            target_rep.unsqueeze(0),
            dim=1
        ).squeeze(0)

        effect = factual_score - counter_score
        return factual_score, counter_score, effect

    def _calculate_counterfactual_loss(self, interaction, reference_loss):
        """Counterfactual target-aware ranking loss.

        delta_pos = s_fact(u,p+) - s_cf(u,p+)
        delta_neg = s_fact(u,p-) - s_cf(u,p-)

        Objective:
            delta_pos > delta_neg + margin

        L_cf = softplus(
            (delta_neg - delta_pos + margin) / temperature
        )
        """
        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        target_pairs_by_user = self._build_batch_pos_neg_targets(interaction)
        if not target_pairs_by_user:
            return reference_loss * 0.0

        sampled_users = self._sample_cf_users(interaction)
        self.cf5_stats['cf_samples'] += len(sampled_users)
        if not sampled_users:
            return reference_loss * 0.0

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()
        losses = []

        for user_id in sampled_users:
            user_pairs = target_pairs_by_user.get(int(user_id), [])
            if not user_pairs:
                continue

            pos_item, neg_item = self._cf5_rng.choice(user_pairs)
            if (
                pos_item < 0
                or neg_item < 0
                or pos_item >= item_rep.size(0)
                or neg_item >= item_rep.size(0)
            ):
                continue

            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf5_min_history:
                continue

            edge_tensor = torch.tensor(
                edge_ids,
                dtype=torch.long,
                device=self.forward_edge_users.device
            )
            history_item_ids = self.forward_edge_items[edge_tensor]

            # Same history for positive and negative; remove either target if
            # already present to avoid trivial cos(z_p, z_p) = 1.
            keep = (
                (history_item_ids != pos_item)
                & (history_item_ids != neg_item)
            )
            history_item_ids = history_item_ids[keep]
            history_edge_tensor = edge_tensor[keep]

            if history_item_ids.numel() < self.cf5_min_history:
                continue

            self.cf5_stats['cf_eligible'] += 1

            history_rep = item_rep[history_item_ids]
            history_mask = mask_weights[history_edge_tensor]
            pos_target_rep = item_rep[pos_item]
            neg_target_rep = item_rep[neg_item]

            pos_fact, pos_counter, pos_effect = self._target_aware_effect(
                history_rep,
                pos_target_rep,
                history_mask
            )
            neg_fact, neg_counter, neg_effect = self._target_aware_effect(
                history_rep,
                neg_target_rep,
                history_mask
            )

            effect_gap = pos_effect - neg_effect
            user_loss = F.softplus(
                (
                    neg_effect
                    - pos_effect
                    + self.cf5_cf_margin
                ) / self.cf5_cf_temperature
            )
            losses.append(user_loss)

            with torch.no_grad():
                self.cf5_stats['cf_used'] += 1
                self.cf5_stats['cf_loss_sum'] += float(user_loss.detach().cpu())
                self.cf5_stats['cf_pos_effect_sum'] += float(pos_effect.detach().cpu())
                self.cf5_stats['cf_neg_effect_sum'] += float(neg_effect.detach().cpu())
                self.cf5_stats['cf_effect_gap_sum'] += float(effect_gap.detach().cpu())
                self.cf5_stats['cf_pos_fact_sum'] += float(pos_fact.detach().cpu())
                self.cf5_stats['cf_pos_counter_sum'] += float(pos_counter.detach().cpu())
                self.cf5_stats['cf_neg_fact_sum'] += float(neg_fact.detach().cpu())
                self.cf5_stats['cf_neg_counter_sum'] += float(neg_counter.detach().cpu())

        if not losses:
            return reference_loss * 0.0

        return torch.stack(losses).mean()