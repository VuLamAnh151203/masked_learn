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


class MASKED_GLORIA_CF3(MASKED_GLORIA):
    """MASKED_GLORIA with broad mask regularization + hard-negative ranking.

    Broad mask regularizer:
        history item -> similarity to the user's history prototype -> mask order.

    Hard-negative ranking regularizer:
        for a sampled user/positive pair, sample unseen candidate items, select
        the highest-scoring candidates under the current final representation,
        and explicitly push the positive above those local hard competitors.

    No graph edge is dropped by either auxiliary objective.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_CF3, self).__init__(config, dataset)

        self.cf3_lambda = float(_cfg(config, 'cf3_lambda', 0.1))
        self.cf3_temperature = float(
            _cfg(config, 'cf3_temperature', 1.0)
        )
        self.cf3_warmup_ratio = float(
            _cfg(config, 'cf3_warmup_ratio', 0.10)
        )
        configured_warmup_epochs = int(
            _cfg(config, 'cf3_warmup_epochs', 50)
        )
        self.cf3_user_ratio = float(_cfg(config, 'cf3_user_ratio', 0.10))
        self.cf3_batch_size = int(_cfg(config, 'cf3_batch_size', 8))
        self.cf3_pair_count = int(_cfg(config, 'cf3_pair_count', 32))
        self.cf3_min_history = int(_cfg(config, 'cf3_min_history', 2))
        self.cf3_similarity_eps = float(
            _cfg(config, 'cf3_similarity_eps', 1e-6)
        )
        self.cf3_seed_offset = int(
            _cfg(config, 'cf3_seed_offset', 20000)
        )
        self.cf3_log_stats = _cfg_bool(config, 'cf3_log_stats', True)

        # Hard-negative local-ranking loss.
        # Start small because the broad mask loss already improves Recall@15/20.
        self.hard_lambda = float(_cfg(config, 'hard_lambda', 0.01))
        self.hard_temperature = float(
            _cfg(config, 'hard_temperature', 1.0)
        )
        self.hard_margin = float(_cfg(config, 'hard_margin', 0.0))
        self.hard_candidate_pool = int(
            _cfg(config, 'hard_candidate_pool', 256)
        )
        self.hard_topk = int(_cfg(config, 'hard_topk', 10))
        self.hard_user_ratio = float(
            _cfg(config, 'hard_user_ratio', self.cf3_user_ratio)
        )
        self.hard_batch_size = int(
            _cfg(config, 'hard_batch_size', self.cf3_batch_size)
        )

        max_epochs = int(_cfg(config, 'epochs', 1000))
        if configured_warmup_epochs >= 0:
            self.cf3_warmup_epochs = configured_warmup_epochs
        else:
            self.cf3_warmup_epochs = int(
                math.ceil(max_epochs * self.cf3_warmup_ratio)
            )

        self._validate_cf3_config()
        self.current_epoch = 0
        self._cf3_rng = random.Random(self.cf3_seed_offset)
        self.user_to_edge_ids = self._build_cf3_history()
        self.cf3_stats = self._new_cf3_stats()

    def _validate_cf3_config(self):
        if self.cf3_lambda < 0.0:
            raise ValueError('cf3_lambda must be non-negative.')
        if self.cf3_temperature <= 0.0:
            raise ValueError('cf3_temperature must be positive.')
        if not 0.0 <= self.cf3_user_ratio <= 1.0:
            raise ValueError('cf3_user_ratio must be in [0, 1].')
        if self.cf3_batch_size <= 0:
            raise ValueError('cf3_batch_size must be positive.')
        if self.cf3_pair_count <= 0:
            raise ValueError('cf3_pair_count must be positive.')
        if self.cf3_min_history < 2:
            raise ValueError('cf3_min_history must be at least 2.')
        if self.cf3_similarity_eps < 0.0:
            raise ValueError('cf3_similarity_eps must be non-negative.')
        if self.cf3_warmup_epochs < 0:
            raise ValueError('cf3_warmup_epochs must be non-negative.')

        if self.hard_lambda < 0.0:
            raise ValueError('hard_lambda must be non-negative.')
        if self.hard_temperature <= 0.0:
            raise ValueError('hard_temperature must be positive.')
        if self.hard_margin < 0.0:
            raise ValueError('hard_margin must be non-negative.')
        if self.hard_candidate_pool <= 0:
            raise ValueError('hard_candidate_pool must be positive.')
        if self.hard_topk <= 0:
            raise ValueError('hard_topk must be positive.')
        if not 0.0 <= self.hard_user_ratio <= 1.0:
            raise ValueError('hard_user_ratio must be in [0, 1].')
        if self.hard_batch_size <= 0:
            raise ValueError('hard_batch_size must be positive.')

    def _build_cf3_history(self):
        user_to_edge_ids = [[] for _ in range(self.num_user)]
        edge_users = self.forward_edge_users.detach().cpu().tolist()
        for edge_id, user_id in enumerate(edge_users):
            user_to_edge_ids[int(user_id)].append(int(edge_id))
        return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

    @staticmethod
    def _new_cf3_stats():
        return {
            # Broad/history-prototype mask regularization.
            'samples': 0,
            'eligible': 0,
            'pairs': 0,
            'used': 0,
            'loss_sum': 0.0,
            'similarity_gap_sum': 0.0,

            # Hard-negative local ranking.
            'hard_samples': 0,
            'hard_eligible': 0,
            'hard_used': 0,
            'hard_loss_sum': 0.0,
            'hard_pos_score_sum': 0.0,
            'hard_neg_score_sum': 0.0,
            'hard_gap_sum': 0.0,
        }

    def set_training_epoch(self, epoch_idx):
        epoch_idx = int(epoch_idx)
        if epoch_idx < 0:
            raise ValueError('epoch_idx must be non-negative.')
        self.current_epoch = epoch_idx
        self._cf3_rng.seed(self.cf3_seed_offset + epoch_idx)

    def pre_epoch_processing(self):
        self.cf3_stats = self._new_cf3_stats()

    def post_epoch_processing(self):
        if not self.cf3_log_stats:
            return None

        used = max(self.cf3_stats['used'], 1)
        hard_used = max(self.cf3_stats['hard_used'], 1)

        return (
            'mask-representation + hard-negative regularization: '
            'epoch={epoch}, warmup_epochs={warmup}, '
            'broad_lambda={lambda_cf3:.6f}, '
            'broad_temperature={temperature:.6f}, '
            'broad_samples={samples}, broad_eligible={eligible}, '
            'broad_pairs={pairs}, broad_used={used_count}, '
            'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
            'hard_lambda={hard_lambda:.6f}, '
            'hard_temperature={hard_temperature:.6f}, '
            'hard_margin={hard_margin:.6f}, '
            'hard_pool={hard_pool}, hard_topk={hard_topk}, '
            'hard_samples={hard_samples}, hard_eligible={hard_eligible}, '
            'hard_used={hard_used_count}, hard_loss={hard_loss:.6f}, '
            'hard_pos_score={hard_pos:.6f}, '
            'hard_neg_score={hard_neg:.6f}, '
            'hard_pos_minus_neg={hard_gap:.6f}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf3_warmup_epochs),

            lambda_cf3=float(self.cf3_lambda),
            temperature=float(self.cf3_temperature),
            samples=int(self.cf3_stats['samples']),
            eligible=int(self.cf3_stats['eligible']),
            pairs=int(self.cf3_stats['pairs']),
            used_count=int(self.cf3_stats['used']),
            loss=float(self.cf3_stats['loss_sum'] / used),
            gap=float(self.cf3_stats['similarity_gap_sum'] / used),

            hard_lambda=float(self.hard_lambda),
            hard_temperature=float(self.hard_temperature),
            hard_margin=float(self.hard_margin),
            hard_pool=int(self.hard_candidate_pool),
            hard_topk=int(self.hard_topk),
            hard_samples=int(self.cf3_stats['hard_samples']),
            hard_eligible=int(self.cf3_stats['hard_eligible']),
            hard_used_count=int(self.cf3_stats['hard_used']),
            hard_loss=float(self.cf3_stats['hard_loss_sum'] / hard_used),
            hard_pos=float(
                self.cf3_stats['hard_pos_score_sum'] / hard_used
            ),
            hard_neg=float(
                self.cf3_stats['hard_neg_score_sum'] / hard_used
            ),
            hard_gap=float(
                self.cf3_stats['hard_gap_sum'] / hard_used
            ),
        )

    def _is_cf3_active(self):
        return (
            self.training
            and (self.cf3_lambda > 0.0 or self.hard_lambda > 0.0)
            and (
                self.cf3_user_ratio > 0.0
                or self.hard_user_ratio > 0.0
            )
            and self.current_epoch >= self.cf3_warmup_epochs
        )

    def _calculate_rec_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        return -torch.mean(
            torch.log2(torch.sigmoid(pos_scores - neg_scores))
        )

    def calculate_loss(self, interaction):
        loss_rec = self._calculate_rec_loss(interaction)
        if not self._is_cf3_active():
            self.result_embed = None
            return loss_rec

        if self.cf3_lambda > 0.0:
            loss_mask_relation = self._calculate_mask_relation_loss(
                interaction,
                loss_rec
            )
        else:
            loss_mask_relation = loss_rec * 0.0

        if self.hard_lambda > 0.0:
            loss_hard_negative = self._calculate_hard_negative_loss(
                interaction,
                loss_rec
            )
        else:
            loss_hard_negative = loss_rec * 0.0

        weighted_auxiliary = (
            self.cf3_lambda * loss_mask_relation
            + self.hard_lambda * loss_hard_negative
        )

        self.result_embed = None

        # Preserve the old trainer API:
        # trainer can keep summing the returned loss terms.
        return loss_rec, weighted_auxiliary

    def _sample_cf3_users(self, interaction):
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.cf3_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(sample_count, self.cf3_batch_size, len(users))
        return self._cf3_rng.sample(users, sample_count)

    def _sample_cf3_pairs(self, history_size):
        total_pairs = history_size * (history_size - 1) // 2
        if total_pairs <= self.cf3_pair_count:
            return list(itertools.combinations(range(history_size), 2))

        pairs = set()
        while len(pairs) < self.cf3_pair_count:
            left = self._cf3_rng.randrange(history_size)
            right = self._cf3_rng.randrange(history_size)
            if left == right:
                continue
            pairs.add(tuple(sorted((left, right))))
        return list(pairs)

    def _calculate_mask_relation_loss(self, interaction, reference_loss):
        """Apply pairwise similarity ordering to current mask weights."""
        sampled_users = self._sample_cf3_users(interaction)
        self.cf3_stats['samples'] += len(sampled_users)
        if not sampled_users or self.result_embed is None:
            return reference_loss * 0.0

        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()
        losses = []

        for user_id in sampled_users:
            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf3_min_history:
                continue

            self.cf3_stats['eligible'] += 1
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

            pair_positions = self._sample_cf3_pairs(len(edge_ids))

            for left_pos, right_pos in pair_positions:
                relevance_gap = (
                    relevance[left_pos] - relevance[right_pos]
                )
                if abs(float(relevance_gap.detach().cpu())) <= self.cf3_similarity_eps:
                    continue

                left_edge = int(edge_ids[left_pos])
                right_edge = int(edge_ids[right_pos])
                mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
                direction = torch.sign(relevance_gap)
                pair_loss = F.softplus(
                    -self.cf3_temperature * direction * mask_gap
                )
                losses.append(pair_loss)

                with torch.no_grad():
                    self.cf3_stats['pairs'] += 1
                    self.cf3_stats['used'] += 1
                    self.cf3_stats['loss_sum'] += float(
                        pair_loss.detach().cpu()
                    )
                    self.cf3_stats['similarity_gap_sum'] += float(
                        relevance_gap.abs().detach().cpu()
                    )

        if not losses:
            return reference_loss * 0.0
        return torch.stack(losses).mean()

    def _sample_hard_users(self, interaction):
        """Sample users for the hard-negative auxiliary objective."""
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.hard_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(
            sample_count,
            self.hard_batch_size,
            len(users)
        )
        return self._cf3_rng.sample(users, sample_count)

    @staticmethod
    def _build_batch_positive_targets(interaction):
        """Map each mini-batch user to the positive item ids in that batch.

        This follows the tuple layout already implied by _calculate_rec_loss:
            interaction[0] = user ids
            interaction[1] = positive item ids
        """
        if interaction is None or len(interaction) < 2:
            return {}

        users = interaction[0].detach().view(-1).cpu().tolist()
        positives = interaction[1].detach().view(-1).cpu().tolist()

        if len(users) != len(positives):
            return {}

        mapping = {}
        for user_id, item_id in zip(users, positives):
            mapping.setdefault(int(user_id), []).append(int(item_id))
        return mapping

    def _user_seen_item_set(self, user_id):
        """Return training items already connected to the user."""
        edge_ids = self.user_to_edge_ids[int(user_id)]
        if not edge_ids:
            return set()

        edge_tensor = torch.tensor(
            edge_ids,
            dtype=torch.long,
            device=self.forward_edge_items.device
        )
        return set(
            int(item_id)
            for item_id in self.forward_edge_items[edge_tensor]
            .detach().cpu().tolist()
        )

    def _sample_unseen_candidates(
        self,
        num_items,
        excluded_items,
        candidate_count
    ):
        """Sample item ids not present in excluded_items.

        Uses Python's range-backed random.sample so it does not materialize the
        whole item catalog. A modest oversampling loop handles filtered items.
        """
        candidate_count = int(min(candidate_count, num_items))
        if candidate_count <= 0:
            return []

        excluded_items = {
            int(item_id)
            for item_id in excluded_items
            if 0 <= int(item_id) < num_items
        }

        available_count = num_items - len(excluded_items)
        target_count = min(candidate_count, available_count)
        if target_count <= 0:
            return []

        selected = set()
        # Keep drawing modest random blocks until enough unseen items are found.
        # This is efficient for the normal recommendation regime where each
        # user's history is much smaller than the full item catalog.
        max_rounds = 20
        for _ in range(max_rounds):
            if len(selected) >= target_count:
                break

            need = target_count - len(selected)
            draw_count = min(
                num_items,
                max(need * 2, need + 16)
            )
            drawn = self._cf3_rng.sample(range(num_items), draw_count)

            for item_id in drawn:
                if item_id in excluded_items or item_id in selected:
                    continue
                selected.add(int(item_id))
                if len(selected) >= target_count:
                    break

        # Rare fallback for very dense users / small item catalogs.
        if len(selected) < target_count:
            for item_id in range(num_items):
                if item_id in excluded_items or item_id in selected:
                    continue
                selected.add(int(item_id))
                if len(selected) >= target_count:
                    break

        return list(selected)

    def _calculate_hard_negative_loss(self, interaction, reference_loss):
        """Local ranking loss against current high-scoring unseen negatives.

        For each sampled user:
          1) choose one positive item from the current training mini-batch;
          2) sample a pool of unseen candidate items;
          3) score the pool with the CURRENT FINAL representation;
          4) take the top-scoring candidates as hard negatives;
          5) push the positive above those hard negatives.

        The top-k selection is performed on detached scores, but the selected
        positive/negative scores retain gradients. Therefore the auxiliary loss
        trains the final user/item representation (and any upstream parameters
        that produced it), rather than supervising mask ordering directly.
        """
        if self.result_embed is None:
            return reference_loss * 0.0

        if not torch.is_tensor(self.result_embed):
            return reference_loss * 0.0

        if self.result_embed.dim() != 2:
            return reference_loss * 0.0

        # result_embed is expected to follow the same packed node layout used
        # throughout this model: [users ; items].
        if self.result_embed.size(0) <= self.num_user:
            return reference_loss * 0.0

        user_rep = self.result_embed[:self.num_user]
        item_rep = self.result_embed[self.num_user:]
        num_items = int(item_rep.size(0))

        positives_by_user = self._build_batch_positive_targets(interaction)
        sampled_users = self._sample_hard_users(interaction)

        self.cf3_stats['hard_samples'] += len(sampled_users)

        if not sampled_users or not positives_by_user:
            return reference_loss * 0.0

        losses = []

        for user_id in sampled_users:
            positive_candidates = positives_by_user.get(int(user_id), [])
            if not positive_candidates:
                continue

            # One target per sampled user keeps the auxiliary branch cheap.
            pos_item = int(self._cf3_rng.choice(positive_candidates))
            if pos_item < 0 or pos_item >= num_items:
                continue

            seen_items = self._user_seen_item_set(user_id)
            excluded_items = set(seen_items)
            excluded_items.add(pos_item)

            candidate_ids = self._sample_unseen_candidates(
                num_items=num_items,
                excluded_items=excluded_items,
                candidate_count=self.hard_candidate_pool
            )
            if not candidate_ids:
                continue

            candidate_tensor = torch.tensor(
                candidate_ids,
                dtype=torch.long,
                device=item_rep.device
            )

            u_vec = user_rep[int(user_id)]
            pos_score = torch.sum(
                u_vec * item_rep[pos_item],
                dim=-1
            )

            candidate_scores = torch.matmul(
                item_rep[candidate_tensor],
                u_vec
            )

            hard_k = min(
                int(self.hard_topk),
                int(candidate_scores.numel())
            )
            if hard_k <= 0:
                continue

            # Selection is discrete; use detached values only to decide which
            # candidates are hard. Gather the original scores afterwards so
            # gradients still flow through the selected negatives.
            hard_positions = torch.topk(
                candidate_scores.detach(),
                k=hard_k,
                largest=True,
                sorted=False
            ).indices
            hard_scores = candidate_scores[hard_positions]

            # Pairwise local ranking:
            #     s_pos >= s_hard_neg + margin
            pair_losses = F.softplus(
                (
                    hard_scores
                    - pos_score
                    + self.hard_margin
                ) / self.hard_temperature
            )
            user_loss = pair_losses.mean()
            losses.append(user_loss)

            with torch.no_grad():
                mean_hard_score = hard_scores.mean()
                self.cf3_stats['hard_eligible'] += 1
                self.cf3_stats['hard_used'] += 1
                self.cf3_stats['hard_loss_sum'] += float(
                    user_loss.detach().cpu()
                )
                self.cf3_stats['hard_pos_score_sum'] += float(
                    pos_score.detach().cpu()
                )
                self.cf3_stats['hard_neg_score_sum'] += float(
                    mean_hard_score.detach().cpu()
                )
                self.cf3_stats['hard_gap_sum'] += float(
                    (pos_score - mean_hard_score).detach().cpu()
                )

        if not losses:
            return reference_loss * 0.0

        return torch.stack(losses).mean()
