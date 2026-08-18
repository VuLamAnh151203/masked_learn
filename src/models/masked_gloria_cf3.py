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
    """MASKED_GLORIA with broad + fine-grained mask regularization.

    Broad regularizer:
        history item -> similarity to the user's history prototype -> mask order.

    Fine-grained regularizer:
        history item -> similarity to a target positive item from the current
        training mini-batch -> mask order.

    Both similarity signals are detached, so these auxiliary losses directly
    supervise the learnable edge masks. They do not directly optimize item
    representations.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_CF3, self).__init__(config, dataset)

        # Existing broad/history-prototype loss.
        self.cf2_lambda = float(_cfg(config, 'cf2_lambda', 0.1))
        self.cf2_temperature = float(
            _cfg(config, 'cf2_temperature', 1.0)
        )

        # New fine-grained/target-specific loss.
        # Keep this smaller than the broad loss at first so that the new
        # objective does not overwrite the behavior that already improves
        # Recall@15/20.
        self.cf2_fine_lambda = float(_cfg(config, 'cf2_fine_lambda', 0.02))
        self.cf2_fine_temperature = float(
            _cfg(config, 'cf2_fine_temperature', 1.0)
        )
        self.cf2_fine_pair_count = int(
            _cfg(config, 'cf2_fine_pair_count', 16)
        )
        self.cf2_warmup_ratio = float(
            _cfg(config, 'cf2_warmup_ratio', 0.10)
        )
        configured_warmup_epochs = int(
            _cfg(config, 'cf2_warmup_epochs', 50)
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
        if self.cf2_fine_lambda < 0.0:
            raise ValueError('cf2_fine_lambda must be non-negative.')
        if self.cf2_fine_temperature <= 0.0:
            raise ValueError('cf2_fine_temperature must be positive.')
        if self.cf2_fine_pair_count <= 0:
            raise ValueError('cf2_fine_pair_count must be positive.')
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
            # Broad/history-prototype stats.
            'samples': 0,
            'eligible': 0,
            'pairs': 0,
            'used': 0,
            'loss_sum': 0.0,
            'similarity_gap_sum': 0.0,

            # Fine-grained/target-specific stats.
            'fine_samples': 0,
            'fine_eligible': 0,
            'fine_pairs': 0,
            'fine_used': 0,
            'fine_loss_sum': 0.0,
            'fine_similarity_gap_sum': 0.0,
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
        fine_used = max(self.cf2_stats['fine_used'], 1)
        return (
            'mask-representation regularization: epoch={epoch}, '
            'warmup_epochs={warmup}, '
            'broad_lambda={lambda_cf2:.6f}, broad_temperature={temperature:.6f}, '
            'broad_samples={samples}, broad_eligible={eligible}, '
            'broad_pairs={pairs}, broad_used={used_count}, '
            'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
            'fine_lambda={fine_lambda:.6f}, fine_temperature={fine_temperature:.6f}, '
            'fine_samples={fine_samples}, fine_eligible={fine_eligible}, '
            'fine_pairs={fine_pairs}, fine_used={fine_used_count}, '
            'fine_loss={fine_loss:.6f}, fine_similarity_gap={fine_gap:.6f}'
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
            fine_lambda=float(self.cf2_fine_lambda),
            fine_temperature=float(self.cf2_fine_temperature),
            fine_samples=int(self.cf2_stats['fine_samples']),
            fine_eligible=int(self.cf2_stats['fine_eligible']),
            fine_pairs=int(self.cf2_stats['fine_pairs']),
            fine_used_count=int(self.cf2_stats['fine_used']),
            fine_loss=float(self.cf2_stats['fine_loss_sum'] / fine_used),
            fine_gap=float(
                self.cf2_stats['fine_similarity_gap_sum'] / fine_used
            ),
        )

    def _is_cf2_active(self):
        return (
            self.training
            and (self.cf2_lambda > 0.0 or self.cf2_fine_lambda > 0.0)
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

        # Keep the broad loss exactly as before.
        if self.cf2_lambda > 0.0:
            loss_mask_relation = self._calculate_mask_relation_loss(
                interaction,
                loss_rec
            )
        else:
            loss_mask_relation = loss_rec * 0.0

        # Add target-specific fine-grained mask supervision.
        if self.cf2_fine_lambda > 0.0:
            loss_fine_relation = self._calculate_fine_mask_relation_loss(
                interaction,
                loss_rec
            )
        else:
            loss_fine_relation = loss_rec * 0.0

        weighted_relation = (
            self.cf2_lambda * loss_mask_relation
            + self.cf2_fine_lambda * loss_fine_relation
        )

        self.result_embed = None

        # Preserve the old two-term return structure so an existing trainer
        # that already handles (loss_rec, auxiliary_loss) does not need change.
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

    def _sample_cf2_pairs(self, history_size, pair_count=None):
        if pair_count is None:
            pair_count = self.cf2_pair_count
        pair_count = int(pair_count)

        total_pairs = history_size * (history_size - 1) // 2
        if total_pairs <= pair_count:
            return list(itertools.combinations(range(history_size), 2))

        pairs = set()
        while len(pairs) < pair_count:
            left = self._cf2_rng.randrange(history_size)
            right = self._cf2_rng.randrange(history_size)
            if left == right:
                continue
            pairs.add(tuple(sorted((left, right))))
        return list(pairs)

    @staticmethod
    def _build_batch_positive_targets(interaction):
        """Map each user in the current mini-batch to its positive item(s).

        This assumes the existing training tuple has the same layout already
        implied by _calculate_rec_loss / forward:
            interaction[0] -> user ids
            interaction[1] -> positive item ids
        """
        if interaction is None or len(interaction) < 2:
            return {}

        users = interaction[0].detach().view(-1).cpu().tolist()
        positives = interaction[1].detach().view(-1).cpu().tolist()
        if len(users) != len(positives):
            return {}

        targets = {}
        for user_id, item_id in zip(users, positives):
            targets.setdefault(int(user_id), []).append(int(item_id))
        return targets

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

    def _calculate_fine_mask_relation_loss(self, interaction, reference_loss):
        """Target-specific pairwise ordering loss for edge masks.

        For a sampled user u:
          1) choose one positive item p from the current training mini-batch;
          2) remove the history edge whose item is p from the candidate set
             (if present), preventing the trivial cos(p, p)=1 supervision;
          3) compute detached cosine similarity between each remaining history
             item representation and the target item representation;
          4) order edge masks according to these target-specific similarities.

        The cosine signal is detached. Therefore this auxiliary loss directly
        updates mask parameters, not item representations.
        """
        target_items_by_user = self._build_batch_positive_targets(interaction)
        if not target_items_by_user:
            return reference_loss * 0.0

        sampled_users = self._sample_cf2_users(interaction)
        self.cf2_stats['fine_samples'] += len(sampled_users)

        if not sampled_users or self.result_embed is None:
            return reference_loss * 0.0

        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()
        losses = []

        for user_id in sampled_users:
            target_candidates = target_items_by_user.get(int(user_id), [])
            if not target_candidates:
                continue

            # One target per sampled user keeps V1 cheap and target-specific.
            target_item = int(self._cf2_rng.choice(target_candidates))
            if target_item < 0 or target_item >= item_rep.size(0):
                continue

            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf2_min_history:
                continue

            edge_tensor = torch.tensor(
                edge_ids,
                dtype=torch.long,
                device=self.forward_edge_users.device
            )
            item_ids = self.forward_edge_items[edge_tensor]

            # Critical: if the positive target itself is one of the user's
            # graph edges, do not let that edge participate in pair ranking.
            # Otherwise its target similarity is trivially 1.
            keep_mask = item_ids != target_item
            candidate_edge_tensor = edge_tensor[keep_mask]
            candidate_item_ids = item_ids[keep_mask]

            if candidate_edge_tensor.numel() < self.cf2_min_history:
                continue

            self.cf2_stats['fine_eligible'] += 1

            # Fine-grained target-specific signal:
            #   edge item <-> current positive target item.
            # Detach the entire signal so it is supervision for masks only.
            with torch.no_grad():
                target_rep = item_rep[target_item]
                relevance = F.cosine_similarity(
                    item_rep[candidate_item_ids],
                    target_rep.unsqueeze(0),
                    dim=1
                )

            pair_positions = self._sample_cf2_pairs(
                int(candidate_edge_tensor.numel()),
                pair_count=self.cf2_fine_pair_count
            )

            for left_pos, right_pos in pair_positions:
                relevance_gap = (
                    relevance[left_pos] - relevance[right_pos]
                )

                if abs(float(relevance_gap.detach().cpu())) <= self.cf2_similarity_eps:
                    continue

                left_edge = int(candidate_edge_tensor[left_pos].item())
                right_edge = int(candidate_edge_tensor[right_pos].item())

                mask_gap = (
                    mask_weights[left_edge]
                    - mask_weights[right_edge]
                )
                direction = torch.sign(relevance_gap)

                # Keep the same pairwise formulation as the working broad
                # loss, so this experiment isolates only the new signal.
                pair_loss = F.softplus(
                    -self.cf2_fine_temperature * direction * mask_gap
                )
                losses.append(pair_loss)

                with torch.no_grad():
                    self.cf2_stats['fine_pairs'] += 1
                    self.cf2_stats['fine_used'] += 1
                    self.cf2_stats['fine_loss_sum'] += float(
                        pair_loss.detach().cpu()
                    )
                    self.cf2_stats['fine_similarity_gap_sum'] += float(
                        relevance_gap.abs().detach().cpu()
                    )

        if not losses:
            return reference_loss * 0.0
        return torch.stack(losses).mean()