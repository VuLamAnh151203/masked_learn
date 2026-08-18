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


class MASKED_GLORIA_CF4(MASKED_GLORIA):
    """MASKED_GLORIA with broad mask regularization + representation distillation.

    Broad loss:
        history item -> similarity to the user's masked-history prototype
        -> pairwise ordering of edge-mask weights.

    Finding-2 representation loss:
        the full-graph branch acts as a teacher and defines target-specific
        item-item similarity. The masked branch is trained to preserve that
        relative similarity structure.

    No hard-negative loss and no explicit counterfactual edge drop are used.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_CF4, self).__init__(config, dataset)

        self.cf4_lambda = float(_cfg(config, 'cf4_lambda', 0.1))
        self.cf4_temperature = float(
            _cfg(config, 'cf4_temperature', 1.0)
        )
        self.cf4_warmup_ratio = float(
            _cfg(config, 'cf4_warmup_ratio', 0.10)
        )
        configured_warmup_epochs = int(
            _cfg(config, 'cf4_warmup_epochs', 50)
        )
        self.cf4_user_ratio = float(_cfg(config, 'cf4_user_ratio', 0.10))
        self.cf4_batch_size = int(_cfg(config, 'cf4_batch_size', 8))
        self.cf4_pair_count = int(_cfg(config, 'cf4_pair_count', 32))
        self.cf4_min_history = int(_cfg(config, 'cf4_min_history', 2))
        self.cf4_similarity_eps = float(
            _cfg(config, 'cf4_similarity_eps', 1e-6)
        )
        self.cf4_seed_offset = int(
            _cfg(config, 'cf4_seed_offset', 20000)
        )
        self.cf4_log_stats = _cfg_bool(config, 'cf4_log_stats', True)

        # -------------------------------------------------------------
        # Finding-2 representation distillation.
        #
        # V1 default:
        #   teacher similarity = cos(z_e^full, z_p^full)
        #   student similarity = cos(z_e^mask, z_p^mask)
        #   L_repr = MSE(student_similarity, stopgrad(teacher_similarity))
        #
        # A pairwise/ranking version is also available by setting:
        #   cf4_repr_mode: pairwise
        # -------------------------------------------------------------
        self.cf4_repr_lambda = float(
            _cfg(config, 'cf4_repr_lambda', 0.01)
        )
        self.cf4_repr_mode = str(
            _cfg(config, 'cf4_repr_mode', 'mse')
        ).strip().lower()
        self.cf4_repr_temperature = float(
            _cfg(config, 'cf4_repr_temperature', 1.0)
        )
        self.cf4_repr_pair_count = int(
            _cfg(config, 'cf4_repr_pair_count', 16)
        )
        self.cf4_repr_user_ratio = float(
            _cfg(config, 'cf4_repr_user_ratio', self.cf4_user_ratio)
        )
        self.cf4_repr_batch_size = int(
            _cfg(config, 'cf4_repr_batch_size', self.cf4_batch_size)
        )

        # If the base class exposes a full-branch tensor directly, put its
        # attribute name here. If absent, the code will try to infer the full
        # branch from result_embed when result_embed is a concat of
        # [full_branch, mask_branch] (or the reverse order).
        self.cf4_teacher_attr = str(
            _cfg(config, 'cf4_teacher_attr', 'full_rep')
        ).strip()

        max_epochs = int(_cfg(config, 'epochs', 1000))
        if configured_warmup_epochs >= 0:
            self.cf4_warmup_epochs = configured_warmup_epochs
        else:
            self.cf4_warmup_epochs = int(
                math.ceil(max_epochs * self.cf4_warmup_ratio)
            )

        self._validate_cf4_config()
        self.current_epoch = 0
        self._cf4_rng = random.Random(self.cf4_seed_offset)
        self.user_to_edge_ids = self._build_cf4_history()
        self.cf4_stats = self._new_cf4_stats()

    def _validate_cf4_config(self):
        if self.cf4_lambda < 0.0:
            raise ValueError('cf4_lambda must be non-negative.')
        if self.cf4_temperature <= 0.0:
            raise ValueError('cf4_temperature must be positive.')
        if not 0.0 <= self.cf4_user_ratio <= 1.0:
            raise ValueError('cf4_user_ratio must be in [0, 1].')
        if self.cf4_batch_size <= 0:
            raise ValueError('cf4_batch_size must be positive.')
        if self.cf4_pair_count <= 0:
            raise ValueError('cf4_pair_count must be positive.')
        if self.cf4_min_history < 2:
            raise ValueError('cf4_min_history must be at least 2.')
        if self.cf4_similarity_eps < 0.0:
            raise ValueError('cf4_similarity_eps must be non-negative.')
        if self.cf4_warmup_epochs < 0:
            raise ValueError('cf4_warmup_epochs must be non-negative.')

        if self.cf4_repr_lambda < 0.0:
            raise ValueError('cf4_repr_lambda must be non-negative.')
        if self.cf4_repr_mode not in ('mse', 'pairwise'):
            raise ValueError(
                "cf4_repr_mode must be either 'mse' or 'pairwise'."
            )
        if self.cf4_repr_temperature <= 0.0:
            raise ValueError('cf4_repr_temperature must be positive.')
        if self.cf4_repr_pair_count <= 0:
            raise ValueError('cf4_repr_pair_count must be positive.')
        if not 0.0 <= self.cf4_repr_user_ratio <= 1.0:
            raise ValueError('cf4_repr_user_ratio must be in [0, 1].')
        if self.cf4_repr_batch_size <= 0:
            raise ValueError('cf4_repr_batch_size must be positive.')

    def _build_cf4_history(self):
        user_to_edge_ids = [[] for _ in range(self.num_user)]
        edge_users = self.forward_edge_users.detach().cpu().tolist()
        for edge_id, user_id in enumerate(edge_users):
            user_to_edge_ids[int(user_id)].append(int(edge_id))
        return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

    @staticmethod
    def _new_cf4_stats():
        return {
            # Broad mask-relation stats.
            'samples': 0,
            'eligible': 0,
            'pairs': 0,
            'used': 0,
            'loss_sum': 0.0,
            'similarity_gap_sum': 0.0,

            # Finding-2 representation stats.
            'repr_samples': 0,
            'repr_eligible': 0,
            'repr_used': 0,
            'repr_loss_sum': 0.0,
            'repr_teacher_sim_sum': 0.0,
            'repr_student_sim_sum': 0.0,
            'repr_abs_error_sum': 0.0,
            'repr_teacher_missing': 0,
        }

    def set_training_epoch(self, epoch_idx):
        epoch_idx = int(epoch_idx)
        if epoch_idx < 0:
            raise ValueError('epoch_idx must be non-negative.')
        self.current_epoch = epoch_idx
        self._cf4_rng.seed(self.cf4_seed_offset + epoch_idx)

    def pre_epoch_processing(self):
        self.cf4_stats = self._new_cf4_stats()

    def post_epoch_processing(self):
        if not self.cf4_log_stats:
            return None

        used = max(self.cf4_stats['used'], 1)
        repr_used = max(self.cf4_stats['repr_used'], 1)

        return (
            'broad-mask + representation-distillation: epoch={epoch}, '
            'warmup_epochs={warmup}, '
            'broad_lambda={lambda_cf4:.6f}, '
            'broad_temperature={temperature:.6f}, '
            'broad_samples={samples}, broad_eligible={eligible}, '
            'broad_pairs={pairs}, broad_used={used_count}, '
            'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
            'repr_lambda={repr_lambda:.6f}, repr_mode={repr_mode}, '
            'repr_temperature={repr_temperature:.6f}, '
            'repr_samples={repr_samples}, repr_eligible={repr_eligible}, '
            'repr_used={repr_used_count}, repr_loss={repr_loss:.6f}, '
            'teacher_sim={teacher_sim:.6f}, '
            'student_sim={student_sim:.6f}, '
            'repr_abs_error={repr_error:.6f}, '
            'teacher_missing={teacher_missing}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf4_warmup_epochs),

            lambda_cf4=float(self.cf4_lambda),
            temperature=float(self.cf4_temperature),
            samples=int(self.cf4_stats['samples']),
            eligible=int(self.cf4_stats['eligible']),
            pairs=int(self.cf4_stats['pairs']),
            used_count=int(self.cf4_stats['used']),
            loss=float(self.cf4_stats['loss_sum'] / used),
            gap=float(self.cf4_stats['similarity_gap_sum'] / used),

            repr_lambda=float(self.cf4_repr_lambda),
            repr_mode=str(self.cf4_repr_mode),
            repr_temperature=float(self.cf4_repr_temperature),
            repr_samples=int(self.cf4_stats['repr_samples']),
            repr_eligible=int(self.cf4_stats['repr_eligible']),
            repr_used_count=int(self.cf4_stats['repr_used']),
            repr_loss=float(
                self.cf4_stats['repr_loss_sum'] / repr_used
            ),
            teacher_sim=float(
                self.cf4_stats['repr_teacher_sim_sum'] / repr_used
            ),
            student_sim=float(
                self.cf4_stats['repr_student_sim_sum'] / repr_used
            ),
            repr_error=float(
                self.cf4_stats['repr_abs_error_sum'] / repr_used
            ),
            teacher_missing=int(self.cf4_stats['repr_teacher_missing']),
        )

    def _is_cf4_active(self):
        return (
            self.training
            and (
                self.cf4_lambda > 0.0
                or self.cf4_repr_lambda > 0.0
            )
            and (
                self.cf4_user_ratio > 0.0
                or self.cf4_repr_user_ratio > 0.0
            )
            and self.current_epoch >= self.cf4_warmup_epochs
        )

    def _calculate_rec_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        return -torch.mean(
            torch.log2(torch.sigmoid(pos_scores - neg_scores))
        )

    def calculate_loss(self, interaction):
        loss_rec = self._calculate_rec_loss(interaction)
        if not self._is_cf4_active():
            self.result_embed = None
            return loss_rec

        if self.cf4_lambda > 0.0:
            loss_mask_relation = self._calculate_mask_relation_loss(
                interaction,
                loss_rec
            )
        else:
            loss_mask_relation = loss_rec * 0.0

        if self.cf4_repr_lambda > 0.0:
            loss_repr = self._calculate_representation_loss(
                interaction,
                loss_rec
            )
        else:
            loss_repr = loss_rec * 0.0

        weighted_auxiliary = (
            self.cf4_lambda * loss_mask_relation
            + self.cf4_repr_lambda * loss_repr
        )

        self.result_embed = None

        # Preserve the old trainer API.
        return loss_rec, weighted_auxiliary

    def _sample_cf4_users(self, interaction):
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.cf4_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(sample_count, self.cf4_batch_size, len(users))
        return self._cf4_rng.sample(users, sample_count)

    def _sample_cf4_pairs(self, history_size, pair_count=None):
        if pair_count is None:
            pair_count = self.cf4_pair_count
        pair_count = int(pair_count)

        total_pairs = history_size * (history_size - 1) // 2
        if total_pairs <= pair_count:
            return list(itertools.combinations(range(history_size), 2))

        pairs = set()
        while len(pairs) < pair_count:
            left = self._cf4_rng.randrange(history_size)
            right = self._cf4_rng.randrange(history_size)
            if left == right:
                continue
            pairs.add(tuple(sorted((left, right))))
        return list(pairs)

    def _calculate_mask_relation_loss(self, interaction, reference_loss):
        """Apply pairwise similarity ordering to current mask weights."""
        sampled_users = self._sample_cf4_users(interaction)
        self.cf4_stats['samples'] += len(sampled_users)
        if not sampled_users or self.result_embed is None:
            return reference_loss * 0.0

        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()
        losses = []

        for user_id in sampled_users:
            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf4_min_history:
                continue

            self.cf4_stats['eligible'] += 1
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

            pair_positions = self._sample_cf4_pairs(len(edge_ids))

            for left_pos, right_pos in pair_positions:
                relevance_gap = (
                    relevance[left_pos] - relevance[right_pos]
                )
                if abs(float(relevance_gap.detach().cpu())) <= self.cf4_similarity_eps:
                    continue

                left_edge = int(edge_ids[left_pos])
                right_edge = int(edge_ids[right_pos])
                mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
                direction = torch.sign(relevance_gap)
                pair_loss = F.softplus(
                    -self.cf4_temperature * direction * mask_gap
                )
                losses.append(pair_loss)

                with torch.no_grad():
                    self.cf4_stats['pairs'] += 1
                    self.cf4_stats['used'] += 1
                    self.cf4_stats['loss_sum'] += float(
                        pair_loss.detach().cpu()
                    )
                    self.cf4_stats['similarity_gap_sum'] += float(
                        relevance_gap.abs().detach().cpu()
                    )

        if not losses:
            return reference_loss * 0.0
        return torch.stack(losses).mean()

    def _sample_repr_users(self, interaction):
        """Sample users for Finding-2 representation supervision."""
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(
            math.ceil(len(users) * self.cf4_repr_user_ratio)
        )
        sample_count = max(1, sample_count)
        sample_count = min(
            sample_count,
            self.cf4_repr_batch_size,
            len(users)
        )
        return self._cf4_rng.sample(users, sample_count)

    @staticmethod
    def _build_batch_positive_targets(interaction):
        """Map each mini-batch user to positive target item ids.

        The existing code already implies:
            interaction[0] -> user ids
            interaction[1] -> positive item ids
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

    def _get_full_teacher_representation(self):
        """Return the full-graph node representation used as teacher.

        Priority:
          1) a tensor exposed directly by the base model via cf4_teacher_attr;
          2) infer the full branch from result_embed when result_embed is a
             concatenation of two equal-size branches and one of those halves
             matches mask_rep more closely.

        Returns:
            Tensor [num_user + num_item, d], or None when it cannot be resolved.
        """
        # -------------------------------------------------------------
        # 1) Direct base-class attribute, e.g. self.full_rep.
        # -------------------------------------------------------------
        if self.cf4_teacher_attr:
            teacher = getattr(self, self.cf4_teacher_attr, None)
            if (
                torch.is_tensor(teacher)
                and teacher.dim() == 2
                and teacher.size(0) >= self.num_user
            ):
                return teacher

        # Common alternative names: harmless fallbacks.
        for attr_name in (
            'full_rep',
            'full_graph_rep',
            'clean_rep',
            'original_rep',
        ):
            teacher = getattr(self, attr_name, None)
            if (
                torch.is_tensor(teacher)
                and teacher.dim() == 2
                and teacher.size(0) >= self.num_user
            ):
                return teacher

        # -------------------------------------------------------------
        # 2) Infer from concatenated result_embed.
        #
        # If result_embed = concat(full_rep, mask_rep) or concat(mask_rep,
        # full_rep), each half has the same dimensionality as mask_rep.
        # We identify which half resembles mask_rep and use the other half
        # as teacher.
        # -------------------------------------------------------------
        if (
            torch.is_tensor(self.result_embed)
            and torch.is_tensor(getattr(self, 'mask_rep', None))
            and self.result_embed.dim() == 2
            and self.mask_rep.dim() == 2
            and self.result_embed.size(0) == self.mask_rep.size(0)
            and self.result_embed.size(1) == 2 * self.mask_rep.size(1)
        ):
            dim = self.mask_rep.size(1)
            left = self.result_embed[:, :dim]
            right = self.result_embed[:, dim:]

            with torch.no_grad():
                mask_ref = self.mask_rep.detach()
                left_error = F.mse_loss(
                    left.detach(),
                    mask_ref
                )
                right_error = F.mse_loss(
                    right.detach(),
                    mask_ref
                )

            # The half closer to mask_rep is considered the masked branch;
            # the other half is the full-graph teacher.
            if float(left_error.cpu()) <= float(right_error.cpu()):
                return right
            return left

        return None

    def _calculate_representation_loss(self, interaction, reference_loss):
        """Finding-2: preserve target-specific similarity in masked branch.

        Teacher:
            r_teacher(e, p) = cos(z_e^full, z_p^full)

        Student:
            r_student(e, p) = cos(z_e^mask, z_p^mask)

        Default MSE objective:
            L_repr = mean_e [
                r_student(e,p) - stopgrad(r_teacher(e,p))
            ]^2

        Optional pairwise objective (cf4_repr_mode='pairwise'):
            if teacher says edge a is more related to target p than edge b,
            the masked representation is trained to preserve that ordering.

        Importantly, this loss does NOT compare target-specific relevance to
        a static mask scalar. It optimizes the masked representation itself.
        """
        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        teacher_rep = self._get_full_teacher_representation()
        if teacher_rep is None:
            self.cf4_stats['repr_teacher_missing'] += 1
            return reference_loss * 0.0

        if (
            teacher_rep.dim() != 2
            or teacher_rep.size(0) != self.mask_rep.size(0)
        ):
            self.cf4_stats['repr_teacher_missing'] += 1
            return reference_loss * 0.0

        mask_item_rep = self.mask_rep[self.num_user:]
        full_item_rep = teacher_rep[self.num_user:]

        positives_by_user = self._build_batch_positive_targets(interaction)
        sampled_users = self._sample_repr_users(interaction)

        self.cf4_stats['repr_samples'] += len(sampled_users)

        if not sampled_users or not positives_by_user:
            return reference_loss * 0.0

        losses = []

        for user_id in sampled_users:
            target_candidates = positives_by_user.get(int(user_id), [])
            if not target_candidates:
                continue

            target_item = int(self._cf4_rng.choice(target_candidates))

            if (
                target_item < 0
                or target_item >= mask_item_rep.size(0)
                or target_item >= full_item_rep.size(0)
            ):
                continue

            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf4_min_history:
                continue

            edge_tensor = torch.tensor(
                edge_ids,
                dtype=torch.long,
                device=self.forward_edge_users.device
            )
            history_item_ids = self.forward_edge_items[edge_tensor]

            # Avoid the trivial relation cos(z_p, z_p) = 1 when the target
            # item is itself already one of the user's graph edges.
            keep_mask = history_item_ids != target_item
            history_item_ids = history_item_ids[keep_mask]

            if history_item_ids.numel() < self.cf4_min_history:
                continue

            self.cf4_stats['repr_eligible'] += 1

            # ---------------------------------------------------------
            # Teacher target: full branch only, stop-gradient.
            # ---------------------------------------------------------
            with torch.no_grad():
                teacher_target = full_item_rep[target_item]
                teacher_similarity = F.cosine_similarity(
                    full_item_rep[history_item_ids],
                    teacher_target.unsqueeze(0),
                    dim=1
                )

            # ---------------------------------------------------------
            # Student prediction: masked branch, gradient enabled.
            # ---------------------------------------------------------
            student_target = mask_item_rep[target_item]
            student_similarity = F.cosine_similarity(
                mask_item_rep[history_item_ids],
                student_target.unsqueeze(0),
                dim=1
            )

            if self.cf4_repr_mode == 'mse':
                user_loss = F.mse_loss(
                    student_similarity,
                    teacher_similarity
                )

            else:
                # Pairwise preservation of teacher relative ordering.
                pair_positions = self._sample_cf4_pairs(
                    int(history_item_ids.numel()),
                    pair_count=self.cf4_repr_pair_count
                )
                pair_losses = []

                for left_pos, right_pos in pair_positions:
                    teacher_gap = (
                        teacher_similarity[left_pos]
                        - teacher_similarity[right_pos]
                    )

                    if abs(
                        float(teacher_gap.detach().cpu())
                    ) <= self.cf4_similarity_eps:
                        continue

                    student_gap = (
                        student_similarity[left_pos]
                        - student_similarity[right_pos]
                    )
                    direction = torch.sign(teacher_gap)

                    pair_loss = F.softplus(
                        -direction
                        * student_gap
                        / self.cf4_repr_temperature
                    )
                    pair_losses.append(pair_loss)

                if not pair_losses:
                    continue

                user_loss = torch.stack(pair_losses).mean()

            losses.append(user_loss)

            with torch.no_grad():
                self.cf4_stats['repr_used'] += 1
                self.cf4_stats['repr_loss_sum'] += float(
                    user_loss.detach().cpu()
                )
                self.cf4_stats['repr_teacher_sim_sum'] += float(
                    teacher_similarity.mean().detach().cpu()
                )
                self.cf4_stats['repr_student_sim_sum'] += float(
                    student_similarity.mean().detach().cpu()
                )
                self.cf4_stats['repr_abs_error_sum'] += float(
                    (
                        student_similarity
                        - teacher_similarity
                    ).abs().mean().detach().cpu()
                )

        if not losses:
            return reference_loss * 0.0

        return torch.stack(losses).mean()
