# coding: utf-8
"""MASKED_GLORIA with listwise MIPD and Full--Masked KL alignment."""

import math

import torch
import torch.nn.functional as F

from models.masked_gloria_mipd import MASKED_GLORIA_MIPD


class MASKED_GLORIA_MIPD_KL(MASKED_GLORIA_MIPD):
    """
    Align Full and Masked listwise interaction distributions.

    The model preserves MASKED_GLORIA_MIPD's concatenated recommendation
    embeddings.  For every unique mini-batch user, MIPD and KL share one fixed
    candidate list containing the positive item and K unseen negatives.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_MIPD_KL, self).__init__(config, dataset)

        self.branch_kl_weight = float(
            self._get_config_value(config, 'branch_kl_weight', 0.01)
        )
        self.branch_kl_temperature = float(
            self._get_config_value(config, 'branch_kl_temperature', 1.0)
        )
        if self.branch_kl_weight < 0.0:
            raise ValueError('branch_kl_weight must be non-negative.')
        if self.branch_kl_temperature <= 0.0:
            raise ValueError('branch_kl_temperature must be positive.')

        self.last_branch_kl_loss = None
        self.last_branch_kl_candidate_count = None
        self.last_branch_kl_user_count = None
        self.branch_kl_epoch_loss_sum = 0.0
        self.branch_kl_epoch_user_count = 0
        self.branch_kl_epoch_batch_count = 0
        self.ranking_epoch_loss_sum = 0.0
        self.ranking_epoch_example_count = 0

    def pre_epoch_processing(self):
        super(MASKED_GLORIA_MIPD_KL, self).pre_epoch_processing()
        self.branch_kl_epoch_loss_sum = 0.0
        self.branch_kl_epoch_user_count = 0
        self.branch_kl_epoch_batch_count = 0
        self.ranking_epoch_loss_sum = 0.0
        self.ranking_epoch_example_count = 0

    def post_epoch_processing(self):
        mipd_info = super(
            MASKED_GLORIA_MIPD_KL, self
        ).post_epoch_processing()
        mean_branch_kl = (
            self.branch_kl_epoch_loss_sum
            / self.branch_kl_epoch_user_count
            if self.branch_kl_epoch_user_count > 0
            else 0.0
        )
        mean_ranking_loss = (
            self.ranking_epoch_loss_sum
            / self.ranking_epoch_example_count
            if self.ranking_epoch_example_count > 0
            else 0.0
        )
        kl_info = (
            'Full-Masked alignment: mean_ranking_loss={:.8f}, '
            'mean_kl={:.8f}, users={}, batches={}, candidates={}, '
            'temperature={}, weight={}'
        ).format(
            mean_ranking_loss,
            mean_branch_kl,
            self.branch_kl_epoch_user_count,
            self.branch_kl_epoch_batch_count,
            self.mipd_num_negatives + 1,
            self.branch_kl_temperature,
            self.branch_kl_weight,
        )
        return '{}\n{}'.format(mipd_info, kl_info)

    @staticmethod
    def _validate_listwise_inputs(user_nodes, candidate_items):
        if user_nodes.dim() != 1:
            raise ValueError('user_nodes must be one-dimensional.')
        if candidate_items.dim() != 2:
            raise ValueError('candidate_items must have shape [B, K+1].')
        if candidate_items.size(0) != user_nodes.size(0):
            raise ValueError('candidate_items and user_nodes must align.')
        if candidate_items.size(1) < 2:
            raise ValueError('Each candidate list needs at least two items.')
        if torch.unique(user_nodes).numel() != user_nodes.numel():
            raise ValueError('Listwise losses need one row per distinct user.')

    def calculate_branch_kl(self, user_nodes, candidate_items):
        """Return KL(P_masked || P_full) on a fixed candidate list."""
        self._validate_listwise_inputs(user_nodes, candidate_items)

        full_users = self.full_user_view[user_nodes]
        masked_users = self.masked_user_view[user_nodes]
        full_items = self.full_item_view[candidate_items]
        masked_items = self.masked_item_view[candidate_items]

        full_logits = torch.sum(
            full_users[:, None, :] * full_items,
            dim=-1,
        ) / self.branch_kl_temperature
        masked_logits = torch.sum(
            masked_users[:, None, :] * masked_items,
            dim=-1,
        ) / self.branch_kl_temperature

        log_p_full = F.log_softmax(full_logits, dim=1)
        log_p_masked = F.log_softmax(masked_logits, dim=1)
        p_masked = log_p_masked.exp()
        return torch.sum(
            p_masked * (log_p_masked - log_p_full),
            dim=1,
        ).mean()

    def _calculate_mipd_from_candidates(self, user_nodes, candidate_items):
        """Calculate MIPD without constructing a second candidate list."""
        batch_size = int(user_nodes.numel())
        if self.mipd_weight == 0.0 or batch_size < 2:
            zero = self.masked_user_view[user_nodes].sum() * 0.0
            self.last_mipd_user_count = 0
            self.last_mipd_candidate_count = 0
            return zero, zero.detach()

        self.last_mipd_user_count = batch_size
        self.last_mipd_candidate_count = int(candidate_items.size(1))
        losses = [
            self.calculate_listwise_mipd(user_nodes, candidate_items)
            for _ in range(self.mipd_num_samples)
        ]
        mipd_loss = torch.stack(losses).mean()
        return mipd_loss, (-mipd_loss).detach()

    def _calculate_branch_kl_from_candidates(
        self,
        user_nodes,
        candidate_items,
    ):
        """Calculate branch KL or a differentiable zero when disabled."""
        batch_size = int(user_nodes.numel())
        if self.branch_kl_weight == 0.0 or batch_size == 0:
            zero = self.masked_user_view[user_nodes].sum() * 0.0
            self.last_branch_kl_user_count = 0
            self.last_branch_kl_candidate_count = 0
            return zero

        self.last_branch_kl_user_count = batch_size
        self.last_branch_kl_candidate_count = int(candidate_items.size(1))
        return self.calculate_branch_kl(user_nodes, candidate_items)

    def calculate_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        ranking_loss = F.softplus(-(pos_scores - neg_scores)).mean()
        ranking_loss = ranking_loss / math.log(2.0)

        if self.mipd_weight > 0.0 or self.branch_kl_weight > 0.0:
            # This is the only candidate-builder call in a training step.  Both
            # listwise objectives below receive these exact item IDs.
            user_nodes, candidate_items = self._build_mipd_candidates(
                interaction
            )
        else:
            user_nodes = interaction[0].detach().new_empty((0,))
            candidate_items = interaction[0].detach().new_empty(
                (0, self.mipd_num_negatives + 1)
            )

        mipd_loss, mipd_jsd = self._calculate_mipd_from_candidates(
            user_nodes,
            candidate_items,
        )
        branch_kl_loss = self._calculate_branch_kl_from_candidates(
            user_nodes,
            candidate_items,
        )
        total_loss = (
            ranking_loss
            + self.mipd_weight * mipd_loss
            + self.branch_kl_weight * branch_kl_loss
        )

        self.last_ranking_loss = ranking_loss.detach()
        self.last_mipd_loss = mipd_loss.detach()
        self.last_mipd_jsd = mipd_jsd
        self.last_branch_kl_loss = branch_kl_loss.detach()

        mipd_user_count = int(self.last_mipd_user_count or 0)
        self.mipd_epoch_jsd_sum += (
            float(mipd_jsd.cpu()) * mipd_user_count
        )
        self.mipd_epoch_user_count += mipd_user_count
        self.mipd_epoch_batch_count += 1

        kl_user_count = int(self.last_branch_kl_user_count or 0)
        self.branch_kl_epoch_loss_sum += (
            float(branch_kl_loss.detach().cpu()) * kl_user_count
        )
        self.branch_kl_epoch_user_count += kl_user_count
        self.branch_kl_epoch_batch_count += 1

        example_count = int(interaction[0].numel())
        self.ranking_epoch_loss_sum += (
            float(ranking_loss.detach().cpu()) * example_count
        )
        self.ranking_epoch_example_count += example_count
        return total_loss
