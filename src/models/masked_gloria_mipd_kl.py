# coding: utf-8
"""MASKED_GLORIA with catalog-wide MIPD and branch alignment."""

import math

import torch
import torch.nn.functional as F

from models.masked_gloria_mipd import MASKED_GLORIA_MIPD


class MASKED_GLORIA_MIPD_KL(MASKED_GLORIA_MIPD):
    """
    Align Full and Masked catalog-wide interaction distributions.

    The model preserves MASKED_GLORIA_MIPD's concatenated recommendation
    embeddings. Both MIPD and KL operate on interaction distributions over
    every catalog item for each unique mini-batch user.
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
        self.last_branch_kl_item_count = None
        self.last_branch_kl_user_count = None
        self.last_mipd_item_count = None
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
        mean_jsd = (
            self.mipd_epoch_jsd_sum / self.mipd_epoch_user_count
            if self.mipd_epoch_user_count > 0
            else 0.0
        )
        mipd_info = (
            'catalog MIPD: mean_jsd={:.8f}, users={}, batches={}, '
            'items={}, permutations={}, temperature={}, weight={}'
        ).format(
            mean_jsd,
            self.mipd_epoch_user_count,
            self.mipd_epoch_batch_count,
            self.num_item,
            self.mipd_num_samples,
            self.mipd_temperature,
            self.mipd_weight,
        )
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
            'mean_kl={:.8f}, users={}, batches={}, items={}, '
            'temperature={}, weight={}'
        ).format(
            mean_ranking_loss,
            mean_branch_kl,
            self.branch_kl_epoch_user_count,
            self.branch_kl_epoch_batch_count,
            self.num_item,
            self.branch_kl_temperature,
            self.branch_kl_weight,
        )
        return '{}\n{}'.format(mipd_info, kl_info)

    @staticmethod
    def _validate_catalog_users(user_nodes):
        if user_nodes.dim() != 1:
            raise ValueError('user_nodes must be one-dimensional.')
        if torch.unique(user_nodes).numel() != user_nodes.numel():
            raise ValueError('Catalog losses need one row per distinct user.')

    def _calculate_catalog_scores(self, user_nodes):
        """Return raw Full and Masked scores against every catalog item."""
        self._validate_catalog_users(user_nodes)

        full_scores = torch.matmul(
            self.full_user_view[user_nodes],
            self.full_item_view.t(),
        )
        masked_scores = torch.matmul(
            self.masked_user_view[user_nodes],
            self.masked_item_view.t(),
        )
        return full_scores, masked_scores

    def _branch_kl_from_scores(self, full_scores, masked_scores):
        """Return KL(P_masked || P_full) from catalog-wide raw scores."""
        full_logits = full_scores / self.branch_kl_temperature
        masked_logits = masked_scores / self.branch_kl_temperature

        log_p_full = F.log_softmax(full_logits, dim=1)
        log_p_masked = F.log_softmax(masked_logits, dim=1)
        p_masked = log_p_masked.exp()
        return torch.sum(
            p_masked * (log_p_masked - log_p_full),
            dim=1,
        ).mean()

    def calculate_branch_kl(self, user_nodes):
        """Return KL(P_masked || P_full) over the entire item catalog."""
        full_scores, masked_scores = self._calculate_catalog_scores(user_nodes)
        return self._branch_kl_from_scores(full_scores, masked_scores)

    def calculate_catalog_mipd(
        self,
        user_nodes,
        full_scores=None,
        masked_scores=None,
        permutation=None,
    ):
        """Return negative catalog-wide JSD after permuting Masked users."""
        self._validate_catalog_users(user_nodes)
        batch_size = int(user_nodes.numel())
        if full_scores is None or masked_scores is None:
            if full_scores is not None or masked_scores is not None:
                raise ValueError(
                    'full_scores and masked_scores must be supplied together.'
                )
            full_scores, masked_scores = self._calculate_catalog_scores(
                user_nodes
            )
        expected_shape = (batch_size, self.num_item)
        if full_scores.shape != expected_shape:
            raise ValueError('full_scores must have shape [B, num_items].')
        if masked_scores.shape != expected_shape:
            raise ValueError('masked_scores must have shape [B, num_items].')
        if batch_size < 2:
            return masked_scores.sum() * 0.0

        masked_users = self.masked_user_view[user_nodes]
        if permutation is None:
            permutation = self._sample_derangement(
                batch_size,
                masked_users.device,
            )
        else:
            permutation = permutation.to(
                device=masked_users.device,
                dtype=torch.long,
            )
            if permutation.shape != (batch_size,):
                raise ValueError('permutation must have shape [B].')
            if not torch.equal(
                torch.sort(permutation).values,
                torch.arange(batch_size, device=masked_users.device),
            ):
                raise ValueError('permutation must contain every row once.')
            if torch.any(
                permutation
                == torch.arange(batch_size, device=masked_users.device)
            ):
                raise ValueError('permutation must not contain fixed points.')

        original_logits = (
            full_scores.detach() + masked_scores
        ) / self.mipd_temperature
        permuted_masked_scores = torch.matmul(
            masked_users[permutation],
            self.masked_item_view.t(),
        )
        permuted_logits = (
            full_scores.detach() + permuted_masked_scores
        ) / self.mipd_temperature
        return -self._js_divergence_from_logits(
            original_logits,
            permuted_logits,
        )

    def _calculate_catalog_mipd_from_scores(
        self,
        user_nodes,
        full_scores,
        masked_scores,
    ):
        """Average catalog-wide MIPD over configured permutations."""
        batch_size = int(user_nodes.numel())
        if self.mipd_weight == 0.0 or batch_size < 2:
            zero = masked_scores.sum() * 0.0
            self.last_mipd_user_count = 0
            self.last_mipd_candidate_count = 0
            self.last_mipd_item_count = 0
            return zero, zero.detach()

        self.last_mipd_user_count = batch_size
        # Retain the parent's diagnostic attribute for compatibility. In this
        # subclass the "candidate" set is the complete item catalog.
        self.last_mipd_candidate_count = int(self.num_item)
        self.last_mipd_item_count = int(self.num_item)
        losses = [
            self.calculate_catalog_mipd(
                user_nodes,
                full_scores=full_scores,
                masked_scores=masked_scores,
            )
            for _ in range(self.mipd_num_samples)
        ]
        mipd_loss = torch.stack(losses).mean()
        return mipd_loss, (-mipd_loss).detach()

    def _calculate_branch_kl_from_scores(
        self,
        user_nodes,
        full_scores,
        masked_scores,
    ):
        """Calculate branch KL or a differentiable zero when disabled."""
        batch_size = int(user_nodes.numel())
        if self.branch_kl_weight == 0.0 or batch_size == 0:
            zero = self.masked_user_view[user_nodes].sum() * 0.0
            self.last_branch_kl_user_count = 0
            self.last_branch_kl_item_count = 0
            return zero

        self.last_branch_kl_user_count = batch_size
        self.last_branch_kl_item_count = int(self.num_item)
        return self._branch_kl_from_scores(full_scores, masked_scores)

    def calculate_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        ranking_loss = F.softplus(-(pos_scores - neg_scores)).mean()
        ranking_loss = ranking_loss / math.log(2.0)

        if self.mipd_weight > 0.0 or self.branch_kl_weight > 0.0:
            all_users = interaction[0].detach().view(-1)
            positions = self._first_unique_user_positions(all_users)
            user_nodes = all_users[positions]
            full_scores, masked_scores = self._calculate_catalog_scores(
                user_nodes
            )
        else:
            user_nodes = interaction[0].detach().new_empty((0,))
            full_scores = self.full_item_view.new_empty((0, self.num_item))
            masked_scores = self.masked_item_view.new_empty((0, self.num_item))

        mipd_loss, mipd_jsd = self._calculate_catalog_mipd_from_scores(
            user_nodes,
            full_scores,
            masked_scores,
        )
        branch_kl_loss = self._calculate_branch_kl_from_scores(
            user_nodes,
            full_scores,
            masked_scores,
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
