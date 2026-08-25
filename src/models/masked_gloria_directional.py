# coding: utf-8
"""MASKED_GLORIA with label-aware directional permutation loss."""

import math

import torch
import torch.nn.functional as F

from models.masked_gloria import MASKED_GLORIA


class MASKED_GLORIA_DIRECTIONAL(MASKED_GLORIA):
    """Train correct Full--Masked pairings to help the positive item."""

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_DIRECTIONAL, self).__init__(config, dataset)

        self.directional_weight = float(
            self._get_config_value(config, 'directional_weight', 0.01)
        )
        self.directional_margin = float(
            self._get_config_value(config, 'directional_margin', 0.1)
        )
        self.directional_num_negatives = int(
            self._get_config_value(config, 'directional_num_negatives', 16)
        )
        self.directional_num_samples = int(
            self._get_config_value(config, 'directional_num_samples', 3)
        )
        self.directional_temperature = float(
            self._get_config_value(config, 'directional_temperature', 0.5)
        )
        self.directional_negative_sampling = str(
            self._get_config_value(
                config,
                'directional_negative_sampling',
                'full_hard',
            )
        ).strip().lower().replace('-', '_')
        self.directional_hard_pool_size = int(
            self._get_config_value(
                config,
                'directional_hard_pool_size',
                256,
            )
        )
        self.directional_permutation_gradient = str(
            self._get_config_value(
                config,
                'directional_permutation_gradient',
                'symmetric',
            )
        ).strip().lower()
        self.directional_loss_type = str(
            self._get_config_value(
                config,
                'directional_loss_type',
                'softplus',
            )
        ).strip().lower()

        if self.directional_weight < 0.0:
            raise ValueError('directional_weight must be non-negative.')
        if self.directional_margin < 0.0:
            raise ValueError('directional_margin must be non-negative.')
        if self.directional_num_negatives < 1:
            raise ValueError('directional_num_negatives must be at least 1.')
        if self.directional_num_samples < 1:
            raise ValueError('directional_num_samples must be at least 1.')
        if self.directional_temperature <= 0.0:
            raise ValueError('directional_temperature must be positive.')
        if self.directional_negative_sampling not in {'random', 'full_hard'}:
            raise ValueError(
                'directional_negative_sampling must be "random" or '
                '"full_hard".'
            )
        if (
            self.directional_negative_sampling == 'full_hard'
            and self.directional_hard_pool_size
            < self.directional_num_negatives
        ):
            raise ValueError(
                'directional_hard_pool_size must be at least '
                'directional_num_negatives.'
            )
        if self.directional_permutation_gradient not in {
            'symmetric',
            'detached',
        }:
            raise ValueError(
                'directional_permutation_gradient must be "symmetric" or '
                '"detached".'
            )
        if self.directional_loss_type not in {'softplus', 'hinge'}:
            raise ValueError(
                'directional_loss_type must be "softplus" or "hinge".'
            )

        self.directional_seen_items = self._build_directional_seen_items()

        self.last_ranking_loss = None
        self.last_directional_loss = None
        self.last_directional_mean_gap = None
        self.last_directional_positive_gap_rate = None
        self.last_directional_margin_rate = None
        self.last_directional_user_count = None
        self.last_directional_candidate_count = None

        self.directional_epoch_loss_sum = 0.0
        self.directional_epoch_gap_sum = 0.0
        self.directional_epoch_positive_gap_sum = 0.0
        self.directional_epoch_margin_sum = 0.0
        self.directional_epoch_user_count = 0
        self.directional_epoch_batch_count = 0
        self.ranking_epoch_loss_sum = 0.0
        self.ranking_epoch_example_count = 0

    @staticmethod
    def _get_config_value(config, key, default):
        try:
            value = config[key]
        except (KeyError, TypeError):
            return default
        return default if value is None else value

    def _build_directional_seen_items(self):
        histories = [set() for _ in range(self.num_user)]
        users = self.forward_edge_users.detach().cpu().tolist()
        items = self.forward_edge_items.detach().cpu().tolist()
        for user_id, item_id in zip(users, items):
            histories[int(user_id)].add(int(item_id))
        return tuple(frozenset(history) for history in histories)

    def pre_epoch_processing(self):
        self.directional_epoch_loss_sum = 0.0
        self.directional_epoch_gap_sum = 0.0
        self.directional_epoch_positive_gap_sum = 0.0
        self.directional_epoch_margin_sum = 0.0
        self.directional_epoch_user_count = 0
        self.directional_epoch_batch_count = 0
        self.ranking_epoch_loss_sum = 0.0
        self.ranking_epoch_example_count = 0

    def post_epoch_processing(self):
        user_count = self.directional_epoch_user_count
        mean_directional_loss = (
            self.directional_epoch_loss_sum / user_count
            if user_count > 0
            else 0.0
        )
        mean_gap = (
            self.directional_epoch_gap_sum / user_count
            if user_count > 0
            else 0.0
        )
        positive_gap_rate = (
            self.directional_epoch_positive_gap_sum / user_count
            if user_count > 0
            else 0.0
        )
        margin_rate = (
            self.directional_epoch_margin_sum / user_count
            if user_count > 0
            else 0.0
        )
        mean_ranking_loss = (
            self.ranking_epoch_loss_sum / self.ranking_epoch_example_count
            if self.ranking_epoch_example_count > 0
            else 0.0
        )
        return (
            'directional permutation: mean_ranking_loss={:.8f}, '
            'mean_loss={:.8f}, mean_gap={:.8f}, positive_gap_rate={:.6f}, '
            'margin_rate={:.6f}, users={}, batches={}, candidates={}, '
            'negatives={}, permutations={}, temperature={}, margin={}, '
            'weight={}, sampling={}, hard_pool={}, permutation_gradient={}, '
            'loss_type={}'
        ).format(
            mean_ranking_loss,
            mean_directional_loss,
            mean_gap,
            positive_gap_rate,
            margin_rate,
            user_count,
            self.directional_epoch_batch_count,
            self.directional_num_negatives + 1,
            self.directional_num_negatives,
            self.directional_num_samples,
            self.directional_temperature,
            self.directional_margin,
            self.directional_weight,
            self.directional_negative_sampling,
            self.directional_hard_pool_size,
            self.directional_permutation_gradient,
            self.directional_loss_type,
        )

    def compute_result_embedding(self, forward_edge_mask=None, full_view=None):
        result_embed = super(
            MASKED_GLORIA_DIRECTIONAL,
            self,
        ).compute_result_embedding(
            forward_edge_mask=forward_edge_mask,
            full_view=full_view,
        )
        if result_embed.size(1) % 2 != 0:
            raise ValueError('Full and Masked embedding dimensions must match.')

        view_dim = result_embed.size(1) // 2
        user_embed = result_embed[:self.num_user]
        item_embed = result_embed[self.num_user:]
        self.full_user_view = user_embed[:, :view_dim]
        self.masked_user_view = user_embed[:, view_dim:]
        self.full_item_view = item_embed[:, :view_dim]
        self.masked_item_view = item_embed[:, view_dim:]
        return result_embed

    @staticmethod
    def _first_unique_user_positions(user_nodes):
        positions = []
        seen = set()
        for position, user_id in enumerate(
            user_nodes.detach().view(-1).cpu().tolist()
        ):
            user_id = int(user_id)
            if user_id in seen:
                continue
            seen.add(user_id)
            positions.append(position)
        return torch.as_tensor(
            positions,
            dtype=torch.long,
            device=user_nodes.device,
        )

    def _sample_unseen_pool(self, user_nodes, positive_items, pool_size):
        """Return padded unseen pools and a mask for their valid positions."""
        batch_size = int(user_nodes.numel())
        negative_count = int(self.directional_num_negatives)
        draw_count = max(pool_size * 3, pool_size + 32)
        random_pool = torch.randint(
            self.num_item,
            (batch_size, draw_count),
            device=user_nodes.device,
        ).cpu().numpy()
        users_cpu = user_nodes.detach().cpu().tolist()
        positives_cpu = positive_items.detach().cpu().tolist()

        pool_rows = []
        valid_lengths = []
        for row, (user_id, positive_id) in enumerate(
            zip(users_cpu, positives_cpu)
        ):
            user_id = int(user_id)
            positive_id = int(positive_id)
            excluded = set(self.directional_seen_items[user_id])
            excluded.add(positive_id)
            available_count = self.num_item - len(excluded)
            if available_count < negative_count:
                raise ValueError(
                    'User {} has only {} unseen items, but directional loss '
                    'needs {}.'.format(
                        user_id,
                        available_count,
                        negative_count,
                    )
                )

            target_size = min(pool_size, available_count)
            selected = []
            selected_set = set()

            def add_item(item_id):
                item_id = int(item_id)
                if (
                    item_id < 0
                    or item_id >= self.num_item
                    or item_id in excluded
                    or item_id in selected_set
                ):
                    return
                selected.append(item_id)
                selected_set.add(item_id)

            for item_id in random_pool[row]:
                add_item(item_id)
                if len(selected) == target_size:
                    break

            if len(selected) < target_size:
                start = int(random_pool[row, 0])
                for offset in range(self.num_item):
                    add_item((start + offset) % self.num_item)
                    if len(selected) == target_size:
                        break

            valid_lengths.append(len(selected))
            selected.extend([selected[0]] * (pool_size - len(selected)))
            pool_rows.append(selected)

        pool_items = torch.as_tensor(
            pool_rows,
            dtype=torch.long,
            device=user_nodes.device,
        )
        valid_lengths = torch.as_tensor(
            valid_lengths,
            dtype=torch.long,
            device=user_nodes.device,
        )
        valid_mask = (
            torch.arange(pool_size, device=user_nodes.device)[None, :]
            < valid_lengths[:, None]
        )
        return pool_items, valid_mask

    def _build_directional_candidates(self, interaction):
        if interaction is None or len(interaction) < 2:
            raise ValueError(
                'interaction must contain users and positive items.'
            )

        all_users = interaction[0].detach().view(-1)
        all_positives = interaction[1].detach().view(-1)
        if all_users.numel() != all_positives.numel():
            raise ValueError('user and positive-item tensors must align.')

        positions = self._first_unique_user_positions(all_users)
        user_nodes = all_users[positions]
        positive_items = all_positives[positions]
        batch_size = int(user_nodes.numel())
        negative_count = int(self.directional_num_negatives)
        if batch_size == 0:
            empty = positive_items.new_empty((0, negative_count + 1))
            return user_nodes, empty

        pool_size = (
            negative_count
            if self.directional_negative_sampling == 'random'
            else int(self.directional_hard_pool_size)
        )
        pool_items, valid_mask = self._sample_unseen_pool(
            user_nodes,
            positive_items,
            pool_size,
        )

        if self.directional_negative_sampling == 'random':
            negatives = pool_items[:, :negative_count]
        else:
            with torch.no_grad():
                full_scores = torch.sum(
                    self.full_user_view[user_nodes, None, :]
                    * self.full_item_view[pool_items],
                    dim=-1,
                )
                full_scores = full_scores.masked_fill(
                    ~valid_mask,
                    float('-inf'),
                )
                hard_positions = torch.topk(
                    full_scores,
                    k=negative_count,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                negatives = torch.gather(
                    pool_items,
                    dim=1,
                    index=hard_positions,
                )

        candidates = torch.cat(
            [positive_items[:, None], negatives],
            dim=1,
        )
        return user_nodes, candidates

    @staticmethod
    def _sample_derangement(batch_size, device):
        order = torch.randperm(batch_size, device=device)
        permutation = torch.empty_like(order)
        permutation[order] = torch.roll(order, shifts=1, dims=0)
        return permutation

    @staticmethod
    def _validate_directional_inputs(user_nodes, candidate_items):
        if user_nodes.dim() != 1:
            raise ValueError('user_nodes must be one-dimensional.')
        if candidate_items.dim() != 2:
            raise ValueError('candidate_items must have shape [B, K+1].')
        if candidate_items.size(0) != user_nodes.size(0):
            raise ValueError('candidate_items and user_nodes must align.')
        if candidate_items.size(1) < 2:
            raise ValueError('Each candidate list needs at least two items.')
        if torch.unique(user_nodes).numel() != user_nodes.numel():
            raise ValueError('Directional loss needs distinct user rows.')

    def calculate_directional_for_permutation(
        self,
        user_nodes,
        candidate_items,
        permutation=None,
    ):
        """Return hinge loss and detached utility gaps for one permutation."""
        self._validate_directional_inputs(user_nodes, candidate_items)
        batch_size = int(user_nodes.numel())
        masked_users = self.masked_user_view[user_nodes]
        if batch_size < 2:
            zero = masked_users.sum() * 0.0
            return zero, zero.detach().expand(batch_size)

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
            expected = torch.arange(batch_size, device=masked_users.device)
            if permutation.shape != (batch_size,):
                raise ValueError('permutation must have shape [B].')
            if not torch.equal(torch.sort(permutation).values, expected):
                raise ValueError('permutation must contain every row once.')
            if torch.any(permutation == expected):
                raise ValueError('permutation must not contain fixed points.')

        full_items = self.full_item_view[candidate_items]
        masked_items = self.masked_item_view[candidate_items]
        with torch.no_grad():
            full_scores = torch.sum(
                self.full_user_view[user_nodes, None, :] * full_items,
                dim=-1,
            )

        masked_scores = torch.sum(
            masked_users[:, None, :] * masked_items,
            dim=-1,
        )
        original_logits = (
            full_scores + masked_scores
        ) / self.directional_temperature
        original_log_prob = F.log_softmax(
            original_logits,
            dim=1,
        )[:, 0]

        def calculate_permuted_log_prob():
            permuted_masked_scores = torch.sum(
                masked_users[permutation, None, :] * masked_items,
                dim=-1,
            )
            permuted_logits = (
                full_scores + permuted_masked_scores
            ) / self.directional_temperature
            return F.log_softmax(
                permuted_logits,
                dim=1,
            )[:, 0]

        if self.directional_permutation_gradient == 'detached':
            # Backward-compatible ablation: permutation only gates/reweights
            # the original listwise gradient.
            with torch.no_grad():
                permuted_log_prob = calculate_permuted_log_prob()
        else:
            # Every permutation now provides its own optimization direction.
            permuted_log_prob = calculate_permuted_log_prob()

        utility_gap = original_log_prob - permuted_log_prob
        margin_violation = self.directional_margin - utility_gap
        if self.directional_loss_type == 'hinge':
            per_user_loss = F.relu(margin_violation)
        else:
            # Unlike hinge, softplus keeps a smooth gradient after the margin.
            per_user_loss = F.softplus(margin_violation)
        directional_loss = per_user_loss.mean()
        return directional_loss, utility_gap.detach()

    def calculate_directional_loss(self, interaction):
        if self.directional_weight == 0.0:
            user_nodes = interaction[0].detach().view(-1)
            zero = self.masked_user_view[user_nodes].sum() * 0.0
            self.last_directional_user_count = 0
            self.last_directional_candidate_count = 0
            return zero, zero.detach(), zero.detach(), zero.detach()

        user_nodes, candidate_items = self._build_directional_candidates(
            interaction
        )
        batch_size = int(user_nodes.numel())
        self.last_directional_user_count = batch_size
        self.last_directional_candidate_count = int(candidate_items.size(1))
        if batch_size < 2:
            zero = self.masked_user_view[user_nodes].sum() * 0.0
            return zero, zero.detach(), zero.detach(), zero.detach()

        losses = []
        gaps = []
        for _ in range(self.directional_num_samples):
            loss, utility_gap = self.calculate_directional_for_permutation(
                user_nodes,
                candidate_items,
            )
            losses.append(loss)
            gaps.append(utility_gap)

        directional_loss = torch.stack(losses).mean()
        utility_gaps = torch.stack(gaps)
        mean_gap = utility_gaps.mean()
        positive_gap_rate = (utility_gaps > 0.0).float().mean()
        margin_rate = (
            utility_gaps >= self.directional_margin
        ).float().mean()
        return (
            directional_loss,
            mean_gap.detach(),
            positive_gap_rate.detach(),
            margin_rate.detach(),
        )

    def calculate_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)
        ranking_loss = F.softplus(-(pos_scores - neg_scores)).mean()
        ranking_loss = ranking_loss / math.log(2.0)

        (
            directional_loss,
            mean_gap,
            positive_gap_rate,
            margin_rate,
        ) = self.calculate_directional_loss(interaction)
        total_loss = (
            ranking_loss
            + self.directional_weight * directional_loss
        )

        self.last_ranking_loss = ranking_loss.detach()
        self.last_directional_loss = directional_loss.detach()
        self.last_directional_mean_gap = mean_gap
        self.last_directional_positive_gap_rate = positive_gap_rate
        self.last_directional_margin_rate = margin_rate

        user_count = int(self.last_directional_user_count or 0)
        self.directional_epoch_loss_sum += (
            float(directional_loss.detach().cpu()) * user_count
        )
        self.directional_epoch_gap_sum += float(mean_gap.cpu()) * user_count
        self.directional_epoch_positive_gap_sum += (
            float(positive_gap_rate.cpu()) * user_count
        )
        self.directional_epoch_margin_sum += (
            float(margin_rate.cpu()) * user_count
        )
        self.directional_epoch_user_count += user_count
        self.directional_epoch_batch_count += 1

        example_count = int(interaction[0].numel())
        self.ranking_epoch_loss_sum += (
            float(ranking_loss.detach().cpu()) * example_count
        )
        self.ranking_epoch_example_count += example_count
        return total_loss
