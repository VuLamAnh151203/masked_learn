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
    """MASKED_GLORIA with boundary-aware counterfactual robustness loss.

    The main recommendation branch is unchanged.  The auxiliary branch uses a
    detached copy of the learned edge mask, so its gradients update only the
    representation parameters and never supervise ``mask_logits``.
    """

    def __init__(self, config, dataset):
        super(MASKED_GLORIA_CF, self).__init__(config, dataset)

        self.cf_lambda = float(_cfg(config, 'cf_lambda', 0.1))
        self.cf_warmup_ratio = float(_cfg(config, 'cf_warmup_ratio', 0.10))
        configured_warmup_epochs = int(_cfg(config, 'cf_warmup_epochs', -1))
        self.cf_user_ratio = float(_cfg(config, 'cf_user_ratio', 0.10))
        self.cf_batch_size = int(_cfg(config, 'cf_batch_size', 8))
        self.cf_k = int(_cfg(config, 'cf_k', 20))
        self.cf_boundary_width = int(_cfg(config, 'cf_boundary_width', 5))
        self.cf_boundary_q = int(_cfg(config, 'cf_boundary_q', 3))
        self.cf_temperature = float(_cfg(config, 'cf_temperature', 1.0))
        self.cf_min_history = int(_cfg(config, 'cf_min_history', 2))
        self.cf_edge_selector = str(
            _cfg(config, 'cf_edge_selector', 'representation')
        ).strip().lower()
        self.cf_selector_top_n = int(
            _cfg(config, 'cf_selector_top_n', 3)
        )
        self.cf_selector_damage_eps = float(
            _cfg(config, 'cf_selector_damage_eps', 1e-8)
        )
        self.cf_drop_bidirectional = _cfg_bool(
            config,
            'cf_drop_bidirectional',
            True
        )
        self.cf_seed_offset = int(_cfg(config, 'cf_seed_offset', 10000))
        self.cf_log_stats = _cfg_bool(config, 'cf_log_stats', True)

        max_epochs = int(_cfg(config, 'epochs', 1000))
        if configured_warmup_epochs > 0:
            self.cf_warmup_epochs = configured_warmup_epochs
        else:
            self.cf_warmup_epochs = int(
                math.ceil(max_epochs * self.cf_warmup_ratio)
            )

        self._validate_cf_config()
        self.current_epoch = 0
        self._cf_rng = random.Random(self.cf_seed_offset)
        self.user_to_edge_ids, self.user_seen_items = self._build_cf_history()
        self.cf_stats = self._new_cf_stats()

    def _validate_cf_config(self):
        if self.cf_lambda < 0.0:
            raise ValueError('cf_lambda must be non-negative.')
        if not 0.0 <= self.cf_user_ratio <= 1.0:
            raise ValueError('cf_user_ratio must be in [0, 1].')
        if self.cf_batch_size <= 0:
            raise ValueError('cf_batch_size must be positive.')
        if self.cf_k <= 0:
            raise ValueError('cf_k must be positive.')
        if self.cf_boundary_width <= 0:
            raise ValueError('cf_boundary_width must be positive.')
        if self.cf_boundary_width > self.cf_k:
            raise ValueError('cf_boundary_width cannot exceed cf_k.')
        if self.cf_boundary_q < 0:
            raise ValueError('cf_boundary_q must be non-negative.')
        if self.cf_temperature <= 0.0:
            raise ValueError('cf_temperature must be positive.')
        if self.cf_min_history < 2:
            raise ValueError('cf_min_history must be at least 2.')
        if self.cf_edge_selector not in ('representation', 'gradient', 'random'):
            raise ValueError(
                'cf_edge_selector must be representation, gradient, or random.'
            )
        if self.cf_selector_top_n <= 0:
            raise ValueError('cf_selector_top_n must be positive.')
        if self.cf_selector_damage_eps < 0.0:
            raise ValueError('cf_selector_damage_eps must be non-negative.')
        if self.cf_warmup_epochs < 0:
            raise ValueError('cf_warmup_epochs must be non-negative.')
        if not self.cf_drop_bidirectional:
            raise ValueError(
                'cf_drop_bidirectional must be True for the bidirectional '
                'MASKED_GLORIA graph.'
            )

    def _build_cf_history(self):
        user_to_edge_ids = [[] for _ in range(self.num_user)]
        user_seen_items = [[] for _ in range(self.num_user)]

        edge_users = self.forward_edge_users.detach().cpu().tolist()
        edge_items = self.forward_edge_items.detach().cpu().tolist()
        for edge_id, (user_id, item_id) in enumerate(
            zip(edge_users, edge_items)
        ):
            user_to_edge_ids[int(user_id)].append(int(edge_id))
            user_seen_items[int(user_id)].append(int(item_id))

        for user_id in range(self.num_user):
            user_to_edge_ids[user_id] = tuple(user_to_edge_ids[user_id])
            user_seen_items[user_id] = tuple(sorted(set(user_seen_items[user_id])))

        return user_to_edge_ids, user_seen_items

    @staticmethod
    def _new_cf_stats():
        return {
            'samples': 0,
            'eligible': 0,
            'fragile': 0,
            'candidates_verified': 0,
            'positive_damage': 0,
            'skipped_no_damage': 0,
            'used': 0,
            'loss_sum': 0.0,
            'damage_sum': 0.0,
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
        mean_loss = self.cf_stats['loss_sum'] / used
        return (
            'boundary CF: epoch={epoch}, warmup_epochs={warmup}, '
            'lambda={lambda_cf:.6f}, selector={selector}, top_n={top_n}, '
            'samples={samples}, eligible={eligible}, fragile={fragile}, '
            'verified={verified}, positive_damage={positive}, '
            'skipped_no_damage={skipped}, used={used_count}, '
            'boundary_loss={loss:.6f}, damage={damage:.6f}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf_warmup_epochs),
            lambda_cf=float(self.cf_lambda),
            selector=str(self.cf_edge_selector),
            top_n=int(self.cf_selector_top_n),
            samples=int(self.cf_stats['samples']),
            eligible=int(self.cf_stats['eligible']),
            fragile=int(self.cf_stats['fragile']),
            verified=int(self.cf_stats['candidates_verified']),
            positive=int(self.cf_stats['positive_damage']),
            skipped=int(self.cf_stats['skipped_no_damage']),
            used_count=int(self.cf_stats['used']),
            loss=float(mean_loss),
            damage=float(
                self.cf_stats['damage_sum'] / used
            ),
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
        loss_value = -torch.mean(
            torch.log2(torch.sigmoid(pos_scores - neg_scores))
        )
        self.result_embed = None
        return loss_value

    def calculate_loss(self, interaction):
        loss_rec = self._calculate_rec_loss(interaction)
        if not self._is_cf_active():
            return loss_rec

        full_view = (
            self.full_rep[:self.num_user],
            self.full_rep[self.num_user:],
        )
        loss_boundary = self._calculate_boundary_loss(
            interaction,
            loss_rec,
            full_view
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

    def _calculate_boundary_loss(self, interaction, reference_loss, full_view):
        sampled_users = self._sample_cf_users(interaction)
        self.cf_stats['samples'] += len(sampled_users)

        if not sampled_users:
            return reference_loss * 0.0

        base_mask = self.get_forward_edge_mask().detach()
        boundary_losses = []

        for user_id in sampled_users:
            history_edges = self.user_to_edge_ids[user_id]
            if len(history_edges) < self.cf_min_history:
                continue

            self.cf_stats['eligible'] += 1
            pseudo_edge_id = self._cf_rng.choice(history_edges)
            pseudo_item_id = int(self.forward_edge_items[pseudo_edge_id].item())

            probe_mask = base_mask.clone()
            probe_mask[pseudo_edge_id] = 0.0

            with torch.no_grad():
                probe_embed = self.compute_result_embedding(
                    forward_edge_mask=probe_mask,
                    full_view=full_view
                )
                probe_scores = self._score_user_items(probe_embed, user_id)
                self._mask_remaining_history(
                    probe_scores,
                    user_id,
                    pseudo_item_id
                )
                rank_p = self._pseudo_positive_rank(
                    probe_scores,
                    pseudo_item_id
                )

                if not self._is_fragile_rank(rank_p):
                    continue

                boundary_item_id = self._select_fixed_boundary_item(
                    probe_scores,
                    pseudo_item_id
                )
                boundary_items = self._select_boundary_items(
                    probe_scores,
                    pseudo_item_id
                )

            if boundary_items.numel() == 0 or boundary_item_id is None:
                continue

            self.cf_stats['fragile'] += 1
            other_edges = [
                edge_id for edge_id in history_edges
                if edge_id != pseudo_edge_id
            ]
            if not other_edges:
                continue

            probe_margin = float(
                (
                    probe_scores[pseudo_item_id]
                    - probe_scores[boundary_item_id]
                ).detach().cpu()
            )
            selector_scores = self._score_cf_candidates(
                base_mask=base_mask,
                full_view=full_view,
                probe_mask=probe_mask,
                user_id=user_id,
                pseudo_item_id=pseudo_item_id,
                boundary_item_id=boundary_item_id,
                candidate_edges=other_edges
            )
            candidate_edges = self._rank_cf_candidates(
                other_edges,
                selector_scores
            )[:self.cf_selector_top_n]
            if not candidate_edges:
                continue

            second_edge_id, best_damage = self._verify_cf_candidates(
                base_mask=base_mask,
                full_view=full_view,
                user_id=user_id,
                pseudo_edge_id=pseudo_edge_id,
                pseudo_item_id=pseudo_item_id,
                boundary_item_id=boundary_item_id,
                candidate_edges=candidate_edges,
                probe_margin=probe_margin
            )
            if second_edge_id is None or best_damage <= self.cf_selector_damage_eps:
                self.cf_stats['skipped_no_damage'] += 1
                continue

            self.cf_stats['positive_damage'] += 1
            self.cf_stats['damage_sum'] += float(best_damage)
            cf_mask = base_mask.clone()
            cf_mask[pseudo_edge_id] = 0.0
            cf_mask[second_edge_id] = 0.0

            cf_embed = self.compute_result_embedding(
                forward_edge_mask=cf_mask,
                full_view=full_view
            )
            loss_u = self._boundary_pairwise_loss(
                cf_embed,
                user_id,
                pseudo_item_id,
                boundary_items
            )
            boundary_losses.append(loss_u)

            with torch.no_grad():
                self.cf_stats['used'] += 1
                self.cf_stats['loss_sum'] += float(loss_u.detach().cpu())

        if not boundary_losses:
            return reference_loss * 0.0

        return torch.stack(boundary_losses).mean()

    def _score_cf_candidates(
        self,
        base_mask,
        full_view,
        probe_mask,
        user_id,
        pseudo_item_id,
        boundary_item_id,
        candidate_edges
    ):
        """Score candidate history edges without supervising mask_logits."""
        candidate_edges = [int(edge_id) for edge_id in candidate_edges]
        if self.cf_edge_selector == 'random':
            return {
                edge_id: float(self._cf_rng.random())
                for edge_id in candidate_edges
            }

        if self.cf_edge_selector == 'representation':
            # ``probe_mask`` has already removed p, so mask_rep is the exact
            # representation state used to detect the fragile pseudo-positive.
            item_vectors = self.mask_rep[self.num_user:]
            pseudo_vector = item_vectors[int(pseudo_item_id)]
            edge_items = self.forward_edge_items[torch.tensor(
                candidate_edges,
                dtype=torch.long,
                device=self.forward_edge_items.device
            )]
            similarities = F.cosine_similarity(
                item_vectors[edge_items],
                pseudo_vector.unsqueeze(0),
                dim=1
            )
            return {
                edge_id: float(score.detach().cpu())
                for edge_id, score in zip(candidate_edges, similarities)
            }

        # Use a detached full view here. The selector obtains gradients only
        # with respect to this leaf mask and never accumulates parameter grads.
        detached_full_view = (
            full_view[0].detach(),
            full_view[1].detach()
        )
        variable_mask = probe_mask.detach().clone()
        variable_mask.requires_grad_(True)
        with torch.enable_grad():
            probe_embed = self.compute_result_embedding(
                forward_edge_mask=variable_mask,
                full_view=detached_full_view
            )
            scores = self._score_user_items(probe_embed, user_id)
            self._mask_remaining_history(
                scores,
                user_id,
                pseudo_item_id
            )
            margin = scores[int(pseudo_item_id)] - scores[int(boundary_item_id)]
            gradient = torch.autograd.grad(
                margin,
                variable_mask,
                allow_unused=False,
                retain_graph=False,
                create_graph=False
            )[0]

        estimated_damage = F.relu(
            variable_mask.detach() * gradient.detach()
        )
        return {
            edge_id: float(estimated_damage[edge_id].cpu())
            for edge_id in candidate_edges
        }

    @staticmethod
    def _rank_cf_candidates(candidate_edges, selector_scores):
        return sorted(
            (int(edge_id) for edge_id in candidate_edges),
            key=lambda edge_id: (
                -float(selector_scores.get(int(edge_id), float('-inf'))),
                int(edge_id)
            )
        )

    def _verify_cf_candidates(
        self,
        base_mask,
        full_view,
        user_id,
        pseudo_edge_id,
        pseudo_item_id,
        boundary_item_id,
        candidate_edges,
        probe_margin
    ):
        best_edge = None
        best_damage = float('-inf')
        with torch.no_grad():
            for edge_id in candidate_edges:
                cf_mask = base_mask.clone()
                cf_mask[int(pseudo_edge_id)] = 0.0
                cf_mask[int(edge_id)] = 0.0
                cf_embed = self.compute_result_embedding(
                    forward_edge_mask=cf_mask,
                    full_view=full_view
                )
                cf_scores = self._score_user_items(cf_embed, user_id)
                self._mask_remaining_history(
                    cf_scores,
                    user_id,
                    pseudo_item_id
                )
                cf_margin = (
                    cf_scores[int(pseudo_item_id)]
                    - cf_scores[int(boundary_item_id)]
                )
                damage = float(
                    (probe_margin - float(cf_margin.detach().cpu()))
                )
                self.cf_stats['candidates_verified'] += 1
                if (
                    damage > best_damage
                    or (
                        damage == best_damage
                        and best_edge is not None
                        and int(edge_id) < int(best_edge)
                    )
                ):
                    best_edge = int(edge_id)
                    best_damage = damage
        return best_edge, best_damage

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

    def _pseudo_positive_rank(self, scores, pseudo_item_id):
        pseudo_score = scores[int(pseudo_item_id)]
        return int(torch.sum(scores > pseudo_score).item()) + 1

    def _is_fragile_rank(self, rank):
        lower = self.cf_k - self.cf_boundary_width + 1
        return lower <= int(rank) <= self.cf_k

    def _select_boundary_items(self, scores, pseudo_item_id):
        top_count = min(self.num_item, self.cf_k + self.cf_boundary_q)
        if top_count <= 0 or self.cf_k > top_count:
            return torch.empty(0, dtype=torch.long, device=scores.device)

        _, ranked_items = torch.topk(scores, k=top_count, dim=0)
        boundary_items = ranked_items[
            self.cf_k - 1:min(top_count, self.cf_k + self.cf_boundary_q)
        ]
        valid_floor = torch.finfo(scores.dtype).min / 2.0
        valid = scores[boundary_items] > valid_floor
        not_pseudo = boundary_items != int(pseudo_item_id)
        return boundary_items[valid & not_pseudo].detach()

    def _select_fixed_boundary_item(self, scores, pseudo_item_id):
        """Return a fixed Top-K competitor, excluding the pseudo-positive."""
        top_count = min(self.num_item, self.cf_k + 1)
        if top_count <= 0 or self.cf_k > top_count:
            return None
        _, ranked_items = torch.topk(scores.detach(), k=top_count, dim=0)
        valid_floor = torch.finfo(scores.dtype).min / 2.0
        target = self.cf_k - 1
        for index in range(target, top_count):
            item_id = int(ranked_items[index].item())
            if item_id == int(pseudo_item_id):
                continue
            if float(scores[item_id].detach().cpu()) <= valid_floor:
                continue
            return item_id
        return None

    def _boundary_pairwise_loss(
        self,
        embedding,
        user_id,
        pseudo_item_id,
        boundary_items
    ):
        user_vector = embedding[int(user_id)]
        item_matrix = embedding[self.num_user:]
        pseudo_score = torch.sum(
            user_vector * item_matrix[int(pseudo_item_id)],
            dim=-1
        )
        boundary_scores = torch.matmul(
            item_matrix[boundary_items],
            user_vector
        )
        return F.softplus(
            self.cf_temperature * (boundary_scores - pseudo_score)
        ).mean()

