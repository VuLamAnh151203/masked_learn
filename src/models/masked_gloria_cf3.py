# coding: utf-8

"""Representation-guided mask-logit regularisation for MASKED_GLORIA.

This variant is deliberately separate from :mod:`masked_gloria_cf2`.  The
regulariser uses a held-out (pseudo-positive) history item as its target and
orders the *raw* edge-mask logits according to representation similarity.
The target similarity is stop-gradient, so the auxiliary objective updates
the mask logits only; the normal recommendation loss still updates all model
parameters as usual.
"""

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
    """MASKED_GLORIA with pseudo-positive representation mask ordering.

    For a user with history ``A, B, C, D``, one deterministic history item
    (``D`` in this example) is used as a pseudo-positive target.  The other
    history edges are candidates.  If ``R_e`` is cosine similarity between a
    candidate item representation and the detached target representation,
    the auxiliary loss is

    ``softplus(-sign(R_a - R_b) * (theta_a - theta_b) / temperature)``.

    ``theta`` is the raw ``mask_logits`` parameter.  The sigmoid mask remains
    used by the base model's graph forward, but is intentionally not used by
    this ordering loss.
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
        # Defaults use every user in the current interaction batch.  The
        # options remain configurable for large-data training.
        self.cf3_user_ratio = float(_cfg(config, 'cf3_user_ratio', 0.1))
        self.cf3_batch_size = int(_cfg(config, 'cf3_batch_size', 1024))
        self.cf3_pair_count = int(_cfg(config, 'cf3_pair_count', 16))
        # One pseudo-positive is removed from the history, hence three edges
        # are needed to leave at least two candidate edges for a pair.
        self.cf3_min_history = int(_cfg(config, 'cf3_min_history', 3))
        self.cf3_similarity_eps = float(
            _cfg(config, 'cf3_similarity_eps', 1e-6)
        )
        self.cf3_seed_offset = int(
            _cfg(config, 'cf3_seed_offset', 30000)
        )
        self.cf3_log_stats = _cfg_bool(config, 'cf3_log_stats', True)

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
        if self.cf3_min_history < 3:
            raise ValueError(
                'cf3_min_history must be at least 3 because one edge is '
                'reserved as the pseudo-positive.'
            )
        if self.cf3_similarity_eps < 0.0:
            raise ValueError('cf3_similarity_eps must be non-negative.')
        if self.cf3_warmup_epochs < 0:
            raise ValueError('cf3_warmup_epochs must be non-negative.')

    def _build_cf3_history(self):
        user_to_edge_ids = [[] for _ in range(self.num_user)]
        edge_users = self.forward_edge_users.detach().cpu().tolist()
        for edge_id, user_id in enumerate(edge_users):
            user_to_edge_ids[int(user_id)].append(int(edge_id))
        return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

    @staticmethod
    def _new_cf3_stats():
        return {
            'samples': 0,
            'eligible': 0,
            'pairs': 0,
            'loss_sum': 0.0,
            'correct_pairs': 0,
            'aligned_logit_gap_sum': 0.0,
            'similarity_gap_sum': 0.0,
        }

    def set_training_epoch(self, epoch_idx):
        epoch_idx = int(epoch_idx)
        if epoch_idx < 0:
            raise ValueError('epoch_idx must be non-negative.')
        self.current_epoch = epoch_idx
        # Kept for API compatibility and configurable deterministic sampling.
        self._cf3_rng.seed(self.cf3_seed_offset + epoch_idx)

    def pre_epoch_processing(self):
        self.cf3_stats = self._new_cf3_stats()

    def post_epoch_processing(self):
        if not self.cf3_log_stats:
            return None

        pairs = max(self.cf3_stats['pairs'], 1)
        return (
            'mask-representation-logit regularization: epoch={epoch}, '
            'warmup_epochs={warmup}, lambda={lambda_cf3:.6f}, '
            'temperature={temperature:.6f}, samples={samples}, '
            'eligible={eligible}, pairs={pairs_count}, loss={loss:.6f}, '
            'pair_accuracy={accuracy:.6f}, aligned_logit_gap={aligned:.6f}, '
            'similarity_gap={similarity:.6f}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf3_warmup_epochs),
            lambda_cf3=float(self.cf3_lambda),
            temperature=float(self.cf3_temperature),
            samples=int(self.cf3_stats['samples']),
            eligible=int(self.cf3_stats['eligible']),
            pairs_count=int(self.cf3_stats['pairs']),
            loss=float(self.cf3_stats['loss_sum'] / pairs),
            accuracy=float(self.cf3_stats['correct_pairs'] / pairs),
            aligned=float(self.cf3_stats['aligned_logit_gap_sum'] / pairs),
            similarity=float(self.cf3_stats['similarity_gap_sum'] / pairs),
        )

    def _is_cf3_active(self):
        return (
            self.training
            and self.cf3_lambda > 0.0
            and self.cf3_user_ratio > 0.0
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

        loss_relation = self._calculate_mask_relation_loss(
            interaction,
            loss_rec
        )
        weighted_relation = self.cf3_lambda * loss_relation
        self.result_embed = None
        return loss_rec, weighted_relation

    def _sample_cf3_users(self, interaction):
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = sorted(int(user_id) for user_id in users)
        if not users:
            return []

        sample_count = int(math.ceil(len(users) * self.cf3_user_ratio))
        sample_count = max(1, sample_count)
        sample_count = min(sample_count, self.cf3_batch_size, len(users))
        if sample_count == len(users):
            return users

        # Sampling is seeded per epoch, but the default ratio is 1.0, making
        # the normal configuration deterministic and free of user subsampling.
        return self._cf3_rng.sample(users, sample_count)

    def _pseudo_edge_for_user(self, user_id, edge_ids):
        """Return a stable pseudo-positive edge for this user."""
        if not edge_ids:
            raise ValueError('edge_ids must not be empty.')
        local_rng = random.Random(self.cf3_seed_offset + int(user_id))
        return int(local_rng.choice(tuple(edge_ids)))

    def _sample_cf3_pairs(self, history_size, user_id=0):
        """Return a stable, seeded pair subset for a user.

        Unlike CF2, the selected pair population does not change every batch
        or epoch.  The seeded subset avoids always favouring the first history
        edges when a user has more pairs than ``cf3_pair_count``.
        """
        pairs = list(itertools.combinations(range(history_size), 2))
        if len(pairs) <= self.cf3_pair_count:
            return pairs
        local_rng = random.Random(
            self.cf3_seed_offset + 1000003 + int(user_id)
        )
        selected = local_rng.sample(pairs, self.cf3_pair_count)
        return sorted(selected)

    def _calculate_mask_relation_loss(self, interaction, reference_loss):
        """Order raw edge logits by similarity to a detached pseudo-positive."""
        sampled_users = self._sample_cf3_users(interaction)
        self.cf3_stats['samples'] += len(sampled_users)
        if not sampled_users or self.result_embed is None:
            return reference_loss * 0.0

        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0
        if not hasattr(self, 'mask_logits'):
            return reference_loss * 0.0

        item_rep = self.mask_rep[self.num_user:]
        raw_mask_logits = self.mask_logits
        losses = []

        for user_id in sampled_users:
            edge_ids = self.user_to_edge_ids[int(user_id)]
            if len(edge_ids) < self.cf3_min_history:
                continue

            pseudo_edge = self._pseudo_edge_for_user(user_id, edge_ids)
            candidate_edges = [
                int(edge_id) for edge_id in edge_ids
                if int(edge_id) != pseudo_edge
            ]
            if len(candidate_edges) < 2:
                continue

            self.cf3_stats['eligible'] += 1
            device = self.forward_edge_users.device
            candidate_tensor = torch.tensor(
                candidate_edges,
                dtype=torch.long,
                device=device
            )
            pseudo_tensor = torch.tensor(
                [pseudo_edge],
                dtype=torch.long,
                device=device
            )
            candidate_item_ids = self.forward_edge_items[candidate_tensor]
            pseudo_item_id = self.forward_edge_items[pseudo_tensor][0]

            # Stop-gradient target and relevance: this auxiliary loss cannot
            # update item representations or GCN parameters through R_e.
            target_rep = item_rep[pseudo_item_id].detach()
            candidate_reps = item_rep[candidate_item_ids].detach()
            relevance = F.cosine_similarity(
                candidate_reps,
                target_rep.unsqueeze(0),
                dim=1
            ).detach()

            for left_pos, right_pos in self._sample_cf3_pairs(
                len(candidate_edges),
                user_id=user_id
            ):
                relevance_gap = relevance[left_pos] - relevance[right_pos]
                if abs(float(relevance_gap.cpu())) <= self.cf3_similarity_eps:
                    continue

                left_edge = candidate_edges[left_pos]
                right_edge = candidate_edges[right_pos]
                logit_gap = (
                    raw_mask_logits[left_edge] - raw_mask_logits[right_edge]
                )
                direction = torch.sign(relevance_gap).detach()
                # Temperature is a divisor, as in a standard soft ranking
                # objective; larger temperature makes ordering softer.
                pair_loss = F.softplus(
                    -direction * logit_gap / self.cf3_temperature
                )
                losses.append(pair_loss)

                with torch.no_grad():
                    aligned_gap = direction * logit_gap
                    self.cf3_stats['pairs'] += 1
                    self.cf3_stats['loss_sum'] += float(
                        pair_loss.detach().cpu()
                    )
                    self.cf3_stats['correct_pairs'] += int(
                        (aligned_gap > 0).item()
                    )
                    self.cf3_stats['aligned_logit_gap_sum'] += float(
                        aligned_gap.detach().cpu()
                    )
                    self.cf3_stats['similarity_gap_sum'] += float(
                        relevance_gap.abs().detach().cpu()
                    )

        if not losses:
            return reference_loss * 0.0
        return torch.stack(losses).mean()
