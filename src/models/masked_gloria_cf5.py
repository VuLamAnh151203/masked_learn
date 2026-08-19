# # coding: utf-8

# import itertools
# import math
# import random

# import torch
# import torch.nn.functional as F

# from models.masked_gloria import MASKED_GLORIA


# def _cfg(config, key, default):
#     try:
#         value = config[key]
#     except Exception:
#         return default
#     return default if value is None else value


# def _cfg_bool(config, key, default):
#     value = _cfg(config, key, default)
#     if isinstance(value, str):
#         return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
#     return bool(value)


# class MASKED_GLORIA_CF5(MASKED_GLORIA):
#     """MASKED_GLORIA with broad mask loss + target-aware counterfactual loss.

#     Broad loss:
#         preserve the existing history-prototype-guided ordering of edge masks.

#     Counterfactual loss:
#         for each sampled (user, positive, negative) training triple, construct
#         a factual target-aware history readout and a counterfactual readout.
#         Both use the same history and the same static mask prior; only the
#         target-specific relevance is intervened from q(e,p) to -q(e,p).

#         The loss encourages the factual-vs-counterfactual effect to be larger
#         for the positive target than for the negative target.

#     No graph-edge dropping and no full-branch teacher are required.
#     """

#     def __init__(self, config, dataset):
#         super(MASKED_GLORIA_CF5, self).__init__(config, dataset)

#         self.cf5_lambda = float(_cfg(config, 'cf5_lambda', 0.1))
#         self.cf5_temperature = float(
#             _cfg(config, 'cf5_temperature', 1.0)
#         )
#         self.cf5_warmup_ratio = float(
#             _cfg(config, 'cf5_warmup_ratio', 0.10)
#         )
#         configured_warmup_epochs = int(
#             _cfg(config, 'cf5_warmup_epochs', 50)
#         )
#         self.cf5_user_ratio = float(_cfg(config, 'cf5_user_ratio', 0.10))
#         self.cf5_batch_size = int(_cfg(config, 'cf5_batch_size', 8))
#         self.cf5_pair_count = int(_cfg(config, 'cf5_pair_count', 32))
#         self.cf5_min_history = int(_cfg(config, 'cf5_min_history', 2))
#         self.cf5_similarity_eps = float(
#             _cfg(config, 'cf5_similarity_eps', 1e-6)
#         )
#         self.cf5_seed_offset = int(
#             _cfg(config, 'cf5_seed_offset', 20000)
#         )
#         self.cf5_log_stats = _cfg_bool(config, 'cf5_log_stats', True)

#         # Target-aware counterfactual readout.
#         self.cf5_cf_lambda = float(
#             _cfg(config, 'cf5_cf_lambda', 0.005)
#         )
#         self.cf5_target_temperature = float(
#             _cfg(config, 'cf5_target_temperature', 1.0)
#         )
#         self.cf5_cf_temperature = float(
#             _cfg(config, 'cf5_cf_temperature', 1.0)
#         )
#         self.cf5_cf_margin = float(
#             _cfg(config, 'cf5_cf_margin', 0.0)
#         )
#         self.cf5_cf_user_ratio = float(
#             _cfg(config, 'cf5_cf_user_ratio', self.cf5_user_ratio)
#         )
#         self.cf5_cf_batch_size = int(
#             _cfg(config, 'cf5_cf_batch_size', self.cf5_batch_size)
#         )
#         self.cf5_cf_use_mask_prior = _cfg_bool(
#             config, 'cf5_cf_use_mask_prior', True
#         )
#         self.cf5_cf_detach_mask_prior = _cfg_bool(
#             config, 'cf5_cf_detach_mask_prior', True
#         )
#         self.cf5_cf_mask_eps = float(
#             _cfg(config, 'cf5_cf_mask_eps', 1e-8)
#         )

#         max_epochs = int(_cfg(config, 'epochs', 1000))
#         if configured_warmup_epochs >= 0:
#             self.cf5_warmup_epochs = configured_warmup_epochs
#         else:
#             self.cf5_warmup_epochs = int(
#                 math.ceil(max_epochs * self.cf5_warmup_ratio)
#             )

#         self._validate_cf5_config()
#         self.current_epoch = 0
#         self._cf5_rng = random.Random(self.cf5_seed_offset)
#         self.user_to_edge_ids = self._build_cf5_history()
#         self.cf5_stats = self._new_cf5_stats()

#     def _validate_cf5_config(self):
#         if self.cf5_lambda < 0.0:
#             raise ValueError('cf5_lambda must be non-negative.')
#         if self.cf5_temperature <= 0.0:
#             raise ValueError('cf5_temperature must be positive.')
#         if not 0.0 <= self.cf5_user_ratio <= 1.0:
#             raise ValueError('cf5_user_ratio must be in [0, 1].')
#         if self.cf5_batch_size <= 0:
#             raise ValueError('cf5_batch_size must be positive.')
#         if self.cf5_pair_count <= 0:
#             raise ValueError('cf5_pair_count must be positive.')
#         if self.cf5_min_history < 2:
#             raise ValueError('cf5_min_history must be at least 2.')
#         if self.cf5_similarity_eps < 0.0:
#             raise ValueError('cf5_similarity_eps must be non-negative.')
#         if self.cf5_warmup_epochs < 0:
#             raise ValueError('cf5_warmup_epochs must be non-negative.')
#         if self.cf5_cf_lambda < 0.0:
#             raise ValueError('cf5_cf_lambda must be non-negative.')
#         if self.cf5_target_temperature <= 0.0:
#             raise ValueError('cf5_target_temperature must be positive.')
#         if self.cf5_cf_temperature <= 0.0:
#             raise ValueError('cf5_cf_temperature must be positive.')
#         if self.cf5_cf_margin < 0.0:
#             raise ValueError('cf5_cf_margin must be non-negative.')
#         if not 0.0 <= self.cf5_cf_user_ratio <= 1.0:
#             raise ValueError('cf5_cf_user_ratio must be in [0, 1].')
#         if self.cf5_cf_batch_size <= 0:
#             raise ValueError('cf5_cf_batch_size must be positive.')
#         if self.cf5_cf_mask_eps <= 0.0:
#             raise ValueError('cf5_cf_mask_eps must be positive.')

#     def _build_cf5_history(self):
#         user_to_edge_ids = [[] for _ in range(self.num_user)]
#         edge_users = self.forward_edge_users.detach().cpu().tolist()
#         for edge_id, user_id in enumerate(edge_users):
#             user_to_edge_ids[int(user_id)].append(int(edge_id))
#         return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

#     @staticmethod
#     def _new_cf5_stats():
#         return {
#             'samples': 0,
#             'eligible': 0,
#             'pairs': 0,
#             'used': 0,
#             'loss_sum': 0.0,
#             'similarity_gap_sum': 0.0,
#             'cf_samples': 0,
#             'cf_eligible': 0,
#             'cf_used': 0,
#             'cf_loss_sum': 0.0,
#             'cf_pos_effect_sum': 0.0,
#             'cf_neg_effect_sum': 0.0,
#             'cf_effect_gap_sum': 0.0,
#             'cf_pos_fact_sum': 0.0,
#             'cf_pos_counter_sum': 0.0,
#             'cf_neg_fact_sum': 0.0,
#             'cf_neg_counter_sum': 0.0,
#         }

#     def set_training_epoch(self, epoch_idx):
#         epoch_idx = int(epoch_idx)
#         if epoch_idx < 0:
#             raise ValueError('epoch_idx must be non-negative.')
#         self.current_epoch = epoch_idx
#         self._cf5_rng.seed(self.cf5_seed_offset + epoch_idx)

#     def pre_epoch_processing(self):
#         self.cf5_stats = self._new_cf5_stats()

#     def post_epoch_processing(self):
#         if not self.cf5_log_stats:
#             return None

#         used = max(self.cf5_stats['used'], 1)
#         cf_used = max(self.cf5_stats['cf_used'], 1)
#         return (
#             'broad-mask + target-aware counterfactual: '
#             'epoch={epoch}, warmup_epochs={warmup}, '
#             'broad_lambda={lambda_cf5:.6f}, '
#             'broad_temperature={temperature:.6f}, '
#             'broad_samples={samples}, broad_eligible={eligible}, '
#             'broad_pairs={pairs}, broad_used={used_count}, '
#             'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
#             'cf_lambda={cf_lambda:.6f}, '
#             'target_temperature={target_temperature:.6f}, '
#             'cf_temperature={cf_temperature:.6f}, '
#             'cf_margin={cf_margin:.6f}, '
#             'cf_samples={cf_samples}, cf_eligible={cf_eligible}, '
#             'cf_used={cf_used_count}, cf_loss={cf_loss:.6f}, '
#             'cf_pos_effect={cf_pos_effect:.6f}, '
#             'cf_neg_effect={cf_neg_effect:.6f}, '
#             'cf_effect_gap={cf_effect_gap:.6f}, '
#             'pos_fact={pos_fact:.6f}, pos_counter={pos_counter:.6f}, '
#             'neg_fact={neg_fact:.6f}, neg_counter={neg_counter:.6f}'
#         ).format(
#             epoch=int(self.current_epoch),
#             warmup=int(self.cf5_warmup_epochs),
#             lambda_cf5=float(self.cf5_lambda),
#             temperature=float(self.cf5_temperature),
#             samples=int(self.cf5_stats['samples']),
#             eligible=int(self.cf5_stats['eligible']),
#             pairs=int(self.cf5_stats['pairs']),
#             used_count=int(self.cf5_stats['used']),
#             loss=float(self.cf5_stats['loss_sum'] / used),
#             gap=float(self.cf5_stats['similarity_gap_sum'] / used),
#             cf_lambda=float(self.cf5_cf_lambda),
#             target_temperature=float(self.cf5_target_temperature),
#             cf_temperature=float(self.cf5_cf_temperature),
#             cf_margin=float(self.cf5_cf_margin),
#             cf_samples=int(self.cf5_stats['cf_samples']),
#             cf_eligible=int(self.cf5_stats['cf_eligible']),
#             cf_used_count=int(self.cf5_stats['cf_used']),
#             cf_loss=float(self.cf5_stats['cf_loss_sum'] / cf_used),
#             cf_pos_effect=float(self.cf5_stats['cf_pos_effect_sum'] / cf_used),
#             cf_neg_effect=float(self.cf5_stats['cf_neg_effect_sum'] / cf_used),
#             cf_effect_gap=float(self.cf5_stats['cf_effect_gap_sum'] / cf_used),
#             pos_fact=float(self.cf5_stats['cf_pos_fact_sum'] / cf_used),
#             pos_counter=float(self.cf5_stats['cf_pos_counter_sum'] / cf_used),
#             neg_fact=float(self.cf5_stats['cf_neg_fact_sum'] / cf_used),
#             neg_counter=float(self.cf5_stats['cf_neg_counter_sum'] / cf_used),
#         )

#     def _is_cf5_active(self):
#         return (
#             self.training
#             and (self.cf5_lambda > 0.0 or self.cf5_cf_lambda > 0.0)
#             and (self.cf5_user_ratio > 0.0 or self.cf5_cf_user_ratio > 0.0)
#             and self.current_epoch >= self.cf5_warmup_epochs
#         )

#     def _calculate_rec_loss(self, interaction):
#         pos_scores, neg_scores = self.forward(interaction)
#         return -torch.mean(
#             torch.log2(torch.sigmoid(pos_scores - neg_scores))
#         )

#     def calculate_loss(self, interaction):
#         loss_rec = self._calculate_rec_loss(interaction)
#         if not self._is_cf5_active():
#             self.result_embed = None
#             return loss_rec

#         if self.cf5_lambda > 0.0:
#             loss_mask_relation = self._calculate_mask_relation_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_mask_relation = loss_rec * 0.0

#         if self.cf5_cf_lambda > 0.0:
#             loss_counterfactual = self._calculate_counterfactual_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_counterfactual = loss_rec * 0.0

#         weighted_auxiliary = (
#             self.cf5_lambda * loss_mask_relation
#             + self.cf5_cf_lambda * loss_counterfactual
#         )
#         self.result_embed = None
#         return loss_rec, weighted_auxiliary

#     def _sample_cf5_users(self, interaction):
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(math.ceil(len(users) * self.cf5_user_ratio))
#         sample_count = max(1, sample_count)
#         sample_count = min(sample_count, self.cf5_batch_size, len(users))
#         return self._cf5_rng.sample(users, sample_count)

#     def _sample_cf5_pairs(self, history_size):
#         total_pairs = history_size * (history_size - 1) // 2
#         if total_pairs <= self.cf5_pair_count:
#             return list(itertools.combinations(range(history_size), 2))

#         pairs = set()
#         while len(pairs) < self.cf5_pair_count:
#             left = self._cf5_rng.randrange(history_size)
#             right = self._cf5_rng.randrange(history_size)
#             if left == right:
#                 continue
#             pairs.add(tuple(sorted((left, right))))
#         return list(pairs)

#     def _calculate_mask_relation_loss(self, interaction, reference_loss):
#         """Apply pairwise similarity ordering to current mask weights."""
#         sampled_users = self._sample_cf5_users(interaction)
#         self.cf5_stats['samples'] += len(sampled_users)
#         if not sampled_users or self.result_embed is None:
#             return reference_loss * 0.0

#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()
#         losses = []

#         for user_id in sampled_users:
#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             self.cf5_stats['eligible'] += 1
#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             item_ids = self.forward_edge_items[edge_tensor]
#             prototype = item_rep[item_ids].mean(dim=0)
#             relevance = F.cosine_similarity(
#                 item_rep[item_ids],
#                 prototype.unsqueeze(0),
#                 dim=1
#             ).detach()

#             pair_positions = self._sample_cf5_pairs(len(edge_ids))

#             for left_pos, right_pos in pair_positions:
#                 relevance_gap = (
#                     relevance[left_pos] - relevance[right_pos]
#                 )
#                 if abs(float(relevance_gap.detach().cpu())) <= self.cf5_similarity_eps:
#                     continue

#                 left_edge = int(edge_ids[left_pos])
#                 right_edge = int(edge_ids[right_pos])
#                 mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
#                 direction = torch.sign(relevance_gap)
#                 pair_loss = F.softplus(
#                     -self.cf5_temperature * direction * mask_gap
#                 )
#                 losses.append(pair_loss)

#                 with torch.no_grad():
#                     self.cf5_stats['pairs'] += 1
#                     self.cf5_stats['used'] += 1
#                     self.cf5_stats['loss_sum'] += float(
#                         pair_loss.detach().cpu()
#                     )
#                     self.cf5_stats['similarity_gap_sum'] += float(
#                         relevance_gap.abs().detach().cpu()
#                     )

#         if not losses:
#             return reference_loss * 0.0
#         return torch.stack(losses).mean()

#     def _sample_cf_users(self, interaction):
#         """Sample users for target-aware counterfactual supervision."""
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(math.ceil(len(users) * self.cf5_cf_user_ratio))
#         sample_count = max(1, sample_count)
#         sample_count = min(sample_count, self.cf5_cf_batch_size, len(users))
#         return self._cf5_rng.sample(users, sample_count)

#     @staticmethod
#     def _build_batch_pos_neg_targets(interaction):
#         """Map each user to its (positive, negative) pairs in the batch.

#         Assumes the standard triplet layout:
#             interaction[0] -> user ids
#             interaction[1] -> positive item ids
#             interaction[2] -> negative item ids
#         """
#         if interaction is None or len(interaction) < 3:
#             return {}

#         users = interaction[0].detach().view(-1).cpu().tolist()
#         positives = interaction[1].detach().view(-1).cpu().tolist()
#         negatives = interaction[2].detach().view(-1).cpu().tolist()

#         if not (len(users) == len(positives) == len(negatives)):
#             return {}

#         mapping = {}
#         for user_id, pos_item, neg_item in zip(users, positives, negatives):
#             mapping.setdefault(int(user_id), []).append(
#                 (int(pos_item), int(neg_item))
#             )
#         return mapping

#     def _target_aware_effect(
#         self,
#         history_rep,
#         target_rep,
#         history_mask_weights=None
#     ):
#         """Compute factual score, counterfactual score, and their effect.

#         The factual view emphasizes history items similar to the target.
#         The counterfactual view reverses only target-specific relevance q -> -q.
#         Static mask prior is identical in both views.
#         """
#         target_similarity = F.cosine_similarity(
#             history_rep,
#             target_rep.unsqueeze(0),
#             dim=1
#         )

#         factual_logits = target_similarity / self.cf5_target_temperature
#         counter_logits = -target_similarity / self.cf5_target_temperature

#         if self.cf5_cf_use_mask_prior and history_mask_weights is not None:
#             mask_prior = history_mask_weights.clamp_min(self.cf5_cf_mask_eps)
#             if self.cf5_cf_detach_mask_prior:
#                 mask_prior = mask_prior.detach()
#             log_mask_prior = torch.log(mask_prior)
#             factual_logits = factual_logits + log_mask_prior
#             counter_logits = counter_logits + log_mask_prior

#         factual_attention = torch.softmax(factual_logits, dim=0)
#         counter_attention = torch.softmax(counter_logits, dim=0)

#         factual_history = torch.sum(
#             factual_attention.unsqueeze(-1) * history_rep,
#             dim=0
#         )
#         counter_history = torch.sum(
#             counter_attention.unsqueeze(-1) * history_rep,
#             dim=0
#         )

#         factual_score = F.cosine_similarity(
#             factual_history.unsqueeze(0),
#             target_rep.unsqueeze(0),
#             dim=1
#         ).squeeze(0)
#         counter_score = F.cosine_similarity(
#             counter_history.unsqueeze(0),
#             target_rep.unsqueeze(0),
#             dim=1
#         ).squeeze(0)

#         effect = factual_score - counter_score
#         return factual_score, counter_score, effect

#     def _calculate_counterfactual_loss(self, interaction, reference_loss):
#         """Counterfactual target-aware ranking loss.

#         delta_pos = s_fact(u,p+) - s_cf(u,p+)
#         delta_neg = s_fact(u,p-) - s_cf(u,p-)

#         Objective:
#             delta_pos > delta_neg + margin

#         L_cf = softplus(
#             (delta_neg - delta_pos + margin) / temperature
#         )
#         """
#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         target_pairs_by_user = self._build_batch_pos_neg_targets(interaction)
#         if not target_pairs_by_user:
#             return reference_loss * 0.0

#         sampled_users = self._sample_cf_users(interaction)
#         self.cf5_stats['cf_samples'] += len(sampled_users)
#         if not sampled_users:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()
#         losses = []

#         for user_id in sampled_users:
#             user_pairs = target_pairs_by_user.get(int(user_id), [])
#             if not user_pairs:
#                 continue

#             pos_item, neg_item = self._cf5_rng.choice(user_pairs)
#             if (
#                 pos_item < 0
#                 or neg_item < 0
#                 or pos_item >= item_rep.size(0)
#                 or neg_item >= item_rep.size(0)
#             ):
#                 continue

#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             history_item_ids = self.forward_edge_items[edge_tensor]

#             # Same history for positive and negative; remove either target if
#             # already present to avoid trivial cos(z_p, z_p) = 1.
#             keep = (
#                 (history_item_ids != pos_item)
#                 & (history_item_ids != neg_item)
#             )
#             history_item_ids = history_item_ids[keep]
#             history_edge_tensor = edge_tensor[keep]

#             if history_item_ids.numel() < self.cf5_min_history:
#                 continue

#             self.cf5_stats['cf_eligible'] += 1

#             history_rep = item_rep[history_item_ids]
#             history_mask = mask_weights[history_edge_tensor]
#             pos_target_rep = item_rep[pos_item]
#             neg_target_rep = item_rep[neg_item]

#             pos_fact, pos_counter, pos_effect = self._target_aware_effect(
#                 history_rep,
#                 pos_target_rep,
#                 history_mask
#             )
#             neg_fact, neg_counter, neg_effect = self._target_aware_effect(
#                 history_rep,
#                 neg_target_rep,
#                 history_mask
#             )

#             effect_gap = pos_effect - neg_effect
#             user_loss = F.softplus(
#                 (
#                     neg_effect
#                     - pos_effect
#                     + self.cf5_cf_margin
#                 ) / self.cf5_cf_temperature
#             )
#             losses.append(user_loss)

#             with torch.no_grad():
#                 self.cf5_stats['cf_used'] += 1
#                 self.cf5_stats['cf_loss_sum'] += float(user_loss.detach().cpu())
#                 self.cf5_stats['cf_pos_effect_sum'] += float(pos_effect.detach().cpu())
#                 self.cf5_stats['cf_neg_effect_sum'] += float(neg_effect.detach().cpu())
#                 self.cf5_stats['cf_effect_gap_sum'] += float(effect_gap.detach().cpu())
#                 self.cf5_stats['cf_pos_fact_sum'] += float(pos_fact.detach().cpu())
#                 self.cf5_stats['cf_pos_counter_sum'] += float(pos_counter.detach().cpu())
#                 self.cf5_stats['cf_neg_fact_sum'] += float(neg_fact.detach().cpu())
#                 self.cf5_stats['cf_neg_counter_sum'] += float(neg_counter.detach().cpu())

#         if not losses:
#             return reference_loss * 0.0

#         return torch.stack(losses).mean()


# coding: utf-8

# import itertools
# import math
# import random

# import torch
# import torch.nn.functional as F

# from models.masked_gloria import MASKED_GLORIA


# def _cfg(config, key, default):
#     try:
#         value = config[key]
#     except Exception:
#         return default
#     return default if value is None else value


# def _cfg_bool(config, key, default):
#     value = _cfg(config, key, default)
#     if isinstance(value, str):
#         return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
#     return bool(value)


# class MASKED_GLORIA_CF5(MASKED_GLORIA):
#     """MASKED_GLORIA with broad mask loss + target-aware counterfactual loss.

#     Broad loss:
#         preserve the existing history-prototype-guided ordering of edge masks.

#     Counterfactual loss:
#         for each sampled (user, positive, negative) training triple, construct
#         a factual target-aware history readout and a target-agnostic
#         counterfactual readout. Both use the same history and the same static
#         mask prior. The intervention removes target-specific relevance:

#             q(e,p) -> 0

#         so factual = global/static importance + target-specific relevance,
#         while counterfactual = global/static importance only.

#         The loss encourages the factual-vs-counterfactual effect to be larger
#         for the positive target than for the negative target.

#     No graph-edge dropping and no full-branch teacher are required.
#     """

#     def __init__(self, config, dataset):
#         super(MASKED_GLORIA_CF5, self).__init__(config, dataset)

#         self.cf5_lambda = float(_cfg(config, 'cf5_lambda', 0.1))
#         self.cf5_temperature = float(
#             _cfg(config, 'cf5_temperature', 1.0)
#         )
#         self.cf5_warmup_ratio = float(
#             _cfg(config, 'cf5_warmup_ratio', 0.10)
#         )
#         configured_warmup_epochs = int(
#             _cfg(config, 'cf5_warmup_epochs', 50)
#         )
#         self.cf5_user_ratio = float(_cfg(config, 'cf5_user_ratio', 0.10))
#         self.cf5_batch_size = int(_cfg(config, 'cf5_batch_size', 8))
#         self.cf5_pair_count = int(_cfg(config, 'cf5_pair_count', 32))
#         self.cf5_min_history = int(_cfg(config, 'cf5_min_history', 2))
#         self.cf5_similarity_eps = float(
#             _cfg(config, 'cf5_similarity_eps', 1e-6)
#         )
#         self.cf5_seed_offset = int(
#             _cfg(config, 'cf5_seed_offset', 20000)
#         )
#         self.cf5_log_stats = _cfg_bool(config, 'cf5_log_stats', True)

#         # Target-aware counterfactual readout.
#         self.cf5_cf_lambda = float(
#             _cfg(config, 'cf5_cf_lambda', 0.005)
#         )
#         self.cf5_target_temperature = float(
#             _cfg(config, 'cf5_target_temperature', 1.0)
#         )
#         self.cf5_cf_temperature = float(
#             _cfg(config, 'cf5_cf_temperature', 1.0)
#         )
#         self.cf5_cf_margin = float(
#             _cfg(config, 'cf5_cf_margin', 0.0)
#         )
#         self.cf5_cf_user_ratio = float(
#             _cfg(config, 'cf5_cf_user_ratio', self.cf5_user_ratio)
#         )
#         self.cf5_cf_batch_size = int(
#             _cfg(config, 'cf5_cf_batch_size', self.cf5_batch_size)
#         )
#         self.cf5_cf_use_mask_prior = _cfg_bool(
#             config, 'cf5_cf_use_mask_prior', True
#         )
#         self.cf5_cf_detach_mask_prior = _cfg_bool(
#             config, 'cf5_cf_detach_mask_prior', True
#         )
#         self.cf5_cf_mask_eps = float(
#             _cfg(config, 'cf5_cf_mask_eps', 1e-8)
#         )

#         max_epochs = int(_cfg(config, 'epochs', 1000))
#         if configured_warmup_epochs >= 0:
#             self.cf5_warmup_epochs = configured_warmup_epochs
#         else:
#             self.cf5_warmup_epochs = int(
#                 math.ceil(max_epochs * self.cf5_warmup_ratio)
#             )

#         self._validate_cf5_config()
#         self.current_epoch = 0
#         self._cf5_rng = random.Random(self.cf5_seed_offset)
#         self.user_to_edge_ids = self._build_cf5_history()
#         self.cf5_stats = self._new_cf5_stats()

#     def _validate_cf5_config(self):
#         if self.cf5_lambda < 0.0:
#             raise ValueError('cf5_lambda must be non-negative.')
#         if self.cf5_temperature <= 0.0:
#             raise ValueError('cf5_temperature must be positive.')
#         if not 0.0 <= self.cf5_user_ratio <= 1.0:
#             raise ValueError('cf5_user_ratio must be in [0, 1].')
#         if self.cf5_batch_size <= 0:
#             raise ValueError('cf5_batch_size must be positive.')
#         if self.cf5_pair_count <= 0:
#             raise ValueError('cf5_pair_count must be positive.')
#         if self.cf5_min_history < 2:
#             raise ValueError('cf5_min_history must be at least 2.')
#         if self.cf5_similarity_eps < 0.0:
#             raise ValueError('cf5_similarity_eps must be non-negative.')
#         if self.cf5_warmup_epochs < 0:
#             raise ValueError('cf5_warmup_epochs must be non-negative.')
#         if self.cf5_cf_lambda < 0.0:
#             raise ValueError('cf5_cf_lambda must be non-negative.')
#         if self.cf5_target_temperature <= 0.0:
#             raise ValueError('cf5_target_temperature must be positive.')
#         if self.cf5_cf_temperature <= 0.0:
#             raise ValueError('cf5_cf_temperature must be positive.')
#         if self.cf5_cf_margin < 0.0:
#             raise ValueError('cf5_cf_margin must be non-negative.')
#         if not 0.0 <= self.cf5_cf_user_ratio <= 1.0:
#             raise ValueError('cf5_cf_user_ratio must be in [0, 1].')
#         if self.cf5_cf_batch_size <= 0:
#             raise ValueError('cf5_cf_batch_size must be positive.')
#         if self.cf5_cf_mask_eps <= 0.0:
#             raise ValueError('cf5_cf_mask_eps must be positive.')

#     def _build_cf5_history(self):
#         user_to_edge_ids = [[] for _ in range(self.num_user)]
#         edge_users = self.forward_edge_users.detach().cpu().tolist()
#         for edge_id, user_id in enumerate(edge_users):
#             user_to_edge_ids[int(user_id)].append(int(edge_id))
#         return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

#     @staticmethod
#     def _new_cf5_stats():
#         return {
#             'samples': 0,
#             'eligible': 0,
#             'pairs': 0,
#             'used': 0,
#             'loss_sum': 0.0,
#             'similarity_gap_sum': 0.0,
#             'cf_samples': 0,
#             'cf_eligible': 0,
#             'cf_used': 0,
#             'cf_loss_sum': 0.0,
#             'cf_pos_effect_sum': 0.0,
#             'cf_neg_effect_sum': 0.0,
#             'cf_effect_gap_sum': 0.0,
#             'cf_pos_fact_sum': 0.0,
#             'cf_pos_counter_sum': 0.0,
#             'cf_neg_fact_sum': 0.0,
#             'cf_neg_counter_sum': 0.0,
#         }

#     def set_training_epoch(self, epoch_idx):
#         epoch_idx = int(epoch_idx)
#         if epoch_idx < 0:
#             raise ValueError('epoch_idx must be non-negative.')
#         self.current_epoch = epoch_idx
#         self._cf5_rng.seed(self.cf5_seed_offset + epoch_idx)

#     def pre_epoch_processing(self):
#         self.cf5_stats = self._new_cf5_stats()

#     def post_epoch_processing(self):
#         if not self.cf5_log_stats:
#             return None

#         used = max(self.cf5_stats['used'], 1)
#         cf_used = max(self.cf5_stats['cf_used'], 1)
#         return (
#             'broad-mask + q0 target-aware counterfactual: '
#             'epoch={epoch}, warmup_epochs={warmup}, '
#             'broad_lambda={lambda_cf5:.6f}, '
#             'broad_temperature={temperature:.6f}, '
#             'broad_samples={samples}, broad_eligible={eligible}, '
#             'broad_pairs={pairs}, broad_used={used_count}, '
#             'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
#             'cf_lambda={cf_lambda:.6f}, '
#             'target_temperature={target_temperature:.6f}, '
#             'cf_temperature={cf_temperature:.6f}, '
#             'cf_margin={cf_margin:.6f}, '
#             'cf_samples={cf_samples}, cf_eligible={cf_eligible}, '
#             'cf_used={cf_used_count}, cf_loss={cf_loss:.6f}, '
#             'cf_pos_effect={cf_pos_effect:.6f}, '
#             'cf_neg_effect={cf_neg_effect:.6f}, '
#             'cf_effect_gap={cf_effect_gap:.6f}, '
#             'pos_fact={pos_fact:.6f}, pos_counter={pos_counter:.6f}, '
#             'neg_fact={neg_fact:.6f}, neg_counter={neg_counter:.6f}'
#         ).format(
#             epoch=int(self.current_epoch),
#             warmup=int(self.cf5_warmup_epochs),
#             lambda_cf5=float(self.cf5_lambda),
#             temperature=float(self.cf5_temperature),
#             samples=int(self.cf5_stats['samples']),
#             eligible=int(self.cf5_stats['eligible']),
#             pairs=int(self.cf5_stats['pairs']),
#             used_count=int(self.cf5_stats['used']),
#             loss=float(self.cf5_stats['loss_sum'] / used),
#             gap=float(self.cf5_stats['similarity_gap_sum'] / used),
#             cf_lambda=float(self.cf5_cf_lambda),
#             target_temperature=float(self.cf5_target_temperature),
#             cf_temperature=float(self.cf5_cf_temperature),
#             cf_margin=float(self.cf5_cf_margin),
#             cf_samples=int(self.cf5_stats['cf_samples']),
#             cf_eligible=int(self.cf5_stats['cf_eligible']),
#             cf_used_count=int(self.cf5_stats['cf_used']),
#             cf_loss=float(self.cf5_stats['cf_loss_sum'] / cf_used),
#             cf_pos_effect=float(self.cf5_stats['cf_pos_effect_sum'] / cf_used),
#             cf_neg_effect=float(self.cf5_stats['cf_neg_effect_sum'] / cf_used),
#             cf_effect_gap=float(self.cf5_stats['cf_effect_gap_sum'] / cf_used),
#             pos_fact=float(self.cf5_stats['cf_pos_fact_sum'] / cf_used),
#             pos_counter=float(self.cf5_stats['cf_pos_counter_sum'] / cf_used),
#             neg_fact=float(self.cf5_stats['cf_neg_fact_sum'] / cf_used),
#             neg_counter=float(self.cf5_stats['cf_neg_counter_sum'] / cf_used),
#         )

#     def _is_cf5_active(self):
#         return (
#             self.training
#             and (self.cf5_lambda > 0.0 or self.cf5_cf_lambda > 0.0)
#             and (self.cf5_user_ratio > 0.0 or self.cf5_cf_user_ratio > 0.0)
#             and self.current_epoch >= self.cf5_warmup_epochs
#         )

#     def _calculate_rec_loss(self, interaction):
#         pos_scores, neg_scores = self.forward(interaction)
#         return -torch.mean(
#             torch.log2(torch.sigmoid(pos_scores - neg_scores))
#         )

#     def calculate_loss(self, interaction):
#         loss_rec = self._calculate_rec_loss(interaction)
#         if not self._is_cf5_active():
#             self.result_embed = None
#             return loss_rec

#         if self.cf5_lambda > 0.0:
#             loss_mask_relation = self._calculate_mask_relation_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_mask_relation = loss_rec * 0.0

#         if self.cf5_cf_lambda > 0.0:
#             loss_counterfactual = self._calculate_counterfactual_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_counterfactual = loss_rec * 0.0

#         weighted_auxiliary = (
#             self.cf5_lambda * loss_mask_relation
#             + self.cf5_cf_lambda * loss_counterfactual
#         )
#         self.result_embed = None
#         return loss_rec, weighted_auxiliary

#     def _sample_cf5_users(self, interaction):
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(math.ceil(len(users) * self.cf5_user_ratio))
#         sample_count = max(1, sample_count)
#         sample_count = min(sample_count, self.cf5_batch_size, len(users))
#         return self._cf5_rng.sample(users, sample_count)

#     def _sample_cf5_pairs(self, history_size):
#         total_pairs = history_size * (history_size - 1) // 2
#         if total_pairs <= self.cf5_pair_count:
#             return list(itertools.combinations(range(history_size), 2))

#         pairs = set()
#         while len(pairs) < self.cf5_pair_count:
#             left = self._cf5_rng.randrange(history_size)
#             right = self._cf5_rng.randrange(history_size)
#             if left == right:
#                 continue
#             pairs.add(tuple(sorted((left, right))))
#         return list(pairs)

#     def _calculate_mask_relation_loss(self, interaction, reference_loss):
#         """Apply pairwise similarity ordering to current mask weights."""
#         sampled_users = self._sample_cf5_users(interaction)
#         self.cf5_stats['samples'] += len(sampled_users)
#         if not sampled_users or self.result_embed is None:
#             return reference_loss * 0.0

#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()
#         losses = []

#         for user_id in sampled_users:
#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             self.cf5_stats['eligible'] += 1
#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             item_ids = self.forward_edge_items[edge_tensor]
#             prototype = item_rep[item_ids].mean(dim=0)
#             relevance = F.cosine_similarity(
#                 item_rep[item_ids],
#                 prototype.unsqueeze(0),
#                 dim=1
#             ).detach()

#             pair_positions = self._sample_cf5_pairs(len(edge_ids))

#             for left_pos, right_pos in pair_positions:
#                 relevance_gap = (
#                     relevance[left_pos] - relevance[right_pos]
#                 )
#                 if abs(float(relevance_gap.detach().cpu())) <= self.cf5_similarity_eps:
#                     continue

#                 left_edge = int(edge_ids[left_pos])
#                 right_edge = int(edge_ids[right_pos])
#                 mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
#                 direction = torch.sign(relevance_gap)
#                 pair_loss = F.softplus(
#                     -self.cf5_temperature * direction * mask_gap
#                 )
#                 losses.append(pair_loss)

#                 with torch.no_grad():
#                     self.cf5_stats['pairs'] += 1
#                     self.cf5_stats['used'] += 1
#                     self.cf5_stats['loss_sum'] += float(
#                         pair_loss.detach().cpu()
#                     )
#                     self.cf5_stats['similarity_gap_sum'] += float(
#                         relevance_gap.abs().detach().cpu()
#                     )

#         if not losses:
#             return reference_loss * 0.0
#         return torch.stack(losses).mean()

#     def _sample_cf_users(self, interaction):
#         """Sample users for target-aware counterfactual supervision."""
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(math.ceil(len(users) * self.cf5_cf_user_ratio))
#         sample_count = max(1, sample_count)
#         sample_count = min(sample_count, self.cf5_cf_batch_size, len(users))
#         return self._cf5_rng.sample(users, sample_count)

#     @staticmethod
#     def _build_batch_pos_neg_targets(interaction):
#         """Map each user to its (positive, negative) pairs in the batch.

#         Assumes the standard triplet layout:
#             interaction[0] -> user ids
#             interaction[1] -> positive item ids
#             interaction[2] -> negative item ids
#         """
#         if interaction is None or len(interaction) < 3:
#             return {}

#         users = interaction[0].detach().view(-1).cpu().tolist()
#         positives = interaction[1].detach().view(-1).cpu().tolist()
#         negatives = interaction[2].detach().view(-1).cpu().tolist()

#         if not (len(users) == len(positives) == len(negatives)):
#             return {}

#         mapping = {}
#         for user_id, pos_item, neg_item in zip(users, positives, negatives):
#             mapping.setdefault(int(user_id), []).append(
#                 (int(pos_item), int(neg_item))
#             )
#         return mapping

#     def _target_aware_effect(
#         self,
#         history_rep,
#         target_rep,
#         history_mask_weights=None
#     ):
#         """Compute factual score, q->0 counterfactual score, and effect.

#         Let
#             q(e,p) = cos(z_e^mask, z_p^mask).

#         Factual view:
#             alpha_fact = softmax(q / T_target + log(m_e))

#         Counterfactual view:
#             remove the target-specific component q(e,p) -> 0, therefore
#             alpha_cf = softmax(log(m_e)).

#         Thus both views keep exactly the same history and global/static mask
#         prior. The only removed information is which history items are related
#         to the current target. If mask-prior use is disabled, alpha_cf becomes
#         uniform attention over the history.
#         """
#         target_similarity = F.cosine_similarity(
#             history_rep,
#             target_rep.unsqueeze(0),
#             dim=1
#         )

#         # factual = target-specific relevance + global/static mask prior
#         factual_logits = target_similarity / self.cf5_target_temperature

#         # counterfactual intervention q -> 0. Start from all-zero logits;
#         # adding log(mask_prior) below leaves only global/static importance.
#         counter_logits = torch.zeros_like(factual_logits)

#         if self.cf5_cf_use_mask_prior and history_mask_weights is not None:
#             mask_prior = history_mask_weights.clamp_min(self.cf5_cf_mask_eps)
#             if self.cf5_cf_detach_mask_prior:
#                 mask_prior = mask_prior.detach()
#             log_mask_prior = torch.log(mask_prior)
#             factual_logits = factual_logits + log_mask_prior
#             counter_logits = counter_logits + log_mask_prior

#         factual_attention = torch.softmax(factual_logits, dim=0)
#         counter_attention = torch.softmax(counter_logits, dim=0)

#         factual_history = torch.sum(
#             factual_attention.unsqueeze(-1) * history_rep,
#             dim=0
#         )
#         counter_history = torch.sum(
#             counter_attention.unsqueeze(-1) * history_rep,
#             dim=0
#         )

#         factual_score = F.cosine_similarity(
#             factual_history.unsqueeze(0),
#             target_rep.unsqueeze(0),
#             dim=1
#         ).squeeze(0)
#         counter_score = F.cosine_similarity(
#             counter_history.unsqueeze(0),
#             target_rep.unsqueeze(0),
#             dim=1
#         ).squeeze(0)

#         effect = factual_score - counter_score
#         return factual_score, counter_score, effect

#     def _calculate_counterfactual_loss(self, interaction, reference_loss):
#         """Counterfactual target-aware ranking loss.

#         delta_pos = s_fact(u,p+) - s_cf(u,p+)
#         delta_neg = s_fact(u,p-) - s_cf(u,p-)

#         Objective:
#             delta_pos > delta_neg + margin

#         L_cf = softplus(
#             (delta_neg - delta_pos + margin) / temperature
#         )
#         """
#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         target_pairs_by_user = self._build_batch_pos_neg_targets(interaction)
#         if not target_pairs_by_user:
#             return reference_loss * 0.0

#         sampled_users = self._sample_cf_users(interaction)
#         self.cf5_stats['cf_samples'] += len(sampled_users)
#         if not sampled_users:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()
#         losses = []

#         for user_id in sampled_users:
#             user_pairs = target_pairs_by_user.get(int(user_id), [])
#             if not user_pairs:
#                 continue

#             pos_item, neg_item = self._cf5_rng.choice(user_pairs)
#             if (
#                 pos_item < 0
#                 or neg_item < 0
#                 or pos_item >= item_rep.size(0)
#                 or neg_item >= item_rep.size(0)
#             ):
#                 continue

#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             history_item_ids = self.forward_edge_items[edge_tensor]

#             # Same history for positive and negative; remove either target if
#             # already present to avoid trivial cos(z_p, z_p) = 1.
#             keep = (
#                 (history_item_ids != pos_item)
#                 & (history_item_ids != neg_item)
#             )
#             history_item_ids = history_item_ids[keep]
#             history_edge_tensor = edge_tensor[keep]

#             if history_item_ids.numel() < self.cf5_min_history:
#                 continue

#             self.cf5_stats['cf_eligible'] += 1

#             history_rep = item_rep[history_item_ids]
#             history_mask = mask_weights[history_edge_tensor]
#             pos_target_rep = item_rep[pos_item]
#             neg_target_rep = item_rep[neg_item]

#             pos_fact, pos_counter, pos_effect = self._target_aware_effect(
#                 history_rep,
#                 pos_target_rep,
#                 history_mask
#             )
#             neg_fact, neg_counter, neg_effect = self._target_aware_effect(
#                 history_rep,
#                 neg_target_rep,
#                 history_mask
#             )

#             effect_gap = pos_effect - neg_effect
#             user_loss = F.softplus(
#                 (
#                     neg_effect
#                     - pos_effect
#                     + self.cf5_cf_margin
#                 ) / self.cf5_cf_temperature
#             )
#             losses.append(user_loss)

#             with torch.no_grad():
#                 self.cf5_stats['cf_used'] += 1
#                 self.cf5_stats['cf_loss_sum'] += float(user_loss.detach().cpu())
#                 self.cf5_stats['cf_pos_effect_sum'] += float(pos_effect.detach().cpu())
#                 self.cf5_stats['cf_neg_effect_sum'] += float(neg_effect.detach().cpu())
#                 self.cf5_stats['cf_effect_gap_sum'] += float(effect_gap.detach().cpu())
#                 self.cf5_stats['cf_pos_fact_sum'] += float(pos_fact.detach().cpu())
#                 self.cf5_stats['cf_pos_counter_sum'] += float(pos_counter.detach().cpu())
#                 self.cf5_stats['cf_neg_fact_sum'] += float(neg_fact.detach().cpu())
#                 self.cf5_stats['cf_neg_counter_sum'] += float(neg_counter.detach().cpu())

#         if not losses:
#             return reference_loss * 0.0

#         return torch.stack(losses).mean()

# coding: utf-8

# import itertools
# import math
# import random

# import torch
# import torch.nn.functional as F

# from models.masked_gloria import MASKED_GLORIA


# def _cfg(config, key, default):
#     try:
#         value = config[key]
#     except Exception:
#         return default
#     return default if value is None else value


# def _cfg_bool(config, key, default):
#     value = _cfg(config, key, default)
#     if isinstance(value, str):
#         return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
#     return bool(value)


# class MASKED_GLORIA_CF5(MASKED_GLORIA):
#     """MASKED_GLORIA with broad mask consistency + target-specific residual.

#     Broad loss:
#         Keep the original history-prototype-guided mask ordering.

#     Target-specific residual:
#         Keep the learned static edge mask as a global/base prior, but create a
#         candidate-conditioned residual in mask-logit space:

#             q(e,p)       = cos(z_e^mask, z_p^mask)
#             delta(e,p)   = beta * tanh(q(e,p) / T_target)
#             m_target     = sigmoid(logit(stopgrad(m_base)) + delta(e,p))

#         The target-conditioned readout is compared with the base/global
#         readout. Their score difference is the target-specific residual score.

#         A small auxiliary BPR-style loss asks the residual correction to help
#         the positive item more than the negative item.

#     This version intentionally contains NO counterfactual loss.
#     """

#     def __init__(self, config, dataset):
#         super(MASKED_GLORIA_CF5, self).__init__(config, dataset)

#         self.cf5_lambda = float(_cfg(config, 'cf5_lambda', 0.1))
#         self.cf5_temperature = float(
#             _cfg(config, 'cf5_temperature', 1.0)
#         )
#         self.cf5_warmup_ratio = float(
#             _cfg(config, 'cf5_warmup_ratio', 0.10)
#         )
#         configured_warmup_epochs = int(
#             _cfg(config, 'cf5_warmup_epochs', 50)
#         )
#         self.cf5_user_ratio = float(_cfg(config, 'cf5_user_ratio', 0.10))
#         self.cf5_batch_size = int(_cfg(config, 'cf5_batch_size', 8))
#         self.cf5_pair_count = int(_cfg(config, 'cf5_pair_count', 32))
#         self.cf5_min_history = int(_cfg(config, 'cf5_min_history', 2))
#         self.cf5_similarity_eps = float(
#             _cfg(config, 'cf5_similarity_eps', 1e-6)
#         )
#         self.cf5_seed_offset = int(
#             _cfg(config, 'cf5_seed_offset', 20000)
#         )
#         self.cf5_log_stats = _cfg_bool(config, 'cf5_log_stats', True)

#         # -------------------------------------------------------------
#         # Target-specific residual configuration.
#         # -------------------------------------------------------------
#         self.cf5_target_lambda = float(
#             _cfg(config, 'cf5_target_lambda', 0.005)
#         )
#         self.cf5_target_beta = float(
#             _cfg(config, 'cf5_target_beta', 0.5)
#         )
#         self.cf5_target_temperature = float(
#             _cfg(config, 'cf5_target_temperature', 1.0)
#         )
#         self.cf5_target_loss_temperature = float(
#             _cfg(config, 'cf5_target_loss_temperature', 1.0)
#         )
#         self.cf5_target_user_ratio = float(
#             _cfg(config, 'cf5_target_user_ratio', self.cf5_user_ratio)
#         )
#         self.cf5_target_batch_size = int(
#             _cfg(config, 'cf5_target_batch_size', self.cf5_batch_size)
#         )
#         self.cf5_target_mask_eps = float(
#             _cfg(config, 'cf5_target_mask_eps', 1e-6)
#         )

#         max_epochs = int(_cfg(config, 'epochs', 1000))
#         if configured_warmup_epochs >= 0:
#             self.cf5_warmup_epochs = configured_warmup_epochs
#         else:
#             self.cf5_warmup_epochs = int(
#                 math.ceil(max_epochs * self.cf5_warmup_ratio)
#             )

#         self._validate_cf5_config()
#         self.current_epoch = 0
#         self._cf5_rng = random.Random(self.cf5_seed_offset)
#         self.user_to_edge_ids = self._build_cf5_history()
#         self.cf5_stats = self._new_cf5_stats()

#     def _validate_cf5_config(self):
#         if self.cf5_lambda < 0.0:
#             raise ValueError('cf5_lambda must be non-negative.')
#         if self.cf5_temperature <= 0.0:
#             raise ValueError('cf5_temperature must be positive.')
#         if not 0.0 <= self.cf5_user_ratio <= 1.0:
#             raise ValueError('cf5_user_ratio must be in [0, 1].')
#         if self.cf5_batch_size <= 0:
#             raise ValueError('cf5_batch_size must be positive.')
#         if self.cf5_pair_count <= 0:
#             raise ValueError('cf5_pair_count must be positive.')
#         if self.cf5_min_history < 2:
#             raise ValueError('cf5_min_history must be at least 2.')
#         if self.cf5_similarity_eps < 0.0:
#             raise ValueError('cf5_similarity_eps must be non-negative.')
#         if self.cf5_warmup_epochs < 0:
#             raise ValueError('cf5_warmup_epochs must be non-negative.')

#         if self.cf5_target_lambda < 0.0:
#             raise ValueError('cf5_target_lambda must be non-negative.')
#         if self.cf5_target_beta < 0.0:
#             raise ValueError('cf5_target_beta must be non-negative.')
#         if self.cf5_target_temperature <= 0.0:
#             raise ValueError('cf5_target_temperature must be positive.')
#         if self.cf5_target_loss_temperature <= 0.0:
#             raise ValueError(
#                 'cf5_target_loss_temperature must be positive.'
#             )
#         if not 0.0 <= self.cf5_target_user_ratio <= 1.0:
#             raise ValueError('cf5_target_user_ratio must be in [0, 1].')
#         if self.cf5_target_batch_size <= 0:
#             raise ValueError('cf5_target_batch_size must be positive.')
#         if not 0.0 < self.cf5_target_mask_eps < 0.5:
#             raise ValueError(
#                 'cf5_target_mask_eps must be in (0, 0.5).'
#             )

#     def _build_cf5_history(self):
#         user_to_edge_ids = [[] for _ in range(self.num_user)]
#         edge_users = self.forward_edge_users.detach().cpu().tolist()
#         for edge_id, user_id in enumerate(edge_users):
#             user_to_edge_ids[int(user_id)].append(int(edge_id))
#         return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

#     @staticmethod
#     def _new_cf5_stats():
#         return {
#             # Broad mask consistency.
#             'samples': 0,
#             'eligible': 0,
#             'pairs': 0,
#             'used': 0,
#             'loss_sum': 0.0,
#             'similarity_gap_sum': 0.0,

#             # Target-specific residual.
#             'target_samples': 0,
#             'target_eligible': 0,
#             'target_used': 0,
#             'target_loss_sum': 0.0,
#             'target_pos_residual_sum': 0.0,
#             'target_neg_residual_sum': 0.0,
#             'target_residual_gap_sum': 0.0,
#             'target_mask_shift_sum': 0.0,
#             'target_pos_similarity_sum': 0.0,
#             'target_neg_similarity_sum': 0.0,
#         }

#     def set_training_epoch(self, epoch_idx):
#         epoch_idx = int(epoch_idx)
#         if epoch_idx < 0:
#             raise ValueError('epoch_idx must be non-negative.')
#         self.current_epoch = epoch_idx
#         self._cf5_rng.seed(self.cf5_seed_offset + epoch_idx)

#     def pre_epoch_processing(self):
#         self.cf5_stats = self._new_cf5_stats()

#     def post_epoch_processing(self):
#         if not self.cf5_log_stats:
#             return None

#         used = max(self.cf5_stats['used'], 1)
#         target_used = max(self.cf5_stats['target_used'], 1)

#         return (
#             'broad-mask + target-specific residual: '
#             'epoch={epoch}, warmup_epochs={warmup}, '
#             'broad_lambda={lambda_cf5:.6f}, '
#             'broad_temperature={temperature:.6f}, '
#             'broad_samples={samples}, broad_eligible={eligible}, '
#             'broad_pairs={pairs}, broad_used={used_count}, '
#             'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
#             'target_lambda={target_lambda:.6f}, '
#             'target_beta={target_beta:.6f}, '
#             'target_temperature={target_temperature:.6f}, '
#             'target_loss_temperature={target_loss_temperature:.6f}, '
#             'target_samples={target_samples}, '
#             'target_eligible={target_eligible}, '
#             'target_used={target_used_count}, '
#             'target_loss={target_loss:.6f}, '
#             'target_pos_residual={target_pos_residual:.6f}, '
#             'target_neg_residual={target_neg_residual:.6f}, '
#             'target_residual_gap={target_residual_gap:.6f}, '
#             'target_mask_shift={target_mask_shift:.6f}, '
#             'target_pos_similarity={target_pos_similarity:.6f}, '
#             'target_neg_similarity={target_neg_similarity:.6f}'
#         ).format(
#             epoch=int(self.current_epoch),
#             warmup=int(self.cf5_warmup_epochs),

#             lambda_cf5=float(self.cf5_lambda),
#             temperature=float(self.cf5_temperature),
#             samples=int(self.cf5_stats['samples']),
#             eligible=int(self.cf5_stats['eligible']),
#             pairs=int(self.cf5_stats['pairs']),
#             used_count=int(self.cf5_stats['used']),
#             loss=float(self.cf5_stats['loss_sum'] / used),
#             gap=float(self.cf5_stats['similarity_gap_sum'] / used),

#             target_lambda=float(self.cf5_target_lambda),
#             target_beta=float(self.cf5_target_beta),
#             target_temperature=float(self.cf5_target_temperature),
#             target_loss_temperature=float(
#                 self.cf5_target_loss_temperature
#             ),
#             target_samples=int(self.cf5_stats['target_samples']),
#             target_eligible=int(self.cf5_stats['target_eligible']),
#             target_used_count=int(self.cf5_stats['target_used']),
#             target_loss=float(
#                 self.cf5_stats['target_loss_sum'] / target_used
#             ),
#             target_pos_residual=float(
#                 self.cf5_stats['target_pos_residual_sum'] / target_used
#             ),
#             target_neg_residual=float(
#                 self.cf5_stats['target_neg_residual_sum'] / target_used
#             ),
#             target_residual_gap=float(
#                 self.cf5_stats['target_residual_gap_sum'] / target_used
#             ),
#             target_mask_shift=float(
#                 self.cf5_stats['target_mask_shift_sum'] / target_used
#             ),
#             target_pos_similarity=float(
#                 self.cf5_stats['target_pos_similarity_sum'] / target_used
#             ),
#             target_neg_similarity=float(
#                 self.cf5_stats['target_neg_similarity_sum'] / target_used
#             ),
#         )

#     def _is_cf5_active(self):
#         return (
#             self.training
#             and (
#                 self.cf5_lambda > 0.0
#                 or self.cf5_target_lambda > 0.0
#             )
#             and (
#                 self.cf5_user_ratio > 0.0
#                 or self.cf5_target_user_ratio > 0.0
#             )
#             and self.current_epoch >= self.cf5_warmup_epochs
#         )

#     def _calculate_rec_loss(self, interaction):
#         pos_scores, neg_scores = self.forward(interaction)
#         return -torch.mean(
#             torch.log2(torch.sigmoid(pos_scores - neg_scores))
#         )

#     def calculate_loss(self, interaction):
#         loss_rec = self._calculate_rec_loss(interaction)
#         if not self._is_cf5_active():
#             self.result_embed = None
#             return loss_rec

#         if self.cf5_lambda > 0.0:
#             loss_mask_relation = self._calculate_mask_relation_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_mask_relation = loss_rec * 0.0

#         if self.cf5_target_lambda > 0.0:
#             loss_target_residual = self._calculate_target_residual_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_target_residual = loss_rec * 0.0

#         weighted_auxiliary = (
#             self.cf5_lambda * loss_mask_relation
#             + self.cf5_target_lambda * loss_target_residual
#         )

#         self.result_embed = None

#         # Preserve the original trainer interface.
#         return loss_rec, weighted_auxiliary

#     def _sample_cf5_users(self, interaction):
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(math.ceil(len(users) * self.cf5_user_ratio))
#         sample_count = max(1, sample_count)
#         sample_count = min(sample_count, self.cf5_batch_size, len(users))
#         return self._cf5_rng.sample(users, sample_count)

#     def _sample_cf5_pairs(self, history_size):
#         total_pairs = history_size * (history_size - 1) // 2
#         if total_pairs <= self.cf5_pair_count:
#             return list(itertools.combinations(range(history_size), 2))

#         pairs = set()
#         while len(pairs) < self.cf5_pair_count:
#             left = self._cf5_rng.randrange(history_size)
#             right = self._cf5_rng.randrange(history_size)
#             if left == right:
#                 continue
#             pairs.add(tuple(sorted((left, right))))
#         return list(pairs)

#     def _calculate_mask_relation_loss(self, interaction, reference_loss):
#         """Apply pairwise similarity ordering to current mask weights."""
#         sampled_users = self._sample_cf5_users(interaction)
#         self.cf5_stats['samples'] += len(sampled_users)
#         if not sampled_users or self.result_embed is None:
#             return reference_loss * 0.0

#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()
#         losses = []

#         for user_id in sampled_users:
#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             self.cf5_stats['eligible'] += 1
#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             item_ids = self.forward_edge_items[edge_tensor]
#             prototype = item_rep[item_ids].mean(dim=0)
#             relevance = F.cosine_similarity(
#                 item_rep[item_ids],
#                 prototype.unsqueeze(0),
#                 dim=1
#             ).detach()

#             pair_positions = self._sample_cf5_pairs(len(edge_ids))

#             for left_pos, right_pos in pair_positions:
#                 relevance_gap = (
#                     relevance[left_pos] - relevance[right_pos]
#                 )
#                 if abs(float(relevance_gap.detach().cpu())) <= self.cf5_similarity_eps:
#                     continue

#                 left_edge = int(edge_ids[left_pos])
#                 right_edge = int(edge_ids[right_pos])
#                 mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
#                 direction = torch.sign(relevance_gap)
#                 pair_loss = F.softplus(
#                     -self.cf5_temperature * direction * mask_gap
#                 )
#                 losses.append(pair_loss)

#                 with torch.no_grad():
#                     self.cf5_stats['pairs'] += 1
#                     self.cf5_stats['used'] += 1
#                     self.cf5_stats['loss_sum'] += float(
#                         pair_loss.detach().cpu()
#                     )
#                     self.cf5_stats['similarity_gap_sum'] += float(
#                         relevance_gap.abs().detach().cpu()
#                     )

#         if not losses:
#             return reference_loss * 0.0
#         return torch.stack(losses).mean()

#     def _sample_target_users(self, interaction):
#         """Sample users for the target-specific residual auxiliary task."""
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(
#             math.ceil(len(users) * self.cf5_target_user_ratio)
#         )
#         sample_count = max(1, sample_count)
#         sample_count = min(
#             sample_count,
#             self.cf5_target_batch_size,
#             len(users)
#         )
#         return self._cf5_rng.sample(users, sample_count)

#     @staticmethod
#     def _build_batch_pos_neg_targets(interaction):
#         """Map batch users to current (positive, negative) item pairs.

#         Assumes the standard layout already used by the base recommender:
#             interaction[0] -> user ids
#             interaction[1] -> positive item ids
#             interaction[2] -> negative item ids
#         """
#         if interaction is None or len(interaction) < 3:
#             return {}

#         users = interaction[0].detach().view(-1).cpu().tolist()
#         positives = interaction[1].detach().view(-1).cpu().tolist()
#         negatives = interaction[2].detach().view(-1).cpu().tolist()

#         if not (
#             len(users) == len(positives) == len(negatives)
#         ):
#             return {}

#         mapping = {}
#         for user_id, pos_item, neg_item in zip(
#             users, positives, negatives
#         ):
#             mapping.setdefault(int(user_id), []).append(
#                 (int(pos_item), int(neg_item))
#             )
#         return mapping

#     def _target_conditioned_residual_score(
#         self,
#         history_rep,
#         target_rep,
#         base_mask
#     ):
#         """Compute a target-specific residual score.

#         Static/base mask:
#             m_e

#         Target relevance:
#             q(e,p) = cos(z_e, z_p)

#         Target residual in logit space:
#             delta(e,p) = beta * tanh(q(e,p) / T)

#         Effective target-conditioned mask:
#             m_target(e,p)
#                 = sigmoid(logit(stopgrad(m_e)) + delta(e,p))

#         We build:
#             h_global  = weighted history using stopgrad(m_e)
#             h_target  = weighted history using m_target(e,p)

#         and define:
#             residual_rep   = h_target - h_global
#             residual_score = <residual_rep, z_p>

#         The base-mask PRIOR is detached in this readout path so the target
#         correction does not directly rewrite the static mask ordering.
#         """
#         if history_rep.numel() == 0:
#             zero = target_rep.sum() * 0.0
#             return zero, zero, zero

#         eps = self.cf5_target_mask_eps

#         # q(e,p): target-conditioned relevance from masked representations.
#         target_similarity = F.cosine_similarity(
#             history_rep,
#             target_rep.unsqueeze(0),
#             dim=1
#         )

#         # Fixed-form V1 residual; no MLP yet.
#         delta = self.cf5_target_beta * torch.tanh(
#             target_similarity / self.cf5_target_temperature
#         )

#         # Keep broad/static mask as a base prior only in this path.
#         base_mask_detached = base_mask.detach().clamp(
#             min=eps,
#             max=1.0 - eps
#         )

#         base_logit = torch.logit(
#             base_mask_detached,
#             eps=eps
#         )

#         target_mask = torch.sigmoid(
#             base_logit + delta
#         )

#         # Normalized weighted readouts.
#         global_weight = (
#             base_mask_detached
#             / base_mask_detached.sum().clamp_min(eps)
#         )
#         target_weight = (
#             target_mask
#             / target_mask.sum().clamp_min(eps)
#         )

#         global_history = torch.sum(
#             global_weight.unsqueeze(-1) * history_rep,
#             dim=0
#         )
#         target_history = torch.sum(
#             target_weight.unsqueeze(-1) * history_rep,
#             dim=0
#         )

#         residual_rep = target_history - global_history

#         # Use dot product to stay closer to ordinary recommendation scoring.
#         residual_score = torch.sum(
#             residual_rep * target_rep,
#             dim=-1
#         )

#         mean_mask_shift = (
#             target_mask - base_mask_detached
#         ).abs().mean()

#         mean_target_similarity = target_similarity.mean()

#         return (
#             residual_score,
#             mean_mask_shift,
#             mean_target_similarity
#         )

#     def _calculate_target_residual_loss(
#         self,
#         interaction,
#         reference_loss
#     ):
#         """Train target-specific residual without any counterfactual objective.

#         For each sampled user and current positive/negative pair:

#             R_pos = residual_score(u, p+)
#             R_neg = residual_score(u, p-)

#         The auxiliary objective asks the target-conditioned correction to help
#         the positive item more than the negative item:

#             L_target = softplus(
#                 (R_neg - R_pos) / T_loss
#             )

#         The ordinary recommendation loss remains unchanged.
#         """
#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         target_pairs_by_user = self._build_batch_pos_neg_targets(
#             interaction
#         )
#         if not target_pairs_by_user:
#             return reference_loss * 0.0

#         sampled_users = self._sample_target_users(interaction)
#         self.cf5_stats['target_samples'] += len(sampled_users)

#         if not sampled_users:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()

#         losses = []

#         for user_id in sampled_users:
#             user_pairs = target_pairs_by_user.get(int(user_id), [])
#             if not user_pairs:
#                 continue

#             pos_item, neg_item = self._cf5_rng.choice(user_pairs)

#             if (
#                 pos_item < 0
#                 or neg_item < 0
#                 or pos_item >= item_rep.size(0)
#                 or neg_item >= item_rep.size(0)
#             ):
#                 continue

#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             history_item_ids = self.forward_edge_items[edge_tensor]

#             # Avoid trivial cos(z_p, z_p)=1 if a batch target is already in
#             # the user's graph history.
#             keep = (
#                 (history_item_ids != pos_item)
#                 & (history_item_ids != neg_item)
#             )
#             history_item_ids = history_item_ids[keep]
#             history_edge_tensor = edge_tensor[keep]

#             if history_item_ids.numel() < self.cf5_min_history:
#                 continue

#             self.cf5_stats['target_eligible'] += 1

#             history_rep = item_rep[history_item_ids]
#             history_mask = mask_weights[history_edge_tensor]

#             pos_score, pos_shift, pos_similarity = (
#                 self._target_conditioned_residual_score(
#                     history_rep,
#                     item_rep[pos_item],
#                     history_mask
#                 )
#             )

#             neg_score, neg_shift, neg_similarity = (
#                 self._target_conditioned_residual_score(
#                     history_rep,
#                     item_rep[neg_item],
#                     history_mask
#                 )
#             )

#             residual_gap = pos_score - neg_score

#             user_loss = F.softplus(
#                 (
#                     neg_score - pos_score
#                 )
#                 / self.cf5_target_loss_temperature
#             )

#             losses.append(user_loss)

#             with torch.no_grad():
#                 self.cf5_stats['target_used'] += 1
#                 self.cf5_stats['target_loss_sum'] += float(
#                     user_loss.detach().cpu()
#                 )
#                 self.cf5_stats['target_pos_residual_sum'] += float(
#                     pos_score.detach().cpu()
#                 )
#                 self.cf5_stats['target_neg_residual_sum'] += float(
#                     neg_score.detach().cpu()
#                 )
#                 self.cf5_stats['target_residual_gap_sum'] += float(
#                     residual_gap.detach().cpu()
#                 )
#                 self.cf5_stats['target_mask_shift_sum'] += float(
#                     (
#                         0.5 * (pos_shift + neg_shift)
#                     ).detach().cpu()
#                 )
#                 self.cf5_stats['target_pos_similarity_sum'] += float(
#                     pos_similarity.detach().cpu()
#                 )
#                 self.cf5_stats['target_neg_similarity_sum'] += float(
#                     neg_similarity.detach().cpu()
#                 )

#         if not losses:
#             return reference_loss * 0.0

#         return torch.stack(losses).mean()


# coding: utf-8

# import itertools
# import math
# import random

# import torch
# import torch.nn.functional as F

# from models.masked_gloria import MASKED_GLORIA


# def _cfg(config, key, default):
#     try:
#         value = config[key]
#     except Exception:
#         return default
#     return default if value is None else value


# def _cfg_bool(config, key, default):
#     value = _cfg(config, key, default)
#     if isinstance(value, str):
#         return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
#     return bool(value)


# class MASKED_GLORIA_CF5(MASKED_GLORIA):
#     """MASKED_GLORIA with broad mask consistency + target-specific residual.

#     Broad loss:
#         Keep the original history-prototype-guided mask ordering.

#     Target-specific residual:
#         Keep the learned static edge mask as a detached global/base prior, and
#         let target similarity change the RELATIVE history contribution:

#             q(e,p) = cos(z_e^mask, z_p^mask)

#             alpha_base(e) ∝ stopgrad(m_e)
#             alpha_target(e,p)
#                 = softmax(log(stopgrad(m_e) + eps)
#                           + gamma * q(e,p) / T_target)

#         Equivalently:

#             alpha_target(e,p) ∝ stopgrad(m_e)
#                                   * exp(gamma * q(e,p) / T_target)

#         This avoids sigmoid saturation when the learned masks are already near
#         one and makes the target signal act directly on relative history
#         weighting. The target-conditioned readout is compared with the
#         base/global readout; their score difference is the target-specific
#         residual score.

#         A small auxiliary BPR-style loss asks the residual correction to help
#         the positive item more than the negative item.

#     This version intentionally contains NO counterfactual loss.
#     """

#     def __init__(self, config, dataset):
#         super(MASKED_GLORIA_CF5, self).__init__(config, dataset)

#         self.cf5_lambda = float(_cfg(config, 'cf5_lambda', 0.1))
#         self.cf5_temperature = float(
#             _cfg(config, 'cf5_temperature', 1.0)
#         )
#         self.cf5_warmup_ratio = float(
#             _cfg(config, 'cf5_warmup_ratio', 0.10)
#         )
#         configured_warmup_epochs = int(
#             _cfg(config, 'cf5_warmup_epochs', 50)
#         )
#         self.cf5_user_ratio = float(_cfg(config, 'cf5_user_ratio', 0.10))
#         self.cf5_batch_size = int(_cfg(config, 'cf5_batch_size', 8))
#         self.cf5_pair_count = int(_cfg(config, 'cf5_pair_count', 32))
#         self.cf5_min_history = int(_cfg(config, 'cf5_min_history', 2))
#         self.cf5_similarity_eps = float(
#             _cfg(config, 'cf5_similarity_eps', 1e-6)
#         )
#         self.cf5_seed_offset = int(
#             _cfg(config, 'cf5_seed_offset', 20000)
#         )
#         self.cf5_log_stats = _cfg_bool(config, 'cf5_log_stats', True)

#         # -------------------------------------------------------------
#         # Target-specific residual configuration.
#         # -------------------------------------------------------------
#         self.cf5_target_lambda = float(
#             _cfg(config, 'cf5_target_lambda', 0.005)
#         )
#         # Strength of target-specific relative reweighting.
#         # Backward-compatible fallback: an old cf5_target_beta value is used
#         # only when cf5_target_gamma is not provided.
#         self.cf5_target_gamma = float(
#             _cfg(
#                 config,
#                 'cf5_target_gamma',
#                 _cfg(config, 'cf5_target_beta', 1.0)
#             )
#         )
#         self.cf5_target_temperature = float(
#             _cfg(config, 'cf5_target_temperature', 1.0)
#         )
#         self.cf5_target_loss_temperature = float(
#             _cfg(config, 'cf5_target_loss_temperature', 1.0)
#         )
#         self.cf5_target_user_ratio = float(
#             _cfg(config, 'cf5_target_user_ratio', self.cf5_user_ratio)
#         )
#         self.cf5_target_batch_size = int(
#             _cfg(config, 'cf5_target_batch_size', self.cf5_batch_size)
#         )
#         self.cf5_target_mask_eps = float(
#             _cfg(config, 'cf5_target_mask_eps', 1e-6)
#         )

#         max_epochs = int(_cfg(config, 'epochs', 1000))
#         if configured_warmup_epochs >= 0:
#             self.cf5_warmup_epochs = configured_warmup_epochs
#         else:
#             self.cf5_warmup_epochs = int(
#                 math.ceil(max_epochs * self.cf5_warmup_ratio)
#             )

#         self._validate_cf5_config()
#         self.current_epoch = 0
#         self._cf5_rng = random.Random(self.cf5_seed_offset)
#         self.user_to_edge_ids = self._build_cf5_history()
#         self.cf5_stats = self._new_cf5_stats()

#     def _validate_cf5_config(self):
#         if self.cf5_lambda < 0.0:
#             raise ValueError('cf5_lambda must be non-negative.')
#         if self.cf5_temperature <= 0.0:
#             raise ValueError('cf5_temperature must be positive.')
#         if not 0.0 <= self.cf5_user_ratio <= 1.0:
#             raise ValueError('cf5_user_ratio must be in [0, 1].')
#         if self.cf5_batch_size <= 0:
#             raise ValueError('cf5_batch_size must be positive.')
#         if self.cf5_pair_count <= 0:
#             raise ValueError('cf5_pair_count must be positive.')
#         if self.cf5_min_history < 2:
#             raise ValueError('cf5_min_history must be at least 2.')
#         if self.cf5_similarity_eps < 0.0:
#             raise ValueError('cf5_similarity_eps must be non-negative.')
#         if self.cf5_warmup_epochs < 0:
#             raise ValueError('cf5_warmup_epochs must be non-negative.')

#         if self.cf5_target_lambda < 0.0:
#             raise ValueError('cf5_target_lambda must be non-negative.')
#         if self.cf5_target_gamma < 0.0:
#             raise ValueError('cf5_target_gamma must be non-negative.')
#         if self.cf5_target_temperature <= 0.0:
#             raise ValueError('cf5_target_temperature must be positive.')
#         if self.cf5_target_loss_temperature <= 0.0:
#             raise ValueError(
#                 'cf5_target_loss_temperature must be positive.'
#             )
#         if not 0.0 <= self.cf5_target_user_ratio <= 1.0:
#             raise ValueError('cf5_target_user_ratio must be in [0, 1].')
#         if self.cf5_target_batch_size <= 0:
#             raise ValueError('cf5_target_batch_size must be positive.')
#         if not 0.0 < self.cf5_target_mask_eps < 0.5:
#             raise ValueError(
#                 'cf5_target_mask_eps must be in (0, 0.5).'
#             )

#     def _build_cf5_history(self):
#         user_to_edge_ids = [[] for _ in range(self.num_user)]
#         edge_users = self.forward_edge_users.detach().cpu().tolist()
#         for edge_id, user_id in enumerate(edge_users):
#             user_to_edge_ids[int(user_id)].append(int(edge_id))
#         return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

#     @staticmethod
#     def _new_cf5_stats():
#         return {
#             # Broad mask consistency.
#             'samples': 0,
#             'eligible': 0,
#             'pairs': 0,
#             'used': 0,
#             'loss_sum': 0.0,
#             'similarity_gap_sum': 0.0,

#             # Target-specific residual.
#             'target_samples': 0,
#             'target_eligible': 0,
#             'target_used': 0,
#             'target_loss_sum': 0.0,
#             'target_pos_residual_sum': 0.0,
#             'target_neg_residual_sum': 0.0,
#             'target_residual_gap_sum': 0.0,
#             'target_weight_shift_sum': 0.0,
#             'target_pos_similarity_sum': 0.0,
#             'target_neg_similarity_sum': 0.0,
#         }

#     def set_training_epoch(self, epoch_idx):
#         epoch_idx = int(epoch_idx)
#         if epoch_idx < 0:
#             raise ValueError('epoch_idx must be non-negative.')
#         self.current_epoch = epoch_idx
#         self._cf5_rng.seed(self.cf5_seed_offset + epoch_idx)

#     def pre_epoch_processing(self):
#         self.cf5_stats = self._new_cf5_stats()

#     def post_epoch_processing(self):
#         if not self.cf5_log_stats:
#             return None

#         used = max(self.cf5_stats['used'], 1)
#         target_used = max(self.cf5_stats['target_used'], 1)

#         return (
#             'broad-mask + relative target residual: '
#             'epoch={epoch}, warmup_epochs={warmup}, '
#             'broad_lambda={lambda_cf5:.6f}, '
#             'broad_temperature={temperature:.6f}, '
#             'broad_samples={samples}, broad_eligible={eligible}, '
#             'broad_pairs={pairs}, broad_used={used_count}, '
#             'broad_loss={loss:.6f}, broad_similarity_gap={gap:.6f}, '
#             'target_lambda={target_lambda:.6f}, '
#             'target_gamma={target_gamma:.6f}, '
#             'target_temperature={target_temperature:.6f}, '
#             'target_loss_temperature={target_loss_temperature:.6f}, '
#             'target_samples={target_samples}, '
#             'target_eligible={target_eligible}, '
#             'target_used={target_used_count}, '
#             'target_loss={target_loss:.6f}, '
#             'target_pos_residual={target_pos_residual:.6f}, '
#             'target_neg_residual={target_neg_residual:.6f}, '
#             'target_residual_gap={target_residual_gap:.6f}, '
#             'target_weight_shift={target_weight_shift:.6f}, '
#             'target_pos_similarity={target_pos_similarity:.6f}, '
#             'target_neg_similarity={target_neg_similarity:.6f}'
#         ).format(
#             epoch=int(self.current_epoch),
#             warmup=int(self.cf5_warmup_epochs),

#             lambda_cf5=float(self.cf5_lambda),
#             temperature=float(self.cf5_temperature),
#             samples=int(self.cf5_stats['samples']),
#             eligible=int(self.cf5_stats['eligible']),
#             pairs=int(self.cf5_stats['pairs']),
#             used_count=int(self.cf5_stats['used']),
#             loss=float(self.cf5_stats['loss_sum'] / used),
#             gap=float(self.cf5_stats['similarity_gap_sum'] / used),

#             target_lambda=float(self.cf5_target_lambda),
#             target_gamma=float(self.cf5_target_gamma),
#             target_temperature=float(self.cf5_target_temperature),
#             target_loss_temperature=float(
#                 self.cf5_target_loss_temperature
#             ),
#             target_samples=int(self.cf5_stats['target_samples']),
#             target_eligible=int(self.cf5_stats['target_eligible']),
#             target_used_count=int(self.cf5_stats['target_used']),
#             target_loss=float(
#                 self.cf5_stats['target_loss_sum'] / target_used
#             ),
#             target_pos_residual=float(
#                 self.cf5_stats['target_pos_residual_sum'] / target_used
#             ),
#             target_neg_residual=float(
#                 self.cf5_stats['target_neg_residual_sum'] / target_used
#             ),
#             target_residual_gap=float(
#                 self.cf5_stats['target_residual_gap_sum'] / target_used
#             ),
#             target_weight_shift=float(
#                 self.cf5_stats['target_weight_shift_sum'] / target_used
#             ),
#             target_pos_similarity=float(
#                 self.cf5_stats['target_pos_similarity_sum'] / target_used
#             ),
#             target_neg_similarity=float(
#                 self.cf5_stats['target_neg_similarity_sum'] / target_used
#             ),
#         )

#     def _is_cf5_active(self):
#         return (
#             self.training
#             and (
#                 self.cf5_lambda > 0.0
#                 or self.cf5_target_lambda > 0.0
#             )
#             and (
#                 self.cf5_user_ratio > 0.0
#                 or self.cf5_target_user_ratio > 0.0
#             )
#             and self.current_epoch >= self.cf5_warmup_epochs
#         )

#     def _calculate_rec_loss(self, interaction):
#         pos_scores, neg_scores = self.forward(interaction)
#         return -torch.mean(
#             torch.log2(torch.sigmoid(pos_scores - neg_scores))
#         )

#     def calculate_loss(self, interaction):
#         loss_rec = self._calculate_rec_loss(interaction)
#         if not self._is_cf5_active():
#             self.result_embed = None
#             return loss_rec

#         if self.cf5_lambda > 0.0:
#             loss_mask_relation = self._calculate_mask_relation_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_mask_relation = loss_rec * 0.0

#         if self.cf5_target_lambda > 0.0:
#             loss_target_residual = self._calculate_target_residual_loss(
#                 interaction,
#                 loss_rec
#             )
#         else:
#             loss_target_residual = loss_rec * 0.0

#         weighted_auxiliary = (
#             self.cf5_lambda * loss_mask_relation
#             + self.cf5_target_lambda * loss_target_residual
#         )

#         self.result_embed = None

#         # Preserve the original trainer interface.
#         return loss_rec, weighted_auxiliary

#     def _sample_cf5_users(self, interaction):
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(math.ceil(len(users) * self.cf5_user_ratio))
#         sample_count = max(1, sample_count)
#         sample_count = min(sample_count, self.cf5_batch_size, len(users))
#         return self._cf5_rng.sample(users, sample_count)

#     def _sample_cf5_pairs(self, history_size):
#         total_pairs = history_size * (history_size - 1) // 2
#         if total_pairs <= self.cf5_pair_count:
#             return list(itertools.combinations(range(history_size), 2))

#         pairs = set()
#         while len(pairs) < self.cf5_pair_count:
#             left = self._cf5_rng.randrange(history_size)
#             right = self._cf5_rng.randrange(history_size)
#             if left == right:
#                 continue
#             pairs.add(tuple(sorted((left, right))))
#         return list(pairs)

#     def _calculate_mask_relation_loss(self, interaction, reference_loss):
#         """Apply pairwise similarity ordering to current mask weights."""
#         sampled_users = self._sample_cf5_users(interaction)
#         self.cf5_stats['samples'] += len(sampled_users)
#         if not sampled_users or self.result_embed is None:
#             return reference_loss * 0.0

#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()
#         losses = []

#         for user_id in sampled_users:
#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             self.cf5_stats['eligible'] += 1
#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             item_ids = self.forward_edge_items[edge_tensor]
#             prototype = item_rep[item_ids].mean(dim=0)
#             relevance = F.cosine_similarity(
#                 item_rep[item_ids],
#                 prototype.unsqueeze(0),
#                 dim=1
#             ).detach()

#             pair_positions = self._sample_cf5_pairs(len(edge_ids))

#             for left_pos, right_pos in pair_positions:
#                 relevance_gap = (
#                     relevance[left_pos] - relevance[right_pos]
#                 )
#                 if abs(float(relevance_gap.detach().cpu())) <= self.cf5_similarity_eps:
#                     continue

#                 left_edge = int(edge_ids[left_pos])
#                 right_edge = int(edge_ids[right_pos])
#                 mask_gap = mask_weights[left_edge] - mask_weights[right_edge]
#                 direction = torch.sign(relevance_gap)
#                 pair_loss = F.softplus(
#                     -self.cf5_temperature * direction * mask_gap
#                 )
#                 losses.append(pair_loss)

#                 with torch.no_grad():
#                     self.cf5_stats['pairs'] += 1
#                     self.cf5_stats['used'] += 1
#                     self.cf5_stats['loss_sum'] += float(
#                         pair_loss.detach().cpu()
#                     )
#                     self.cf5_stats['similarity_gap_sum'] += float(
#                         relevance_gap.abs().detach().cpu()
#                     )

#         if not losses:
#             return reference_loss * 0.0
#         return torch.stack(losses).mean()

#     def _sample_target_users(self, interaction):
#         """Sample users for the target-specific residual auxiliary task."""
#         if interaction is None or len(interaction) == 0:
#             return []

#         users = torch.unique(interaction[0].detach()).cpu().tolist()
#         users = [int(user_id) for user_id in users]
#         if not users:
#             return []

#         sample_count = int(
#             math.ceil(len(users) * self.cf5_target_user_ratio)
#         )
#         sample_count = max(1, sample_count)
#         sample_count = min(
#             sample_count,
#             self.cf5_target_batch_size,
#             len(users)
#         )
#         return self._cf5_rng.sample(users, sample_count)

#     @staticmethod
#     def _build_batch_pos_neg_targets(interaction):
#         """Map batch users to current (positive, negative) item pairs.

#         Assumes the standard layout already used by the base recommender:
#             interaction[0] -> user ids
#             interaction[1] -> positive item ids
#             interaction[2] -> negative item ids
#         """
#         if interaction is None or len(interaction) < 3:
#             return {}

#         users = interaction[0].detach().view(-1).cpu().tolist()
#         positives = interaction[1].detach().view(-1).cpu().tolist()
#         negatives = interaction[2].detach().view(-1).cpu().tolist()

#         if not (
#             len(users) == len(positives) == len(negatives)
#         ):
#             return {}

#         mapping = {}
#         for user_id, pos_item, neg_item in zip(
#             users, positives, negatives
#         ):
#             mapping.setdefault(int(user_id), []).append(
#                 (int(pos_item), int(neg_item))
#             )
#         return mapping

#     def _target_conditioned_residual_score(
#         self,
#         history_rep,
#         target_rep,
#         base_mask
#     ):
#         """Compute target-specific residual using relative mask weighting.

#         Target relevance:
#             q(e,p) = cos(z_e, z_p)

#         Base/global history weights:
#             alpha_base(e)
#                 = stopgrad(m_e) / sum_j stopgrad(m_j)

#         Target-conditioned history weights:
#             alpha_target(e,p)
#                 = softmax(
#                     log(stopgrad(m_e) + eps)
#                     + gamma * q(e,p) / T_target
#                   )

#         Equivalently:
#             alpha_target(e,p)
#                 proportional to
#                 stopgrad(m_e) * exp(gamma * q(e,p) / T_target)

#         Thus the broad/static mask remains a GLOBAL PRIOR, while target
#         similarity only changes the RELATIVE contribution of history items.
#         This avoids the old sigmoid(logit(m)+delta) saturation when m is near 1.

#         We then define:
#             h_global     = sum_e alpha_base(e) z_e
#             h_target     = sum_e alpha_target(e,p) z_e
#             residual_rep = h_target - h_global
#             residual_score = <residual_rep, z_p>

#         The base mask is detached in this auxiliary path, so the target loss
#         does not directly rewrite the static mask values. Gradients still flow
#         through q(e,p), history_rep, target_rep, and therefore shape the masked
#         representations.
#         """
#         if history_rep.numel() == 0:
#             zero = target_rep.sum() * 0.0
#             return zero, zero, zero

#         eps = self.cf5_target_mask_eps

#         # Candidate-conditioned relevance from the current masked branch.
#         target_similarity = F.cosine_similarity(
#             history_rep,
#             target_rep.unsqueeze(0),
#             dim=1
#         )

#         # The broad/static mask is only a detached global prior here.
#         base_mask_detached = base_mask.detach().clamp_min(eps)

#         # Base/global normalized history weights. Using softmax(log(m)) is
#         # numerically equivalent to m / sum(m), while making the relationship
#         # to target-conditioned logits explicit.
#         base_logits = torch.log(base_mask_detached)
#         global_weight = torch.softmax(base_logits, dim=0)

#         # New relative target correction.
#         # A positive q increases an edge's RELATIVE contribution; a negative q
#         # decreases it. There is no extra sigmoid over an already-near-one mask.
#         target_logits = (
#             base_logits
#             + self.cf5_target_gamma
#             * target_similarity
#             / self.cf5_target_temperature
#         )
#         target_weight = torch.softmax(target_logits, dim=0)

#         global_history = torch.sum(
#             global_weight.unsqueeze(-1) * history_rep,
#             dim=0
#         )
#         target_history = torch.sum(
#             target_weight.unsqueeze(-1) * history_rep,
#             dim=0
#         )

#         residual_rep = target_history - global_history

#         # Auxiliary contribution score: how much the target-specific readout
#         # moves the history representation in the target item's direction.
#         residual_score = torch.sum(
#             residual_rep * target_rep,
#             dim=-1
#         )

#         # Diagnostic: actual change in NORMALIZED history contribution, which
#         # is the relevant quantity for this formulation.
#         mean_weight_shift = (
#             target_weight - global_weight
#         ).abs().mean()

#         mean_target_similarity = target_similarity.mean()

#         return (
#             residual_score,
#             mean_weight_shift,
#             mean_target_similarity
#         )

#     def _calculate_target_residual_loss(

#         self,
#         interaction,
#         reference_loss
#     ):
#         """Train target-specific residual without any counterfactual objective.

#         For each sampled user and current positive/negative pair:

#             R_pos = residual_score(u, p+)
#             R_neg = residual_score(u, p-)

#         The auxiliary objective asks the target-conditioned correction to help
#         the positive item more than the negative item:

#             L_target = softplus(
#                 (R_neg - R_pos) / T_loss
#             )

#         The ordinary recommendation loss remains unchanged. The change from
#         the previous version is the target-conditioned weighting mechanism, not
#         the pairwise residual ranking objective itself.
#         """
#         if not hasattr(self, 'mask_rep') or self.mask_rep is None:
#             return reference_loss * 0.0

#         target_pairs_by_user = self._build_batch_pos_neg_targets(
#             interaction
#         )
#         if not target_pairs_by_user:
#             return reference_loss * 0.0

#         sampled_users = self._sample_target_users(interaction)
#         self.cf5_stats['target_samples'] += len(sampled_users)

#         if not sampled_users:
#             return reference_loss * 0.0

#         item_rep = self.mask_rep[self.num_user:]
#         mask_weights = self.get_forward_edge_mask()

#         losses = []

#         for user_id in sampled_users:
#             user_pairs = target_pairs_by_user.get(int(user_id), [])
#             if not user_pairs:
#                 continue

#             pos_item, neg_item = self._cf5_rng.choice(user_pairs)

#             if (
#                 pos_item < 0
#                 or neg_item < 0
#                 or pos_item >= item_rep.size(0)
#                 or neg_item >= item_rep.size(0)
#             ):
#                 continue

#             edge_ids = self.user_to_edge_ids[int(user_id)]
#             if len(edge_ids) < self.cf5_min_history:
#                 continue

#             edge_tensor = torch.tensor(
#                 edge_ids,
#                 dtype=torch.long,
#                 device=self.forward_edge_users.device
#             )
#             history_item_ids = self.forward_edge_items[edge_tensor]

#             # Avoid trivial cos(z_p, z_p)=1 if a batch target is already in
#             # the user's graph history.
#             keep = (
#                 (history_item_ids != pos_item)
#                 & (history_item_ids != neg_item)
#             )
#             history_item_ids = history_item_ids[keep]
#             history_edge_tensor = edge_tensor[keep]

#             if history_item_ids.numel() < self.cf5_min_history:
#                 continue

#             self.cf5_stats['target_eligible'] += 1

#             history_rep = item_rep[history_item_ids]
#             history_mask = mask_weights[history_edge_tensor]

#             pos_score, pos_weight_shift, pos_similarity = (
#                 self._target_conditioned_residual_score(
#                     history_rep,
#                     item_rep[pos_item],
#                     history_mask
#                 )
#             )

#             neg_score, neg_weight_shift, neg_similarity = (
#                 self._target_conditioned_residual_score(
#                     history_rep,
#                     item_rep[neg_item],
#                     history_mask
#                 )
#             )

#             residual_gap = pos_score - neg_score

#             user_loss = F.softplus(
#                 (
#                     neg_score - pos_score
#                 )
#                 / self.cf5_target_loss_temperature
#             )

#             losses.append(user_loss)

#             with torch.no_grad():
#                 self.cf5_stats['target_used'] += 1
#                 self.cf5_stats['target_loss_sum'] += float(
#                     user_loss.detach().cpu()
#                 )
#                 self.cf5_stats['target_pos_residual_sum'] += float(
#                     pos_score.detach().cpu()
#                 )
#                 self.cf5_stats['target_neg_residual_sum'] += float(
#                     neg_score.detach().cpu()
#                 )
#                 self.cf5_stats['target_residual_gap_sum'] += float(
#                     residual_gap.detach().cpu()
#                 )
#                 self.cf5_stats['target_weight_shift_sum'] += float(
#                     (
#                         0.5 * (pos_weight_shift + neg_weight_shift)
#                     ).detach().cpu()
#                 )
#                 self.cf5_stats['target_pos_similarity_sum'] += float(
#                     pos_similarity.detach().cpu()
#                 )
#                 self.cf5_stats['target_neg_similarity_sum'] += float(
#                     neg_similarity.detach().cpu()
#                 )

#         if not losses:
#             return reference_loss * 0.0

#         return torch.stack(losses).mean()


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
    """MASKED_GLORIA with broad mask consistency + target-specific residual.

    Broad loss:
        Keep the original history-prototype-guided mask ordering.

    Target-specific residual:
        Keep the learned static edge mask as a detached global/base prior, and
        let target similarity change the RELATIVE history contribution:

            q(e,p) = cos(z_e^mask, z_p^mask)

            alpha_base(e) ∝ stopgrad(m_e)
            alpha_target(e,p)
                = softmax(log(stopgrad(m_e) + eps)
                          + gamma * q(e,p) / T_target)

        Equivalently:

            alpha_target(e,p) ∝ stopgrad(m_e)
                                  * exp(gamma * q(e,p) / T_target)

        This avoids sigmoid saturation when the learned masks are already near
        one and makes the target signal act directly on relative history
        weighting. The target-conditioned readout is compared with the
        base/global readout; their score difference is the target-specific
        residual score.

        The residual is also injected DIRECTLY into the recommendation score:

            s_new(u,p) = s_base(u,p) + eta * R(u,p)

        where:

            R(u,p) = <h_target(u,p) - h_global(u), z_p^mask>

        Thus the main ranking loss directly supervises whether the target-aware
        correction improves the positive-vs-negative ranking margin.

        The old residual-ranking auxiliary loss L_target is kept OPTIONAL for
        ablation, but defaults to zero in this version.

    This version intentionally contains NO counterfactual loss.
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

        # -------------------------------------------------------------
        # Target-specific residual configuration.
        # -------------------------------------------------------------
        # Optional old auxiliary residual-ranking loss. Keep this at 0.0 for
        # the clean "direct residual scoring" experiment.
        self.cf5_target_lambda = float(
            _cfg(config, 'cf5_target_lambda', 0.0)
        )
        # Strength of target-specific relative reweighting.
        # Backward-compatible fallback: an old cf5_target_beta value is used
        # only when cf5_target_gamma is not provided.
        self.cf5_target_gamma = float(
            _cfg(
                config,
                'cf5_target_gamma',
                _cfg(config, 'cf5_target_beta', 1.0)
            )
        )
        self.cf5_target_temperature = float(
            _cfg(config, 'cf5_target_temperature', 1.0)
        )
        self.cf5_target_loss_temperature = float(
            _cfg(config, 'cf5_target_loss_temperature', 1.0)
        )
        self.cf5_target_user_ratio = float(
            _cfg(config, 'cf5_target_user_ratio', self.cf5_user_ratio)
        )
        self.cf5_target_batch_size = int(
            _cfg(config, 'cf5_target_batch_size', self.cf5_batch_size)
        )
        self.cf5_target_mask_eps = float(
            _cfg(config, 'cf5_target_mask_eps', 1e-6)
        )

        # -------------------------------------------------------------
        # Direct target-residual scoring configuration.
        # -------------------------------------------------------------
        # Final training/evaluation score:
        #     s_new = s_base + eta * residual_score
        self.cf5_target_score_eta = float(
            _cfg(config, 'cf5_target_score_eta', 0.1)
        )
        self.cf5_target_full_sort_chunk_size = int(
            _cfg(config, 'cf5_target_full_sort_chunk_size', 2048)
        )

        max_epochs = int(_cfg(config, 'epochs', 1000))
        if configured_warmup_epochs >= 0:
            self.cf5_warmup_epochs = configured_warmup_epochs
        else:
            self.cf5_warmup_epochs = int(
                math.ceil(max_epochs * self.cf5_warmup_ratio)
            )

        configured_score_warmup = int(
            _cfg(
                config,
                'cf5_target_score_warmup_epochs',
                self.cf5_warmup_epochs
            )
        )
        self.cf5_target_score_warmup_epochs = configured_score_warmup

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

        if self.cf5_target_lambda < 0.0:
            raise ValueError('cf5_target_lambda must be non-negative.')
        if self.cf5_target_gamma < 0.0:
            raise ValueError('cf5_target_gamma must be non-negative.')
        if self.cf5_target_temperature <= 0.0:
            raise ValueError('cf5_target_temperature must be positive.')
        if self.cf5_target_loss_temperature <= 0.0:
            raise ValueError(
                'cf5_target_loss_temperature must be positive.'
            )
        if not 0.0 <= self.cf5_target_user_ratio <= 1.0:
            raise ValueError('cf5_target_user_ratio must be in [0, 1].')
        if self.cf5_target_batch_size <= 0:
            raise ValueError('cf5_target_batch_size must be positive.')
        if not 0.0 < self.cf5_target_mask_eps < 0.5:
            raise ValueError(
                'cf5_target_mask_eps must be in (0, 0.5).'
            )
        if self.cf5_target_score_eta < 0.0:
            raise ValueError('cf5_target_score_eta must be non-negative.')
        if self.cf5_target_score_warmup_epochs < 0:
            raise ValueError(
                'cf5_target_score_warmup_epochs must be non-negative.'
            )
        if self.cf5_target_full_sort_chunk_size <= 0:
            raise ValueError(
                'cf5_target_full_sort_chunk_size must be positive.'
            )

    def _build_cf5_history(self):
        user_to_edge_ids = [[] for _ in range(self.num_user)]
        edge_users = self.forward_edge_users.detach().cpu().tolist()
        for edge_id, user_id in enumerate(edge_users):
            user_to_edge_ids[int(user_id)].append(int(edge_id))
        return tuple(tuple(edge_ids) for edge_ids in user_to_edge_ids)

    @staticmethod
    def _new_cf5_stats():
        return {
            # Broad mask consistency.
            'samples': 0,
            'eligible': 0,
            'pairs': 0,
            'used': 0,
            'loss_sum': 0.0,
            'similarity_gap_sum': 0.0,

            # Target-specific residual.
            'target_samples': 0,
            'target_eligible': 0,
            'target_used': 0,
            'target_loss_sum': 0.0,
            'target_pos_residual_sum': 0.0,
            'target_neg_residual_sum': 0.0,
            'target_residual_gap_sum': 0.0,
            'target_weight_shift_sum': 0.0,
            'target_pos_similarity_sum': 0.0,
            'target_neg_similarity_sum': 0.0,

            # Direct residual scoring inside main recommendation loss.
            'score_used': 0,
            'score_base_margin_sum': 0.0,
            'score_pos_residual_sum': 0.0,
            'score_neg_residual_sum': 0.0,
            'score_residual_gap_sum': 0.0,
            'score_corrected_margin_sum': 0.0,
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
        target_used = max(self.cf5_stats['target_used'], 1)
        score_used = max(self.cf5_stats['score_used'], 1)

        return (
            'broad-mask + direct relative target score: '
            'epoch={epoch}, warmup_epochs={warmup}, '
            'score_warmup={score_warmup}, '
            'broad_lambda={lambda_cf5:.6f}, '
            'broad_loss={broad_loss:.6f}, '
            'broad_similarity_gap={broad_gap:.6f}, '
            'target_gamma={target_gamma:.6f}, '
            'target_temperature={target_temperature:.6f}, '
            'target_score_eta={score_eta:.6f}, '
            'score_used={score_used_count}, '
            'score_base_margin={base_margin:.6f}, '
            'score_pos_residual={score_pos_residual:.6f}, '
            'score_neg_residual={score_neg_residual:.6f}, '
            'score_residual_gap={score_residual_gap:.6f}, '
            'score_corrected_margin={corrected_margin:.6f}, '
            'aux_target_lambda={target_lambda:.6f}, '
            'aux_target_loss={target_loss:.6f}, '
            'aux_target_residual_gap={target_residual_gap:.6f}, '
            'aux_target_weight_shift={target_weight_shift:.6f}, '
            'aux_target_pos_similarity={target_pos_similarity:.6f}, '
            'aux_target_neg_similarity={target_neg_similarity:.6f}'
        ).format(
            epoch=int(self.current_epoch),
            warmup=int(self.cf5_warmup_epochs),
            score_warmup=int(self.cf5_target_score_warmup_epochs),
            lambda_cf5=float(self.cf5_lambda),
            broad_loss=float(self.cf5_stats['loss_sum'] / used),
            broad_gap=float(self.cf5_stats['similarity_gap_sum'] / used),
            target_gamma=float(self.cf5_target_gamma),
            target_temperature=float(self.cf5_target_temperature),
            score_eta=float(self.cf5_target_score_eta),
            score_used_count=int(self.cf5_stats['score_used']),
            base_margin=float(
                self.cf5_stats['score_base_margin_sum'] / score_used
            ),
            score_pos_residual=float(
                self.cf5_stats['score_pos_residual_sum'] / score_used
            ),
            score_neg_residual=float(
                self.cf5_stats['score_neg_residual_sum'] / score_used
            ),
            score_residual_gap=float(
                self.cf5_stats['score_residual_gap_sum'] / score_used
            ),
            corrected_margin=float(
                self.cf5_stats['score_corrected_margin_sum'] / score_used
            ),
            target_lambda=float(self.cf5_target_lambda),
            target_loss=float(
                self.cf5_stats['target_loss_sum'] / target_used
            ),
            target_residual_gap=float(
                self.cf5_stats['target_residual_gap_sum'] / target_used
            ),
            target_weight_shift=float(
                self.cf5_stats['target_weight_shift_sum'] / target_used
            ),
            target_pos_similarity=float(
                self.cf5_stats['target_pos_similarity_sum'] / target_used
            ),
            target_neg_similarity=float(
                self.cf5_stats['target_neg_similarity_sum'] / target_used
            ),
        )

    def _is_cf5_active(self):
        return (
            self.training
            and (
                self.cf5_lambda > 0.0
                or self.cf5_target_lambda > 0.0
            )
            and (
                self.cf5_user_ratio > 0.0
                or self.cf5_target_user_ratio > 0.0
            )
            and self.current_epoch >= self.cf5_warmup_epochs
        )

    def _target_score_active_for_training(self):
        return (
            self.training
            and self.cf5_target_score_eta > 0.0
            and self.current_epoch >= self.cf5_target_score_warmup_epochs
        )

    def _calculate_rec_loss(self, interaction):
        """Main BPR loss with optional target-specific residual correction.

        Before score warmup:
            margin = s_pos_base - s_neg_base

        After score warmup:
            s_pos_new = s_pos_base + eta * R_pos
            s_neg_new = s_neg_base + eta * R_neg

            margin_new = margin_base + eta * (R_pos - R_neg)

        Therefore the target-specific residual is trained DIRECTLY by the main
        recommendation objective, rather than only through L_target.
        """
        pos_scores, neg_scores = self.forward(interaction)

        if not self._target_score_active_for_training():
            return -torch.mean(
                torch.log2(torch.sigmoid(pos_scores - neg_scores))
            )

        pos_residual, neg_residual, valid = (
            self._calculate_batch_target_residual_scores(interaction)
        )

        # Invalid/too-short histories receive zero correction, so their score
        # exactly falls back to the base recommender.
        corrected_pos = (
            pos_scores
            + self.cf5_target_score_eta * pos_residual
        )
        corrected_neg = (
            neg_scores
            + self.cf5_target_score_eta * neg_residual
        )

        with torch.no_grad():
            valid_count = int(valid.sum().item())
            if valid_count > 0:
                base_margin = pos_scores[valid] - neg_scores[valid]
                residual_gap = (
                    pos_residual[valid] - neg_residual[valid]
                )
                corrected_margin = (
                    corrected_pos[valid] - corrected_neg[valid]
                )

                self.cf5_stats['score_used'] += valid_count
                self.cf5_stats['score_base_margin_sum'] += float(
                    base_margin.sum().detach().cpu()
                )
                self.cf5_stats['score_pos_residual_sum'] += float(
                    pos_residual[valid].sum().detach().cpu()
                )
                self.cf5_stats['score_neg_residual_sum'] += float(
                    neg_residual[valid].sum().detach().cpu()
                )
                self.cf5_stats['score_residual_gap_sum'] += float(
                    residual_gap.sum().detach().cpu()
                )
                self.cf5_stats['score_corrected_margin_sum'] += float(
                    corrected_margin.sum().detach().cpu()
                )

        return -torch.mean(
            torch.log2(
                torch.sigmoid(corrected_pos - corrected_neg)
            )
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

        if self.cf5_target_lambda > 0.0:
            loss_target_residual = self._calculate_target_residual_loss(
                interaction,
                loss_rec
            )
        else:
            loss_target_residual = loss_rec * 0.0

        weighted_auxiliary = (
            self.cf5_lambda * loss_mask_relation
            + self.cf5_target_lambda * loss_target_residual
        )

        self.result_embed = None

        # Preserve the original trainer interface.
        return loss_rec, weighted_auxiliary

    def _calculate_batch_target_residual_scores(self, interaction):
        """Return R_pos and R_neg for every row in a training mini-batch.

        Unlike the optional L_target auxiliary task, this function DOES NOT
        subsample users because its outputs are part of the main BPR score.
        Rows with an unavailable/too-short history simply receive zero
        residual correction.
        """
        users = interaction[0].view(-1)
        positives = interaction[1].view(-1)
        negatives = interaction[2].view(-1)

        batch_size = int(users.numel())
        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            zero = torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=users.device
            )
            valid = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=users.device
            )
            return zero, zero, valid

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()

        pos_residuals = []
        neg_residuals = []
        valid_rows = []

        for row in range(batch_size):
            user_id = int(users[row].detach().item())
            pos_item = int(positives[row].detach().item())
            neg_item = int(negatives[row].detach().item())

            edge_ids = self.user_to_edge_ids[user_id]
            if len(edge_ids) < self.cf5_min_history:
                zero = item_rep[pos_item].sum() * 0.0
                pos_residuals.append(zero)
                neg_residuals.append(zero)
                valid_rows.append(False)
                continue

            edge_tensor = torch.tensor(
                edge_ids,
                dtype=torch.long,
                device=self.forward_edge_users.device
            )
            history_item_ids = self.forward_edge_items[edge_tensor]

            # Prevent trivial target self-match when a current target is
            # already present in the graph history.
            keep = (
                (history_item_ids != pos_item)
                & (history_item_ids != neg_item)
            )
            history_item_ids = history_item_ids[keep]
            history_edge_tensor = edge_tensor[keep]

            if history_item_ids.numel() < self.cf5_min_history:
                zero = item_rep[pos_item].sum() * 0.0
                pos_residuals.append(zero)
                neg_residuals.append(zero)
                valid_rows.append(False)
                continue

            history_rep = item_rep[history_item_ids]
            history_mask = mask_weights[history_edge_tensor]

            pos_residual, _, _ = self._target_conditioned_residual_score(
                history_rep,
                item_rep[pos_item],
                history_mask
            )
            neg_residual, _, _ = self._target_conditioned_residual_score(
                history_rep,
                item_rep[neg_item],
                history_mask
            )

            pos_residuals.append(pos_residual)
            neg_residuals.append(neg_residual)
            valid_rows.append(True)

        pos_residual = torch.stack(pos_residuals)
        neg_residual = torch.stack(neg_residuals)
        valid = torch.tensor(
            valid_rows,
            dtype=torch.bool,
            device=pos_residual.device
        )
        return pos_residual, neg_residual, valid

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

    def _sample_target_users(self, interaction):
        """Sample users for the target-specific residual auxiliary task."""
        if interaction is None or len(interaction) == 0:
            return []

        users = torch.unique(interaction[0].detach()).cpu().tolist()
        users = [int(user_id) for user_id in users]
        if not users:
            return []

        sample_count = int(
            math.ceil(len(users) * self.cf5_target_user_ratio)
        )
        sample_count = max(1, sample_count)
        sample_count = min(
            sample_count,
            self.cf5_target_batch_size,
            len(users)
        )
        return self._cf5_rng.sample(users, sample_count)

    @staticmethod
    def _build_batch_pos_neg_targets(interaction):
        """Map batch users to current (positive, negative) item pairs.

        Assumes the standard layout already used by the base recommender:
            interaction[0] -> user ids
            interaction[1] -> positive item ids
            interaction[2] -> negative item ids
        """
        if interaction is None or len(interaction) < 3:
            return {}

        users = interaction[0].detach().view(-1).cpu().tolist()
        positives = interaction[1].detach().view(-1).cpu().tolist()
        negatives = interaction[2].detach().view(-1).cpu().tolist()

        if not (
            len(users) == len(positives) == len(negatives)
        ):
            return {}

        mapping = {}
        for user_id, pos_item, neg_item in zip(
            users, positives, negatives
        ):
            mapping.setdefault(int(user_id), []).append(
                (int(pos_item), int(neg_item))
            )
        return mapping

    def _target_conditioned_residual_score(
        self,
        history_rep,
        target_rep,
        base_mask
    ):
        """Compute target-specific residual using relative mask weighting.

        Target relevance:
            q(e,p) = cos(z_e, z_p)

        Base/global history weights:
            alpha_base(e)
                = stopgrad(m_e) / sum_j stopgrad(m_j)

        Target-conditioned history weights:
            alpha_target(e,p)
                = softmax(
                    log(stopgrad(m_e) + eps)
                    + gamma * q(e,p) / T_target
                  )

        Equivalently:
            alpha_target(e,p)
                proportional to
                stopgrad(m_e) * exp(gamma * q(e,p) / T_target)

        Thus the broad/static mask remains a GLOBAL PRIOR, while target
        similarity only changes the RELATIVE contribution of history items.
        This avoids the old sigmoid(logit(m)+delta) saturation when m is near 1.

        We then define:
            h_global     = sum_e alpha_base(e) z_e
            h_target     = sum_e alpha_target(e,p) z_e
            residual_rep = h_target - h_global
            residual_score = <residual_rep, z_p>

        The base mask is detached in this auxiliary path, so the target loss
        does not directly rewrite the static mask values. Gradients still flow
        through q(e,p), history_rep, target_rep, and therefore shape the masked
        representations.
        """
        if history_rep.numel() == 0:
            zero = target_rep.sum() * 0.0
            return zero, zero, zero

        eps = self.cf5_target_mask_eps

        # Candidate-conditioned relevance from the current masked branch.
        target_similarity = F.cosine_similarity(
            history_rep,
            target_rep.unsqueeze(0),
            dim=1
        )

        # The broad/static mask is only a detached global prior here.
        base_mask_detached = base_mask.detach().clamp_min(eps)

        # Base/global normalized history weights. Using softmax(log(m)) is
        # numerically equivalent to m / sum(m), while making the relationship
        # to target-conditioned logits explicit.
        base_logits = torch.log(base_mask_detached)
        global_weight = torch.softmax(base_logits, dim=0)

        # New relative target correction.
        # A positive q increases an edge's RELATIVE contribution; a negative q
        # decreases it. There is no extra sigmoid over an already-near-one mask.
        target_logits = (
            base_logits
            + self.cf5_target_gamma
            * target_similarity
            / self.cf5_target_temperature
        )
        target_weight = torch.softmax(target_logits, dim=0)

        global_history = torch.sum(
            global_weight.unsqueeze(-1) * history_rep,
            dim=0
        )
        target_history = torch.sum(
            target_weight.unsqueeze(-1) * history_rep,
            dim=0
        )

        residual_rep = target_history - global_history

        # Auxiliary contribution score: how much the target-specific readout
        # moves the history representation in the target item's direction.
        residual_score = torch.sum(
            residual_rep * target_rep,
            dim=-1
        )

        # Diagnostic: actual change in NORMALIZED history contribution, which
        # is the relevant quantity for this formulation.
        mean_weight_shift = (
            target_weight - global_weight
        ).abs().mean()

        mean_target_similarity = target_similarity.mean()

        return (
            residual_score,
            mean_weight_shift,
            mean_target_similarity
        )

    def _calculate_target_residual_loss(

        self,
        interaction,
        reference_loss
    ):
        """Train target-specific residual without any counterfactual objective.

        For each sampled user and current positive/negative pair:

            R_pos = residual_score(u, p+)
            R_neg = residual_score(u, p-)

        The auxiliary objective asks the target-conditioned correction to help
        the positive item more than the negative item:

            L_target = softplus(
                (R_neg - R_pos) / T_loss
            )

        This auxiliary objective is OPTIONAL in this file. The main
        recommendation loss already consumes the residual score directly via
        s_new = s_base + eta * R. Set cf5_target_lambda=0.0 for the clean
        direct-scoring experiment.
        """
        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            return reference_loss * 0.0

        target_pairs_by_user = self._build_batch_pos_neg_targets(
            interaction
        )
        if not target_pairs_by_user:
            return reference_loss * 0.0

        sampled_users = self._sample_target_users(interaction)
        self.cf5_stats['target_samples'] += len(sampled_users)

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

            # Avoid trivial cos(z_p, z_p)=1 if a batch target is already in
            # the user's graph history.
            keep = (
                (history_item_ids != pos_item)
                & (history_item_ids != neg_item)
            )
            history_item_ids = history_item_ids[keep]
            history_edge_tensor = edge_tensor[keep]

            if history_item_ids.numel() < self.cf5_min_history:
                continue

            self.cf5_stats['target_eligible'] += 1

            history_rep = item_rep[history_item_ids]
            history_mask = mask_weights[history_edge_tensor]

            pos_score, pos_weight_shift, pos_similarity = (
                self._target_conditioned_residual_score(
                    history_rep,
                    item_rep[pos_item],
                    history_mask
                )
            )

            neg_score, neg_weight_shift, neg_similarity = (
                self._target_conditioned_residual_score(
                    history_rep,
                    item_rep[neg_item],
                    history_mask
                )
            )

            residual_gap = pos_score - neg_score

            user_loss = F.softplus(
                (
                    neg_score - pos_score
                )
                / self.cf5_target_loss_temperature
            )

            losses.append(user_loss)

            with torch.no_grad():
                self.cf5_stats['target_used'] += 1
                self.cf5_stats['target_loss_sum'] += float(
                    user_loss.detach().cpu()
                )
                self.cf5_stats['target_pos_residual_sum'] += float(
                    pos_score.detach().cpu()
                )
                self.cf5_stats['target_neg_residual_sum'] += float(
                    neg_score.detach().cpu()
                )
                self.cf5_stats['target_residual_gap_sum'] += float(
                    residual_gap.detach().cpu()
                )
                self.cf5_stats['target_weight_shift_sum'] += float(
                    (
                        0.5 * (pos_weight_shift + neg_weight_shift)
                    ).detach().cpu()
                )
                self.cf5_stats['target_pos_similarity_sum'] += float(
                    pos_similarity.detach().cpu()
                )
                self.cf5_stats['target_neg_similarity_sum'] += float(
                    neg_similarity.detach().cpu()
                )

        if not losses:
            return reference_loss * 0.0

        return torch.stack(losses).mean()

    def target_residual_for_pairs(self, user_ids, item_ids):
        """Compute target residual R(u,p) for arbitrary user-item pairs.

        This is an inference helper that does NOT rerun GCN. It assumes the
        current forward/predict path has already produced self.mask_rep.

        Args:
            user_ids: 1-D tensor of user ids.
            item_ids: 1-D tensor of item ids, same length as user_ids.
        Returns:
            Tensor [B] of residual scores.
        """
        user_ids = user_ids.view(-1)
        item_ids = item_ids.view(-1)
        if user_ids.numel() != item_ids.numel():
            raise ValueError('user_ids and item_ids must have equal length.')
        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            raise RuntimeError(
                'mask_rep is unavailable. Run the base embedding/forward path '
                'before calling target_residual_for_pairs().'
            )

        item_rep = self.mask_rep[self.num_user:]
        mask_weights = self.get_forward_edge_mask()
        outputs = []

        for user_tensor, item_tensor in zip(user_ids, item_ids):
            user_id = int(user_tensor.detach().item())
            item_id = int(item_tensor.detach().item())
            edge_ids = self.user_to_edge_ids[user_id]

            if len(edge_ids) < self.cf5_min_history:
                outputs.append(item_rep[item_id].sum() * 0.0)
                continue

            edge_tensor = torch.tensor(
                edge_ids,
                dtype=torch.long,
                device=self.forward_edge_users.device
            )
            history_item_ids = self.forward_edge_items[edge_tensor]
            keep = history_item_ids != item_id
            history_item_ids = history_item_ids[keep]
            history_edge_tensor = edge_tensor[keep]

            if history_item_ids.numel() < self.cf5_min_history:
                outputs.append(item_rep[item_id].sum() * 0.0)
                continue

            residual, _, _ = self._target_conditioned_residual_score(
                item_rep[history_item_ids],
                item_rep[item_id],
                mask_weights[history_edge_tensor]
            )
            outputs.append(residual)

        return torch.stack(outputs)

    def target_residual_for_all_items(self, user_id):
        """Compute R(u,p) for every catalog item for one user, chunked.

        This helper is intended for wiring the same correction into a
        full-sort evaluator once the base MASKED_GLORIA full_sort_predict
        signature is known. It uses the already-computed masked embeddings and
        does not rerun GCN per candidate.
        """
        if not hasattr(self, 'mask_rep') or self.mask_rep is None:
            raise RuntimeError(
                'mask_rep is unavailable. Run the base embedding path first.'
            )

        user_id = int(user_id)
        item_rep = self.mask_rep[self.num_user:]
        num_items = int(item_rep.size(0))
        edge_ids = self.user_to_edge_ids[user_id]

        if len(edge_ids) < self.cf5_min_history:
            return torch.zeros(
                num_items,
                dtype=item_rep.dtype,
                device=item_rep.device
            )

        edge_tensor = torch.tensor(
            edge_ids,
            dtype=torch.long,
            device=self.forward_edge_users.device
        )
        history_item_ids = self.forward_edge_items[edge_tensor]
        history_rep = item_rep[history_item_ids]
        history_mask = self.get_forward_edge_mask()[edge_tensor]

        eps = self.cf5_target_mask_eps
        base_mask = history_mask.detach().clamp_min(eps)
        base_logits = torch.log(base_mask)
        base_weight = torch.softmax(base_logits, dim=0)
        global_history = torch.sum(
            base_weight.unsqueeze(-1) * history_rep,
            dim=0
        )

        history_norm = F.normalize(history_rep, dim=1)
        residual_chunks = []
        chunk_size = self.cf5_target_full_sort_chunk_size

        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            candidates = item_rep[start:end]
            candidate_norm = F.normalize(candidates, dim=1)

            # [history_len, chunk]
            similarity = torch.matmul(
                history_norm,
                candidate_norm.transpose(0, 1)
            )
            logits = (
                base_logits.unsqueeze(1)
                + self.cf5_target_gamma
                * similarity
                / self.cf5_target_temperature
            )
            weights = torch.softmax(logits, dim=0)

            # [chunk, dim]
            target_history = torch.matmul(
                weights.transpose(0, 1),
                history_rep
            )
            residual_rep = target_history - global_history.unsqueeze(0)
            residual_score = torch.sum(
                residual_rep * candidates,
                dim=1
            )
            residual_chunks.append(residual_score)

        return torch.cat(residual_chunks, dim=0)