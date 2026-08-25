# coding: utf-8
"""Exact pair-level Full/Mask/Joint analysis for an original MASKED_GLORIA checkpoint.

This is an offline, read-only diagnostic.  It reconstructs the two propagated
branches from the checkpoint, ranks every validation/test positive against the
whole item catalogue (after masking training history), and writes auditable
tables, post-hoc permutation diagnostics, regressions, and figures.
"""

import argparse
import csv
import json
import math
import os
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camure_matplotlib")
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERACTIONS = PROJECT_ROOT / "data" / "book" / "book.inter"
DEFAULT_TEXT_FEATURES = PROJECT_ROOT / "data" / "book" / "text_feat.npy"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "saved"
    / "MASKED_GLORIA-book-seed999-Aug-12-2026-13-46-25.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "original_checkpoint_branch_case_analysis"
BRANCHES = ("full", "masked", "joint")
PRIMARY_GROUPS = ("mask_only", "full_only", "both_win", "both_fail")
LOGISTIC_PREDICTORS = (
    "log1p_user_degree",
    "log1p_item_degree",
    "history_text_diversity",
    "target_history_text_cosine_max",
    "user_mask_mean",
    "user_mask_std",
    "item_mask_mean",
    "user_full_mask_cosine",
    "single_item_history",
)
CONTINUOUS_LOGISTIC_PREDICTORS = LOGISTIC_PREDICTORS[:-1]


def load_interaction_splits(path):
    # Lazy imports keep numeric unit tests usable in minimal PyTorch
    # environments; the reused legacy module imports matplotlib at module load.
    from analyze_branch_test_performance import load_interaction_splits as implementation
    return implementation(path)


def compute_knn_indices(text_feature_path, k, chunk_size, device):
    from analyze_branch_test_performance import compute_knn_indices as implementation
    return implementation(text_feature_path, k, chunk_size, device)


def build_branch_embeddings(state, train_rows, knn_indices, device):
    from analyze_branch_test_performance import build_branch_embeddings as implementation
    return implementation(state, train_rows, knn_indices, device)


def load_state_dict(checkpoint_path):
    from plot_same_user_embedding_similarity import load_state_dict as implementation
    return implementation(checkpoint_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze exact Full/Mask/Joint branch cases from MASKED_GLORIA."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--text-features", type=Path, default=DEFAULT_TEXT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits", nargs="+", choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--cutoffs", nargs="+", type=int, default=(5, 10, 20))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-chunk-size", type=int, default=1024)
    parser.add_argument("--directional-num-negatives", type=int, default=16)
    parser.add_argument("--directional-num-permutations", type=int, default=3)
    parser.add_argument("--directional-temperature", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--example-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args(argv)
    args.cutoffs = tuple(sorted(set(args.cutoffs)))
    for name in (
        "score_batch_size", "knn_k", "knn_chunk_size",
        "directional_num_negatives", "directional_num_permutations",
        "example_count",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if not args.cutoffs or min(args.cutoffs) <= 0:
        parser.error("--cutoffs must contain positive integers")
    if args.directional_temperature <= 0:
        parser.error("--directional-temperature must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples cannot be negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this environment")
    return args


def _finite_numpy(name, value):
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        raise RuntimeError("{} contains NaN or Inf.".format(name))


def _safe_stats(values, size):
    """Per-node mean/std/min/max for edge-aligned values."""
    ids, weights = values
    ids = np.asarray(ids, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    count = np.bincount(ids, minlength=size).astype(np.int64)
    total = np.bincount(ids, weights=weights, minlength=size)
    square = np.bincount(ids, weights=weights * weights, minlength=size)
    mean = np.divide(total, count, out=np.zeros(size), where=count > 0)
    variance = np.divide(square, count, out=np.zeros(size), where=count > 0) - mean ** 2
    std = np.sqrt(np.maximum(variance, 0.0))
    minimum = np.full(size, np.inf)
    maximum = np.full(size, -np.inf)
    np.minimum.at(minimum, ids, weights)
    np.maximum.at(maximum, ids, weights)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    return count, mean, std, minimum, maximum


def embedding_cosine_l2(first, second):
    first = first.detach().cpu().float()
    second = second.detach().cpu().float()
    cosine = F.cosine_similarity(first, second, dim=1, eps=1e-12).numpy()
    distance = torch.linalg.vector_norm(first - second, dim=1).numpy()
    return cosine, distance


def quantile_strata(values, quantiles=(0.2, 0.4, 0.6, 0.8)):
    """Return fixed quantile thresholds and bins without splitting degree ties."""
    values = np.asarray(values)
    thresholds = np.quantile(values, quantiles, method="linear")
    # side='left' keeps values equal to a threshold in the lower stratum.
    bins = np.searchsorted(thresholds, values, side="left").astype(np.int64) + 1
    return thresholds, bins


def text_history_features(train_users, train_items, num_users, text_normalized):
    counts = np.bincount(train_users, minlength=num_users).astype(np.int64)
    sums = np.zeros((num_users, text_normalized.shape[1]), dtype=np.float64)
    np.add.at(sums, train_users, text_normalized[train_items])
    numerator = np.sum(sums * sums, axis=1) - counts
    denominator = counts * np.maximum(counts - 1, 0)
    mean_pair_cosine = np.divide(
        numerator, denominator, out=np.ones(num_users), where=denominator > 0
    )
    diversity = 1.0 - mean_pair_cosine
    diversity[counts <= 1] = 0.0
    return diversity, (counts == 1).astype(np.int64)


def _pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def build_graph_features(state, embeddings, split_rows, text_feature_path):
    num_users = int(embeddings["num_users"])
    num_items = int(embeddings["num_items"])
    train = split_rows[0]
    train_users = np.asarray([u for u, _ in train], dtype=np.int64)
    train_items = np.asarray([i for _, i in train], dtype=np.int64)
    mask = torch.sigmoid(state["mask_logits"].detach().cpu()).numpy().astype(np.float64)
    if len(mask) != len(train):
        raise RuntimeError("mask_logits do not preserve the training-edge ordering.")

    user_degree, user_mean, user_std, user_min, user_max = _safe_stats(
        (train_users, mask), num_users
    )
    item_degree, item_mean, item_std, item_min, item_max = _safe_stats(
        (train_items, mask), num_items
    )
    user_lt09 = np.divide(
        np.bincount(train_users, weights=mask < 0.9, minlength=num_users),
        user_degree, out=np.zeros(num_users), where=user_degree > 0,
    )
    user_lt05 = np.divide(
        np.bincount(train_users, weights=mask < 0.5, minlength=num_users),
        user_degree, out=np.zeros(num_users), where=user_degree > 0,
    )
    val_count = np.bincount(
        np.asarray([u for u, _ in split_rows.get(1, [])], dtype=np.int64),
        minlength=num_users,
    )
    test_count = np.bincount(
        np.asarray([u for u, _ in split_rows.get(2, [])], dtype=np.int64),
        minlength=num_users,
    )

    text = np.load(text_feature_path.expanduser().resolve()).astype(np.float64)
    if text.shape[0] != num_items:
        raise RuntimeError("Text feature item count differs from the checkpoint.")
    norms = np.linalg.norm(text, axis=1, keepdims=True)
    text = np.divide(text, norms, out=np.zeros_like(text), where=norms > 0)
    diversity, single = text_history_features(
        train_users, train_items, num_users, text
    )
    user_cos, user_l2 = embedding_cosine_l2(
        embeddings["full_user"], embeddings["masked_user"]
    )
    item_cos, item_l2 = embedding_cosine_l2(
        embeddings["full_item"], embeddings["masked_item"]
    )

    top_user_threshold = float(np.quantile(user_degree, 0.8, method="linear"))
    neighbor_degree_sum = np.bincount(
        train_items, weights=user_degree[train_users], minlength=num_items
    )
    neighbor_top_sum = np.bincount(
        train_items, weights=user_degree[train_users] > top_user_threshold,
        minlength=num_items,
    )
    neighbor_mean = np.divide(
        neighbor_degree_sum, item_degree, out=np.zeros(num_items), where=item_degree > 0
    )
    neighbor_top_rate = np.divide(
        neighbor_top_sum, item_degree, out=np.zeros(num_items), where=item_degree > 0
    )
    popularity_20, popularity_80 = np.quantile(
        item_degree, (0.2, 0.8), method="linear"
    )
    popularity_band = np.full(num_items, "mid", dtype=object)
    popularity_band[item_degree <= popularity_20] = "tail"
    popularity_band[item_degree > popularity_80] = "head"
    user_thresholds, user_quintile = quantile_strata(user_degree)
    item_thresholds, item_quintile = quantile_strata(item_degree)

    user_features = []
    for user_id in range(num_users):
        user_features.append({
            "user_id": user_id,
            "user_degree": int(user_degree[user_id]),
            "log1p_user_degree": float(np.log1p(user_degree[user_id])),
            "validation_positive_count": int(val_count[user_id]),
            "test_positive_count": int(test_count[user_id]),
            "user_mask_mean": float(user_mean[user_id]),
            "user_mask_std": float(user_std[user_id]),
            "user_mask_min": float(user_min[user_id]),
            "user_mask_max": float(user_max[user_id]),
            "user_mask_removed_strength": float(1.0 - user_mean[user_id]),
            "user_mask_ratio_lt_0_9": float(user_lt09[user_id]),
            "user_mask_ratio_lt_0_5": float(user_lt05[user_id]),
            "user_full_mask_cosine": float(user_cos[user_id]),
            "user_full_mask_l2": float(user_l2[user_id]),
            "history_text_diversity": float(diversity[user_id]),
            "single_item_history": int(single[user_id]),
            "user_degree_quintile": int(user_quintile[user_id]),
        })
    item_features = []
    for item_id in range(num_items):
        item_features.append({
            "item_id": item_id,
            "item_degree": int(item_degree[item_id]),
            "log1p_item_degree": float(np.log1p(item_degree[item_id])),
            "item_popularity_band": str(popularity_band[item_id]),
            "neighbor_user_degree_mean": float(neighbor_mean[item_id]),
            "neighbor_top20_user_rate": float(neighbor_top_rate[item_id]),
            "item_mask_mean": float(item_mean[item_id]),
            "item_mask_std": float(item_std[item_id]),
            "item_mask_min": float(item_min[item_id]),
            "item_mask_max": float(item_max[item_id]),
            "item_full_mask_cosine": float(item_cos[item_id]),
            "item_full_mask_l2": float(item_l2[item_id]),
            "item_degree_quintile": int(item_quintile[item_id]),
        })

    train_groups = defaultdict(list)
    known_groups = defaultdict(set)
    for label in (0, 1, 2):
        for user_id, item_id in split_rows.get(label, []):
            known_groups[int(user_id)].add(int(item_id))
            if label == 0:
                train_groups[int(user_id)].append(int(item_id))
    rank_degree = _rankdata_average(user_degree)
    correlations = {}
    for name, values in (
        ("user_mask_mean", user_mean),
        ("user_mask_std", user_std),
        ("user_mask_removed_strength", 1.0 - user_mean),
        ("user_mask_ratio_lt_0_9", user_lt09),
        ("user_mask_ratio_lt_0_5", user_lt05),
    ):
        correlations[name] = {
            "pearson": _pearson(user_degree, values),
            "spearman": _pearson(rank_degree, _rankdata_average(values)),
        }
    metadata = {
        "user_degree_quintile_thresholds": user_thresholds.tolist(),
        "item_degree_quintile_thresholds": item_thresholds.tolist(),
        "item_popularity_p20": float(popularity_20),
        "item_popularity_p80": float(popularity_80),
        "top20_user_degree_threshold": top_user_threshold,
        "mask_weight_quantiles": {
            str(q): float(np.quantile(mask, q))
            for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "user_degree_mask_correlations": correlations,
    }
    return user_features, item_features, train_groups, known_groups, text, metadata


def exact_rank_batch(scores, histories, user_ids, cutoffs):
    """Rank a batch exactly; equal scores are ordered by ascending item ID."""
    scores = scores.clone()
    batch_size, num_items = scores.shape
    candidate = torch.ones_like(scores, dtype=torch.bool)
    for row, user_id in enumerate(user_ids):
        history = histories.get(int(user_id), ())
        if history:
            history_tensor = torch.as_tensor(history, dtype=torch.long, device=scores.device)
            scores[row, history_tensor] = -torch.inf
            candidate[row, history_tensor] = False
    candidate_count = candidate.sum(dim=1)
    if torch.any(candidate_count < max(cutoffs)):
        raise RuntimeError("A user has fewer candidates than the maximum cutoff.")
    safe = torch.where(candidate, scores, torch.zeros_like(scores))
    mean = safe.sum(dim=1) / candidate_count
    variance = (
        torch.where(candidate, (scores - mean[:, None]) ** 2, torch.zeros_like(scores)).sum(dim=1)
        / candidate_count
    )
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    # Columns enter in ascending item-ID order. A stable descending sort thus
    # implements the required smaller-item-ID tie policy. PyTorch versions
    # before the stable-sort argument was introduced use the exact NumPy
    # mergesort fallback below; this is slower because it crosses to CPU, but
    # it never perturbs close non-tied scores with an artificial epsilon.
    try:
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
    except TypeError:
        scores_numpy = scores.detach().cpu().numpy()
        order_numpy = np.argsort(-scores_numpy, axis=1, kind="mergesort")
        order = torch.from_numpy(order_numpy.copy()).to(device=scores.device)
    inverse = torch.empty_like(order)
    ranks = torch.arange(1, num_items + 1, device=scores.device).expand(batch_size, -1)
    inverse.scatter_(1, order, ranks)
    boundaries = {int(k): scores.gather(1, order[:, k - 1:k]).squeeze(1) for k in cutoffs}
    return {
        "scores": scores,
        "rank": inverse,
        "std": std,
        "boundary": boundaries,
        "candidate_count": candidate_count,
    }


def assign_group(full_hit, masked_hit):
    if masked_hit and not full_hit:
        return "mask_only"
    if full_hit and not masked_hit:
        return "full_only"
    if full_hit and masked_hit:
        return "both_win"
    return "both_fail"


def _target_history_similarity(user_id, item_id, train_groups, text):
    history = train_groups.get(int(user_id), ())
    if not history:
        raise RuntimeError("Heldout user {} has no training history.".format(user_id))
    similarity = text[int(item_id)] @ text[np.asarray(history, dtype=np.int64)].T
    return float(np.max(similarity)), float(np.mean(similarity))


@torch.no_grad()
def analyze_split_pairs(
    split_name, heldout_rows, embeddings, user_features, item_features,
    train_groups, text, cutoffs, batch_size,
):
    by_user = defaultdict(list)
    for user_id, item_id in heldout_rows:
        by_user[int(user_id)].append(int(item_id))
    users = np.asarray(sorted(by_user), dtype=np.int64)
    device = embeddings["full_user"].device
    result = []
    for start in range(0, len(users), batch_size):
        batch_users_np = users[start:start + batch_size]
        batch_users = torch.as_tensor(batch_users_np, dtype=torch.long, device=device)
        full_scores = embeddings["full_user"][batch_users] @ embeddings["full_item"].T
        masked_scores = embeddings["masked_user"][batch_users] @ embeddings["masked_item"].T
        joint_scores = full_scores + masked_scores
        if not all(torch.isfinite(x).all() for x in (full_scores, masked_scores, joint_scores)):
            raise RuntimeError("Non-finite unmasked score detected in {}.".format(split_name))
        ranked = {
            "full": exact_rank_batch(full_scores, train_groups, batch_users_np, cutoffs),
            "masked": exact_rank_batch(masked_scores, train_groups, batch_users_np, cutoffs),
            "joint": exact_rank_batch(joint_scores, train_groups, batch_users_np, cutoffs),
        }
        for local_row, user_id in enumerate(batch_users_np):
            for item_id in by_user[int(user_id)]:
                row = {"split": split_name, "user_id": int(user_id), "positive_item_id": item_id}
                row.update(user_features[int(user_id)])
                row.update(item_features[item_id])
                target_max, target_mean = _target_history_similarity(
                    user_id, item_id, train_groups, text
                )
                row["target_history_text_cosine_max"] = target_max
                row["target_history_text_cosine_mean"] = target_mean
                row["target_history_text_domain_shift_proxy"] = 1.0 - target_max
                for branch in BRANCHES:
                    data = ranked[branch]
                    positive_score = float(data["scores"][local_row, item_id].item())
                    rank = int(data["rank"][local_row, item_id].item())
                    row["{}_positive_score".format(branch)] = positive_score
                    row["{}_rank".format(branch)] = rank
                    row["{}_hard_negative_count".format(branch)] = rank - 1
                    row["{}_candidate_score_std".format(branch)] = float(data["std"][local_row].item())
                    for k in cutoffs:
                        boundary = float(data["boundary"][k][local_row].item())
                        margin = positive_score - boundary
                        row["{}_boundary_score_at_{}".format(branch, k)] = boundary
                        row["{}_boundary_margin_at_{}".format(branch, k)] = margin
                        row["{}_normalized_boundary_margin_at_{}".format(branch, k)] = (
                            margin / (float(data["std"][local_row].item()) + 1e-12)
                        )
                        row["{}_hit_at_{}".format(branch, k)] = int(rank <= k)
                row["full_minus_mask_rank"] = row["full_rank"] - row["masked_rank"]
                row["full_minus_joint_rank"] = row["full_rank"] - row["joint_rank"]
                row["joint_rank_gain_vs_full"] = row["full_rank"] - row["joint_rank"]
                row["joint_rank_gain_vs_masked"] = row["masked_rank"] - row["joint_rank"]
                row["mask_positive_score_contribution"] = (
                    row["joint_positive_score"] - row["full_positive_score"]
                )
                if not np.isclose(
                    row["mask_positive_score_contribution"],
                    row["masked_positive_score"], rtol=1e-5, atol=1e-5,
                ):
                    raise RuntimeError("Joint score does not equal Full + Mask.")
                for k in cutoffs:
                    full_hit = bool(row["full_hit_at_{}".format(k)])
                    mask_hit = bool(row["masked_hit_at_{}".format(k)])
                    joint_hit = bool(row["joint_hit_at_{}".format(k)])
                    row["group_at_{}".format(k)] = assign_group(full_hit, mask_hit)
                    row["joint_rescue_at_{}".format(k)] = int(
                        not full_hit and not mask_hit and joint_hit
                    )
                    row["fusion_harm_at_{}".format(k)] = int(
                        (full_hit or mask_hit) and not joint_hit
                    )
                result.append(row)
    return result


def make_derangement(unique_users, rng):
    unique_users = np.asarray(unique_users, dtype=np.int64)
    if len(unique_users) < 2:
        return {int(user): int(user) for user in unique_users}
    shuffled = rng.permutation(unique_users)
    shifted = np.roll(shuffled, 1)
    mapping = dict(zip(shuffled.tolist(), shifted.tolist()))
    if any(user == other for user, other in mapping.items()):
        raise RuntimeError("Failed to construct a derangement.")
    return mapping


def sample_unseen_candidates(rows, known_groups, num_items, num_negatives, seed):
    rng = np.random.RandomState(seed)
    candidates = np.empty((len(rows), num_negatives + 1), dtype=np.int64)
    for row_index, row in enumerate(rows):
        user_id = int(row["user_id"])
        positive = int(row["positive_item_id"])
        excluded = known_groups[user_id]
        if num_items - len(excluded) < num_negatives:
            raise RuntimeError("Not enough unseen negatives for user {}.".format(user_id))
        selected = set()
        while len(selected) < num_negatives:
            proposal = int(rng.randint(0, num_items))
            if proposal not in excluded and proposal != positive:
                selected.add(proposal)
        candidates[row_index, 0] = positive
        candidates[row_index, 1:] = np.asarray(sorted(selected), dtype=np.int64)
    return candidates


def js_divergence_from_logits(original_logits, permuted_logits):
    log_p = F.log_softmax(original_logits, dim=1)
    log_q = F.log_softmax(permuted_logits, dim=1)
    p, q = log_p.exp(), log_q.exp()
    mixture = 0.5 * (p + q)
    log_mixture = torch.log(torch.clamp(mixture, min=torch.finfo(mixture.dtype).tiny))
    return 0.5 * torch.sum(p * (log_p - log_mixture), dim=1) + 0.5 * torch.sum(
        q * (log_q - log_mixture), dim=1
    )


@torch.no_grad()
def add_directional_diagnostics(
    rows, embeddings, known_groups, num_negatives=16, num_permutations=3,
    temperature=0.5, seed=999, batch_size=256,
):
    """Add post-hoc directional gap/JSD columns; does not train the checkpoint."""
    if not rows:
        return
    candidates = sample_unseen_candidates(
        rows, known_groups, embeddings["num_items"], num_negatives, seed
    )
    users = np.asarray([int(row["user_id"]) for row in rows], dtype=np.int64)
    unique_users = np.unique(users)
    if len(unique_users) < 2:
        for row in rows:
            for name in (
                "directional_gap_mean", "directional_gap_std", "directional_correct_win_rate",
                "directional_jsd_mean", "directional_jsd_std",
                "directional_original_positive_probability_mean",
                "directional_permuted_positive_probability_mean",
            ):
                row[name] = 0.0
        return
    rng = np.random.RandomState(seed + 7919)
    maps = [make_derangement(unique_users, rng) for _ in range(num_permutations)]
    gaps = np.empty((len(rows), num_permutations), dtype=np.float64)
    jsds = np.empty_like(gaps)
    orig_probs = np.empty_like(gaps)
    perm_probs = np.empty_like(gaps)
    device = embeddings["full_user"].device
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        batch_users_np = users[start:end]
        batch_users = torch.as_tensor(batch_users_np, dtype=torch.long, device=device)
        batch_items = torch.as_tensor(candidates[start:end], dtype=torch.long, device=device)
        full_score = torch.sum(
            embeddings["full_user"][batch_users, None, :] * embeddings["full_item"][batch_items],
            dim=2,
        )
        mask_score = torch.sum(
            embeddings["masked_user"][batch_users, None, :] * embeddings["masked_item"][batch_items],
            dim=2,
        )
        original_logits = (full_score + mask_score) / temperature
        original_log_probability = F.log_softmax(original_logits, dim=1)
        for permutation_index, mapping in enumerate(maps):
            perm_users = torch.as_tensor(
                [mapping[int(user)] for user in batch_users_np],
                dtype=torch.long, device=device,
            )
            perm_mask_score = torch.sum(
                embeddings["masked_user"][perm_users, None, :]
                * embeddings["masked_item"][batch_items], dim=2,
            )
            permuted_logits = (full_score + perm_mask_score) / temperature
            perm_log_probability = F.log_softmax(permuted_logits, dim=1)
            gaps[start:end, permutation_index] = (
                original_log_probability[:, 0] - perm_log_probability[:, 0]
            ).cpu().numpy()
            jsds[start:end, permutation_index] = js_divergence_from_logits(
                original_logits, permuted_logits
            ).cpu().numpy()
            orig_probs[start:end, permutation_index] = original_log_probability[:, 0].exp().cpu().numpy()
            perm_probs[start:end, permutation_index] = perm_log_probability[:, 0].exp().cpu().numpy()
    for index, row in enumerate(rows):
        row["directional_posthoc_diagnostic"] = 1
        row["directional_gap_mean"] = float(np.mean(gaps[index]))
        row["directional_gap_std"] = float(np.std(gaps[index]))
        row["directional_correct_win_rate"] = float(np.mean(gaps[index] > 0))
        row["directional_jsd_mean"] = float(np.mean(jsds[index]))
        row["directional_jsd_std"] = float(np.std(jsds[index]))
        row["directional_original_positive_probability_mean"] = float(np.mean(orig_probs[index]))
        row["directional_permuted_positive_probability_mean"] = float(np.mean(perm_probs[index]))


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if len(values) else float("nan")


def _median(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)) if len(values) else float("nan")


def cluster_bootstrap_mean_ci(values, user_ids, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    user_ids = np.asarray(user_ids, dtype=np.int64)
    if samples <= 0 or len(values) == 0:
        return float("nan"), float("nan")
    _, inverse = np.unique(user_ids, return_inverse=True)
    user_count = int(inverse.max()) + 1
    cluster_count = np.bincount(inverse, minlength=user_count).astype(np.float64)
    cluster_sum = np.bincount(inverse, weights=values, minlength=user_count)
    rng = np.random.RandomState(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 64):
        end = min(start + 64, samples)
        selected = rng.randint(0, user_count, size=(end - start, user_count))
        estimates[start:end] = (
            cluster_sum[selected].sum(axis=1) / cluster_count[selected].sum(axis=1)
        )
    return tuple(float(x) for x in np.percentile(estimates, (2.5, 97.5)))


def cluster_bootstrap_metric_cis(rows, metrics, samples, seed):
    """Vectorized user-cluster bootstrap for several pair-level metrics."""
    if not rows or samples <= 0:
        return {metric: (float("nan"), float("nan")) for metric in metrics}
    user_ids = np.asarray([row["user_id"] for row in rows], dtype=np.int64)
    _, inverse = np.unique(user_ids, return_inverse=True)
    user_count = int(inverse.max()) + 1
    cluster_count = np.bincount(inverse, minlength=user_count).astype(np.float64)
    cluster_sums = np.zeros((user_count, len(metrics)), dtype=np.float64)
    for metric_index, metric in enumerate(metrics):
        cluster_sums[:, metric_index] = np.bincount(
            inverse,
            weights=np.asarray([row[metric] for row in rows], dtype=np.float64),
            minlength=user_count,
        )
    rng = np.random.RandomState(seed)
    estimates = np.empty((samples, len(metrics)), dtype=np.float64)
    for start in range(0, samples, 64):
        end = min(start + 64, samples)
        selected = rng.randint(0, user_count, size=(end - start, user_count))
        denominator = cluster_count[selected].sum(axis=1)
        for metric_index in range(len(metrics)):
            estimates[start:end, metric_index] = (
                cluster_sums[selected, metric_index].sum(axis=1) / denominator
            )
    return {
        metric: tuple(float(value) for value in np.percentile(estimates[:, index], (2.5, 97.5)))
        for index, metric in enumerate(metrics)
    }


def summarize_groups(rows, cutoffs, bootstrap_samples, seed):
    counts, feature_summary, directional_summary = [], [], []
    feature_names = (
        "log1p_user_degree", "log1p_item_degree", "history_text_diversity",
        "target_history_text_cosine_max", "user_mask_mean", "item_mask_mean",
        "user_full_mask_cosine", "item_full_mask_cosine",
    )
    directional_metrics = (
        "directional_gap_mean", "directional_correct_win_rate", "directional_jsd_mean",
        "directional_original_positive_probability_mean",
        "directional_permuted_positive_probability_mean",
    )
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for k in cutoffs:
            for group in PRIMARY_GROUPS:
                subset = [row for row in split_rows if row["group_at_{}".format(k)] == group]
                counts.append({
                    "split": split, "cutoff": k, "group": group,
                    "pair_count": len(subset),
                    "pair_percentage": 100.0 * len(subset) / max(len(split_rows), 1),
                    "unique_user_count": len({row["user_id"] for row in subset}),
                })
                for feature in feature_names:
                    values = [row[feature] for row in subset]
                    feature_summary.append({
                        "split": split, "cutoff": k, "group": group,
                        "feature": feature, "pair_count": len(subset),
                        "mean": _mean(values), "median": _median(values),
                        "q25": float(np.quantile(values, 0.25)) if values else float("nan"),
                        "q75": float(np.quantile(values, 0.75)) if values else float("nan"),
                    })
                confidence_intervals = cluster_bootstrap_metric_cis(
                    subset, directional_metrics, bootstrap_samples, seed + k
                )
                for metric in directional_metrics:
                    values = [row[metric] for row in subset]
                    lower, upper = confidence_intervals[metric]
                    directional_summary.append({
                        "split": split, "cutoff": k, "group": group,
                        "metric": metric, "pair_count": len(subset),
                        "mean": _mean(values), "median": _median(values),
                        "user_cluster_bootstrap_ci_lower": lower,
                        "user_cluster_bootstrap_ci_upper": upper,
                    })
            for overlay in ("joint_rescue", "fusion_harm"):
                subset = [row for row in split_rows if row["{}_at_{}".format(overlay, k)]]
                counts.append({
                    "split": split, "cutoff": k, "group": overlay,
                    "pair_count": len(subset),
                    "pair_percentage": 100.0 * len(subset) / max(len(split_rows), 1),
                    "unique_user_count": len({row["user_id"] for row in subset}),
                })
    return counts, feature_summary, directional_summary


def summarize_strata(rows, cutoffs):
    output = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for k in cutoffs:
            for user_q in range(1, 6):
                for item_q in range(1, 6):
                    subset = [
                        row for row in split_rows
                        if row["user_degree_quintile"] == user_q
                        and row["item_degree_quintile"] == item_q
                    ]
                    count = len(subset)
                    group_counts = {
                        group: sum(row["group_at_{}".format(k)] == group for row in subset)
                        for group in PRIMARY_GROUPS
                    }
                    record = {
                        "split": split, "cutoff": k,
                        "user_degree_quintile": user_q,
                        "item_degree_quintile": item_q, "pair_count": count,
                    }
                    for group in PRIMARY_GROUPS:
                        record["{}_count".format(group)] = group_counts[group]
                        record["{}_rate".format(group)] = group_counts[group] / count if count else float("nan")
                    for overlay in ("joint_rescue", "fusion_harm"):
                        record["{}_rate".format(overlay)] = (
                            _mean([row["{}_at_{}".format(overlay, k)] for row in subset])
                        )
                    for feature in (
                        "history_text_diversity", "target_history_text_cosine_max",
                        "user_mask_removed_strength",
                        "full_normalized_boundary_margin_at_{}".format(k),
                        "masked_normalized_boundary_margin_at_{}".format(k),
                        "joint_normalized_boundary_margin_at_{}".format(k),
                    ):
                        record["mean_{}".format(feature)] = _mean([row[feature] for row in subset])
                    output.append(record)
    return output


def validate_logistic_predictors(predictors):
    forbidden = ("rank", "margin", "hard_negative", "group", "hit", "rescue", "harm")
    leaking = [name for name in predictors if any(token in name for token in forbidden)]
    if leaking:
        raise ValueError("Target-leaking logistic predictors: {}".format(leaking))
    return True


def _standardize_matrix(matrix, continuous_count):
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    mean = matrix[:, :continuous_count].mean(axis=0)
    std = matrix[:, :continuous_count].std(axis=0)
    std[std < 1e-12] = 1.0
    matrix[:, :continuous_count] = (matrix[:, :continuous_count] - mean) / std
    return matrix, mean, std


def fit_logistic_associations(rows, cutoffs, bootstrap_samples, seed):
    validate_logistic_predictors(LOGISTIC_PREDICTORS)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for logistic analysis.") from exc
    output = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for k in cutoffs:
            selected = [
                row for row in split_rows
                if row["group_at_{}".format(k)] in ("mask_only", "full_only")
            ]
            if not selected:
                continue
            y = np.asarray([
                int(row["group_at_{}".format(k)] == "mask_only") for row in selected
            ], dtype=np.int64)
            groups = np.asarray([row["user_id"] for row in selected], dtype=np.int64)
            x_raw = np.asarray([
                [float(row[name]) for name in LOGISTIC_PREDICTORS] for row in selected
            ], dtype=np.float64)
            if np.unique(y).size < 2 or min(np.bincount(y)) < 5:
                warnings.warn("Too few observations in one class for {}@{}.".format(split, k))
                continue
            x, _, _ = _standardize_matrix(x_raw, len(CONTINUOUS_LOGISTIC_PREDICTORS))
            # LogisticRegression defaults to L2; omitting the explicit penalty
            # also avoids the sklearn >=1.8 deprecation warning.
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            model.fit(x, y)
            aucs = []
            n_splits = min(5, len(np.unique(groups)))
            if n_splits >= 2:
                for train_index, test_index in GroupKFold(n_splits=n_splits).split(x_raw, y, groups):
                    if np.unique(y[train_index]).size < 2 or np.unique(y[test_index]).size < 2:
                        continue
                    x_train, mean, std = _standardize_matrix(
                        x_raw[train_index], len(CONTINUOUS_LOGISTIC_PREDICTORS)
                    )
                    x_test = x_raw[test_index].copy()
                    x_test[:, :len(CONTINUOUS_LOGISTIC_PREDICTORS)] = (
                        x_test[:, :len(CONTINUOUS_LOGISTIC_PREDICTORS)] - mean
                    ) / std
                    fold_model = LogisticRegression(
                        C=1.0, solver="lbfgs", max_iter=2000
                    ).fit(x_train, y[train_index])
                    aucs.append(roc_auc_score(y[test_index], fold_model.predict_proba(x_test)[:, 1]))
            rng = np.random.RandomState(seed + k + (0 if split == "validation" else 10000))
            unique_users = np.unique(groups)
            boot = []
            for _ in range(bootstrap_samples):
                sampled_users = rng.choice(unique_users, size=len(unique_users), replace=True)
                indices = np.concatenate([np.flatnonzero(groups == user) for user in sampled_users])
                if np.unique(y[indices]).size < 2:
                    continue
                x_boot, _, _ = _standardize_matrix(
                    x_raw[indices], len(CONTINUOUS_LOGISTIC_PREDICTORS)
                )
                try:
                    fitted = LogisticRegression(
                        C=1.0, solver="lbfgs", max_iter=2000
                    ).fit(x_boot, y[indices])
                    boot.append(fitted.coef_[0])
                except (ValueError, FloatingPointError):
                    continue
            boot = np.asarray(boot, dtype=np.float64)
            if bootstrap_samples >= 950 and len(boot) < 950:
                warnings.warn(
                    "Only {} successful bootstrap fits for {}@{}.".format(len(boot), split, k)
                )
            for predictor_index, predictor in enumerate(LOGISTIC_PREDICTORS):
                coefficient = float(model.coef_[0, predictor_index])
                lower, upper = (float("nan"), float("nan"))
                if len(boot):
                    lower, upper = (float(v) for v in np.percentile(boot[:, predictor_index], (2.5, 97.5)))
                output.append({
                    "split": split, "cutoff": k, "predictor": predictor,
                    "coefficient_scale": (
                        "per_one_standard_deviation"
                        if predictor in CONTINUOUS_LOGISTIC_PREDICTORS
                        else "binary_zero_to_one"
                    ),
                    "coefficient_per_sd": coefficient,
                    "odds_ratio": float(np.exp(coefficient)),
                    "coefficient_ci_lower": lower,
                    "coefficient_ci_upper": upper,
                    "odds_ratio_ci_lower": float(np.exp(lower)),
                    "odds_ratio_ci_upper": float(np.exp(upper)),
                    "groupkfold_roc_auc_mean": _mean(aucs),
                    "groupkfold_successful_folds": len(aucs),
                    "bootstrap_successful_fits": len(boot),
                    "observation_count": len(selected),
                    "unique_user_count": len(unique_users),
                    "mask_only_count": int(y.sum()),
                    "full_only_count": int((1 - y).sum()),
                    "interpretation": "association_not_causal",
                })
    return output


def select_case_examples(rows, cutoffs, example_count):
    examples = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for k in cutoffs:
            definitions = {
                "mask_only": lambda r: r["group_at_{}".format(k)] == "mask_only",
                "full_only": lambda r: r["group_at_{}".format(k)] == "full_only",
                "joint_rescue": lambda r: bool(r["joint_rescue_at_{}".format(k)]),
                "fusion_harm": lambda r: bool(r["fusion_harm_at_{}".format(k)]),
            }
            for case_type, predicate in definitions.items():
                subset = [row for row in split_rows if predicate(row)]
                if case_type == "mask_only":
                    strength = lambda r: r["masked_normalized_boundary_margin_at_{}".format(k)] - r["full_normalized_boundary_margin_at_{}".format(k)]
                elif case_type == "full_only":
                    strength = lambda r: r["full_normalized_boundary_margin_at_{}".format(k)] - r["masked_normalized_boundary_margin_at_{}".format(k)]
                elif case_type == "joint_rescue":
                    strength = lambda r: r["joint_normalized_boundary_margin_at_{}".format(k)] - max(r["full_normalized_boundary_margin_at_{}".format(k)], r["masked_normalized_boundary_margin_at_{}".format(k)])
                else:
                    strength = lambda r: max(r["full_normalized_boundary_margin_at_{}".format(k)], r["masked_normalized_boundary_margin_at_{}".format(k)]) - r["joint_normalized_boundary_margin_at_{}".format(k)]
                chosen = sorted(
                    subset,
                    key=lambda row: (-float(strength(row)), int(row["user_id"]), int(row["positive_item_id"])),
                )[:example_count]
                for priority, row in enumerate(chosen, start=1):
                    record = dict(row)
                    record.update({
                        "case_type": case_type, "case_cutoff": k,
                        "case_priority": priority, "case_strength": float(strength(row)),
                    })
                    examples.append(record)
    return examples


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(plt, path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def create_plots(rows, counts, cutoffs, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create analysis plots.") from exc
    colors = {"mask_only": "#2ca02c", "full_only": "#d62728", "both_win": "#1f77b4", "both_fail": "#7f7f7f"}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        # 1. Stacked four-way partition.
        bottom = np.zeros(len(cutoffs))
        plt.figure(figsize=(7, 4.5))
        for group in PRIMARY_GROUPS:
            values = [
                next(record["pair_percentage"] for record in counts if record["split"] == split and record["cutoff"] == k and record["group"] == group)
                for k in cutoffs
            ]
            plt.bar([str(k) for k in cutoffs], values, bottom=bottom, label=group, color=colors[group])
            bottom += values
        plt.xlabel("K"); plt.ylabel("Pair percentage (%)"); plt.title("{} branch cases".format(split.title())); plt.legend(fontsize=8)
        _save_figure(plt, output_dir / "{}_primary_groups.png".format(split))

        base_k = max(cutoffs)
        mask_rows = [row for row in split_rows if row["group_at_{}".format(base_k)] == "mask_only"]
        full_rows = [row for row in split_rows if row["group_at_{}".format(base_k)] == "full_only"]
        # 2/3. Degree and popularity distributions.
        for field, label, suffix in (
            ("log1p_user_degree", "log(1 + user degree)", "user_degree"),
            ("log1p_item_degree", "log(1 + item popularity)", "item_popularity"),
        ):
            plt.figure(figsize=(6.5, 4.5))
            if mask_rows:
                plt.hist([row[field] for row in mask_rows], bins=30, density=True, alpha=0.55, label="mask_only")
            if full_rows:
                plt.hist([row[field] for row in full_rows], bins=30, density=True, alpha=0.55, label="full_only")
            plt.xlabel(label); plt.ylabel("Density"); plt.title("{} at K={}".format(split.title(), base_k)); plt.legend()
            _save_figure(plt, output_dir / "{}_{}.png".format(split, suffix))

        # 4. Rank difference hexbin plus per-degree-bin median/IQR.
        x = np.asarray([row["log1p_user_degree"] for row in split_rows])
        y = np.asarray([row["full_minus_mask_rank"] for row in split_rows])
        plt.figure(figsize=(7, 4.8)); plt.hexbin(x, y, gridsize=45, bins="log", mincnt=1, cmap="viridis")
        plt.colorbar(label="log count")
        if len(x):
            edges = np.linspace(x.min(), x.max(), 16)
            centers, medians, lowers, uppers = [], [], [], []
            for left, right in zip(edges[:-1], edges[1:]):
                values = y[(x >= left) & (x <= right)]
                if len(values):
                    centers.append((left + right) / 2); medians.append(np.median(values)); lowers.append(np.quantile(values, .25)); uppers.append(np.quantile(values, .75))
            plt.plot(centers, medians, color="red", linewidth=2, label="median")
            plt.fill_between(centers, lowers, uppers, color="red", alpha=.18, label="IQR")
        plt.axhline(0, color="black", linewidth=.8); plt.xlabel("log(1 + user degree)"); plt.ylabel("Full rank - Mask rank"); plt.legend()
        _save_figure(plt, output_dir / "{}_rank_difference_hexbin.png".format(split))

        # 5. Directional gap and JSD by primary group at max K.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
        data = [[row["directional_gap_mean"] for row in split_rows if row["group_at_{}".format(base_k)] == group] for group in PRIMARY_GROUPS]
        axes[0].boxplot(data, labels=PRIMARY_GROUPS, showfliers=False); axes[0].set_title("Post-hoc directional gap"); axes[0].tick_params(axis="x", rotation=25)
        data = [[row["directional_jsd_mean"] for row in split_rows if row["group_at_{}".format(base_k)] == group] for group in PRIMARY_GROUPS]
        axes[1].boxplot(data, labels=PRIMARY_GROUPS, showfliers=False); axes[1].set_title("Post-hoc listwise JSD"); axes[1].tick_params(axis="x", rotation=25)
        _save_figure(plt, output_dir / "{}_directional_by_group.png".format(split))

        # 6. Overlay rates.
        plt.figure(figsize=(6.5, 4.3)); positions = np.arange(len(cutoffs)); width = .36
        for offset, overlay in ((-.5, "joint_rescue"), (.5, "fusion_harm")):
            values = [next(record["pair_percentage"] for record in counts if record["split"] == split and record["cutoff"] == k and record["group"] == overlay) for k in cutoffs]
            plt.bar(positions + offset * width, values, width=width, label=overlay)
        plt.xticks(positions, cutoffs); plt.xlabel("K"); plt.ylabel("Pair percentage (%)"); plt.legend(); plt.title(split.title())
        _save_figure(plt, output_dir / "{}_fusion_overlays.png".format(split))

        # 7. User mask distributions by primary group.
        plt.figure(figsize=(7, 4.5))
        data = [[row["user_mask_mean"] for row in split_rows if row["group_at_{}".format(base_k)] == group] for group in PRIMARY_GROUPS]
        plt.boxplot(data, labels=PRIMARY_GROUPS, showfliers=False); plt.xticks(rotation=25); plt.ylabel("Mean incident edge mask"); plt.title("{} at K={}".format(split.title(), base_k))
        _save_figure(plt, output_dir / "{}_user_mask_by_group.png".format(split))


def validate_checkpoint(state, embeddings, split_rows, expected_users=11000, expected_items=9332, expected_dim=64, expected_edges=120464):
    num_users, num_items = embeddings["num_users"], embeddings["num_items"]
    if (num_users, num_items) != (expected_users, expected_items):
        raise RuntimeError("Expected {}/{} users/items, found {}/{}.".format(expected_users, expected_items, num_users, num_items))
    for key in ("full_user", "masked_user", "full_item", "masked_item"):
        if embeddings[key].shape[1] != expected_dim:
            raise RuntimeError("{} does not have dimension {}.".format(key, expected_dim))
        if not torch.isfinite(embeddings[key]).all():
            raise RuntimeError("{} contains NaN/Inf.".format(key))
    if len(split_rows[0]) != expected_edges or state["mask_logits"].numel() != expected_edges:
        raise RuntimeError("Expected {} ordered training edges.".format(expected_edges))


def _recall_audit(rows, cutoffs):
    output = {}
    for split in sorted({row["split"] for row in rows}):
        output[split] = {}
        split_rows = [row for row in rows if row["split"] == split]
        for branch in BRANCHES:
            output[split][branch] = {}
            for k in cutoffs:
                hits = [row["{}_hit_at_{}".format(branch, k)] for row in split_rows]
                per_user_hits = defaultdict(list)
                for row in split_rows:
                    per_user_hits[int(row["user_id"])].append(
                        row["{}_hit_at_{}".format(branch, k)]
                    )
                macro_user_recall = _mean([_mean(values) for values in per_user_hits.values()])
                output[split][branch][str(k)] = {
                    "pair_hit_rate": _mean(hits),
                    "macro_user_recall": macro_user_recall,
                    "hit_count": int(np.sum(hits)),
                }
    return output


def run(args):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = load_interaction_splits(args.interactions)
    requested = {"validation": 1, "test": 2}
    for split in args.splits:
        if not split_rows.get(requested[split]):
            raise RuntimeError("Requested split '{}' is empty.".format(split))
    state = load_state_dict(args.checkpoint)
    knn = compute_knn_indices(args.text_features, args.knn_k, args.knn_chunk_size, device)
    embeddings = build_branch_embeddings(state, split_rows[0], knn, device)
    validate_checkpoint(state, embeddings, split_rows)
    user_features, item_features, train_groups, known_groups, text, metadata = build_graph_features(
        state, embeddings, split_rows, args.text_features
    )
    pair_rows = []
    for split in args.splits:
        current = analyze_split_pairs(
            split, split_rows[requested[split]], embeddings, user_features,
            item_features, train_groups, text, args.cutoffs, args.score_batch_size,
        )
        add_directional_diagnostics(
            current, embeddings, known_groups,
            num_negatives=args.directional_num_negatives,
            num_permutations=args.directional_num_permutations,
            temperature=args.directional_temperature,
            seed=args.seed,
            batch_size=args.score_batch_size,
        )
        pair_rows.extend(current)
    counts, features, directional = summarize_groups(
        pair_rows, args.cutoffs, args.bootstrap_samples, args.seed
    )
    strata = summarize_strata(pair_rows, args.cutoffs)
    logistic = fit_logistic_associations(
        pair_rows, args.cutoffs, args.bootstrap_samples, args.seed
    )
    examples = select_case_examples(pair_rows, args.cutoffs, args.example_count)
    write_csv(output_dir / "branch_pair_analysis.csv", pair_rows)
    write_csv(output_dir / "user_features.csv", user_features)
    write_csv(output_dir / "item_features.csv", item_features)
    write_csv(output_dir / "group_counts_by_k.csv", counts)
    write_csv(output_dir / "group_feature_summary.csv", features)
    write_csv(output_dir / "directional_group_summary.csv", directional)
    write_csv(output_dir / "degree_popularity_strata.csv", strata)
    write_csv(output_dir / "logistic_coefficients.csv", logistic)
    write_csv(output_dir / "case_examples.csv", examples)
    create_plots(pair_rows, counts, args.cutoffs, output_dir)
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "interactions": str(args.interactions.expanduser().resolve()),
        "text_features": str(args.text_features.expanduser().resolve()),
        "splits": list(args.splits), "cutoffs": list(args.cutoffs),
        "num_users": embeddings["num_users"], "num_items": embeddings["num_items"],
        "embedding_dimension": int(embeddings["full_user"].shape[1]),
        "training_edge_count": len(split_rows[0]),
        "pair_counts": {split: sum(row["split"] == split for row in pair_rows) for split in args.splits},
        "candidate_mask_policy": "training_history_only_for_validation_and_test",
        "rank_tie_policy": "stable_descending_score_then_ascending_item_id",
        "directional_diagnostic": {
            "post_hoc_not_training_loss": True,
            "negative_sampling": "random_unseen_excluding_train_validation_test",
            "num_negatives": args.directional_num_negatives,
            "num_permutations": args.directional_num_permutations,
            "temperature": args.directional_temperature,
        },
        "logistic_interpretation": "association_after_measured_adjustment_not_causal",
        "recall_audit": _recall_audit(pair_rows, args.cutoffs),
        "feature_metadata": metadata,
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    print("Saved exact branch analysis to {}".format(args.output_dir.expanduser().resolve()))
    print(json.dumps(summary["pair_counts"], indent=2))


if __name__ == "__main__":
    main()
