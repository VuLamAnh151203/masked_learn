# coding: utf-8
"""Post-hoc adaptive Full-Mask gated fusion for MASKED_GLORIA.

The checkpoint is frozen.  Fusion weights are selected exclusively on the
validation split and then applied unchanged to test.  Training history is the
only catalogue mask for both validation and test, matching the project
evaluator.
"""

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camure_matplotlib")
)

from analyze_original_checkpoint_branch_cases import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INTERACTIONS,
    DEFAULT_TEXT_FEATURES,
    build_branch_embeddings,
    compute_knn_indices,
    load_interaction_splits,
    load_state_dict,
    validate_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "original_checkpoint_gated_fusion_results"
DEFAULT_CUTOFFS = (5, 10, 20)
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BASE_METHODS = ("full_only", "original_fusion")
TUNABLE_METHODS = (
    "normalized_static", "popularity_gate", "confidence_gate", "combined_gate"
)
GATED_METHODS = ("popularity_gate", "confidence_gate", "combined_gate")
DIAGNOSTIC_METHODS = (
    "full_only", "masked_only", "original_fusion", "normalized_static",
    "popularity_gate", "confidence_gate", "combined_gate",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune and evaluate post-hoc adaptive Full-Mask fusion."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--text-features", type=Path, default=DEFAULT_TEXT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=DEFAULT_CUTOFFS)
    parser.add_argument("--boundary-k", type=int, default=20)
    parser.add_argument("--uncertainty-temperature", type=float, default=0.5)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--selection-metric", choices=("ndcg@20",), default="ndcg@20")
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-chunk-size", type=int, default=1024)
    parser.add_argument("--rank-pair-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args(argv)
    args.cutoffs = tuple(sorted(set(args.cutoffs)))
    args.lambdas = tuple(sorted(set(float(value) for value in args.lambdas)))
    if not args.cutoffs or min(args.cutoffs) <= 0:
        parser.error("--cutoffs must contain positive integers")
    if 20 not in args.cutoffs or 10 not in args.cutoffs:
        parser.error("--cutoffs must include 10 and 20 for selection tie-breaks")
    if not args.lambdas or min(args.lambdas) < 0:
        parser.error("--lambdas must contain non-negative values")
    if args.boundary_k <= 0:
        parser.error("--boundary-k must be positive")
    if args.uncertainty_temperature <= 0:
        parser.error("--uncertainty-temperature must be positive")
    for name in ("score_batch_size", "knn_k", "knn_chunk_size", "rank_pair_chunk_size"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def group_user_items(rows):
    grouped = defaultdict(list)
    for user_id, item_id in rows:
        grouped[int(user_id)].append(int(item_id))
    return grouped


def average_rank_tailness(item_degree):
    """Average-tie ascending percentile rank, inverted so rare items approach 1."""
    degree = np.asarray(item_degree, dtype=np.float64)
    if degree.ndim != 1 or degree.size == 0:
        raise ValueError("item_degree must be a non-empty one-dimensional array")
    if degree.size == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(degree, kind="mergesort")
    rank = np.empty(degree.size, dtype=np.float64)
    start = 0
    while start < degree.size:
        end = start + 1
        while end < degree.size and degree[order[end]] == degree[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        rank[order[start:end]] = average
        start = end
    return 1.0 - (rank - 1.0) / (degree.size - 1.0)


def build_item_popularity(train_rows, num_items):
    train_items = np.asarray([item for _, item in train_rows], dtype=np.int64)
    degree = np.bincount(train_items, minlength=num_items).astype(np.int64)
    tailness = average_rank_tailness(degree)
    try:
        p20, p80 = np.quantile(degree, (0.2, 0.8), method="linear")
    except TypeError:  # NumPy < 1.22
        p20, p80 = np.quantile(degree, (0.2, 0.8), interpolation="linear")
    band = np.full(num_items, "mid", dtype=object)
    band[degree <= p20] = "tail"
    band[degree > p80] = "head"
    return degree, tailness, band, float(p20), float(p80)


def build_candidate_mask(scores, train_groups, user_ids):
    candidate = torch.ones_like(scores, dtype=torch.bool)
    for row, user_id in enumerate(user_ids):
        history = train_groups.get(int(user_id), ())
        if history:
            item_ids = torch.as_tensor(history, dtype=torch.long, device=scores.device)
            candidate[row, item_ids] = False
    return candidate


def normalize_unseen_scores(scores, candidate_mask, epsilon=1e-12):
    """Per-user population z-score using only training-unseen candidates."""
    if scores.shape != candidate_mask.shape:
        raise ValueError("scores and candidate_mask must have identical shapes")
    count = candidate_mask.sum(dim=1).to(scores.dtype)
    if torch.any(count <= 0):
        raise RuntimeError("A user has no unseen catalogue candidates")
    safe = torch.where(candidate_mask, scores, torch.zeros_like(scores))
    mean = safe.sum(dim=1) / count
    centered = scores - mean[:, None]
    variance = torch.where(
        candidate_mask, centered * centered, torch.zeros_like(centered)
    ).sum(dim=1) / count
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    normalized = centered / (std[:, None] + epsilon)
    return normalized, mean, std


def mask_seen(scores, candidate_mask):
    return scores.masked_fill(~candidate_mask, -torch.inf)


def full_boundary(normalized_full, candidate_mask, boundary_k):
    masked = mask_seen(normalized_full, candidate_mask)
    if torch.any(candidate_mask.sum(dim=1) < boundary_k):
        raise RuntimeError("A user has fewer unseen candidates than boundary-k")
    return torch.topk(masked, boundary_k, dim=1, largest=True, sorted=False).values.min(dim=1).values


def compute_uncertainty(normalized_full, boundary, temperature):
    return torch.exp(-torch.abs(normalized_full - boundary[:, None]) / temperature)


def method_gate(method, tailness, uncertainty):
    if method == "normalized_static":
        return torch.ones_like(uncertainty)
    if method == "popularity_gate":
        return tailness[None, :].expand_as(uncertainty)
    if method == "confidence_gate":
        return uncertainty
    if method == "combined_gate":
        return tailness[None, :] * uncertainty
    raise ValueError("Unknown tunable method: {}".format(method))


def fused_normalized_score(normalized_full, normalized_mask, gate, weight, candidate_mask):
    score = normalized_full + float(weight) * gate * normalized_mask
    return mask_seen(score, candidate_mask)


@torch.no_grad()
def prepare_score_batch(
    embeddings, user_ids, train_groups, tailness, boundary_k, temperature,
):
    device = embeddings["full_user"].device
    users = torch.as_tensor(user_ids, dtype=torch.long, device=device)
    raw_full = embeddings["full_user"][users] @ embeddings["full_item"].T
    raw_mask = embeddings["masked_user"][users] @ embeddings["masked_item"].T
    if not torch.isfinite(raw_full).all() or not torch.isfinite(raw_mask).all():
        raise RuntimeError("Branch scores contain NaN or Inf")
    candidate = build_candidate_mask(raw_full, train_groups, user_ids)
    normalized_full, full_mean, full_std = normalize_unseen_scores(raw_full, candidate)
    normalized_mask, mask_mean, mask_std = normalize_unseen_scores(raw_mask, candidate)
    boundary = full_boundary(normalized_full, candidate, boundary_k)
    uncertainty = compute_uncertainty(normalized_full, boundary, temperature)
    tailness_tensor = torch.as_tensor(tailness, dtype=raw_full.dtype, device=device)
    return {
        "raw_full": raw_full,
        "raw_mask": raw_mask,
        "normalized_full": normalized_full,
        "normalized_mask": normalized_mask,
        "candidate": candidate,
        "boundary": boundary,
        "uncertainty": uncertainty,
        "tailness": tailness_tensor,
        "full_mean": full_mean,
        "full_std": full_std,
        "mask_mean": mask_mean,
        "mask_std": mask_std,
    }


def score_method(batch, method, weight=None):
    candidate = batch["candidate"]
    if method == "full_only":
        return mask_seen(batch["raw_full"], candidate)
    if method == "masked_only":
        return mask_seen(batch["raw_mask"], candidate)
    if method == "original_fusion":
        return mask_seen(batch["raw_full"] + batch["raw_mask"], candidate)
    gate = method_gate(method, batch["tailness"], batch["uncertainty"])
    return fused_normalized_score(
        batch["normalized_full"], batch["normalized_mask"], gate, weight, candidate
    )


def deterministic_topk(scores, k):
    """Exact lexicographic top-k: descending score then ascending item ID.

    It uses torch.topk for speed, repairs boundary ties against the full row,
    and stable-sorts only the selected k values on CPU.  It therefore works on
    PyTorch releases that do not accept argsort(stable=True).
    """
    if k <= 0 or k > scores.shape[1]:
        raise ValueError("k must be in [1, num_items]")
    values, item_ids = torch.topk(scores, k, dim=1, largest=True, sorted=True)
    thresholds = values[:, -1]
    total_boundary_ties = (scores == thresholds[:, None]).sum(dim=1)
    selected_boundary_ties = (values == thresholds[:, None]).sum(dim=1)
    affected = torch.nonzero(
        total_boundary_ties > selected_boundary_ties, as_tuple=False
    ).flatten().cpu().numpy()
    ids_numpy = item_ids.detach().cpu().numpy().copy()
    values_numpy = values.detach().cpu().numpy().copy()
    for row in affected:
        threshold = thresholds[row]
        strict = torch.nonzero(scores[row] > threshold, as_tuple=False).flatten()
        tied = torch.nonzero(scores[row] == threshold, as_tuple=False).flatten()
        needed = k - strict.numel()
        selected = torch.cat((strict, tied[:needed]), dim=0)
        ids_numpy[row] = selected.cpu().numpy()
        values_numpy[row] = scores[row, selected].cpu().numpy()
    for row in range(ids_numpy.shape[0]):
        order = np.lexsort((ids_numpy[row], -values_numpy[row]))
        ids_numpy[row] = ids_numpy[row, order]
    return ids_numpy.astype(np.int64, copy=False)


def _metric_arrays(topk_items, eval_users, heldout_groups, cutoffs, item_band=None, band=None):
    max_k = max(cutoffs)
    recall = {k: [] for k in cutoffs}
    ndcg = {k: [] for k in cutoffs}
    positive_count = 0
    eligible_users = 0
    for row, user_id in enumerate(eval_users):
        positives = heldout_groups[int(user_id)]
        if band is not None:
            positives = [item for item in positives if item_band[item] == band]
            if not positives:
                continue
        eligible_users += 1
        positive_count += len(positives)
        positive_set = set(positives)
        hit = np.asarray(
            [int(item) in positive_set for item in topk_items[row, :max_k]],
            dtype=np.float64,
        )
        cumulative = np.cumsum(hit)
        discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
        cumulative_dcg = np.cumsum(hit * discounts)
        for k in cutoffs:
            ideal = min(len(positives), k)
            recall[k].append(cumulative[k - 1] / len(positives))
            ndcg[k].append(cumulative_dcg[k - 1] / discounts[:ideal].sum())
    return {
        k: {
            "recall": float(np.mean(recall[k])) if recall[k] else float("nan"),
            "ndcg": float(np.mean(ndcg[k])) if ndcg[k] else float("nan"),
            "eligible_user_count": eligible_users,
            "positive_count": positive_count,
        }
        for k in cutoffs
    }


def metrics_from_topk(topk_items, eval_users, heldout_groups, cutoffs, item_band):
    return {
        "overall": _metric_arrays(topk_items, eval_users, heldout_groups, cutoffs),
        "tail": _metric_arrays(
            topk_items, eval_users, heldout_groups, cutoffs, item_band, "tail"
        ),
        "head": _metric_arrays(
            topk_items, eval_users, heldout_groups, cutoffs, item_band, "head"
        ),
    }


def spec_key(method, weight=None):
    return method if weight is None else "{}|{:.12g}".format(method, float(weight))


@torch.no_grad()
def validation_grid_sweep(
    embeddings, train_groups, validation_groups, tailness, item_band, cutoffs,
    boundary_k, temperature, lambdas, batch_size,
):
    eval_users = np.asarray(sorted(validation_groups), dtype=np.int64)
    max_k = max(cutoffs)
    keys = [spec_key(method) for method in BASE_METHODS]
    keys += [spec_key(method, weight) for method in TUNABLE_METHODS for weight in lambdas]
    topk_batches = {key: [] for key in keys}
    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start + batch_size]
        batch = prepare_score_batch(
            embeddings, batch_users, train_groups, tailness, boundary_k, temperature
        )
        full_topk = deterministic_topk(score_method(batch, "full_only"), max_k)
        topk_batches["full_only"].append(full_topk)
        topk_batches["original_fusion"].append(
            deterministic_topk(score_method(batch, "original_fusion"), max_k)
        )
        for method in TUNABLE_METHODS:
            for weight in lambdas:
                key = spec_key(method, weight)
                if weight == 0:
                    topk_batches[key].append(full_topk.copy())
                else:
                    topk_batches[key].append(
                        deterministic_topk(score_method(batch, method, weight), max_k)
                    )
    rows = []
    for key, batches in topk_batches.items():
        topk = np.concatenate(batches, axis=0)
        method, separator, weight_text = key.partition("|")
        weight = float(weight_text) if separator else None
        metrics = metrics_from_topk(
            topk, eval_users, validation_groups, cutoffs, item_band
        )["overall"]
        for k in cutoffs:
            rows.append({
                "split": "validation", "method": method,
                "lambda": "" if weight is None else weight,
                "cutoff": k, "recall": metrics[k]["recall"],
                "ndcg": metrics[k]["ndcg"],
                "eligible_user_count": metrics[k]["eligible_user_count"],
                "positive_count": metrics[k]["positive_count"],
            })
    return rows


def select_hyperparameters(grid_rows):
    """Select independently by NDCG@20, Recall@20, NDCG@10, smaller lambda."""
    validation_rows = [
        row for row in grid_rows if row.get("split", "validation") == "validation"
    ]
    if not validation_rows:
        raise ValueError("No validation rows were provided for hyperparameter selection")
    selected = {}
    for method in TUNABLE_METHODS:
        candidates = []
        weights = sorted({
            float(row["lambda"]) for row in validation_rows if row["method"] == method
        })
        if not weights:
            raise ValueError("No validation candidates for {}".format(method))
        for weight in weights:
            relevant = {
                int(row["cutoff"]): row
                for row in validation_rows
                if row["method"] == method and float(row["lambda"]) == weight
            }
            if 10 not in relevant or 20 not in relevant:
                raise ValueError("{} lambda {} lacks cutoff 10 or 20".format(method, weight))
            candidates.append({
                "lambda": weight,
                "ndcg_at_20": float(relevant[20]["ndcg"]),
                "recall_at_20": float(relevant[20]["recall"]),
                "ndcg_at_10": float(relevant[10]["ndcg"]),
            })
        ranked = sorted(
            candidates,
            key=lambda row: (
                -row["ndcg_at_20"], -row["recall_at_20"],
                -row["ndcg_at_10"], row["lambda"],
            ),
        )
        selected[method] = {
            "selected_lambda": ranked[0]["lambda"],
            "selection_order": (
                "max_ndcg@20_then_recall@20_then_ndcg@10_then_smaller_lambda"
            ),
            "winner_metrics": ranked[0],
            "ranked_candidates": ranked,
        }
    return selected


def exact_positive_ranks(scores, local_rows, positive_items, chunk_size=256):
    local_rows = np.asarray(local_rows, dtype=np.int64)
    positive_items = np.asarray(positive_items, dtype=np.int64)
    device = scores.device
    item_ids = torch.arange(scores.shape[1], device=device)
    ranks = np.empty(len(local_rows), dtype=np.int64)
    positive_scores = np.empty(len(local_rows), dtype=np.float64)
    for start in range(0, len(local_rows), chunk_size):
        end = min(start + chunk_size, len(local_rows))
        rows = torch.as_tensor(local_rows[start:end], dtype=torch.long, device=device)
        items = torch.as_tensor(positive_items[start:end], dtype=torch.long, device=device)
        selected_scores = scores[rows, items]
        row_scores = scores[rows]
        greater = (row_scores > selected_scores[:, None]).sum(dim=1)
        tied_lower_id = (
            (row_scores == selected_scores[:, None]) & (item_ids[None, :] < items[:, None])
        ).sum(dim=1)
        ranks[start:end] = (1 + greater + tied_lower_id).cpu().numpy()
        positive_scores[start:end] = selected_scores.cpu().numpy()
    return ranks, positive_scores


@torch.no_grad()
def evaluate_selected_split(
    split_name, embeddings, train_groups, heldout_groups, selected, tailness,
    item_degree, item_band, cutoffs, boundary_k, temperature, batch_size,
    rank_pair_chunk_size,
):
    eval_users = np.asarray(sorted(heldout_groups), dtype=np.int64)
    max_k = max(cutoffs)
    method_weights = {
        method: selected[method]["selected_lambda"] for method in TUNABLE_METHODS
    }
    topk_batches = {method: [] for method in DIAGNOSTIC_METHODS}
    pair_rows = []
    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start + batch_size]
        batch = prepare_score_batch(
            embeddings, batch_users, train_groups, tailness, boundary_k, temperature
        )
        score_by_method = {
            "full_only": score_method(batch, "full_only"),
            "masked_only": score_method(batch, "masked_only"),
            "original_fusion": score_method(batch, "original_fusion"),
        }
        for method in TUNABLE_METHODS:
            score_by_method[method] = score_method(batch, method, method_weights[method])
        for method, scores in score_by_method.items():
            topk_batches[method].append(deterministic_topk(scores, max_k))

        local_rows, users, positives = [], [], []
        for local_row, user_id in enumerate(batch_users):
            for item_id in heldout_groups[int(user_id)]:
                local_rows.append(local_row); users.append(int(user_id)); positives.append(int(item_id))
        common = []
        for local_row, user_id, item_id in zip(local_rows, users, positives):
            common.append({
                "split": split_name, "user_id": user_id,
                "positive_item_id": item_id,
                "item_degree": int(item_degree[item_id]),
                "item_popularity_band": str(item_band[item_id]),
                "item_tailness": float(tailness[item_id]),
                "full_boundary_k": boundary_k,
                "full_boundary_score": float(batch["boundary"][local_row].item()),
                "full_raw_positive_score": float(batch["raw_full"][local_row, item_id].item()),
                "masked_raw_positive_score": float(batch["raw_mask"][local_row, item_id].item()),
                "full_normalized_positive_score": float(batch["normalized_full"][local_row, item_id].item()),
                "masked_normalized_positive_score": float(batch["normalized_mask"][local_row, item_id].item()),
                "full_uncertainty": float(batch["uncertainty"][local_row, item_id].item()),
                "popularity_gate": float(tailness[item_id]),
                "confidence_gate": float(batch["uncertainty"][local_row, item_id].item()),
                "combined_gate": float(tailness[item_id] * batch["uncertainty"][local_row, item_id].item()),
            })
        for method, scores in score_by_method.items():
            ranks, positive_scores = exact_positive_ranks(
                scores, local_rows, positives, rank_pair_chunk_size
            )
            for index, row in enumerate(common):
                row["{}_rank".format(method)] = int(ranks[index])
                row["{}_positive_score".format(method)] = float(positive_scores[index])
                for k in cutoffs:
                    row["{}_hit_at_{}".format(method, k)] = int(ranks[index] <= k)
        for row in common:
            for k in cutoffs:
                full_hit = bool(row["full_only_hit_at_{}".format(k)])
                mask_hit = bool(row["masked_only_hit_at_{}".format(k)])
                row["branch_group_at_{}".format(k)] = (
                    "mask_only" if mask_hit and not full_hit else
                    "full_only" if full_hit and not mask_hit else
                    "both_win" if full_hit and mask_hit else "both_fail"
                )
                for method in (
                    "original_fusion", "normalized_static", "popularity_gate",
                    "confidence_gate", "combined_gate",
                ):
                    final_hit = bool(row["{}_hit_at_{}".format(method, k)])
                    row["{}_joint_rescue_at_{}".format(method, k)] = int(
                        not full_hit and not mask_hit and final_hit
                    )
                    row["{}_fusion_harm_at_{}".format(method, k)] = int(
                        (full_hit or mask_hit) and not final_hit
                    )
        pair_rows.extend(common)
    topk = {
        method: np.concatenate(batches, axis=0)
        for method, batches in topk_batches.items()
    }
    metrics = {
        method: metrics_from_topk(
            values, eval_users, heldout_groups, cutoffs, item_band
        )
        for method, values in topk.items()
    }
    return metrics, pair_rows


def flatten_selected_metrics(split, metrics, selected, cutoffs):
    rows, tail_head = [], []
    reported_methods = (
        "full_only", "original_fusion", "normalized_static", "popularity_gate",
        "confidence_gate", "combined_gate",
    )
    for method in reported_methods:
        weight = "" if method in BASE_METHODS else selected[method]["selected_lambda"]
        for k in cutoffs:
            overall = metrics[method]["overall"][k]
            rows.append({
                "split": split, "method": method, "selected_lambda": weight,
                "cutoff": k, "recall": overall["recall"], "ndcg": overall["ndcg"],
            })
            for band in ("tail", "head"):
                values = metrics[method][band][k]
                tail_head.append({
                    "split": split, "method": method, "selected_lambda": weight,
                    "band": band, "cutoff": k, "recall": values["recall"],
                    "eligible_user_count": values["eligible_user_count"],
                    "positive_count": values["positive_count"],
                })
    lookup = {(row["split"], row["method"], row["cutoff"]): row for row in rows}
    for row in rows:
        for baseline in ("full_only", "original_fusion", "normalized_static"):
            reference = lookup[(split, baseline, row["cutoff"])]
            row["recall_delta_vs_{}".format(baseline)] = row["recall"] - reference["recall"]
            row["ndcg_delta_vs_{}".format(baseline)] = row["ndcg"] - reference["ndcg"]
    return rows, tail_head


def summarize_fusion_cases(pair_rows, cutoffs):
    output = []
    methods = (
        "original_fusion", "normalized_static", "popularity_gate",
        "confidence_gate", "combined_gate",
    )
    for split in sorted({row["split"] for row in pair_rows}):
        split_rows = [row for row in pair_rows if row["split"] == split]
        for method in methods:
            for k in cutoffs:
                for case in ("joint_rescue", "fusion_harm"):
                    field = "{}_{}_at_{}".format(method, case, k)
                    subset = [row for row in split_rows if row[field]]
                    output.append({
                        "split": split, "method": method, "cutoff": k,
                        "case": case, "pair_count": len(subset),
                        "pair_percentage": 100.0 * len(subset) / max(len(split_rows), 1),
                        "unique_user_count": len({row["user_id"] for row in subset}),
                    })
    return output


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {name: float("nan") for name in ("mean", "median", "p10", "p25", "p50", "p75", "p90")}
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)), "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_gate_groups(pair_rows, cutoffs):
    output = []
    for split in sorted({row["split"] for row in pair_rows}):
        split_rows = [row for row in pair_rows if row["split"] == split]
        for method in GATED_METHODS:
            for k in cutoffs:
                distributions = {}
                for group in ("mask_only", "full_only"):
                    values = [
                        row[method] for row in split_rows
                        if row["branch_group_at_{}".format(k)] == group
                    ]
                    distributions[group] = _distribution(values)
                    record = {
                        "split": split, "method": method, "cutoff": k,
                        "branch_group": group, "pair_count": len(values),
                    }
                    record.update(distributions[group])
                    output.append(record)
                difference = distributions["mask_only"]["mean"] - distributions["full_only"]["mean"]
                for record in output[-2:]:
                    record["mask_only_minus_full_only_mean"] = difference
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field); seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def create_plots(output_dir, grid_rows, selected_metrics, tail_head, fusion_cases, gate_groups, cutoffs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create plots") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for method in TUNABLE_METHODS:
        rows = sorted(
            [row for row in grid_rows if row["method"] == method and row["cutoff"] == 20],
            key=lambda row: float(row["lambda"]),
        )
        plt.plot([row["lambda"] for row in rows], [row["ndcg"] for row in rows], marker="o", label=method)
    plt.xlabel("Lambda"); plt.ylabel("Validation NDCG@20"); plt.legend(fontsize=8); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(output_dir / "validation_lambda_sweep.png", dpi=180); plt.close()

    test_rows = [row for row in selected_metrics if row["split"] == "test"]
    methods = list(dict.fromkeys(row["method"] for row in test_rows))
    x = np.arange(len(cutoffs)); width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, method in enumerate(methods):
        rows = {row["cutoff"]: row for row in test_rows if row["method"] == method}
        for axis, metric in zip(axes, ("recall", "ndcg")):
            axis.bar(x + (index - (len(methods) - 1) / 2) * width, [rows[k][metric] for k in cutoffs], width, label=method)
    for axis, metric in zip(axes, ("Recall", "NDCG")):
        axis.set_xticks(x, cutoffs); axis.set_title("Test {}".format(metric)); axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(output_dir / "test_overall_metrics.png", dpi=180); plt.close(fig)

    rows = [row for row in tail_head if row["split"] == "test" and row["cutoff"] == 20]
    plt.figure(figsize=(9, 4.8)); positions = np.arange(len(methods)); width = .35
    for offset, band in ((-.5, "tail"), (.5, "head")):
        lookup = {row["method"]: row["recall"] for row in rows if row["band"] == band}
        plt.bar(positions + offset * width, [lookup[m] for m in methods], width, label=band)
    plt.xticks(positions, methods, rotation=25, ha="right"); plt.ylabel("Recall@20"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "test_tail_head_recall.png", dpi=180); plt.close()

    rows = [row for row in fusion_cases if row["split"] == "test" and row["cutoff"] == 20]
    fusion_methods = list(dict.fromkeys(row["method"] for row in rows))
    plt.figure(figsize=(9, 4.8)); positions = np.arange(len(fusion_methods)); width = .35
    for offset, case in ((-.5, "joint_rescue"), (.5, "fusion_harm")):
        lookup = {row["method"]: row["pair_percentage"] for row in rows if row["case"] == case}
        plt.bar(positions + offset * width, [lookup[m] for m in fusion_methods], width, label=case)
    plt.xticks(positions, fusion_methods, rotation=25, ha="right"); plt.ylabel("Positive pairs (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "test_rescue_harm.png", dpi=180); plt.close()

    rows = [row for row in gate_groups if row["split"] == "test" and row["cutoff"] == 20]
    plt.figure(figsize=(8, 4.8)); positions = np.arange(len(GATED_METHODS)); width = .35
    for offset, group in ((-.5, "mask_only"), (.5, "full_only")):
        lookup = {row["method"]: row["mean"] for row in rows if row["branch_group"] == group}
        plt.bar(positions + offset * width, [lookup[m] for m in GATED_METHODS], width, label=group)
    plt.xticks(positions, GATED_METHODS, rotation=20); plt.ylabel("Mean positive-item gate"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "gate_by_branch_case.png", dpi=180); plt.close()


def run(args):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split_rows = load_interaction_splits(args.interactions)
    if not split_rows.get(1) or not split_rows.get(2):
        raise RuntimeError("Both validation and test splits are required")
    state = load_state_dict(args.checkpoint)
    knn = compute_knn_indices(args.text_features, args.knn_k, args.knn_chunk_size, device)
    embeddings = build_branch_embeddings(state, split_rows[0], knn, device)
    validate_checkpoint(state, embeddings, split_rows)
    train_groups = group_user_items(split_rows[0])
    validation_groups = group_user_items(split_rows[1])
    test_groups = group_user_items(split_rows[2])
    item_degree, tailness, item_band, p20, p80 = build_item_popularity(
        split_rows[0], embeddings["num_items"]
    )

    grid_rows = validation_grid_sweep(
        embeddings, train_groups, validation_groups, tailness, item_band,
        args.cutoffs, args.boundary_k, args.uncertainty_temperature,
        args.lambdas, args.score_batch_size,
    )
    selected = select_hyperparameters(grid_rows)
    all_pair_rows, selected_metric_rows, tail_head_rows = [], [], []
    for split_name, heldout in (("validation", validation_groups), ("test", test_groups)):
        metrics, pairs = evaluate_selected_split(
            split_name, embeddings, train_groups, heldout, selected, tailness,
            item_degree, item_band, args.cutoffs, args.boundary_k,
            args.uncertainty_temperature, args.score_batch_size,
            args.rank_pair_chunk_size,
        )
        metric_rows, band_rows = flatten_selected_metrics(
            split_name, metrics, selected, args.cutoffs
        )
        selected_metric_rows.extend(metric_rows); tail_head_rows.extend(band_rows)
        all_pair_rows.extend(pairs)
    fusion_cases = summarize_fusion_cases(all_pair_rows, args.cutoffs)
    gate_groups = summarize_gate_groups(all_pair_rows, args.cutoffs)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "validation_grid_metrics.csv", grid_rows)
    write_csv(output_dir / "selected_split_metrics.csv", selected_metric_rows)
    write_csv(output_dir / "tail_head_recall.csv", tail_head_rows)
    write_csv(output_dir / "fusion_case_metrics.csv", fusion_cases)
    write_csv(output_dir / "gate_group_summary.csv", gate_groups)
    write_csv(output_dir / "pair_diagnostics.csv", all_pair_rows)
    with (output_dir / "selected_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    create_plots(
        output_dir, grid_rows, selected_metric_rows, tail_head_rows,
        fusion_cases, gate_groups, args.cutoffs,
    )
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "num_users": embeddings["num_users"], "num_items": embeddings["num_items"],
        "embedding_dimension": int(embeddings["full_user"].shape[1]),
        "training_edge_count": len(split_rows[0]),
        "validation_pair_count": len(split_rows[1]), "test_pair_count": len(split_rows[2]),
        "cutoffs": list(args.cutoffs), "boundary_k": args.boundary_k,
        "uncertainty_temperature": args.uncertainty_temperature,
        "lambda_grid": list(args.lambdas), "selection_metric": args.selection_metric,
        "selected_hyperparameters": selected,
        "candidate_policy": "mask_training_history_only_for_validation_and_test",
        "score_normalization": "per_user_per_branch_population_zscore_over_training_unseen_catalog",
        "tailness": "one_minus_average_ascending_degree_percentile_rank",
        "tail_degree_threshold_p20": p20, "head_degree_threshold_p80": p80,
        "uncertainty": "exp(-abs(normalized_full-boundary)/temperature)",
        "rank_tie_policy": "descending_score_then_ascending_item_id",
        "checkpoint_frozen": True, "test_used_for_tuning": False,
        "test_metrics": [row for row in selected_metric_rows if row["split"] == "test"],
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    print("Saved gated-fusion experiment to {}".format(args.output_dir.expanduser().resolve()))
    print(json.dumps(summary["selected_hyperparameters"], indent=2))


if __name__ == "__main__":
    main()
