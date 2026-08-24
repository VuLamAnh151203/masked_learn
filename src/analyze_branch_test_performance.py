# coding: utf-8
"""Evaluate MASKED_GLORIA full, masked, and fused branches on the test set."""

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camure_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_same_user_embedding_similarity import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INTERACTIONS,
    load_state_dict,
    propagate_three_layers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_FEATURES = PROJECT_ROOT / "data" / "book" / "text_feat.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "branch_performance_results"
DEFAULT_CUTOFFS = (5, 10, 15, 20)
BRANCHES = ("full", "masked", "fused")
METRICS = ("recall", "ndcg", "precision", "map")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full-only, masked-only, and full+masked recommendation "
            "scores on the book test split."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--text-features", type=Path, default=DEFAULT_TEXT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="ranking cutoffs to report",
    )
    parser.add_argument("--case-cutoff", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument("--example-count", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu, cuda, or cuda:0",
    )
    args = parser.parse_args(argv)

    args.cutoffs = tuple(sorted(set(args.cutoffs)))
    if not args.cutoffs or min(args.cutoffs) <= 0:
        parser.error("--cutoffs must contain positive integers")
    if args.case_cutoff not in args.cutoffs:
        parser.error("--case-cutoff must be included in --cutoffs")
    for name in (
        "eval_batch_size",
        "knn_k",
        "knn_chunk_size",
        "example_count",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    return args


def load_interaction_splits(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Interaction file does not exist: {}".format(path))

    split_rows = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"userID", "itemID", "x_label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                "Interaction file is missing columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        for row in reader:
            split_rows[int(row["x_label"])].append(
                (int(row["userID"]), int(row["itemID"]))
            )
    if not split_rows[0] or not split_rows[2]:
        raise RuntimeError("Both training (0) and test (2) rows are required.")
    return split_rows


@torch.no_grad()
def compute_knn_indices(text_feature_path, k, chunk_size, device):
    text_feature_path = text_feature_path.expanduser().resolve()
    if not text_feature_path.is_file():
        raise FileNotFoundError(
            "Text feature file does not exist: {}".format(text_feature_path)
        )
    features = torch.from_numpy(np.load(text_feature_path)).to(
        device=device, dtype=torch.float32
    )
    if k > features.shape[0]:
        raise ValueError("knn-k cannot exceed the number of items")
    features = F.normalize(features, dim=1)
    feature_transpose = features.t().contiguous()
    neighbor_batches = []
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        similarity = torch.matmul(features[start:end], feature_transpose)
        neighbor_batches.append(torch.topk(similarity, k, dim=1).indices)
    return torch.cat(neighbor_batches, dim=0)


@torch.no_grad()
def build_branch_embeddings(state, train_rows, knn_indices, device):
    full_preference = state["full_gcn.preference"].to(device)
    mask_preference = state["mask_gcn.preference"].to(device)
    full_items = state["id_embedding_full.weight"].to(device)
    mask_items = state["id_embedding_masked.weight"].to(device)
    mask_logits = state["mask_logits"].to(device)

    num_users = full_preference.shape[0]
    num_items = full_items.shape[0]
    if knn_indices.shape[0] != num_items:
        raise RuntimeError("Text features and checkpoint item counts differ.")
    train_users_np = np.asarray([row[0] for row in train_rows], dtype=np.int64)
    train_items_np = np.asarray([row[1] for row in train_rows], dtype=np.int64)
    if mask_logits.numel() != train_users_np.size:
        raise RuntimeError(
            "Checkpoint has {} mask logits but training has {} interactions."
            .format(mask_logits.numel(), train_users_np.size)
        )

    train_users = torch.as_tensor(train_users_np, dtype=torch.long, device=device)
    item_nodes = (
        torch.as_tensor(train_items_np, dtype=torch.long, device=device)
        + num_users
    )
    src = torch.cat((train_users, item_nodes), dim=0)
    dst = torch.cat((item_nodes, train_users), dim=0)
    degree = torch.bincount(src, minlength=num_users + num_items).to(
        full_items.dtype
    )
    normalization = degree[src].pow(-0.5) * degree[dst].pow(-0.5)
    normalization[~torch.isfinite(normalization)] = 0.0

    _, _, _, full_result = propagate_three_layers(
        full_preference, full_items, src, dst, normalization
    )
    mask_weights = torch.sigmoid(mask_logits)
    mask_scale = torch.cat(
        (
            normalization[: train_users.numel()] * mask_weights,
            normalization[train_users.numel() :] * mask_weights,
        ),
        dim=0,
    )
    _, _, _, mask_result = propagate_three_layers(
        mask_preference, mask_items, src, dst, mask_scale
    )

    # MASKED_GLORIA.item_item returns rep + mm_adj @ rep.  Its KNN graph has
    # exactly k outgoing entries per item, so the normalization is 1/k.
    full_item_result = full_result[num_users:]
    mask_item_result = mask_result[num_users:]
    full_item_result = full_item_result + full_item_result[knn_indices].mean(dim=1)
    mask_item_result = mask_item_result + mask_item_result[knn_indices].mean(dim=1)

    return {
        "full_user": full_result[:num_users],
        "full_item": full_item_result,
        "masked_user": mask_result[:num_users],
        "masked_item": mask_item_result,
        "num_users": num_users,
        "num_items": num_items,
    }


def group_user_items(rows):
    groups = defaultdict(list)
    for user_id, item_id in rows:
        groups[int(user_id)].append(int(item_id))
    return groups


@torch.no_grad()
def rank_test_users(embeddings, train_groups, test_groups, max_k, batch_size):
    eval_users = np.asarray(sorted(test_groups), dtype=np.int64)
    if max_k > embeddings["num_items"]:
        raise ValueError("Maximum cutoff exceeds the number of items.")

    topk_by_branch = {branch: [] for branch in BRANCHES}
    device = embeddings["full_user"].device
    for start in range(0, eval_users.size, batch_size):
        batch_users_np = eval_users[start : start + batch_size]
        batch_users = torch.as_tensor(
            batch_users_np, dtype=torch.long, device=device
        )
        full_scores = torch.matmul(
            embeddings["full_user"][batch_users],
            embeddings["full_item"].t(),
        )
        masked_scores = torch.matmul(
            embeddings["masked_user"][batch_users],
            embeddings["masked_item"].t(),
        )
        fused_scores = full_scores + masked_scores

        history_lengths = np.asarray(
            [len(train_groups[int(user_id)]) for user_id in batch_users_np],
            dtype=np.int64,
        )
        if np.any(history_lengths == 0):
            missing_user = int(batch_users_np[np.flatnonzero(history_lengths == 0)[0]])
            raise RuntimeError("Test user {} has no training history.".format(missing_user))
        mask_rows = torch.as_tensor(
            np.repeat(np.arange(batch_users_np.size), history_lengths),
            dtype=torch.long,
            device=device,
        )
        mask_items = torch.as_tensor(
            np.concatenate(
                [train_groups[int(user_id)] for user_id in batch_users_np]
            ),
            dtype=torch.long,
            device=device,
        )
        for scores in (full_scores, masked_scores, fused_scores):
            scores[mask_rows, mask_items] = -1e10

        for branch, scores in zip(
            BRANCHES, (full_scores, masked_scores, fused_scores)
        ):
            topk_by_branch[branch].append(
                torch.topk(scores, max_k, dim=1).indices.cpu().numpy()
            )

    return eval_users, {
        branch: np.concatenate(batches, axis=0)
        for branch, batches in topk_by_branch.items()
    }


def per_user_metrics(topk_items, eval_users, test_groups, cutoffs):
    max_k = max(cutoffs)
    hits = np.zeros((eval_users.size, max_k), dtype=np.float64)
    positive_counts = np.empty(eval_users.size, dtype=np.int64)
    for row, user_id in enumerate(eval_users):
        positives = np.asarray(test_groups[int(user_id)], dtype=np.int64)
        positive_counts[row] = positives.size
        hits[row] = np.isin(topk_items[row], positives)

    metrics = {}
    cumulative_hits = np.cumsum(hits, axis=1)
    ranks = np.arange(1, max_k + 1, dtype=np.float64)
    discounts = 1.0 / np.log2(ranks + 1.0)
    cumulative_dcg = np.cumsum(hits * discounts, axis=1)
    precision_at_rank = cumulative_hits / ranks
    cumulative_ap = np.cumsum(precision_at_rank * hits, axis=1)

    for cutoff in cutoffs:
        index = cutoff - 1
        ideal_lengths = np.minimum(positive_counts, cutoff)
        idcg = np.asarray(
            [discounts[:length].sum() for length in ideal_lengths],
            dtype=np.float64,
        )
        metrics[cutoff] = {
            "recall": cumulative_hits[:, index] / positive_counts,
            "ndcg": cumulative_dcg[:, index] / idcg,
            "precision": cumulative_hits[:, index] / cutoff,
            "map": cumulative_ap[:, index] / ideal_lengths,
            "hit_count": cumulative_hits[:, index].astype(np.int64),
        }
    return hits.astype(bool), metrics


def classify_cases(branch_metrics, cutoff):
    full_recall = branch_metrics["full"][cutoff]["recall"]
    masked_recall = branch_metrics["masked"][cutoff]["recall"]
    fused_recall = branch_metrics["fused"][cutoff]["recall"]
    full_hits = branch_metrics["full"][cutoff]["hit_count"]
    masked_hits = branch_metrics["masked"][cutoff]["hit_count"]
    fused_hits = branch_metrics["fused"][cutoff]["hit_count"]
    return {
        "full_wins": full_recall > masked_recall,
        "masked_wins": masked_recall > full_recall,
        "branches_tie": full_recall == masked_recall,
        "full_only_success": (full_hits > 0) & (masked_hits == 0),
        "masked_only_success": (masked_hits > 0) & (full_hits == 0),
        "both_branches_succeed": (full_hits > 0) & (masked_hits > 0),
        "both_fail_fusion_wins":
            (full_hits == 0) & (masked_hits == 0) & (fused_hits > 0),
        "both_fail_all":
            (full_hits == 0) & (masked_hits == 0) & (fused_hits == 0),
        "fusion_beats_both":
            (fused_recall > full_recall) & (fused_recall > masked_recall),
    }


def join_ids(values):
    return "|".join(str(int(value)) for value in values)


def write_aggregate_csv(path, branch_metrics, cutoffs):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("branch", "cutoff", *METRICS))
        for branch in BRANCHES:
            for cutoff in cutoffs:
                writer.writerow(
                    (
                        branch,
                        cutoff,
                        *(
                            "{:.12g}".format(
                                float(branch_metrics[branch][cutoff][metric].mean())
                            )
                            for metric in METRICS
                        ),
                    )
                )


def write_per_user_csv(
    path,
    eval_users,
    test_groups,
    topk_by_branch,
    branch_metrics,
    cases,
    cutoff,
):
    fields = [
        "user_id",
        "test_positive_count",
        "test_positive_items",
        "case_label",
        "full_wins",
        "masked_wins",
        "both_fail_fusion_wins",
    ]
    for branch in BRANCHES:
        fields.extend(
            [
                "{}_recall_at_{}".format(branch, cutoff),
                "{}_ndcg_at_{}".format(branch, cutoff),
                "{}_hit_items_at_{}".format(branch, cutoff),
                "{}_top_{}".format(branch, cutoff),
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, user_id in enumerate(eval_users):
            positives = set(test_groups[int(user_id)])
            if cases["both_fail_fusion_wins"][row]:
                label = "both_fail_fusion_wins"
            elif cases["full_wins"][row]:
                label = "full_wins"
            elif cases["masked_wins"][row]:
                label = "masked_wins"
            else:
                label = "branches_tie"
            output = {
                "user_id": int(user_id),
                "test_positive_count": len(positives),
                "test_positive_items": join_ids(sorted(positives)),
                "case_label": label,
                "full_wins": int(cases["full_wins"][row]),
                "masked_wins": int(cases["masked_wins"][row]),
                "both_fail_fusion_wins": int(
                    cases["both_fail_fusion_wins"][row]
                ),
            }
            for branch in BRANCHES:
                topk = topk_by_branch[branch][row, :cutoff]
                output["{}_recall_at_{}".format(branch, cutoff)] = (
                    "{:.12g}".format(
                        branch_metrics[branch][cutoff]["recall"][row]
                    )
                )
                output["{}_ndcg_at_{}".format(branch, cutoff)] = (
                    "{:.12g}".format(
                        branch_metrics[branch][cutoff]["ndcg"][row]
                    )
                )
                output["{}_hit_items_at_{}".format(branch, cutoff)] = join_ids(
                    [item for item in topk if int(item) in positives]
                )
                output["{}_top_{}".format(branch, cutoff)] = join_ids(topk)
            writer.writerow(output)


def select_examples(cases, branch_metrics, example_count, cutoff):
    selections = {}
    priorities = {
        "full_wins": (
            branch_metrics["full"][cutoff]["recall"]
            - branch_metrics["masked"][cutoff]["recall"]
        ),
        "masked_wins": (
            branch_metrics["masked"][cutoff]["recall"]
            - branch_metrics["full"][cutoff]["recall"]
        ),
        "both_fail_fusion_wins": branch_metrics["fused"][cutoff]["recall"],
    }
    for name, priority in priorities.items():
        indices = np.flatnonzero(cases[name])
        order = np.lexsort((indices, -priority[indices]))
        selections[name] = indices[order[:example_count]]
    return selections


def write_examples_csv(
    path,
    selections,
    eval_users,
    test_groups,
    topk_by_branch,
    branch_metrics,
    cutoff,
):
    fields = ["case", "user_id", "test_positive_items"]
    for branch in BRANCHES:
        fields.extend(
            [
                "{}_recall_at_{}".format(branch, cutoff),
                "{}_hit_items_at_{}".format(branch, cutoff),
                "{}_top_{}".format(branch, cutoff),
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_name, indices in selections.items():
            for row in indices:
                positives = set(test_groups[int(eval_users[row])])
                output = {
                    "case": case_name,
                    "user_id": int(eval_users[row]),
                    "test_positive_items": join_ids(sorted(positives)),
                }
                for branch in BRANCHES:
                    topk = topk_by_branch[branch][row, :cutoff]
                    output["{}_recall_at_{}".format(branch, cutoff)] = (
                        "{:.12g}".format(
                            branch_metrics[branch][cutoff]["recall"][row]
                        )
                    )
                    output["{}_hit_items_at_{}".format(branch, cutoff)] = join_ids(
                        [item for item in topk if int(item) in positives]
                    )
                    output["{}_top_{}".format(branch, cutoff)] = join_ids(topk)
                writer.writerow(output)


def save_performance_plot(path, branch_metrics, cutoffs):
    colors = {"full": "#4c78a8", "masked": "#f58518", "fused": "#54a24b"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    x = np.arange(len(cutoffs))
    width = 0.25
    for axis, metric in zip(axes.flat, METRICS):
        for offset, branch in enumerate(BRANCHES):
            values = [
                branch_metrics[branch][cutoff][metric].mean()
                for cutoff in cutoffs
            ]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                label=branch,
                color=colors[branch],
            )
        axis.set_title(metric.upper())
        axis.set_xticks(x, [str(cutoff) for cutoff in cutoffs])
        axis.grid(axis="y", alpha=0.25)
    axes[1, 0].set_xlabel("K")
    axes[1, 1].set_xlabel("K")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[0, 0].legend()
    figure.suptitle("Book test performance by MASKED_GLORIA branch")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_case_plot(path, case_counts, cutoff):
    names = ("full_wins", "masked_wins", "both_fail_fusion_wins")
    labels = (
        "Full Recall >\nMasked",
        "Masked Recall >\nFull",
        "Both fail,\nfusion wins",
    )
    values = [case_counts[name] for name in names]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(labels, values, color=("#4c78a8", "#f58518", "#54a24b"))
    axis.set_ylim(0, max(values) * 1.15)
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Test users")
    axis.set_title("Branch cases at Recall@{}".format(cutoff))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    state = load_state_dict(args.checkpoint)
    split_rows = load_interaction_splits(args.interactions)
    print("Computing the item text KNN graph...")
    knn_indices = compute_knn_indices(
        args.text_features, args.knn_k, args.knn_chunk_size, device
    )
    print("Building full and masked branch embeddings...")
    embeddings = build_branch_embeddings(
        state, split_rows[0], knn_indices, device
    )
    train_groups = group_user_items(split_rows[0])
    test_groups = group_user_items(split_rows[2])
    print("Ranking test items for full, masked, and fused scorers...")
    eval_users, topk_by_branch = rank_test_users(
        embeddings,
        train_groups,
        test_groups,
        max(args.cutoffs),
        args.eval_batch_size,
    )

    branch_metrics = {}
    for branch in BRANCHES:
        _, branch_metrics[branch] = per_user_metrics(
            topk_by_branch[branch], eval_users, test_groups, args.cutoffs
        )
    cases = classify_cases(branch_metrics, args.case_cutoff)
    case_counts = {name: int(values.sum()) for name, values in cases.items()}
    selections = select_examples(
        cases, branch_metrics, args.example_count, args.case_cutoff
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "branch_test_metrics.csv"
    per_user_path = output_dir / "branch_test_per_user.csv"
    examples_path = output_dir / "branch_case_examples.csv"
    summary_path = output_dir / "branch_test_summary.json"
    performance_plot_path = output_dir / "branch_test_performance.png"
    case_plot_path = output_dir / "branch_case_counts.png"

    write_aggregate_csv(aggregate_path, branch_metrics, args.cutoffs)
    write_per_user_csv(
        per_user_path,
        eval_users,
        test_groups,
        topk_by_branch,
        branch_metrics,
        cases,
        args.case_cutoff,
    )
    write_examples_csv(
        examples_path,
        selections,
        eval_users,
        test_groups,
        topk_by_branch,
        branch_metrics,
        args.case_cutoff,
    )
    save_performance_plot(performance_plot_path, branch_metrics, args.cutoffs)
    save_case_plot(case_plot_path, case_counts, args.case_cutoff)

    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "test_user_count": int(eval_users.size),
        "test_interaction_count": int(len(split_rows[2])),
        "cutoffs": list(args.cutoffs),
        "case_cutoff": args.case_cutoff,
        "case_definitions": {
            "full_wins": "Recall(full) > Recall(masked)",
            "masked_wins": "Recall(masked) > Recall(full)",
            "full_only_success": "full has a hit and masked has zero hits",
            "masked_only_success": "masked has a hit and full has zero hits",
            "both_fail_fusion_wins": (
                "full and masked each have zero hits, fused has at least one hit"
            ),
        },
        "case_counts": case_counts,
        "case_percentages_of_test_users": {
            name: 100.0 * count / eval_users.size
            for name, count in case_counts.items()
        },
        "metrics": {
            branch: {
                str(cutoff): {
                    metric: float(branch_metrics[branch][cutoff][metric].mean())
                    for metric in METRICS
                }
                for cutoff in args.cutoffs
            }
            for branch in BRANCHES
        },
        "example_user_ids": {
            name: [int(eval_users[row]) for row in rows]
            for name, rows in selections.items()
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("Evaluated {:,} test users.".format(eval_users.size))
    for branch in BRANCHES:
        values = branch_metrics[branch][args.case_cutoff]
        print(
            "{} @{}: Recall={:.6f}, NDCG={:.6f}, Precision={:.6f}, MAP={:.6f}"
            .format(
                branch,
                args.case_cutoff,
                values["recall"].mean(),
                values["ndcg"].mean(),
                values["precision"].mean(),
                values["map"].mean(),
            )
        )
    print("Case counts @{}: {}".format(args.case_cutoff, case_counts))
    print("Output directory: {}".format(output_dir))


if __name__ == "__main__":
    main()
