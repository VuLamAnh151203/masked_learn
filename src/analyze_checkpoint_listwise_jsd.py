# coding: utf-8
"""Measure listwise Mask-user permutation JSD for a trained checkpoint."""

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

from analyze_branch_test_performance import (
    DEFAULT_TEXT_FEATURES,
    build_branch_embeddings,
    compute_knn_indices,
    load_interaction_splits,
)
from plot_same_user_embedding_similarity import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INTERACTIONS,
    load_state_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoint_listwise_jsd_results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute listwise JSD between original and Mask-user-permuted "
            "ranking distributions from a trained MASKED_GLORIA checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--text-features", type=Path, default=DEFAULT_TEXT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-negatives", type=int, default=32)
    parser.add_argument("--num-permutations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-chunk-size", type=int, default=10000)
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device such as cpu or cuda:0"
    )
    args = parser.parse_args(argv)

    for name in (
        "num_negatives",
        "num_permutations",
        "score_batch_size",
        "knn_k",
        "knn_chunk_size",
        "histogram_bins",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    return args


def group_interactions(rows):
    groups = defaultdict(list)
    for user_id, item_id in rows:
        groups[int(user_id)].append(int(item_id))
    return groups


def select_users_and_positives(split_rows, rng):
    """Choose one reproducible test-positive item for every test user."""
    test_groups = group_interactions(split_rows[2])
    user_ids = np.asarray(sorted(test_groups), dtype=np.int64)
    positive_items = np.asarray(
        [
            test_groups[int(user_id)][
                int(rng.randint(len(test_groups[int(user_id)])))
            ]
            for user_id in user_ids
        ],
        dtype=np.int64,
    )
    positive_counts = np.asarray(
        [len(test_groups[int(user_id)]) for user_id in user_ids],
        dtype=np.int64,
    )
    return user_ids, positive_items, positive_counts


def build_known_item_sets(split_rows, num_users):
    """Exclude train, validation, and test positives from negative sampling."""
    known = [set() for _ in range(num_users)]
    train_degree = np.zeros(num_users, dtype=np.int64)
    for split_id, rows in split_rows.items():
        for user_id, item_id in rows:
            known[int(user_id)].add(int(item_id))
            if split_id == 0:
                train_degree[int(user_id)] += 1
    return tuple(frozenset(items) for items in known), train_degree


def sample_candidate_items(
    user_ids,
    positive_items,
    known_items,
    num_items,
    num_negatives,
    rng,
):
    """Return ``[positive, unique unknown negatives]`` for every user."""
    candidates = np.empty(
        (user_ids.size, num_negatives + 1), dtype=np.int64
    )
    candidates[:, 0] = positive_items

    draw_count = max(num_negatives * 3, num_negatives + 32)
    random_pool = rng.randint(
        0, num_items, size=(user_ids.size, draw_count), dtype=np.int64
    )
    for row, user_id in enumerate(user_ids):
        excluded = known_items[int(user_id)]
        available_count = num_items - len(excluded)
        if available_count < num_negatives:
            raise ValueError(
                "User {} has only {} unknown items; {} negatives requested."
                .format(int(user_id), available_count, num_negatives)
            )

        selected = []
        selected_set = set()
        for item_id in random_pool[row]:
            item_id = int(item_id)
            if item_id in excluded or item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
            if len(selected) == num_negatives:
                break

        if len(selected) < num_negatives:
            start = int(random_pool[row, 0])
            for offset in range(num_items):
                item_id = (start + offset) % num_items
                if item_id in excluded or item_id in selected_set:
                    continue
                selected.append(item_id)
                selected_set.add(item_id)
                if len(selected) == num_negatives:
                    break
        candidates[row, 1:] = selected
    return candidates


def sample_derangement(size, rng):
    """Return a random permutation with no fixed points."""
    if size < 2:
        raise ValueError("At least two users are required for permutation.")
    order = rng.permutation(size)
    permutation = np.empty(size, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    return permutation


@torch.no_grad()
def calculate_checkpoint_listwise_jsd(
    embeddings,
    user_ids_np,
    candidate_items_np,
    num_permutations,
    temperature,
    score_batch_size,
    rng,
):
    """Compute per-user JSD and positive probabilities."""
    device = embeddings["full_user"].device
    user_ids = torch.as_tensor(user_ids_np, dtype=torch.long, device=device)
    candidates = torch.as_tensor(
        candidate_items_np, dtype=torch.long, device=device
    )
    user_count = user_ids.numel()

    original_probabilities = []
    full_score_batches = []
    masked_score_batches = []
    for start in range(0, user_count, score_batch_size):
        end = min(start + score_batch_size, user_count)
        batch_users = user_ids[start:end]
        batch_candidates = candidates[start:end]
        full_scores = torch.sum(
            embeddings["full_user"][batch_users, None, :]
            * embeddings["full_item"][batch_candidates],
            dim=-1,
        )
        masked_scores = torch.sum(
            embeddings["masked_user"][batch_users, None, :]
            * embeddings["masked_item"][batch_candidates],
            dim=-1,
        )
        full_score_batches.append(full_scores.cpu())
        masked_score_batches.append(masked_scores.cpu())
        original_probabilities.append(
            F.softmax((full_scores + masked_scores) / temperature, dim=1).cpu()
        )

    # Scores are small compared with embeddings and keeping them on CPU makes
    # the experiment work on both CPU-only and memory-constrained GPU systems.
    full_scores = torch.cat(full_score_batches, dim=0)
    masked_scores = torch.cat(masked_score_batches, dim=0)
    p = torch.cat(original_probabilities, dim=0)
    log_p = p.clamp_min(torch.finfo(p.dtype).eps).log()

    jsd_sum = torch.zeros(user_count, dtype=p.dtype)
    jsd_square_sum = torch.zeros(user_count, dtype=p.dtype)
    permuted_positive_probability_sum = torch.zeros(user_count, dtype=p.dtype)

    for _ in range(num_permutations):
        permutation_np = sample_derangement(user_count, rng)
        permuted_mask_score_batches = []
        for start in range(0, user_count, score_batch_size):
            end = min(start + score_batch_size, user_count)
            batch_candidates = candidates[start:end]
            permuted_users = torch.as_tensor(
                user_ids_np[permutation_np[start:end]],
                dtype=torch.long,
                device=device,
            )
            permuted_scores = torch.sum(
                embeddings["masked_user"][permuted_users, None, :]
                * embeddings["masked_item"][batch_candidates],
                dim=-1,
            )
            permuted_mask_score_batches.append(permuted_scores.cpu())

        permuted_masked_scores = torch.cat(
            permuted_mask_score_batches, dim=0
        )
        q = F.softmax(
            (full_scores + permuted_masked_scores) / temperature, dim=1
        )
        mixture = 0.5 * (p + q)
        log_mixture = mixture.clamp_min(torch.finfo(p.dtype).eps).log()
        log_q = q.clamp_min(torch.finfo(q.dtype).eps).log()
        jsd = 0.5 * torch.sum(
            p * (log_p - log_mixture)
            + q * (log_q - log_mixture),
            dim=1,
        )
        jsd_sum += jsd
        jsd_square_sum += jsd.square()
        permuted_positive_probability_sum += q[:, 0]

    jsd_mean = jsd_sum / num_permutations
    jsd_variance = (
        jsd_square_sum / num_permutations - jsd_mean.square()
    ).clamp_min(0.0)
    return {
        "jsd_mean": jsd_mean.numpy(),
        "jsd_std_across_permutations": jsd_variance.sqrt().numpy(),
        "original_positive_probability": p[:, 0].numpy(),
        "permuted_positive_probability_mean": (
            permuted_positive_probability_sum / num_permutations
        ).numpy(),
    }


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    levels = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    quantiles = np.quantile(values, levels)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "quantiles": {
            "{:.2f}".format(level): float(value)
            for level, value in zip(levels, quantiles)
        },
    }


def write_results_csv(
    path,
    user_ids,
    positive_items,
    positive_counts,
    train_degree,
    results,
):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "user_id",
                "selected_test_positive_item",
                "test_positive_count",
                "train_degree",
                "listwise_jsd_mean",
                "listwise_jsd_std_across_permutations",
                "original_positive_probability",
                "permuted_positive_probability_mean",
                "positive_probability_change",
            )
        )
        for row, user_id in enumerate(user_ids):
            original_probability = results["original_positive_probability"][row]
            permuted_probability = results[
                "permuted_positive_probability_mean"
            ][row]
            writer.writerow(
                (
                    int(user_id),
                    int(positive_items[row]),
                    int(positive_counts[row]),
                    int(train_degree[int(user_id)]),
                    "{:.12g}".format(results["jsd_mean"][row]),
                    "{:.12g}".format(
                        results["jsd_std_across_permutations"][row]
                    ),
                    "{:.12g}".format(original_probability),
                    "{:.12g}".format(permuted_probability),
                    "{:.12g}".format(
                        permuted_probability - original_probability
                    ),
                )
            )


def save_figure(path, results, train_degrees, bins):
    jsd = results["jsd_mean"]
    original_probability = results["original_positive_probability"]
    permuted_probability = results["permuted_positive_probability_mean"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(jsd, bins=bins, color="#4c78a8", edgecolor="none")
    axes[0].axvline(
        float(np.mean(jsd)),
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Mean = {:.6f}".format(float(np.mean(jsd))),
    )
    axes[0].set_xlabel("Listwise JSD")
    axes[0].set_ylabel("Test users")
    axes[0].set_title("Original vs. permuted Mask-user ranking")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    probability_change = permuted_probability - original_probability
    scatter = axes[1].hexbin(
        np.log1p(train_degrees),
        probability_change,
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("log(1 + training degree)")
    axes[1].set_ylabel("Permuted − original positive probability")
    axes[1].set_title("Permutation effect by user history size")
    figure.colorbar(scatter, ax=axes[1], label="User count")

    figure.suptitle("Checkpoint listwise Mask-user permutation experiment")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    rng = np.random.RandomState(args.seed)
    state = load_state_dict(args.checkpoint)
    split_rows = load_interaction_splits(args.interactions)
    print("Computing item text KNN graph...")
    knn_indices = compute_knn_indices(
        args.text_features, args.knn_k, args.knn_chunk_size, device
    )
    print("Building full and masked checkpoint embeddings...")
    embeddings = build_branch_embeddings(
        state, split_rows[0], knn_indices, device
    )

    user_ids, positive_items, positive_counts = select_users_and_positives(
        split_rows, rng
    )
    known_items, train_degree = build_known_item_sets(
        split_rows, embeddings["num_users"]
    )
    candidates = sample_candidate_items(
        user_ids,
        positive_items,
        known_items,
        embeddings["num_items"],
        args.num_negatives,
        rng,
    )
    print(
        "Calculating listwise JSD for {:,} test users...".format(
            user_ids.size
        )
    )
    results = calculate_checkpoint_listwise_jsd(
        embeddings,
        user_ids,
        candidates,
        args.num_permutations,
        args.temperature,
        args.score_batch_size,
        rng,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "checkpoint_listwise_jsd_per_user.csv"
    summary_path = output_dir / "checkpoint_listwise_jsd_summary.json"
    figure_path = output_dir / "checkpoint_listwise_jsd_distribution.png"
    candidate_path = output_dir / "checkpoint_listwise_jsd_candidates.npz"

    write_results_csv(
        csv_path,
        user_ids,
        positive_items,
        positive_counts,
        train_degree,
        results,
    )
    np.savez_compressed(
        candidate_path,
        user_ids=user_ids,
        candidate_items=candidates,
    )
    selected_degrees = train_degree[user_ids]
    save_figure(
        figure_path, results, selected_degrees, args.histogram_bins
    )

    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "protocol": {
            "evaluation_split": "test",
            "one_random_test_positive_per_user": True,
            "negative_items": (
                "unique catalog items absent from train, validation, and test"
            ),
            "num_negatives": args.num_negatives,
            "candidate_count": args.num_negatives + 1,
            "num_mask_user_permutations": args.num_permutations,
            "permutation_has_no_fixed_points": True,
            "temperature": args.temperature,
            "seed": args.seed,
            "full_scores_fixed_across_permutations": True,
            "candidate_items_and_labels_fixed_across_permutations": True,
        },
        "listwise_jsd": describe(results["jsd_mean"]),
        "original_positive_probability": describe(
            results["original_positive_probability"]
        ),
        "permuted_positive_probability_mean": describe(
            results["permuted_positive_probability_mean"]
        ),
        "permuted_minus_original_positive_probability": describe(
            results["permuted_positive_probability_mean"]
            - results["original_positive_probability"]
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    jsd_summary = summary["listwise_jsd"]
    print(
        "JSD: mean={:.9f}, std={:.9f}, median={:.9f}, max={:.9f}"
        .format(
            jsd_summary["mean"],
            jsd_summary["standard_deviation"],
            jsd_summary["quantiles"]["0.50"],
            jsd_summary["maximum"],
        )
    )
    print("Figure: {}".format(figure_path))
    print("CSV: {}".format(csv_path))
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
