# coding: utf-8
"""Compare Full--Masked PID statistics for two MASKED_GLORIA checkpoints."""

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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
from pid_estimators import MultimodalDataset, critic_ce_alignment
from plot_same_user_embedding_similarity import (
    DEFAULT_INTERACTIONS,
    load_state_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_CHECKPOINT = (
    PROJECT_ROOT
    / "saved"
    / "MASKED_GLORIA-book-seed999-Aug-12-2026-13-46-25.pth"
)
DEFAULT_NEW_CHECKPOINT = (
    PROJECT_ROOT
    / "saved"
    / "MASKED_GLORIA_MIPD-book-seed999-Aug-25-2026-04-52-41.pth"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "checkpoint_pid_results" / "base_vs_mipd_seed999"
)
CHECKPOINT_NAMES = ("original", "new")
PID_COMPONENTS = ("redundancy", "unique_full", "unique_mask", "synergy")
SUMMARY_METRICS = PID_COMPONENTS + (
    "mask_complementary",
    "total_information",
    "mask_complementary_ratio",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Estimate BATCH partial information decomposition for the Full "
            "and Masked branches of two recommendation checkpoints."
        )
    )
    parser.add_argument(
        "--checkpoint-original",
        type=Path,
        default=DEFAULT_ORIGINAL_CHECKPOINT,
    )
    parser.add_argument(
        "--checkpoint-new", type=Path, default=DEFAULT_NEW_CHECKPOINT
    )
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--text-features", type=Path, default=DEFAULT_TEXT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target",
        choices=("both", "ground_truth", "predicted"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--estimator-seeds", type=int, nargs="+", default=(999, 1000, 1001)
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--discrim-epochs", type=int, default=40)
    parser.add_argument("--ce-epochs", type=int, default=10)
    parser.add_argument("--sinkhorn-iterations", type=int, default=500)
    parser.add_argument("--sinkhorn-tolerance", type=float, default=0.01)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--device", default="cuda:0", help="PyTorch device such as cuda:0 or cpu"
    )
    args = parser.parse_args(argv)

    for name in (
        "batch_size",
        "discrim_epochs",
        "ce_epochs",
        "sinkhorn_iterations",
        "knn_k",
        "knn_chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be between 0 and 1")
    if args.sinkhorn_tolerance <= 0.0:
        parser.error("--sinkhorn-tolerance must be positive")
    if len(set(args.estimator_seeds)) != len(args.estimator_seeds):
        parser.error("--estimator-seeds must not contain duplicates")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def group_interactions(rows):
    groups = defaultdict(list)
    for user_id, item_id in rows:
        groups[int(user_id)].append(int(item_id))
    return groups


def build_known_items(split_rows, num_users, num_items):
    known = [set() for _ in range(num_users)]
    for rows in split_rows.values():
        for user_id, item_id in rows:
            if not 0 <= int(user_id) < num_users:
                raise ValueError("Interaction user ID is outside checkpoint range.")
            if not 0 <= int(item_id) < num_items:
                raise ValueError("Interaction item ID is outside checkpoint range.")
            known[int(user_id)].add(int(item_id))
    return tuple(frozenset(items) for items in known)


def sample_unseen_item(user_id, known_items, num_items, rng):
    excluded = known_items[int(user_id)]
    if len(excluded) >= num_items:
        raise ValueError("User {} has no unseen item.".format(int(user_id)))
    for _ in range(128):
        candidate = int(rng.randint(num_items))
        if candidate not in excluded:
            return candidate
    start = int(rng.randint(num_items))
    for offset in range(num_items):
        candidate = (start + offset) % num_items
        if candidate not in excluded:
            return candidate
    raise RuntimeError("Unable to sample an unseen item.")


def build_pid_manifest(
    split_rows,
    num_users,
    num_items,
    seed=999,
    train_ratio=0.8,
):
    """Create one deterministic ranking sample for every test user."""
    rng = np.random.RandomState(seed)
    test_groups = group_interactions(split_rows[2])
    user_ids = np.asarray(sorted(test_groups), dtype=np.int64)
    if user_ids.size < 2:
        raise ValueError("At least two test users are required.")
    known_items = build_known_items(split_rows, num_users, num_items)

    positive_items = np.asarray(
        [
            test_groups[int(user_id)][
                int(rng.randint(len(test_groups[int(user_id)])))
            ]
            for user_id in user_ids
        ],
        dtype=np.int64,
    )
    negative_items = np.asarray(
        [
            sample_unseen_item(user_id, known_items, num_items, rng)
            for user_id in user_ids
        ],
        dtype=np.int64,
    )
    flipped = rng.rand(user_ids.size) < 0.5
    item_a = np.where(flipped, negative_items, positive_items).astype(np.int64)
    item_b = np.where(flipped, positive_items, negative_items).astype(np.int64)
    ground_truth_labels = (~flipped).astype(np.int64)

    permutation = rng.permutation(user_ids.size)
    split_at = int(train_ratio * user_ids.size)
    if split_at <= 0 or split_at >= user_ids.size:
        raise ValueError("PID split produced an empty partition.")
    train_indices = permutation[:split_at].astype(np.int64)
    test_indices = permutation[split_at:].astype(np.int64)
    partition = np.full(user_ids.size, "test", dtype="<U5")
    partition[train_indices] = "train"

    manifest = {
        "user_ids": user_ids,
        "positive_items": positive_items,
        "negative_items": negative_items,
        "flipped": flipped,
        "item_a": item_a,
        "item_b": item_b,
        "ground_truth_labels": ground_truth_labels,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "partition": partition,
    }
    validate_manifest(manifest, known_items, num_items)
    return manifest


def validate_manifest(manifest, known_items, num_items):
    sample_count = int(manifest["user_ids"].size)
    per_sample_fields = (
        "positive_items",
        "negative_items",
        "flipped",
        "item_a",
        "item_b",
        "ground_truth_labels",
        "partition",
    )
    if any(int(manifest[name].size) != sample_count for name in per_sample_fields):
        raise ValueError("Manifest fields have inconsistent sample counts.")
    if np.unique(manifest["user_ids"]).size != sample_count:
        raise ValueError("PID manifest must contain exactly one sample per user.")
    for user_id, positive, negative in zip(
        manifest["user_ids"],
        manifest["positive_items"],
        manifest["negative_items"],
    ):
        if int(positive) == int(negative):
            raise ValueError("A PID negative equals its positive item.")
        if int(negative) in known_items[int(user_id)]:
            raise ValueError("PID negative occurs in the user's known history.")
        if not 0 <= int(negative) < num_items:
            raise ValueError("PID negative is outside the item catalog.")
    all_indices = np.concatenate(
        (manifest["train_indices"], manifest["test_indices"])
    )
    if not np.array_equal(np.sort(all_indices), np.arange(sample_count)):
        raise ValueError("PID train/test indices are not a complete partition.")
    if set(np.unique(manifest["ground_truth_labels"]).tolist()) != {0, 1}:
        raise ValueError("Ground-truth PID labels must contain both classes.")


def state_dimensions(state):
    required = (
        "full_gcn.preference",
        "mask_gcn.preference",
        "id_embedding_full.weight",
        "id_embedding_masked.weight",
        "mask_logits",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(
            "Checkpoint is missing required tensors: {}".format(
                ", ".join(missing)
            )
        )
    full_user = state["full_gcn.preference"]
    masked_user = state["mask_gcn.preference"]
    full_item = state["id_embedding_full.weight"]
    masked_item = state["id_embedding_masked.weight"]
    if full_user.shape != masked_user.shape:
        raise RuntimeError("Full and Masked user parameter shapes differ.")
    if full_item.shape != masked_item.shape:
        raise RuntimeError("Full and Masked item parameter shapes differ.")
    if full_user.shape[1] != full_item.shape[1]:
        raise RuntimeError("User and item embedding dimensions differ.")
    return {
        "num_users": int(full_user.shape[0]),
        "num_items": int(full_item.shape[0]),
        "embedding_dim": int(full_user.shape[1]),
        "num_mask_logits": int(state["mask_logits"].numel()),
    }


def validate_checkpoint_pair(original_state, new_state, train_count):
    original_dimensions = state_dimensions(original_state)
    new_dimensions = state_dimensions(new_state)
    if original_dimensions != new_dimensions:
        raise RuntimeError(
            "Checkpoint architecture mismatch: original={} new={}".format(
                original_dimensions, new_dimensions
            )
        )
    if original_dimensions["num_mask_logits"] != int(train_count):
        raise RuntimeError(
            "Checkpoint mask_logits count does not match training edge order."
        )
    return original_dimensions


@torch.no_grad()
def extract_pid_features(embeddings, manifest):
    device = embeddings["full_user"].device
    users = torch.as_tensor(manifest["user_ids"], dtype=torch.long, device=device)
    item_a = torch.as_tensor(manifest["item_a"], dtype=torch.long, device=device)
    item_b = torch.as_tensor(manifest["item_b"], dtype=torch.long, device=device)

    x_full = embeddings["full_user"][users] * (
        embeddings["full_item"][item_a] - embeddings["full_item"][item_b]
    )
    x_mask = embeddings["masked_user"][users] * (
        embeddings["masked_item"][item_a]
        - embeddings["masked_item"][item_b]
    )
    full_margin = x_full.sum(dim=-1)
    mask_margin = x_mask.sum(dim=-1)
    joint_margin = full_margin + mask_margin
    predicted_labels = (joint_margin > 0.0).long()

    tensors = {
        "x_full": x_full.float().cpu(),
        "x_mask": x_mask.float().cpu(),
        "full_margin": full_margin.float().cpu(),
        "mask_margin": mask_margin.float().cpu(),
        "joint_margin": joint_margin.float().cpu(),
        "predicted_labels": predicted_labels.cpu(),
    }
    if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise FloatingPointError("Extracted PID features contain non-finite values.")
    return tensors


def verify_feature_margins(embeddings, manifest, features, tolerance=1e-5):
    device = embeddings["full_user"].device
    users = torch.as_tensor(manifest["user_ids"], dtype=torch.long, device=device)
    item_a = torch.as_tensor(manifest["item_a"], dtype=torch.long, device=device)
    item_b = torch.as_tensor(manifest["item_b"], dtype=torch.long, device=device)
    with torch.no_grad():
        expected_full = (
            embeddings["full_user"][users]
            * (
                embeddings["full_item"][item_a]
                - embeddings["full_item"][item_b]
            )
        ).sum(dim=-1).cpu()
        expected_mask = (
            embeddings["masked_user"][users]
            * (
                embeddings["masked_item"][item_a]
                - embeddings["masked_item"][item_b]
            )
        ).sum(dim=-1).cpu()
    if not torch.allclose(
        features["full_margin"], expected_full, atol=tolerance, rtol=tolerance
    ):
        raise RuntimeError("Full contribution sum does not match score margin.")
    if not torch.allclose(
        features["mask_margin"], expected_mask, atol=tolerance, rtol=tolerance
    ):
        raise RuntimeError("Masked contribution sum does not match score margin.")
    if not torch.allclose(
        features["joint_margin"],
        expected_full + expected_mask,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise RuntimeError("Joint contribution does not match fused score margin.")


def label_counts(labels):
    counts = torch.bincount(labels.long().view(-1), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def feature_metadata(features):
    return {
        "sample_count": int(features["x_full"].shape[0]),
        "embedding_dim": int(features["x_full"].shape[1]),
        "predicted_label_counts": label_counts(features["predicted_labels"]),
        "full_margin_mean": float(features["full_margin"].mean()),
        "mask_margin_mean": float(features["mask_margin"].mean()),
        "joint_margin_mean": float(features["joint_margin"].mean()),
        "joint_pairwise_accuracy": float(
            (
                features["predicted_labels"]
                == torch.as_tensor(features["ground_truth_labels"])
            )
            .float()
            .mean()
        )
        if "ground_truth_labels" in features
        else None,
    }


def save_manifest(path, manifest):
    train_position = {
        int(index): position
        for position, index in enumerate(manifest["train_indices"].tolist())
    }
    test_position = {
        int(index): position
        for position, index in enumerate(manifest["test_indices"].tolist())
    }
    fields = (
        "sample_id",
        "user_id",
        "positive_item",
        "negative_item",
        "item_a",
        "item_b",
        "flipped",
        "ground_truth_label",
        "pid_split",
        "split_position",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(manifest["user_ids"].size):
            partition = str(manifest["partition"][index])
            positions = train_position if partition == "train" else test_position
            writer.writerow(
                {
                    "sample_id": index,
                    "user_id": int(manifest["user_ids"][index]),
                    "positive_item": int(manifest["positive_items"][index]),
                    "negative_item": int(manifest["negative_items"][index]),
                    "item_a": int(manifest["item_a"][index]),
                    "item_b": int(manifest["item_b"][index]),
                    "flipped": int(manifest["flipped"][index]),
                    "ground_truth_label": int(
                        manifest["ground_truth_labels"][index]
                    ),
                    "pid_split": partition,
                    "split_position": positions[index],
                }
            )


def add_derived_pid_values(row):
    total = sum(float(row[name]) for name in PID_COMPONENTS)
    complementary = float(row["unique_mask"]) + float(row["synergy"])
    row["mask_complementary"] = complementary
    row["total_information"] = total
    row["mask_complementary_ratio"] = complementary / max(total, 1e-12)
    row["has_negative_atom"] = any(float(row[name]) < 0.0 for name in PID_COMPONENTS)
    return row


def run_pid_estimator(
    features,
    labels,
    manifest,
    estimator_seed,
    args,
):
    labels = labels.long().view(-1, 1)
    if set(torch.unique(labels).tolist()) != {0, 1}:
        raise ValueError("PID target must contain both binary labels.")
    train_indices = torch.as_tensor(manifest["train_indices"], dtype=torch.long)
    test_indices = torch.as_tensor(manifest["test_indices"], dtype=torch.long)
    train_dataset = MultimodalDataset(
        (
            features["x_full"][train_indices],
            features["x_mask"][train_indices],
        ),
        labels[train_indices],
    )
    test_dataset = MultimodalDataset(
        (
            features["x_full"][test_indices],
            features["x_mask"][test_indices],
        ),
        labels[test_indices],
    )
    results, alignments, models = critic_ce_alignment(
        x1=features["x_full"],
        x2=features["x_mask"],
        labels=labels,
        num_labels=2,
        train_ds=train_dataset,
        test_ds=test_dataset,
        discrim_epochs=args.discrim_epochs,
        ce_epochs=args.ce_epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=estimator_seed,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_tolerance=args.sinkhorn_tolerance,
    )
    pid = results.mean(dim=0)
    values = {name: float(pid[index]) for index, name in enumerate(PID_COMPONENTS)}
    values["eval_batch_count"] = int(results.shape[0])
    values["evaluated_sample_count"] = int(results.shape[0] * args.batch_size)
    del alignments, models, results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return add_derived_pid_values(values)


def write_runs_csv(path, rows):
    fields = (
        "target",
        "checkpoint",
        "estimator_seed",
        *PID_COMPONENTS,
        "mask_complementary",
        "total_information",
        "mask_complementary_ratio",
        "has_negative_atom",
        "eval_batch_count",
        "evaluated_sample_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_values(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def summarize_runs(rows, estimator_seeds, targets):
    summary = {}
    for target in targets:
        summary[target] = {}
        grouped = {
            checkpoint: [
                row
                for row in rows
                if row["target"] == target and row["checkpoint"] == checkpoint
            ]
            for checkpoint in CHECKPOINT_NAMES
        }
        for checkpoint, checkpoint_rows in grouped.items():
            by_seed = {int(row["estimator_seed"]): row for row in checkpoint_rows}
            if set(by_seed) != set(estimator_seeds):
                raise RuntimeError("PID run seeds are incomplete.")
            summary[target][checkpoint] = {
                metric: summarize_values(
                    [by_seed[seed][metric] for seed in estimator_seeds]
                )
                for metric in SUMMARY_METRICS
            }

        original_by_seed = {
            int(row["estimator_seed"]): row for row in grouped["original"]
        }
        new_by_seed = {int(row["estimator_seed"]): row for row in grouped["new"]}
        summary[target]["delta_new_minus_original"] = {
            metric: summarize_values(
                [
                    new_by_seed[seed][metric] - original_by_seed[seed][metric]
                    for seed in estimator_seeds
                ]
            )
            for metric in SUMMARY_METRICS
        }
    return summary


def write_summary_csv(path, summary):
    fields = ("target", "entity", "metric", "mean", "std", "values")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target, target_values in summary.items():
            for entity, entity_values in target_values.items():
                for metric, statistics in entity_values.items():
                    writer.writerow(
                        {
                            "target": target,
                            "entity": entity,
                            "metric": metric,
                            "mean": statistics["mean"],
                            "std": statistics["std"],
                            "values": json.dumps(statistics["values"]),
                        }
                    )


def save_pid_plot(path, target, target_summary):
    x = np.arange(len(PID_COMPONENTS))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for offset, checkpoint, label, color in (
        (-width / 2, "original", "MASKED_GLORIA", "#4c78a8"),
        (width / 2, "new", "MASKED_GLORIA_MIPD", "#f58518"),
    ):
        means = [target_summary[checkpoint][name]["mean"] for name in PID_COMPONENTS]
        errors = [target_summary[checkpoint][name]["std"] for name in PID_COMPONENTS]
        axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=label,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, ("R", "U Full", "U Mask", "S"))
    axis.set_ylabel("PID information (nats)")
    axis.set_title("Full--Masked PID ({})".format(target.replace("_", " ")))
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_complementarity_plot(path, targets, summary):
    x = np.arange(len(targets))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for offset, checkpoint, label, color in (
        (-width / 2, "original", "MASKED_GLORIA", "#4c78a8"),
        (width / 2, "new", "MASKED_GLORIA_MIPD", "#f58518"),
    ):
        means = [summary[target][checkpoint]["mask_complementary"]["mean"] for target in targets]
        errors = [summary[target][checkpoint]["mask_complementary"]["std"] for target in targets]
        axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=label,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [target.replace("_", " ") for target in targets])
    axis.set_ylabel("U Mask + S (nats)")
    axis.set_title("Masked complementary information")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def print_summary(summary):
    for target, target_values in summary.items():
        print("\nPID target: {}".format(target))
        for entity in ("original", "new", "delta_new_minus_original"):
            metrics = target_values[entity]
            print(
                "  {}: R={:.6f}+/-{:.6f}, U_full={:.6f}+/-{:.6f}, "
                "U_mask={:.6f}+/-{:.6f}, S={:.6f}+/-{:.6f}, "
                "U_mask+S={:.6f}+/-{:.6f}".format(
                    entity,
                    metrics["redundancy"]["mean"],
                    metrics["redundancy"]["std"],
                    metrics["unique_full"]["mean"],
                    metrics["unique_full"]["std"],
                    metrics["unique_mask"]["mean"],
                    metrics["unique_mask"]["std"],
                    metrics["synergy"]["mean"],
                    metrics["synergy"]["std"],
                    metrics["mask_complementary"]["mean"],
                    metrics["mask_complementary"]["std"],
                )
            )


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Python uses a CPU-only PyTorch build. "
            "Run the experiment from a CUDA-enabled PyTorch environment."
        )
    seed_everything(args.seed)

    checkpoint_paths = {
        "original": args.checkpoint_original.expanduser().resolve(),
        "new": args.checkpoint_new.expanduser().resolve(),
    }
    fingerprints = {
        name: checkpoint_fingerprint(path)
        for name, path in checkpoint_paths.items()
    }
    states = {name: load_state_dict(path) for name, path in checkpoint_paths.items()}
    split_rows = load_interaction_splits(args.interactions)
    dimensions = validate_checkpoint_pair(
        states["original"], states["new"], len(split_rows[0])
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_pid_manifest(
        split_rows,
        dimensions["num_users"],
        dimensions["num_items"],
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    if len(manifest["train_indices"]) < args.batch_size:
        raise ValueError("PID train split is smaller than one estimator batch.")
    if len(manifest["test_indices"]) < args.batch_size:
        raise ValueError("PID test split is smaller than one estimator batch.")
    save_manifest(output_dir / "pid_sample_manifest.csv", manifest)

    print("Computing the shared item text KNN graph...")
    knn_indices = compute_knn_indices(
        args.text_features,
        args.knn_k,
        args.knn_chunk_size,
        device,
    )
    features_by_checkpoint = {}
    feature_summaries = {}
    ground_truth_tensor = torch.as_tensor(
        manifest["ground_truth_labels"], dtype=torch.long
    )
    for checkpoint_name in CHECKPOINT_NAMES:
        print("Building {} branch embeddings...".format(checkpoint_name))
        embeddings = build_branch_embeddings(
            states[checkpoint_name], split_rows[0], knn_indices, device
        )
        features = extract_pid_features(embeddings, manifest)
        verify_feature_margins(embeddings, manifest, features)
        features["ground_truth_labels"] = ground_truth_tensor
        features_by_checkpoint[checkpoint_name] = features
        feature_summaries[checkpoint_name] = feature_metadata(features)
        torch.save(
            {
                "x_full": features["x_full"],
                "x_mask": features["x_mask"],
                "ground_truth_labels": ground_truth_tensor,
                "predicted_labels": features["predicted_labels"],
                "full_margin": features["full_margin"],
                "mask_margin": features["mask_margin"],
                "joint_margin": features["joint_margin"],
            },
            output_dir / "pid_features_{}.pt".format(checkpoint_name),
        )
        del embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    targets = (
        ("ground_truth", "predicted")
        if args.target == "both"
        else (args.target,)
    )
    run_rows = []
    total_runs = len(targets) * len(args.estimator_seeds) * 2
    completed_runs = 0
    for target in targets:
        for estimator_seed in args.estimator_seeds:
            for checkpoint_name in CHECKPOINT_NAMES:
                completed_runs += 1
                print(
                    "\nPID run {}/{}: target={}, checkpoint={}, seed={}".format(
                        completed_runs,
                        total_runs,
                        target,
                        checkpoint_name,
                        estimator_seed,
                    )
                )
                labels = (
                    ground_truth_tensor
                    if target == "ground_truth"
                    else features_by_checkpoint[checkpoint_name]["predicted_labels"]
                )
                values = run_pid_estimator(
                    features_by_checkpoint[checkpoint_name],
                    labels,
                    manifest,
                    estimator_seed,
                    args,
                )
                run_rows.append(
                    {
                        "target": target,
                        "checkpoint": checkpoint_name,
                        "estimator_seed": int(estimator_seed),
                        **values,
                    }
                )
                write_runs_csv(output_dir / "pid_runs.csv", run_rows)

    summary = summarize_runs(run_rows, args.estimator_seeds, targets)
    write_summary_csv(output_dir / "pid_summary.csv", summary)
    for target in targets:
        save_pid_plot(
            output_dir / "pid_{}.png".format(target),
            target,
            summary[target],
        )
    save_complementarity_plot(
        output_dir / "mask_complementarity.png", targets, summary
    )

    metadata = {
        "method": "BATCH partial information decomposition",
        "pid_order": ["R", "U_full", "U_mask", "S"],
        "units": "nats",
        "checkpoint_fingerprints": fingerprints,
        "dimensions": dimensions,
        "interaction_counts": {
            str(split_id): len(rows) for split_id, rows in split_rows.items()
        },
        "sample_protocol": {
            "seed": args.seed,
            "one_test_positive_per_user": True,
            "negative_sampling": "one random item unseen in train/validation/test",
            "sample_count": int(manifest["user_ids"].size),
            "train_count": int(manifest["train_indices"].size),
            "test_count": int(manifest["test_indices"].size),
            "ground_truth_label_counts": label_counts(ground_truth_tensor),
        },
        "estimator": {
            "targets": list(targets),
            "seeds": list(args.estimator_seeds),
            "device": str(device),
            "batch_size": args.batch_size,
            "discriminator_epochs": args.discrim_epochs,
            "alignment_epochs": args.ce_epochs,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_tolerance": args.sinkhorn_tolerance,
            "learning_rate": 1e-3,
            "hidden_dim": 32,
            "alignment_embed_dim": 10,
            "num_hidden_layers": 3,
        },
        "feature_summaries": feature_summaries,
        "negative_pid_atom_warning": any(
            bool(row["has_negative_atom"]) for row in run_rows
        ),
        "summary": summary,
    }
    write_json(output_dir / "pid_summary.json", metadata)
    print_summary(summary)
    if metadata["negative_pid_atom_warning"]:
        print(
            "Warning: at least one raw PID atom is negative due to estimator "
            "noise; raw values were retained."
        )
    print("\nOutput directory: {}".format(output_dir))


if __name__ == "__main__":
    main()
