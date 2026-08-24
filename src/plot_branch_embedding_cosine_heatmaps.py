# coding: utf-8
"""Create cosine-similarity heatmaps for both MASKED_GLORIA branches."""

import argparse
import csv
import json
import os
import tempfile
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "branch_embedding_heatmaps"
BRANCHES = ("full", "masked")
ENTITY_TYPES = ("user", "item")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot user-user, item-item, and user-item cosine similarities for "
            "the full and masked MASKED_GLORIA branches."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--text-features", type=Path, default=DEFAULT_TEXT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--user-sample-size", type=int, default=250)
    parser.add_argument("--item-sample-size", type=int, default=250)
    parser.add_argument("--sample-seed", type=int, default=999)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument(
        "--knn-chunk-size",
        type=int,
        default=10000,
        help="rows per text-similarity batch; 10000 reproduces one-shot book KNN",
    )
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device such as cpu or cuda:0"
    )
    args = parser.parse_args(argv)

    for name in (
        "user_sample_size",
        "item_sample_size",
        "knn_k",
        "knn_chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    return args


def sample_ids(count, sample_size, rng):
    if sample_size > count:
        raise ValueError(
            "Requested sample size {} exceeds entity count {}.".format(
                sample_size, count
            )
        )
    return np.sort(rng.choice(count, size=sample_size, replace=False))


@torch.no_grad()
def compute_similarity_matrices(embeddings, user_ids, item_ids):
    user_index = torch.as_tensor(
        user_ids, dtype=torch.long, device=embeddings["full_user"].device
    )
    item_index = torch.as_tensor(
        item_ids, dtype=torch.long, device=embeddings["full_item"].device
    )
    normalized = {}
    matrices = {}
    for branch in BRANCHES:
        users = F.normalize(embeddings[branch + "_user"][user_index], dim=1)
        items = F.normalize(embeddings[branch + "_item"][item_index], dim=1)
        normalized[branch] = (users, items)
        matrices[branch + "_user_user"] = torch.matmul(users, users.t()).cpu().numpy()
        matrices[branch + "_item_item"] = torch.matmul(items, items.t()).cpu().numpy()
        matrices[branch + "_user_item"] = torch.matmul(users, items.t()).cpu().numpy()

    matched = {
        "user": F.cosine_similarity(
            embeddings["full_user"], embeddings["masked_user"], dim=1
        ).cpu().numpy(),
        "item": F.cosine_similarity(
            embeddings["full_item"], embeddings["masked_item"], dim=1
        ).cpu().numpy(),
    }
    return matrices, matched


def off_diagonal_values(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return matrix.reshape(-1)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "quantile_05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "quantile_95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def shared_color_limit(matrices):
    values = np.concatenate(
        [off_diagonal_values(matrix) for matrix in matrices.values()]
    )
    limit = float(np.quantile(np.abs(values), 0.995))
    return min(1.0, max(limit, 0.05))


def axis_ticks(ids, count=6):
    positions = np.linspace(0, len(ids) - 1, min(count, len(ids))).astype(int)
    return positions, [str(int(ids[position])) for position in positions]


def draw_heatmap(axis, matrix, title, row_ids, column_ids, color_limit):
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="auto",
    )
    x_positions, x_labels = axis_ticks(column_ids)
    y_positions, y_labels = axis_ticks(row_ids)
    axis.set_xticks(x_positions, x_labels, rotation=45, ha="right")
    axis.set_yticks(y_positions, y_labels)
    axis.set_title(title)
    return image


def save_individual_heatmap(
    path, matrix, title, row_ids, column_ids, row_label, column_label, color_limit
):
    figure, axis = plt.subplots(figsize=(7, 6))
    image = draw_heatmap(
        axis, matrix, title, row_ids, column_ids, color_limit
    )
    axis.set_xlabel(column_label)
    axis.set_ylabel(row_label)
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_four_panel(path, matrices, user_ids, item_ids, color_limit):
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    image = None
    for row, branch in enumerate(BRANCHES):
        image = draw_heatmap(
            axes[row, 0],
            matrices[branch + "_user_user"],
            "{} branch: user-user".format(branch.title()),
            user_ids,
            user_ids,
            color_limit,
        )
        axes[row, 0].set_xlabel("User ID")
        axes[row, 0].set_ylabel("User ID")
        draw_heatmap(
            axes[row, 1],
            matrices[branch + "_item_item"],
            "{} branch: item-item".format(branch.title()),
            item_ids,
            item_ids,
            color_limit,
        )
        axes[row, 1].set_xlabel("Item ID")
        axes[row, 1].set_ylabel("Item ID")
    figure.suptitle("Within-branch embedding cosine similarities")
    figure.subplots_adjust(top=0.92, right=0.86, wspace=0.28, hspace=0.30)
    color_axis = figure.add_axes((0.89, 0.15, 0.018, 0.70))
    figure.colorbar(image, cax=color_axis, label="Cosine similarity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_cross_panel(path, matrices, user_ids, item_ids, color_limit):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    image = None
    for axis, branch in zip(axes, BRANCHES):
        image = draw_heatmap(
            axis,
            matrices[branch + "_user_item"],
            "{} branch: user-item".format(branch.title()),
            user_ids,
            item_ids,
            color_limit,
        )
        axis.set_xlabel("Item ID")
        axis.set_ylabel("User ID")
    figure.suptitle("User-item cosine similarities by branch")
    figure.subplots_adjust(top=0.88, right=0.87, wspace=0.28)
    color_axis = figure.add_axes((0.90, 0.15, 0.016, 0.70))
    figure.colorbar(image, cax=color_axis, label="Cosine similarity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_matched_csv(path, matched):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("entity_type", "entity_id", "full_vs_masked_cosine"))
        for entity_type in ENTITY_TYPES:
            for entity_id, similarity in enumerate(matched[entity_type]):
                writer.writerow(
                    (entity_type, entity_id, "{:.9g}".format(float(similarity)))
                )


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    state = load_state_dict(args.checkpoint)
    splits = load_interaction_splits(args.interactions)
    print("Computing the item text KNN graph...")
    knn_indices = compute_knn_indices(
        args.text_features, args.knn_k, args.knn_chunk_size, device
    )
    print("Building full and masked branch embeddings...")
    embeddings = build_branch_embeddings(state, splits[0], knn_indices, device)

    rng = np.random.RandomState(args.sample_seed)
    user_ids = sample_ids(
        embeddings["num_users"], args.user_sample_size, rng
    )
    item_ids = sample_ids(
        embeddings["num_items"], args.item_sample_size, rng
    )
    matrices, matched = compute_similarity_matrices(
        embeddings, user_ids, item_ids
    )
    color_limit = shared_color_limit(matrices)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    four_panel_path = output_dir / "branch_user_item_self_similarity_heatmaps.png"
    cross_panel_path = output_dir / "branch_user_item_cross_similarity_heatmaps.png"
    matrix_path = output_dir / "sampled_cosine_similarity_matrices.npz"
    matched_path = output_dir / "full_vs_masked_matched_node_cosine.csv"
    summary_path = output_dir / "branch_embedding_cosine_summary.json"

    save_four_panel(
        four_panel_path, matrices, user_ids, item_ids, color_limit
    )
    save_cross_panel(
        cross_panel_path, matrices, user_ids, item_ids, color_limit
    )
    for branch in BRANCHES:
        for entity_type, ids in (("user", user_ids), ("item", item_ids)):
            key = "{}_{}_{}".format(branch, entity_type, entity_type)
            save_individual_heatmap(
                output_dir / "{}_{}_cosine_heatmap.png".format(branch, entity_type),
                matrices[key],
                "{} branch: {}-{} cosine similarity".format(
                    branch.title(), entity_type, entity_type
                ),
                ids,
                ids,
                "{} ID".format(entity_type.title()),
                "{} ID".format(entity_type.title()),
                color_limit,
            )

    np.savez_compressed(
        matrix_path,
        user_ids=user_ids,
        item_ids=item_ids,
        **matrices,
    )
    write_matched_csv(matched_path, matched)

    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "embedding_definition": {
            "user": "three-layer branch GCN output",
            "item": "three-layer branch GCN output plus text-KNN item propagation",
        },
        "sampling": {
            "seed": args.sample_seed,
            "user_sample_size": int(user_ids.size),
            "item_sample_size": int(item_ids.size),
            "user_ids": user_ids.tolist(),
            "item_ids": item_ids.tolist(),
        },
        "shared_heatmap_color_limit": color_limit,
        "sampled_similarity_statistics": {
            key: describe(off_diagonal_values(matrix))
            for key, matrix in matrices.items()
        },
        "all_node_full_vs_masked_matched_cosine": {
            entity_type: describe(values)
            for entity_type, values in matched.items()
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(
        "Computed heatmaps for {} users and {} items.".format(
            user_ids.size, item_ids.size
        )
    )
    for branch in BRANCHES:
        user_stats = summary["sampled_similarity_statistics"][
            branch + "_user_user"
        ]
        item_stats = summary["sampled_similarity_statistics"][
            branch + "_item_item"
        ]
        cross_stats = summary["sampled_similarity_statistics"][
            branch + "_user_item"
        ]
        print(
            "{} mean cosine: user-user={:.6f}, item-item={:.6f}, "
            "user-item={:.6f}".format(
                branch,
                user_stats["mean"],
                item_stats["mean"],
                cross_stats["mean"],
            )
        )
    print("Output directory: {}".format(output_dir))


if __name__ == "__main__":
    main()
