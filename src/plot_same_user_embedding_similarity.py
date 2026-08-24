# coding: utf-8
"""Plot same-user embedding similarity under single-edge interventions.

For every training interaction ``e = (user, item)``, this script compares:

1. the user's baseline final embedding; and
2. the same user's final embedding after setting only edge ``e`` to zero in
   the learned masked branch.

The final MASKED_GLORIA user embedding is the concatenation of the unchanged
full-graph branch and the learned masked-graph branch.  The counterfactual
masked embedding is evaluated exactly.  A closed-form rank-two update avoids
rerunning the entire three-layer GCN once per edge.
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camure_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "saved"
    / "MASKED_GLORIA-book-seed999-Aug-12-2026-13-46-25.pth"
)
DEFAULT_INTERACTIONS = PROJECT_ROOT / "data" / "book" / "book.inter"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "embedding_similarity_results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarity between each user's baseline embedding "
            "and its single-edge counterfactual embedding."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MASKED_GLORIA checkpoint path",
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=DEFAULT_INTERACTIONS,
        help="tab-separated interaction file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for PNG, CSV, and JSON results",
    )
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--x-min", type=float, default=0.95)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu, cuda, or cuda:0",
    )
    args = parser.parse_args(argv)

    if args.bins <= 0:
        parser.error("--bins must be positive")
    if args.x_min >= args.x_max:
        parser.error("--x-min must be smaller than --x-max")
    return args


def load_state_dict(checkpoint_path):
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint_path))

    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must contain a dictionary.")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint state_dict must be a dictionary.")

    # The supplied historical checkpoint uses the short alias names.  Newer
    # checkpoints may use the canonical nested-module names.
    aliases = {
        "full_gcn.preference": "full_preference",
        "mask_gcn.preference": "mask_preference",
    }
    resolved = dict(state)
    for canonical, legacy in aliases.items():
        if canonical not in resolved and legacy in resolved:
            resolved[canonical] = resolved[legacy]

    required = (
        "mask_logits",
        "full_gcn.preference",
        "mask_gcn.preference",
        "id_embedding_full.weight",
        "id_embedding_masked.weight",
    )
    missing = [name for name in required if name not in resolved]
    if missing:
        raise RuntimeError(
            "Checkpoint is missing required tensors: {}".format(", ".join(missing))
        )
    return resolved


def load_training_edges(interaction_path):
    interaction_path = interaction_path.expanduser().resolve()
    if not interaction_path.is_file():
        raise FileNotFoundError(
            "Interaction file does not exist: {}".format(interaction_path)
        )

    users = []
    items = []
    with interaction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"userID", "itemID", "x_label"}
        actual = set(reader.fieldnames or ())
        missing = required - actual
        if missing:
            raise RuntimeError(
                "Interaction file is missing columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        for row in reader:
            if int(row["x_label"]) == 0:
                users.append(int(row["userID"]))
                items.append(int(row["itemID"]))

    if not users:
        raise RuntimeError("No training interactions (x_label=0) were found.")
    return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)


def propagate_three_layers(preference, item_embedding, src, dst, edge_scale):
    """Reproduce MASKED_GLORIA.GCN.forward exactly."""
    node_embedding = F.normalize(
        torch.cat((preference, item_embedding), dim=0), dim=1
    )

    def propagate(features):
        output = torch.zeros_like(features)
        messages = features[src] * edge_scale.unsqueeze(1)
        output.index_add_(0, dst, messages)
        return output

    layer_1 = propagate(node_embedding)
    layer_2 = propagate(layer_1)
    layer_3 = propagate(layer_2)
    result = node_embedding + layer_1 + layer_2 + layer_3
    return node_embedding, layer_1, layer_2, result


@torch.no_grad()
def compute_similarities(state, edge_users_np, edge_items_np, device):
    full_preference = state["full_gcn.preference"].to(device)
    mask_preference = state["mask_gcn.preference"].to(device)
    full_items = state["id_embedding_full.weight"].to(device)
    mask_items = state["id_embedding_masked.weight"].to(device)
    mask_logits = state["mask_logits"].to(device)

    num_users = full_preference.shape[0]
    num_items = full_items.shape[0]
    num_edges = edge_users_np.size
    if mask_logits.numel() != num_edges:
        raise RuntimeError(
            "Checkpoint has {} mask logits but the training split has {} edges."
            .format(mask_logits.numel(), num_edges)
        )
    if edge_users_np.min() < 0 or edge_users_np.max() >= num_users:
        raise RuntimeError("Training user IDs do not match the checkpoint.")
    if edge_items_np.min() < 0 or edge_items_np.max() >= num_items:
        raise RuntimeError("Training item IDs do not match the checkpoint.")
    edge_pairs = np.column_stack((edge_users_np, edge_items_np))
    if np.unique(edge_pairs, axis=0).shape[0] != num_edges:
        raise RuntimeError(
            "Duplicate training user-item pairs are not supported by the "
            "exact single-edge update."
        )

    edge_users = torch.as_tensor(edge_users_np, dtype=torch.long, device=device)
    edge_items = torch.as_tensor(edge_items_np, dtype=torch.long, device=device)
    item_nodes = edge_items + num_users

    # The model creates a forward user->item edge followed by its reverse.
    src = torch.cat((edge_users, item_nodes), dim=0)
    dst = torch.cat((item_nodes, edge_users), dim=0)
    node_count = num_users + num_items
    degree = torch.bincount(src, minlength=node_count).to(full_items.dtype)
    normalization = degree[src].pow(-0.5) * degree[dst].pow(-0.5)
    normalization[~torch.isfinite(normalization)] = 0.0

    _, _, _, full_result = propagate_three_layers(
        full_preference,
        full_items,
        src,
        dst,
        normalization,
    )

    mask_weights = torch.sigmoid(mask_logits)
    forward_coefficient = normalization[:num_edges] * mask_weights
    masked_scale = torch.cat((forward_coefficient, forward_coefficient), dim=0)
    mask_x, mask_h, mask_h_2, mask_result = propagate_three_layers(
        mask_preference,
        mask_items,
        src,
        dst,
        masked_scale,
    )

    # Exact target-user update for deleting one undirected edge from the fixed
    # propagation matrix M.  The GCN output is (I + M + M^2 + M^3)X.
    coefficient_square_sum = torch.zeros(
        num_users, dtype=forward_coefficient.dtype, device=device
    )
    coefficient_square_sum.index_add_(
        0, edge_users, forward_coefficient.square()
    )
    coefficient = forward_coefficient.unsqueeze(1)
    delta = -coefficient * (mask_h[item_nodes] + mask_h_2[item_nodes])
    delta -= coefficient * (
        1.0
        + coefficient_square_sum[edge_users]
        - forward_coefficient.square()
    ).unsqueeze(1) * mask_x[item_nodes]

    baseline_user = torch.cat(
        (full_result[edge_users], mask_result[edge_users]), dim=1
    )
    counterfactual_user = torch.cat(
        (full_result[edge_users], mask_result[edge_users] + delta), dim=1
    )
    similarities = F.cosine_similarity(
        baseline_user, counterfactual_user, dim=1
    )
    return similarities.cpu().numpy(), mask_weights.cpu().numpy()


def write_csv(path, users, items, similarities, mask_weights):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "edge_id",
                "user_id",
                "item_id",
                "original_edge_weight",
                "cosine_similarity",
            )
        )
        for edge_id, (user_id, item_id, weight, similarity) in enumerate(
            zip(users, items, mask_weights, similarities)
        ):
            writer.writerow(
                (
                    edge_id,
                    int(user_id),
                    int(item_id),
                    "{:.9g}".format(float(weight)),
                    "{:.9g}".format(float(similarity)),
                )
            )


def build_summary(similarities, x_min):
    quantile_levels = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    quantile_values = np.quantile(similarities, quantile_levels)
    return {
        "count": int(similarities.size),
        "mean": float(np.mean(similarities)),
        "standard_deviation": float(np.std(similarities, ddof=1)),
        "minimum": float(np.min(similarities)),
        "maximum": float(np.max(similarities)),
        "count_below_plot_x_min": int(np.sum(similarities < x_min)),
        "quantiles": {
            "{:.2f}".format(level): float(value)
            for level, value in zip(quantile_levels, quantile_values)
        },
    }


def save_plot(path, similarities, bins, x_min, x_max):
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.hist(
        similarities,
        bins=bins,
        range=(x_min, x_max),
        color="#1f77b4",
        edgecolor="none",
    )
    axis.set_xlim(x_min, x_max)
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Count")
    axis.set_title("Same-user baseline vs. counterfactual embeddings")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    state = load_state_dict(args.checkpoint)
    users, items = load_training_edges(args.interactions)
    similarities, mask_weights = compute_similarities(
        state, users, items, device
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "same_user_embedding_similarity.csv"
    json_path = output_dir / "same_user_embedding_similarity_summary.json"
    plot_path = output_dir / "same_user_embedding_similarity_histogram.png"

    write_csv(csv_path, users, items, similarities, mask_weights)
    summary = build_summary(similarities, args.x_min)
    summary.update(
        {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "interactions": str(args.interactions.expanduser().resolve()),
            "comparison": (
                "baseline final user embedding vs. final user embedding after "
                "setting one learned masked-branch edge to zero"
            ),
        }
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    save_plot(
        plot_path, similarities, args.bins, args.x_min, args.x_max
    )

    print("Computed {:,} same-user similarities.".format(similarities.size))
    print(
        "mean={:.9f}, std={:.9f}, min={:.9f}, max={:.9f}".format(
            summary["mean"],
            summary["standard_deviation"],
            summary["minimum"],
            summary["maximum"],
        )
    )
    print("Plot: {}".format(plot_path))
    print("CSV: {}".format(csv_path))
    print("Summary: {}".format(json_path))


if __name__ == "__main__":
    main()
