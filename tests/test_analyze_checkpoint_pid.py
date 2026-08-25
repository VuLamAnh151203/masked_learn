import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from analyze_checkpoint_pid import (
    PID_COMPONENTS,
    add_derived_pid_values,
    build_known_items,
    build_pid_manifest,
    extract_pid_features,
    save_manifest,
    summarize_runs,
    validate_checkpoint_pair,
    verify_feature_margins,
)
from pid_estimators import MultimodalDataset, critic_ce_alignment


class AnalyzeCheckpointPIDTest(unittest.TestCase):
    def setUp(self):
        self.num_users = 20
        self.num_items = 50
        self.split_rows = {
            0: [(user_id, user_id) for user_id in range(self.num_users)],
            1: [
                (user_id, (user_id + 20) % self.num_items)
                for user_id in range(self.num_users)
            ],
            2: [
                (user_id, (user_id + 30) % self.num_items)
                for user_id in range(self.num_users)
            ],
        }

    def test_manifest_is_reproducible_and_negatives_are_unseen(self):
        first = build_pid_manifest(
            self.split_rows,
            self.num_users,
            self.num_items,
            seed=999,
            train_ratio=0.8,
        )
        second = build_pid_manifest(
            self.split_rows,
            self.num_users,
            self.num_items,
            seed=999,
            train_ratio=0.8,
        )
        for name in first:
            self.assertTrue(np.array_equal(first[name], second[name]), name)

        known = build_known_items(
            self.split_rows, self.num_users, self.num_items
        )
        self.assertEqual(np.unique(first["user_ids"]).size, self.num_users)
        for user_id, positive, negative, item_a, item_b, label in zip(
            first["user_ids"],
            first["positive_items"],
            first["negative_items"],
            first["item_a"],
            first["item_b"],
            first["ground_truth_labels"],
        ):
            self.assertNotEqual(int(positive), int(negative))
            self.assertNotIn(int(negative), known[int(user_id)])
            if int(label) == 1:
                self.assertEqual(int(item_a), int(positive))
                self.assertEqual(int(item_b), int(negative))
            else:
                self.assertEqual(int(item_a), int(negative))
                self.assertEqual(int(item_b), int(positive))

    def test_saved_manifest_contains_complete_split_order(self):
        manifest = build_pid_manifest(
            self.split_rows,
            self.num_users,
            self.num_items,
            seed=100,
            train_ratio=0.8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            save_manifest(path, manifest)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), self.num_users + 1)
        self.assertIn("split_position", lines[0])

    def test_contribution_vectors_equal_branch_score_margins(self):
        manifest = build_pid_manifest(
            self.split_rows,
            self.num_users,
            self.num_items,
            seed=999,
            train_ratio=0.8,
        )
        generator = torch.Generator().manual_seed(7)
        embeddings = {
            "full_user": torch.randn(self.num_users, 4, generator=generator),
            "masked_user": torch.randn(self.num_users, 4, generator=generator),
            "full_item": torch.randn(self.num_items, 4, generator=generator),
            "masked_item": torch.randn(self.num_items, 4, generator=generator),
        }
        features = extract_pid_features(embeddings, manifest)
        verify_feature_margins(embeddings, manifest, features)
        self.assertTrue(
            torch.allclose(
                features["joint_margin"],
                features["full_margin"] + features["mask_margin"],
            )
        )
        self.assertTrue(
            torch.equal(
                features["predicted_labels"],
                (features["joint_margin"] > 0).long(),
            )
        )

    def test_checkpoint_pair_validation(self):
        def make_state(item_count=7):
            return {
                "full_gcn.preference": torch.zeros(3, 2),
                "mask_gcn.preference": torch.zeros(3, 2),
                "id_embedding_full.weight": torch.zeros(item_count, 2),
                "id_embedding_masked.weight": torch.zeros(item_count, 2),
                "mask_logits": torch.zeros(4),
            }

        dimensions = validate_checkpoint_pair(make_state(), make_state(), 4)
        self.assertEqual(dimensions["num_users"], 3)
        self.assertEqual(dimensions["num_items"], 7)
        with self.assertRaises(RuntimeError):
            validate_checkpoint_pair(make_state(), make_state(8), 4)
        with self.assertRaises(RuntimeError):
            validate_checkpoint_pair(make_state(), make_state(), 5)

    def test_summary_uses_paired_seed_deltas(self):
        rows = []
        for checkpoint, offset in (("original", 0.0), ("new", 1.0)):
            for seed, base in ((10, 2.0), (11, 4.0)):
                values = {
                    name: base + offset + index
                    for index, name in enumerate(PID_COMPONENTS)
                }
                rows.append(
                    {
                        "target": "ground_truth",
                        "checkpoint": checkpoint,
                        "estimator_seed": seed,
                        **add_derived_pid_values(values),
                    }
                )
        summary = summarize_runs(rows, (10, 11), ("ground_truth",))
        delta = summary["ground_truth"]["delta_new_minus_original"]
        self.assertEqual(delta["redundancy"]["values"], [1.0, 1.0])
        self.assertAlmostEqual(delta["mask_complementary"]["mean"], 2.0)


class BatchEstimatorSmokeTest(unittest.TestCase):
    def test_cpu_estimator_returns_finite_r_u1_u2_s(self):
        generator = torch.Generator().manual_seed(123)
        sample_count = 32
        labels = (torch.arange(sample_count) % 2).long().view(-1, 1)
        signal = labels.float() * 2.0 - 1.0
        x1 = signal + 0.2 * torch.randn(sample_count, 4, generator=generator)
        x2 = signal + 0.2 * torch.randn(sample_count, 4, generator=generator)
        train_indices = torch.arange(24)
        test_indices = torch.arange(24, 32)
        train_dataset = MultimodalDataset(
            (x1[train_indices], x2[train_indices]), labels[train_indices]
        )
        test_dataset = MultimodalDataset(
            (x1[test_indices], x2[test_indices]), labels[test_indices]
        )
        results, alignments, _ = critic_ce_alignment(
            x1=x1,
            x2=x2,
            labels=labels,
            num_labels=2,
            train_ds=train_dataset,
            test_ds=test_dataset,
            discrim_epochs=1,
            ce_epochs=1,
            batch_size=8,
            device="cpu",
            seed=999,
            sinkhorn_iterations=5,
            verbose=False,
        )
        self.assertEqual(tuple(results.shape), (1, 4))
        self.assertEqual(len(alignments), 1)
        self.assertTrue(torch.isfinite(results).all())


if __name__ == "__main__":
    unittest.main()
