import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from analyze_original_checkpoint_branch_cases import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INTERACTIONS,
    DEFAULT_TEXT_FEATURES,
    LOGISTIC_PREDICTORS,
    PRIMARY_GROUPS,
    _recall_audit,
    add_directional_diagnostics,
    assign_group,
    build_graph_features,
    cluster_bootstrap_mean_ci,
    compute_knn_indices,
    exact_rank_batch,
    fit_logistic_associations,
    js_divergence_from_logits,
    load_interaction_splits,
    load_state_dict,
    make_derangement,
    sample_unseen_candidates,
    select_case_examples,
    summarize_groups,
    text_history_features,
    validate_checkpoint,
    validate_logistic_predictors,
    write_csv,
    build_branch_embeddings,
)


class ExactRankingTest(unittest.TestCase):
    def test_training_items_are_masked_and_ties_use_smaller_item_id(self):
        # Items 1 and 2 tie, so item 1 must be before item 2. Item 0 is masked.
        scores = torch.tensor([[100.0, 4.0, 4.0, 3.0, 2.0]])
        result = exact_rank_batch(scores, {7: [0]}, [7], (1, 2, 3))
        self.assertTrue(torch.isneginf(result["scores"][0, 0]))
        self.assertEqual(int(result["rank"][0, 1]), 1)
        self.assertEqual(int(result["rank"][0, 2]), 2)
        self.assertEqual(float(result["boundary"][2][0]), 4.0)

    def test_old_torch_fallback_preserves_stable_tie_policy(self):
        scores = torch.tensor([[1.0, 4.0, 4.0, 3.0]])
        with mock.patch("torch.argsort", side_effect=TypeError("stable unsupported")):
            result = exact_rank_batch(scores, {}, [0], (1, 2))
        self.assertEqual(int(result["rank"][0, 1]), 1)
        self.assertEqual(int(result["rank"][0, 2]), 2)

    def test_boundary_margin_and_population_std_match_manual_values(self):
        scores = torch.tensor([[5.0, 4.0, 3.0, 2.0]], dtype=torch.float64)
        result = exact_rank_batch(scores, {}, [0], (2,))
        self.assertAlmostEqual(float(result["std"][0]), float(np.std([5, 4, 3, 2])))
        positive_score = float(result["scores"][0, 2])
        margin = positive_score - float(result["boundary"][2][0])
        self.assertEqual(margin, -1.0)

    def test_primary_groups_are_exclusive(self):
        combinations = {
            (False, True): "mask_only",
            (True, False): "full_only",
            (True, True): "both_win",
            (False, False): "both_fail",
        }
        observed = {assign_group(full, mask) for (full, mask), _ in combinations.items()}
        self.assertEqual(observed, set(PRIMARY_GROUPS))
        for flags, expected in combinations.items():
            self.assertEqual(assign_group(*flags), expected)

    def test_overlay_definitions(self):
        full_hit, mask_hit, joint_hit = False, False, True
        rescue = (not full_hit and not mask_hit and joint_hit)
        harm = ((full_hit or mask_hit) and not joint_hit)
        self.assertTrue(rescue)
        self.assertFalse(harm)
        # An overlay coexists with a primary label; it does not create a fifth group.
        self.assertEqual(assign_group(full_hit, mask_hit), "both_fail")


class FeatureTest(unittest.TestCase):
    def test_text_diversity_matches_pairwise_cosine_and_degree_one_rule(self):
        root_two = np.sqrt(2.0)
        text = np.asarray([[1, 0], [0, 1], [1 / root_two, 1 / root_two]], dtype=np.float64)
        users = np.asarray([0, 0, 1], dtype=np.int64)
        items = np.asarray([0, 1, 2], dtype=np.int64)
        diversity, single = text_history_features(users, items, 2, text)
        self.assertAlmostEqual(diversity[0], 1.0)
        self.assertEqual(diversity[1], 0.0)
        self.assertEqual(single.tolist(), [0, 1])

    def test_graph_mask_statistics_preserve_edge_order(self):
        split_rows = {0: [(0, 0), (0, 1), (1, 1)], 1: [(0, 2)], 2: [(1, 2)]}
        state = {"mask_logits": torch.logit(torch.tensor([0.2, 0.8, 0.5]))}
        embeddings = {
            "num_users": 2,
            "num_items": 3,
            "full_user": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "masked_user": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "full_item": torch.eye(3, 2),
            "masked_item": torch.eye(3, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.npy"
            np.save(path, np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
            users, items, train, known, _, _ = build_graph_features(
                state, embeddings, split_rows, path
            )
        self.assertAlmostEqual(users[0]["user_mask_mean"], 0.5, places=6)
        self.assertAlmostEqual(items[1]["item_mask_mean"], 0.65, places=6)
        self.assertEqual(train[0], [0, 1])
        self.assertEqual(known[0], {0, 1, 2})


class DirectionalDiagnosticTest(unittest.TestCase):
    def test_unseen_candidates_are_unique_and_positive_is_first(self):
        rows = [{"user_id": 0, "positive_item_id": 3}]
        known = {0: {0, 1, 3}}
        candidates = sample_unseen_candidates(rows, known, 10, 4, seed=999)
        self.assertEqual(int(candidates[0, 0]), 3)
        negatives = candidates[0, 1:].tolist()
        self.assertEqual(len(negatives), len(set(negatives)))
        self.assertTrue(all(item not in known[0] for item in negatives))

    def test_derangement_has_no_fixed_point_and_is_user_level(self):
        users = np.asarray([2, 4, 8, 16])
        mapping = make_derangement(users, np.random.RandomState(999))
        self.assertEqual(set(mapping), set(users.tolist()))
        self.assertEqual(set(mapping.values()), set(users.tolist()))
        self.assertTrue(all(user != mapped for user, mapped in mapping.items()))

    def test_jsd_matches_manual_computation_and_is_finite(self):
        original = torch.tensor([[2.0, 0.0], [1000.0, -1000.0]], dtype=torch.float64)
        permuted = torch.tensor([[0.0, 2.0], [-1000.0, 1000.0]], dtype=torch.float64)
        actual = js_divergence_from_logits(original, permuted)
        p = torch.softmax(original[0], dim=0)
        q = torch.softmax(permuted[0], dim=0)
        m = 0.5 * (p + q)
        manual = 0.5 * torch.sum(p * torch.log(p / m)) + 0.5 * torch.sum(q * torch.log(q / m))
        self.assertAlmostEqual(float(actual[0]), float(manual), places=12)
        self.assertTrue(torch.isfinite(actual).all())

    def test_directional_gap_matches_manual_positive_log_probability(self):
        rows = [
            {"user_id": 0, "positive_item_id": 0},
            {"user_id": 1, "positive_item_id": 1},
        ]
        known = {0: {0}, 1: {1}}
        embeddings = {
            "num_items": 4,
            "full_user": torch.tensor([[1.0], [0.5]]),
            "masked_user": torch.tensor([[2.0], [-1.0]]),
            "full_item": torch.tensor([[1.0], [0.5], [-0.5], [0.25]]),
            "masked_item": torch.tensor([[1.0], [-0.5], [0.25], [0.75]]),
        }
        candidates = sample_unseen_candidates(rows, known, 4, 2, seed=999)
        add_directional_diagnostics(
            rows, embeddings, known, num_negatives=2,
            num_permutations=1, temperature=0.5, seed=999, batch_size=2,
        )
        item_ids = torch.tensor(candidates[0])
        full = embeddings["full_user"][0] * embeddings["full_item"][item_ids, 0]
        original_mask = embeddings["masked_user"][0] * embeddings["masked_item"][item_ids, 0]
        permuted_mask = embeddings["masked_user"][1] * embeddings["masked_item"][item_ids, 0]
        original_logp = torch.log_softmax((full + original_mask) / 0.5, dim=0)[0]
        permuted_logp = torch.log_softmax((full + permuted_mask) / 0.5, dim=0)[0]
        self.assertAlmostEqual(
            rows[0]["directional_gap_mean"],
            float(original_logp - permuted_logp), places=6,
        )


class SummaryAndSafetyTest(unittest.TestCase):
    def _pair_rows(self):
        rows = []
        for index, group in enumerate(PRIMARY_GROUPS):
            row = {
                "split": "test", "user_id": index // 2, "positive_item_id": index,
                "group_at_5": group, "joint_rescue_at_5": int(group == "both_fail"),
                "fusion_harm_at_5": int(group == "mask_only"),
                "directional_gap_mean": float(index),
                "directional_correct_win_rate": float(index > 0),
                "directional_jsd_mean": 0.01 * index,
                "directional_original_positive_probability_mean": 0.2,
                "directional_permuted_positive_probability_mean": 0.1,
            }
            for name in (
                "log1p_user_degree", "log1p_item_degree", "history_text_diversity",
                "target_history_text_cosine_max", "user_mask_mean", "item_mask_mean",
                "user_full_mask_cosine", "item_full_mask_cosine",
            ):
                row[name] = float(index)
            rows.append(row)
        return rows

    def test_group_counts_sum_to_all_pairs_and_multiple_rows_share_user(self):
        rows = self._pair_rows()
        counts, _, directional = summarize_groups(rows, (5,), 20, 999)
        primary = [record for record in counts if record["group"] in PRIMARY_GROUPS]
        self.assertEqual(sum(record["pair_count"] for record in primary), len(rows))
        self.assertEqual(sum(record["pair_percentage"] for record in primary), 100.0)
        self.assertTrue(directional)

    def test_logistic_leakage_guard(self):
        self.assertTrue(validate_logistic_predictors(LOGISTIC_PREDICTORS))
        with self.assertRaises(ValueError):
            validate_logistic_predictors(("log1p_user_degree", "full_rank"))

    def test_cluster_bootstrap_operates_on_whole_users(self):
        values = np.asarray([0.0, 2.0, 10.0])
        users = np.asarray([1, 1, 2])
        lower, upper = cluster_bootstrap_mean_ci(values, users, 100, 999)
        self.assertTrue(np.isfinite(lower))
        self.assertLessEqual(lower, upper)

    def test_case_examples_use_strength_before_user_id(self):
        rows = []
        for user, strength in ((1, 0.2), (99, 2.0)):
            rows.append({
                "split": "test", "user_id": user, "positive_item_id": user,
                "group_at_5": "mask_only", "joint_rescue_at_5": 0,
                "fusion_harm_at_5": 0,
                "masked_normalized_boundary_margin_at_5": strength,
                "full_normalized_boundary_margin_at_5": 0.0,
                "joint_normalized_boundary_margin_at_5": 0.0,
            })
        examples = select_case_examples(rows, (5,), 1)
        mask_example = next(row for row in examples if row["case_type"] == "mask_only")
        self.assertEqual(mask_example["user_id"], 99)

    def test_csv_schema_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.csv"
            write_csv(path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
            text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("a,b"))
        self.assertEqual(len(text.splitlines()), 3)

    def test_recall_audit_reports_pair_and_macro_user_values(self):
        rows = [
            {"split": "test", "user_id": 0, "full_hit_at_5": 1, "masked_hit_at_5": 0, "joint_hit_at_5": 1},
            {"split": "test", "user_id": 0, "full_hit_at_5": 0, "masked_hit_at_5": 0, "joint_hit_at_5": 0},
            {"split": "test", "user_id": 1, "full_hit_at_5": 1, "masked_hit_at_5": 1, "joint_hit_at_5": 1},
        ]
        audit = _recall_audit(rows, (5,))["test"]["full"]["5"]
        self.assertAlmostEqual(audit["pair_hit_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(audit["macro_user_recall"], 0.75)
        self.assertEqual(audit["hit_count"], 2)

    def test_logistic_regression_outputs_association_coefficients(self):
        rows = []
        for user_id in range(24):
            group = "mask_only" if user_id >= 12 else "full_only"
            row = {"split": "test", "user_id": user_id, "group_at_5": group}
            signal = float(user_id >= 12)
            for predictor_index, predictor in enumerate(LOGISTIC_PREDICTORS):
                row[predictor] = signal + 0.01 * predictor_index + 0.001 * user_id
            row["single_item_history"] = int(user_id % 2)
            rows.append(row)
        result = fit_logistic_associations(rows, (5,), 10, 999)
        self.assertEqual(len(result), len(LOGISTIC_PREDICTORS))
        self.assertTrue(all(record["interpretation"] == "association_not_causal" for record in result))
        self.assertTrue(all(np.isfinite(record["coefficient_per_sd"]) for record in result))


@unittest.skipUnless(
    os.environ.get("CAMURE_RUN_CHECKPOINT_SMOKE") == "1"
    and DEFAULT_CHECKPOINT.is_file()
    and DEFAULT_INTERACTIONS.is_file()
    and DEFAULT_TEXT_FEATURES.is_file(),
    "set CAMURE_RUN_CHECKPOINT_SMOKE=1 to run the actual-checkpoint smoke test",
)
class ActualCheckpointSmokeTest(unittest.TestCase):
    def test_dimensions_finite_embeddings_and_one_score_batch(self):
        split_rows = load_interaction_splits(DEFAULT_INTERACTIONS)
        state = load_state_dict(DEFAULT_CHECKPOINT)
        knn = compute_knn_indices(DEFAULT_TEXT_FEATURES, 10, 1024, torch.device("cpu"))
        embeddings = build_branch_embeddings(state, split_rows[0], knn, torch.device("cpu"))
        validate_checkpoint(state, embeddings, split_rows)
        users = torch.tensor([split_rows[2][0][0]], dtype=torch.long)
        score = embeddings["full_user"][users] @ embeddings["full_item"].T
        self.assertEqual(score.shape, (1, 9332))
        self.assertTrue(torch.isfinite(score).all())


if __name__ == "__main__":
    unittest.main()
