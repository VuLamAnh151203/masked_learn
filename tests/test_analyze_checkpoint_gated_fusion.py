import os
import sys
import unittest

import numpy as np
import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from analyze_checkpoint_gated_fusion import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INTERACTIONS,
    DEFAULT_TEXT_FEATURES,
    TUNABLE_METHODS,
    average_rank_tailness,
    build_branch_embeddings,
    build_candidate_mask,
    build_item_popularity,
    compute_knn_indices,
    compute_uncertainty,
    deterministic_topk,
    exact_positive_ranks,
    evaluate_selected_split,
    full_boundary,
    fused_normalized_score,
    load_interaction_splits,
    load_state_dict,
    method_gate,
    metrics_from_topk,
    normalize_unseen_scores,
    select_hyperparameters,
    summarize_fusion_cases,
    summarize_gate_groups,
    validate_checkpoint,
    validation_grid_sweep,
)


class ScoreConstructionTest(unittest.TestCase):
    def test_zscore_uses_only_unseen_items(self):
        scores = torch.tensor([[1000.0, 1.0, 2.0, 3.0]])
        candidate = torch.tensor([[False, True, True, True]])
        normalized, mean, std = normalize_unseen_scores(scores, candidate)
        self.assertAlmostEqual(float(mean[0]), 2.0)
        self.assertAlmostEqual(float(std[0]), float(np.std([1.0, 2.0, 3.0])), places=6)
        self.assertAlmostEqual(float(normalized[0, 2]), 0.0)

    def test_zscore_is_finite_with_zero_variance(self):
        scores = torch.tensor([[7.0, 7.0, 7.0]])
        candidate = torch.ones_like(scores, dtype=torch.bool)
        normalized, _, std = normalize_unseen_scores(scores, candidate)
        self.assertEqual(float(std[0]), 0.0)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized, torch.zeros_like(normalized)))

    def test_candidate_mask_excludes_only_training_history(self):
        scores = torch.zeros((2, 5))
        mask = build_candidate_mask(scores, {2: [0, 3], 4: [1]}, [2, 4])
        self.assertEqual(mask[0].tolist(), [False, True, True, False, True])
        self.assertEqual(mask[1].tolist(), [True, False, True, True, True])

    def test_boundary_excludes_seen_items(self):
        normalized = torch.tensor([[100.0, 4.0, 3.0, 2.0, 1.0]])
        candidate = torch.tensor([[False, True, True, True, True]])
        boundary = full_boundary(normalized, candidate, 2)
        self.assertEqual(float(boundary[0]), 3.0)

    def test_uncertainty_is_symmetric_and_peaks_at_boundary(self):
        scores = torch.tensor([[-1.0, 0.0, 1.0]])
        uncertainty = compute_uncertainty(scores, torch.tensor([0.0]), 0.5)
        self.assertEqual(float(uncertainty[0, 1]), 1.0)
        self.assertAlmostEqual(float(uncertainty[0, 0]), float(uncertainty[0, 2]))
        self.assertLess(float(uncertainty[0, 0]), 1.0)

    def test_gate_and_final_score_formulas(self):
        full = torch.tensor([[1.0, 2.0]])
        masked = torch.tensor([[0.5, -1.0]])
        uncertainty = torch.tensor([[0.8, 0.2]])
        tailness = torch.tensor([0.25, 1.0])
        candidate = torch.ones_like(full, dtype=torch.bool)
        expected_gates = {
            "normalized_static": torch.ones_like(full),
            "popularity_gate": torch.tensor([[0.25, 1.0]]),
            "confidence_gate": uncertainty,
            "combined_gate": torch.tensor([[0.2, 0.2]]),
        }
        for method, expected in expected_gates.items():
            gate = method_gate(method, tailness, uncertainty)
            self.assertTrue(torch.allclose(gate, expected))
            actual = fused_normalized_score(full, masked, gate, 2.0, candidate)
            self.assertTrue(torch.allclose(actual, full + 2.0 * expected * masked))


class PopularityTest(unittest.TestCase):
    def test_tailness_is_monotonic_and_average_ties_match(self):
        degree = np.asarray([1, 1, 3, 10])
        tailness = average_rank_tailness(degree)
        self.assertAlmostEqual(tailness[0], tailness[1])
        self.assertGreater(tailness[0], tailness[2])
        self.assertGreater(tailness[2], tailness[3])
        self.assertTrue(np.all((tailness >= 0) & (tailness <= 1)))
        self.assertEqual(tailness[3], 0.0)

    def test_popularity_bands_preserve_degree_ties(self):
        train = []
        for item_id, degree in enumerate([1, 1, 2, 4, 8]):
            train.extend((edge, item_id) for edge in range(degree))
        item_degree, _, band, p20, p80 = build_item_popularity(train, 5)
        self.assertEqual(item_degree.tolist(), [1, 1, 2, 4, 8])
        self.assertTrue(all(band[index] == "tail" for index in (0, 1)))
        self.assertEqual(band[4], "head")
        self.assertLessEqual(p20, p80)


class RankingAndMetricTest(unittest.TestCase):
    def test_deterministic_topk_repairs_boundary_and_internal_ties(self):
        scores = torch.tensor([[5.0, 5.0, 4.0, 4.0, 4.0, 1.0]])
        top = deterministic_topk(scores, 4)
        self.assertEqual(top.tolist(), [[0, 1, 2, 3]])

    def test_exact_positive_rank_uses_item_id_tie_break(self):
        scores = torch.tensor([[4.0, 3.0, 3.0, 1.0]])
        ranks, values = exact_positive_ranks(scores, [0, 0], [1, 2], 2)
        self.assertEqual(ranks.tolist(), [2, 3])
        self.assertEqual(values.tolist(), [3.0, 3.0])

    def test_tail_and_head_recall_use_only_eligible_users(self):
        topk = np.asarray([[0, 2], [3, 1]])
        users = np.asarray([0, 1])
        positives = {0: [0, 1], 1: [2, 3]}
        band = np.asarray(["tail", "head", "tail", "head"], dtype=object)
        metrics = metrics_from_topk(topk, users, positives, (1, 2), band)
        self.assertAlmostEqual(metrics["tail"][1]["recall"], 0.5)
        self.assertAlmostEqual(metrics["head"][1]["recall"], 0.5)
        self.assertEqual(metrics["tail"][1]["eligible_user_count"], 2)
        self.assertEqual(metrics["head"][1]["positive_count"], 2)


class SelectionAndDiagnosticTest(unittest.TestCase):
    def _grid_rows(self):
        rows = []
        for method in TUNABLE_METHODS:
            # lambda .5 and 1 tie at NDCG@20; .5 wins through Recall@20.
            for weight, ndcg20, recall20, ndcg10 in (
                (0.0, .1, .1, .1), (.5, .2, .3, .1), (1.0, .2, .2, .9)
            ):
                rows.extend([
                    {"method": method, "lambda": weight, "cutoff": 20, "ndcg": ndcg20, "recall": recall20},
                    {"method": method, "lambda": weight, "cutoff": 10, "ndcg": ndcg10, "recall": .1},
                ])
        return rows

    def test_validation_selection_follows_locked_tie_break(self):
        rows = self._grid_rows()
        # A deliberately excellent test row must never influence selection.
        for method in TUNABLE_METHODS:
            rows.extend([
                {"split": "test", "method": method, "lambda": 8.0, "cutoff": 20, "ndcg": 1.0, "recall": 1.0},
                {"split": "test", "method": method, "lambda": 8.0, "cutoff": 10, "ndcg": 1.0, "recall": 1.0},
            ])
        selected = select_hyperparameters(rows)
        self.assertTrue(all(value["selected_lambda"] == 0.5 for value in selected.values()))
        self.assertTrue(all("test" not in value for value in selected.values()))

    def test_rescue_and_harm_are_pair_overlays(self):
        row = {"split": "test", "user_id": 1}
        for method in (
            "original_fusion", "normalized_static", "popularity_gate",
            "confidence_gate", "combined_gate",
        ):
            row["{}_joint_rescue_at_5".format(method)] = int(method == "combined_gate")
            row["{}_fusion_harm_at_5".format(method)] = int(method == "confidence_gate")
        output = summarize_fusion_cases([row], (5,))
        combined = next(x for x in output if x["method"] == "combined_gate" and x["case"] == "joint_rescue")
        confidence = next(x for x in output if x["method"] == "confidence_gate" and x["case"] == "fusion_harm")
        self.assertEqual(combined["pair_percentage"], 100.0)
        self.assertEqual(confidence["pair_count"], 1)

    def test_gate_summary_reports_mask_minus_full_difference(self):
        rows = []
        for group, value in (("mask_only", .8), ("full_only", .2)):
            row = {"split": "test", "branch_group_at_5": group}
            for method in ("popularity_gate", "confidence_gate", "combined_gate"):
                row[method] = value
            rows.append(row)
        output = summarize_gate_groups(rows, (5,))
        self.assertTrue(all(abs(row["mask_only_minus_full_only_mean"] - .6) < 1e-12 for row in output))

    def test_streaming_grid_and_selected_diagnostic_pipeline(self):
        generator = torch.Generator().manual_seed(4)
        embeddings = {
            "num_users": 3, "num_items": 8,
            "full_user": torch.randn(3, 3, generator=generator),
            "masked_user": torch.randn(3, 3, generator=generator),
            "full_item": torch.randn(8, 3, generator=generator),
            "masked_item": torch.randn(8, 3, generator=generator),
        }
        train = {0: [0], 1: [1], 2: [2]}
        heldout = {0: [3], 1: [4], 2: [5]}
        degree = np.asarray([1, 1, 1, 0, 0, 0, 0, 0])
        tailness = average_rank_tailness(degree)
        band = np.asarray(["head", "head", "head", "tail", "tail", "tail", "tail", "tail"], dtype=object)
        grid = validation_grid_sweep(
            embeddings, train, heldout, tailness, band, (1, 2),
            boundary_k=2, temperature=.5, lambdas=(0.0, 1.0), batch_size=2,
        )
        self.assertEqual(len(grid), (2 + 4 * 2) * 2)
        selected = {
            method: {"selected_lambda": 1.0} for method in TUNABLE_METHODS
        }
        metrics, pairs = evaluate_selected_split(
            "validation", embeddings, train, heldout, selected, tailness,
            degree, band, (1, 2), 2, .5, 2, 4,
        )
        self.assertEqual(len(pairs), 3)
        self.assertIn("combined_gate", metrics)
        self.assertTrue(all("combined_gate_rank" in row for row in pairs))


@unittest.skipUnless(
    os.environ.get("CAMURE_RUN_CHECKPOINT_SMOKE") == "1"
    and DEFAULT_CHECKPOINT.is_file()
    and DEFAULT_INTERACTIONS.is_file()
    and DEFAULT_TEXT_FEATURES.is_file(),
    "set CAMURE_RUN_CHECKPOINT_SMOKE=1 to run the actual checkpoint",
)
class ActualCheckpointSmokeTest(unittest.TestCase):
    def test_checkpoint_dimensions_and_finite_scores(self):
        split_rows = load_interaction_splits(DEFAULT_INTERACTIONS)
        state = load_state_dict(DEFAULT_CHECKPOINT)
        device = torch.device("cpu")
        knn = compute_knn_indices(DEFAULT_TEXT_FEATURES, 10, 1024, device)
        embeddings = build_branch_embeddings(state, split_rows[0], knn, device)
        validate_checkpoint(state, embeddings, split_rows)
        user = torch.tensor([split_rows[1][0][0]], dtype=torch.long)
        full = embeddings["full_user"][user] @ embeddings["full_item"].T
        masked = embeddings["masked_user"][user] @ embeddings["masked_item"].T
        self.assertEqual(full.shape, (1, 9332))
        self.assertTrue(torch.isfinite(full).all() and torch.isfinite(masked).all())


if __name__ == "__main__":
    unittest.main()
