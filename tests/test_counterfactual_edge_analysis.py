import csv
import os
import sys
import tempfile
import types
import unittest

import numpy as np
import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from counterfactual_edge_analysis import (
    build_user_edge_map,
    compute_metrics_from_topk,
    load_checkpoint_strict,
    load_completed_edge_ids,
    result_fieldnames,
    safe_correlation,
    select_target_users,
)
from models.masked_gloria import MASKED_GLORIA


class CaptureGCN(torch.nn.Module):
    def __init__(self, representation):
        super().__init__()
        self.representation = representation
        self.last_edge_mask = None

    def forward(self, edge_index, features, edge_mask=None):
        self.last_edge_mask = (
            edge_mask.detach().clone() if edge_mask is not None else None
        )
        return self.representation, None


def make_inference_only_model():
    model = MASKED_GLORIA.__new__(MASKED_GLORIA)
    torch.nn.Module.__init__(model)
    model.num_user = 2
    model.num_item = 2
    model.n_users = 2
    model.n_items = 2
    model.num_interactions = 2
    model.forward_edge_users = torch.tensor([0, 1], dtype=torch.long)
    model.forward_edge_items = torch.tensor([0, 1], dtype=torch.long)
    forward_edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    model.forward_edge_index = forward_edges
    model.edge_index = torch.cat(
        [forward_edges, forward_edges[[1, 0]]],
        dim=1
    )
    base_weights = torch.tensor([0.8, 0.3], dtype=torch.float32)
    model.mask_logits = torch.nn.Parameter(torch.logit(base_weights))
    model.id_embedding_full = torch.nn.Embedding(2, 2)
    model.id_embedding_masked = torch.nn.Embedding(2, 2)
    full_representation = torch.tensor(
        [
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
            [16.0, 17.0],
        ]
    )
    masked_representation = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    model.full_gcn = CaptureGCN(full_representation)
    model.mask_gcn = CaptureGCN(masked_representation)
    item_operator = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    model.item_item = types.MethodType(
        lambda self, value: value + torch.matmul(item_operator, value),
        model
    )
    return model


class CounterfactualMetricTest(unittest.TestCase):
    def test_recall_and_ndcg_are_per_user_and_unrounded(self):
        topk = np.asarray(
            [
                [1, 2, 3] + list(range(10, 27)),
                list(range(20)),
            ],
            dtype=np.int64
        )
        metrics = compute_metrics_from_topk(topk, [[1, 3], [9]])

        discounts = 1.0 / np.log2(np.arange(2, 22))
        expected_user_0_ndcg = (
            discounts[0] + discounts[2]
        ) / (discounts[0] + discounts[1])
        self.assertTrue(np.allclose(metrics[0, :2], [1.0, 1.0]))
        self.assertAlmostEqual(metrics[0, 2], expected_user_0_ndcg)
        self.assertAlmostEqual(metrics[0, 3], expected_user_0_ndcg)
        self.assertEqual(metrics[1, 0], 0.0)
        self.assertEqual(metrics[1, 1], 1.0)
        self.assertEqual(metrics[1, 2], 0.0)
        self.assertAlmostEqual(metrics[1, 3], discounts[9])

    def test_correlations_handle_ties_and_constant_drops(self):
        self.assertAlmostEqual(
            safe_correlation([1, 2, 3], [2, 4, 6], 'pearson'),
            1.0
        )
        self.assertAlmostEqual(
            safe_correlation([1, 1, 2], [3, 3, 5], 'spearman'),
            1.0
        )
        self.assertIsNone(
            safe_correlation([1, 2, 3], [0, 0, 0], 'pearson')
        )


class UserAndEdgeSelectionTest(unittest.TestCase):
    def test_user_sampling_is_reproducible_and_eligible(self):
        eval_users = np.asarray([0, 1, 2, 3, 4])
        edge_users = np.asarray([0, 0, 1, 2, 4])
        first = select_target_users(eval_users, edge_users, 3, seed=999)
        second = select_target_users(eval_users, edge_users, 3, seed=999)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 3)
        self.assertTrue(set(first).issubset({0, 1, 2, 4}))
        self.assertTrue(
            np.array_equal(
                select_target_users(eval_users, edge_users, None, seed=1),
                np.asarray([0, 1, 2, 4])
            )
        )
        with self.assertRaises(ValueError):
            select_target_users(eval_users, edge_users, 5, seed=999)

    def test_users_can_be_ranked_by_descending_recall(self):
        eval_users = np.asarray([10, 11, 12, 13])
        edge_users = np.asarray([10, 11, 12, 13])
        recall_scores = {
            10: 0.5,
            11: 0.5,
            12: 1.0,
            13: 0.0,
        }
        selected = select_target_users(
            eval_users,
            edge_users,
            3,
            seed=999,
            strategy='recall_desc',
            user_scores=recall_scores
        )
        self.assertTrue(np.array_equal(selected, [12, 10, 11]))
        all_ranked = select_target_users(
            eval_users,
            edge_users,
            None,
            seed=999,
            strategy='recall_desc',
            user_scores=recall_scores
        )
        self.assertTrue(np.array_equal(all_ranked, [12, 10, 11, 13]))

    def test_recall_selection_requires_scores(self):
        with self.assertRaises(ValueError):
            select_target_users(
                [0, 1],
                [0, 1],
                1,
                seed=999,
                strategy='recall_desc'
            )

    def test_zero_recall_users_are_excluded_before_selection(self):
        scores = {0: 0.0, 1: 0.5, 2: 0.0, 3: 1.0}
        selected = select_target_users(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            None,
            seed=999,
            strategy='recall_desc',
            user_scores=scores,
            exclude_zero_scores=True
        )
        self.assertTrue(np.array_equal(selected, [3, 1]))
        with self.assertRaises(ValueError):
            select_target_users(
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                3,
                seed=999,
                strategy='random',
                user_scores=scores,
                exclude_zero_scores=True
            )

    def test_every_incident_edge_is_grouped_once(self):
        mapping = build_user_edge_map([0, 0, 1, 2, 2], [0, 2])
        self.assertEqual(mapping, {0: [0, 1], 2: [3, 4]})


class MaskedGloriaInferenceApiTest(unittest.TestCase):
    def test_edge_mapping_and_counterfactual_mask(self):
        model = make_inference_only_model()
        self.assertTrue(
            torch.equal(model.get_user_edge_ids(1), torch.tensor([1]))
        )
        base_mask = model.get_forward_edge_mask().detach()
        counterfactual = model.get_counterfactual_forward_mask(0, base_mask)
        self.assertEqual(counterfactual[0].item(), 0.0)
        self.assertAlmostEqual(counterfactual[1].item(), 0.3, places=6)
        self.assertAlmostEqual(base_mask[0].item(), 0.8, places=6)

    def test_embedding_matches_original_concatenation_and_zeros_both_directions(self):
        model = make_inference_only_model()
        full_view = model.compute_full_view()
        counterfactual = model.get_counterfactual_forward_mask(0)
        result = model.compute_result_embedding(
            counterfactual,
            full_view=full_view
        )

        full_representation = model.full_gcn.representation
        masked_representation = model.mask_gcn.representation
        expected_user = torch.cat(
            [full_representation[:2], masked_representation[:2]],
            dim=1
        )
        expected_item = model.item_item(
            torch.cat(
                [full_representation[2:], masked_representation[2:]],
                dim=1
            )
        )
        expected = torch.cat([expected_user, expected_item], dim=0)
        self.assertTrue(torch.allclose(result, expected))
        applied_mask = model.mask_gcn.last_edge_mask
        self.assertEqual(applied_mask[0].item(), 0.0)
        self.assertEqual(applied_mask[2].item(), 0.0)
        self.assertAlmostEqual(applied_mask[1].item(), 0.3, places=6)
        self.assertAlmostEqual(applied_mask[3].item(), 0.3, places=6)


class CheckpointAndResumeTest(unittest.TestCase):
    def test_strict_checkpoint_load_and_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = os.path.join(temporary_directory, 'valid.pth')
            model = torch.nn.Linear(2, 1, bias=False)
            expected_weight = torch.tensor([[2.0, 3.0]])
            torch.save(
                {'state_dict': {'weight': expected_weight}},
                checkpoint_path
            )
            load_checkpoint_strict(model, checkpoint_path)
            self.assertTrue(torch.equal(model.weight, expected_weight))

            mismatch_path = os.path.join(temporary_directory, 'mismatch.pth')
            torch.save(
                {'state_dict': {'weight': torch.ones(1, 3)}},
                mismatch_path
            )
            with self.assertRaises(RuntimeError):
                load_checkpoint_strict(model, mismatch_path)

    def test_legacy_preference_aliases_are_canonicalized(self):
        class LegacyPreferenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.full_gcn = torch.nn.Module()
                self.mask_gcn = torch.nn.Module()
                self.full_gcn.preference = torch.nn.Parameter(torch.zeros(2, 3))
                self.mask_gcn.preference = torch.nn.Parameter(torch.zeros(2, 3))

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = os.path.join(temporary_directory, 'legacy.pth')
            full_value = torch.full((2, 3), 2.0)
            mask_value = torch.full((2, 3), 3.0)
            torch.save(
                {
                    'state_dict': {
                        'full_preference': full_value,
                        'mask_preference': mask_value,
                    }
                },
                checkpoint_path
            )
            model = LegacyPreferenceModel()
            load_checkpoint_strict(model, checkpoint_path)
            self.assertTrue(torch.equal(model.full_gcn.preference, full_value))
            self.assertTrue(torch.equal(model.mask_gcn.preference, mask_value))

    def test_resume_repairs_invalid_tail_without_duplicate_edges(self):
        fields = result_fieldnames()
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = os.path.join(temporary_directory, 'results.csv')
            row = {field: 0 for field in fields}
            row.update({
                'user_id': 2,
                'edge_id': 7,
                'item_id': 4,
                'user_train_degree': 1,
                'original_edge_weight': 0.8,
            })
            with open(results_path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
                handle.write('interrupted,incomplete,row\n')

            completed = load_completed_edge_ids(results_path, fields, repair=True)
            self.assertEqual(completed, {7})
            self.assertEqual(
                load_completed_edge_ids(results_path, fields, repair=False),
                {7}
            )

    def test_resume_repairs_an_empty_csv(self):
        fields = result_fieldnames()
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = os.path.join(temporary_directory, 'results.csv')
            open(results_path, 'w', encoding='utf-8').close()
            self.assertEqual(
                load_completed_edge_ids(results_path, fields, repair=True),
                set()
            )
            with open(results_path, 'r', encoding='utf-8', newline='') as handle:
                self.assertEqual(next(csv.reader(handle)), fields)


if __name__ == '__main__':
    unittest.main()
