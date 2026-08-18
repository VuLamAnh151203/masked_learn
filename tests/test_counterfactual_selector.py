import os
import sys
import types
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from counterfactual_selector_analysis import recall_at_n, summarize_rows
try:
    from models.masked_gloria_cf import MASKED_GLORIA_CF
except ModuleNotFoundError as error:
    if error.name != 'torch_geometric':
        raise
    MASKED_GLORIA_CF = None


def make_selector_model():
    model = MASKED_GLORIA_CF.__new__(MASKED_GLORIA_CF)
    torch.nn.Module.__init__(model)
    model.num_user = 1
    model.num_item = 4
    model.cf_k = 2
    model.cf_boundary_width = 2
    model.cf_selector_top_n = 3
    model.cf_selector_damage_eps = 1e-8
    model.forward_edge_items = torch.tensor([0, 1, 2], dtype=torch.long)
    model.user_seen_items = [(0, 1, 2)]
    model.mask_logits = torch.nn.Parameter(torch.zeros(3))
    model.cf_stats = model._new_cf_stats()
    return model


@unittest.skipIf(
    MASKED_GLORIA_CF is None,
    'torch_geometric is required for model selector tests'
)
class SelectorScoringTest(unittest.TestCase):
    def test_selector_config_validation(self):
        model = make_selector_model()
        model.cf_lambda = 0.1
        model.cf_user_ratio = 0.1
        model.cf_batch_size = 8
        model.cf_boundary_q = 3
        model.cf_temperature = 1.0
        model.cf_min_history = 2
        model.cf_edge_selector = 'representation'
        model.cf_warmup_epochs = 0
        model.cf_drop_bidirectional = True
        model._validate_cf_config()

        model.cf_edge_selector = 'unknown'
        with self.assertRaises(ValueError):
            model._validate_cf_config()
        model.cf_edge_selector = 'gradient'
        model.cf_selector_top_n = 0
        with self.assertRaises(ValueError):
            model._validate_cf_config()

    def test_representation_selector_uses_probe_masked_items(self):
        model = make_selector_model()
        model.cf_edge_selector = 'representation'
        model.mask_rep = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        scores = model._score_cf_candidates(
            base_mask=torch.ones(3),
            full_view=(torch.zeros(1, 1), torch.zeros(4, 1)),
            probe_mask=torch.tensor([0.0, 1.0, 1.0]),
            user_id=0,
            pseudo_item_id=0,
            boundary_item_id=3,
            candidate_edges=[1, 2]
        )
        self.assertGreater(scores[1], scores[2])

    def test_gradient_selector_matches_first_order_damage(self):
        model = make_selector_model()
        model.cf_edge_selector = 'gradient'

        def compute_result_embedding(self, forward_edge_mask, full_view=None):
            zero = forward_edge_mask.sum() * 0.0
            pseudo = 3.0 * forward_edge_mask[1] - forward_edge_mask[2]
            return torch.stack([
                zero + 1.0,
                pseudo,
                zero,
                zero,
                zero,
            ]).reshape(5, 1)

        model.compute_result_embedding = types.MethodType(
            compute_result_embedding, model
        )
        mask_before = model.mask_logits.detach().clone()
        scores = model._score_cf_candidates(
            base_mask=torch.tensor([0.8, 0.5, 0.25]),
            full_view=(torch.zeros(1, 1), torch.zeros(4, 1)),
            probe_mask=torch.tensor([0.0, 0.5, 0.25]),
            user_id=0,
            pseudo_item_id=0,
            boundary_item_id=3,
            candidate_edges=[1, 2]
        )
        self.assertAlmostEqual(scores[1], 1.5, places=6)
        self.assertEqual(scores[2], 0.0)
        self.assertTrue(torch.equal(model.mask_logits.detach(), mask_before))
        self.assertIsNone(model.mask_logits.grad)

    def test_stable_ranking_breaks_score_ties_by_edge_id(self):
        ranked = MASKED_GLORIA_CF._rank_cf_candidates(
            [7, 3, 5],
            {7: 0.4, 3: 0.4, 5: 0.9}
        )
        self.assertEqual(ranked, [5, 3, 7])

    def test_fixed_boundary_skips_pseudo_at_rank_k(self):
        model = make_selector_model()
        scores = torch.tensor([10.0, 9.0, 8.0, 7.0])
        self.assertEqual(model._select_fixed_boundary_item(scores, 1), 2)

    def test_actual_verification_selects_largest_margin_damage(self):
        model = make_selector_model()

        def compute_result_embedding(self, forward_edge_mask, full_view=None):
            zero = forward_edge_mask.sum() * 0.0
            pseudo = 0.5 * forward_edge_mask[1] + 0.2 * forward_edge_mask[2]
            return torch.stack([
                zero + 1.0,
                pseudo,
                zero,
                zero,
                zero,
            ]).reshape(5, 1)

        model.compute_result_embedding = types.MethodType(
            compute_result_embedding, model
        )
        edge_id, damage = model._verify_cf_candidates(
            base_mask=torch.ones(3),
            full_view=(torch.zeros(1, 1), torch.zeros(4, 1)),
            user_id=0,
            pseudo_edge_id=0,
            pseudo_item_id=0,
            boundary_item_id=3,
            candidate_edges=[1, 2],
            probe_margin=0.7
        )
        self.assertEqual(edge_id, 1)
        self.assertAlmostEqual(damage, 0.5, places=6)
        self.assertEqual(model.cf_stats['candidates_verified'], 2)


class OfflineMetricTest(unittest.TestCase):
    def test_recall_at_n_and_empty_harmful_set(self):
        ranked = [4, 2, 9, 7]
        harmful = {2, 7}
        self.assertEqual(recall_at_n(ranked, harmful, 1), 0.0)
        self.assertEqual(recall_at_n(ranked, harmful, 3), 0.5)
        self.assertEqual(recall_at_n(ranked, harmful, 5), 1.0)
        self.assertIsNone(recall_at_n(ranked, set(), 3))

    def test_summary_excludes_pseudos_without_harmful_edges(self):
        rows = []
        for selector in ('representation', 'gradient', 'random'):
            rows.extend([
                {
                    'selector': selector,
                    'candidate_count': 4,
                    'harmful_count': 2,
                    'recall_at_1': 0.5,
                    'recall_at_3': 1.0,
                    'recall_at_5': 1.0,
                    'hit_at_1': 1.0,
                    'hit_at_3': 1.0,
                    'hit_at_5': 1.0,
                    'captured_at_1': 1,
                    'captured_at_3': 2,
                    'captured_at_5': 2,
                },
                {
                    'selector': selector,
                    'candidate_count': 4,
                    'harmful_count': 0,
                    'recall_at_1': None,
                    'recall_at_3': None,
                    'recall_at_5': None,
                    'hit_at_1': None,
                    'hit_at_3': None,
                    'hit_at_5': None,
                    'captured_at_1': 0,
                    'captured_at_3': 0,
                    'captured_at_5': 0,
                },
            ])
        summary = summarize_rows(rows)
        representation_at_1 = next(
            row for row in summary
            if row['selector'] == 'representation' and row['top_n'] == 1
        )
        self.assertEqual(representation_at_1['pseudo_count'], 2)
        self.assertEqual(representation_at_1['eligible_pseudo_count'], 1)
        self.assertEqual(representation_at_1['edge_recall_micro'], 0.5)


if __name__ == '__main__':
    unittest.main()
