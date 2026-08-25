import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


try:
    import torch_geometric  # noqa: F401
except ModuleNotFoundError:
    torch_geometric = types.ModuleType('torch_geometric')
    geometric_nn = types.ModuleType('torch_geometric.nn')
    geometric_conv = types.ModuleType('torch_geometric.nn.conv')
    geometric_utils = types.ModuleType('torch_geometric.utils')

    class DummyMessagePassing(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    geometric_conv.MessagePassing = DummyMessagePassing
    geometric_utils.remove_self_loops = lambda edge_index: (edge_index, None)
    geometric_utils.add_self_loops = (
        lambda edge_index, **kwargs: (edge_index, None)
    )
    geometric_utils.degree = lambda *args, **kwargs: None
    torch_geometric.nn = geometric_nn
    torch_geometric.utils = geometric_utils
    geometric_nn.conv = geometric_conv
    sys.modules['torch_geometric'] = torch_geometric
    sys.modules['torch_geometric.nn'] = geometric_nn
    sys.modules['torch_geometric.nn.conv'] = geometric_conv
    sys.modules['torch_geometric.utils'] = geometric_utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.masked_gloria import MASKED_GLORIA
from models.masked_gloria_directional import MASKED_GLORIA_DIRECTIONAL
from utils.utils import get_model


def make_model():
    model = MASKED_GLORIA_DIRECTIONAL.__new__(MASKED_GLORIA_DIRECTIONAL)
    torch.nn.Module.__init__(model)
    model.num_user = 3
    model.num_item = 6
    model.n_users = 3
    model.n_items = 6
    model.directional_weight = 0.25
    model.directional_margin = 0.1
    model.directional_num_negatives = 2
    model.directional_num_samples = 2
    model.directional_temperature = 0.7
    model.directional_negative_sampling = 'random'
    model.directional_hard_pool_size = 6
    model.directional_permutation_gradient = 'symmetric'
    model.directional_loss_type = 'softplus'
    model.full_user_view = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        requires_grad=True,
    )
    model.masked_user_view = torch.tensor(
        [[1.0, 0.5], [0.2, 1.2], [-0.7, 0.4]],
        requires_grad=True,
    )
    model.full_item_view = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
            [0.4, -0.8],
            [0.3, 0.6],
        ],
        requires_grad=True,
    )
    model.masked_item_view = torch.tensor(
        [
            [0.5, 0.2],
            [0.1, 0.9],
            [0.8, 0.7],
            [-0.6, 0.3],
            [0.5, -0.4],
            [0.2, 0.8],
        ],
        requires_grad=True,
    )
    model.directional_seen_items = (
        frozenset((0, 1)),
        frozenset((2, 3)),
        frozenset((4,)),
    )
    model.last_ranking_loss = None
    model.last_directional_loss = None
    model.last_directional_mean_gap = None
    model.last_directional_positive_gap_rate = None
    model.last_directional_margin_rate = None
    model.last_directional_user_count = None
    model.last_directional_candidate_count = None
    model.directional_epoch_loss_sum = 0.0
    model.directional_epoch_gap_sum = 0.0
    model.directional_epoch_positive_gap_sum = 0.0
    model.directional_epoch_margin_sum = 0.0
    model.directional_epoch_user_count = 0
    model.directional_epoch_batch_count = 0
    model.ranking_epoch_loss_sum = 0.0
    model.ranking_epoch_example_count = 0
    return model


class DirectionalPermutationTest(unittest.TestCase):
    def setUp(self):
        self.users = torch.tensor([0, 1, 2])
        self.candidates = torch.tensor(
            [[0, 2, 3], [2, 0, 5], [4, 1, 3]]
        )
        self.permutation = torch.tensor([1, 2, 0])

    def test_compute_result_embedding_splits_existing_result(self):
        model = make_model()
        model.num_user = 2
        result = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        def fake_base_compute(this, forward_edge_mask=None, full_view=None):
            this.result_embed = result
            return result

        with patch.object(
            MASKED_GLORIA,
            'compute_result_embedding',
            fake_base_compute,
        ):
            actual = model.compute_result_embedding()

        self.assertIs(actual, result)
        self.assertTrue(torch.equal(model.full_user_view, result[:2, :2]))
        self.assertTrue(torch.equal(model.masked_user_view, result[:2, 2:]))
        self.assertTrue(torch.equal(model.full_item_view, result[2:, :2]))
        self.assertTrue(torch.equal(model.masked_item_view, result[2:, 2:]))

    def test_random_candidates_are_unique_unseen(self):
        model = make_model()
        interaction = torch.tensor(
            [[0, 0, 1, 2], [0, 1, 2, 4], [2, 3, 5, 0]]
        )

        torch.manual_seed(11)
        users, candidates = model._build_directional_candidates(interaction)

        self.assertEqual(users.tolist(), [0, 1, 2])
        self.assertEqual(candidates[:, 0].tolist(), [0, 2, 4])
        self.assertEqual(candidates.shape, (3, 3))
        for user_id, row in zip(users.tolist(), candidates.tolist()):
            negatives = row[1:]
            self.assertEqual(len(negatives), len(set(negatives)))
            self.assertNotIn(row[0], negatives)
            self.assertTrue(
                set(negatives).isdisjoint(
                    model.directional_seen_items[user_id]
                )
            )

    def test_full_hard_selects_highest_full_score(self):
        model = make_model()
        model.directional_negative_sampling = 'full_hard'
        model.directional_hard_pool_size = 6
        interaction = torch.tensor([[0, 1], [0, 2], [3, 4]])

        torch.manual_seed(17)
        users, candidates = model._build_directional_candidates(interaction)

        self.assertEqual(users.tolist(), [0, 1])
        self.assertEqual(candidates.tolist(), [[0, 2, 4], [2, 1, 5]])

    def test_directional_loss_matches_manual_calculation(self):
        model = make_model()
        actual_loss, actual_gap = (
            model.calculate_directional_for_permutation(
                self.users,
                self.candidates,
                permutation=self.permutation,
            )
        )

        full_scores = torch.sum(
            model.full_user_view[self.users, None, :]
            * model.full_item_view[self.candidates],
            dim=-1,
        ).detach()
        masked_items = model.masked_item_view[self.candidates]
        masked_users = model.masked_user_view[self.users]
        original_masked_scores = torch.sum(
            masked_users[:, None, :] * masked_items,
            dim=-1,
        )
        permuted_masked_scores = torch.sum(
            masked_users[self.permutation, None, :] * masked_items,
            dim=-1,
        )
        original_log_prob = F.log_softmax(
            (full_scores + original_masked_scores)
            / model.directional_temperature,
            dim=1,
        )[:, 0]
        permuted_log_prob = F.log_softmax(
            (full_scores + permuted_masked_scores)
            / model.directional_temperature,
            dim=1,
        )[:, 0]
        expected_gap = original_log_prob - permuted_log_prob
        expected_loss = F.softplus(
            model.directional_margin - expected_gap
        ).mean()

        self.assertTrue(torch.allclose(actual_gap, expected_gap, atol=1e-7))
        self.assertTrue(torch.allclose(actual_loss, expected_loss, atol=1e-7))

    def test_symmetric_gradient_updates_mask_but_not_full(self):
        model = make_model()
        loss, _ = model.calculate_directional_for_permutation(
            self.users,
            self.candidates,
            permutation=self.permutation,
        )
        loss.backward()

        self.assertIsNone(model.full_user_view.grad)
        self.assertIsNone(model.full_item_view.grad)
        self.assertIsNotNone(model.masked_user_view.grad)
        self.assertIsNotNone(model.masked_item_view.grad)
        self.assertGreater(model.masked_user_view.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.masked_item_view.grad.abs().sum().item(), 0.0)

    def test_symmetric_and_detached_permutation_gradients_differ(self):
        symmetric = make_model()
        detached = make_model()
        detached.directional_permutation_gradient = 'detached'

        symmetric_loss, _ = symmetric.calculate_directional_for_permutation(
            self.users,
            self.candidates,
            permutation=self.permutation,
        )
        detached_loss, _ = detached.calculate_directional_for_permutation(
            self.users,
            self.candidates,
            permutation=self.permutation,
        )
        symmetric_loss.backward()
        detached_loss.backward()

        self.assertFalse(
            torch.allclose(
                symmetric.masked_user_view.grad,
                detached.masked_user_view.grad,
            )
        )
        self.assertFalse(
            torch.allclose(
                symmetric.masked_item_view.grad,
                detached.masked_item_view.grad,
            )
        )

    def test_helpful_pairings_satisfy_margin_and_have_zero_loss(self):
        model = make_model()
        model.directional_loss_type = 'hinge'
        model.num_user = 2
        model.num_item = 2
        model.full_user_view = torch.zeros(2, 1, requires_grad=True)
        model.full_item_view = torch.zeros(2, 1, requires_grad=True)
        model.masked_user_view = torch.tensor(
            [[2.0], [-2.0]],
            requires_grad=True,
        )
        model.masked_item_view = torch.tensor(
            [[1.0], [-1.0]],
            requires_grad=True,
        )
        users = torch.tensor([0, 1])
        candidates = torch.tensor([[0, 1], [1, 0]])

        loss, gaps = model.calculate_directional_for_permutation(
            users,
            candidates,
            permutation=torch.tensor([1, 0]),
        )

        self.assertTrue(torch.all(gaps > model.directional_margin))
        self.assertEqual(loss.item(), 0.0)

    def test_same_candidates_are_reused_for_all_permutations(self):
        model = make_model()
        fixed_candidates = self.candidates.clone()
        calls = []

        def fake_builder(this, interaction):
            return self.users, fixed_candidates

        def fake_directional(this, users, candidates, permutation=None):
            calls.append(candidates)
            value = float(len(calls))
            loss = this.masked_user_view[users].sum() * 0.0 + value
            gaps = torch.full((users.numel(),), value)
            return loss, gaps

        model._build_directional_candidates = types.MethodType(
            fake_builder,
            model,
        )
        model.calculate_directional_for_permutation = types.MethodType(
            fake_directional,
            model,
        )

        loss, mean_gap, _, _ = model.calculate_directional_loss(
            torch.tensor([[0, 1, 2], [0, 2, 4], [3, 5, 1]])
        )

        self.assertEqual(len(calls), model.directional_num_samples)
        self.assertTrue(all(candidate is fixed_candidates for candidate in calls))
        self.assertEqual(loss.item(), 1.5)
        self.assertEqual(mean_gap.item(), 1.5)

    def test_total_loss_and_zero_weight_fast_path(self):
        model = make_model()
        interaction = torch.tensor([[0, 1, 2], [0, 2, 4], [3, 5, 1]])

        def fake_forward(this, supplied_interaction):
            return (
                torch.tensor([1.2, 0.8, 1.5], requires_grad=True),
                torch.tensor([0.1, 0.4, 0.2], requires_grad=True),
            )

        def fake_directional(this, supplied_interaction):
            this.last_directional_user_count = 3
            this.last_directional_candidate_count = 3
            loss = this.masked_user_view.sum() * 0.0 + 0.3
            metric = torch.tensor(0.2)
            return loss, metric, metric, metric

        model.forward = types.MethodType(fake_forward, model)
        model.calculate_directional_loss = types.MethodType(
            fake_directional,
            model,
        )
        actual = model.calculate_loss(interaction)
        ranking = F.softplus(
            -(
                torch.tensor([1.2, 0.8, 1.5])
                - torch.tensor([0.1, 0.4, 0.2])
            )
        ).mean() / math.log(2.0)
        expected = ranking + model.directional_weight * 0.3
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

        disabled = make_model()
        disabled.directional_weight = 0.0
        disabled.forward = types.MethodType(fake_forward, disabled)

        def forbidden_builder(this, supplied_interaction):
            raise AssertionError('Disabled directional loss must not sample.')

        disabled._build_directional_candidates = types.MethodType(
            forbidden_builder,
            disabled,
        )
        disabled_loss = disabled.calculate_loss(interaction)
        self.assertTrue(torch.allclose(disabled_loss, ranking, atol=1e-7))

    def test_dynamic_model_import(self):
        self.assertIs(
            get_model('MASKED_GLORIA_DIRECTIONAL'),
            MASKED_GLORIA_DIRECTIONAL,
        )


if __name__ == '__main__':
    unittest.main()
