import math
import sys
import types
import unittest
from pathlib import Path

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

from models.masked_gloria_mipd_kl import MASKED_GLORIA_MIPD_KL
from utils.utils import get_model


def make_model():
    model = MASKED_GLORIA_MIPD_KL.__new__(MASKED_GLORIA_MIPD_KL)
    torch.nn.Module.__init__(model)
    model.num_user = 3
    model.num_item = 6
    model.n_users = 3
    model.n_items = 6
    model.mipd_weight = 0.5
    model.mipd_num_samples = 2
    model.mipd_num_negatives = 2
    model.mipd_temperature = 0.7
    model.branch_kl_weight = 0.25
    model.branch_kl_temperature = 0.8
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
    model.last_ranking_loss = None
    model.last_mipd_loss = None
    model.last_mipd_jsd = None
    model.last_mipd_candidate_count = None
    model.last_mipd_user_count = None
    model.last_branch_kl_loss = None
    model.last_branch_kl_item_count = None
    model.last_branch_kl_user_count = None
    model.last_mipd_item_count = None
    model.mipd_epoch_jsd_sum = 0.0
    model.mipd_epoch_user_count = 0
    model.mipd_epoch_batch_count = 0
    model.branch_kl_epoch_loss_sum = 0.0
    model.branch_kl_epoch_user_count = 0
    model.branch_kl_epoch_batch_count = 0
    model.ranking_epoch_loss_sum = 0.0
    model.ranking_epoch_example_count = 0
    return model


class BranchKlTest(unittest.TestCase):
    def setUp(self):
        self.users = torch.tensor([0, 1, 2])

    def test_matches_manual_kl(self):
        model = make_model()
        actual = model.calculate_branch_kl(self.users)

        full_logits = torch.matmul(
            model.full_user_view[self.users],
            model.full_item_view.t(),
        ) / model.branch_kl_temperature
        masked_logits = torch.matmul(
            model.masked_user_view[self.users],
            model.masked_item_view.t(),
        ) / model.branch_kl_temperature
        log_p_full = F.log_softmax(full_logits, dim=1)
        log_p_masked = F.log_softmax(masked_logits, dim=1)
        expected = torch.sum(
            log_p_masked.exp() * (log_p_masked - log_p_full),
            dim=1,
        ).mean()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_identical_branch_logits_have_zero_kl(self):
        model = make_model()
        model.masked_user_view = model.full_user_view.detach().clone()
        model.masked_item_view = model.full_item_view.detach().clone()

        loss = model.calculate_branch_kl(self.users)

        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1e-7))

    def test_extreme_logits_remain_finite(self):
        model = make_model()
        model.full_user_view = model.full_user_view * 1e6
        model.masked_user_view = model.masked_user_view * -1e6

        loss = model.calculate_branch_kl(self.users)

        self.assertTrue(torch.isfinite(loss))

    def test_kl_updates_both_branches(self):
        model = make_model()
        loss = model.calculate_branch_kl(self.users)
        loss.backward()

        for view in (
            model.full_user_view,
            model.full_item_view,
            model.masked_user_view,
            model.masked_item_view,
        ):
            self.assertIsNotNone(view.grad)
            self.assertGreater(view.grad.abs().sum().item(), 0.0)

    def test_catalog_mipd_matches_manual_jsd(self):
        model = make_model()
        permutation = torch.tensor([1, 2, 0])
        full_scores, masked_scores = model._calculate_catalog_scores(
            self.users
        )

        actual = model.calculate_catalog_mipd(
            self.users,
            full_scores=full_scores,
            masked_scores=masked_scores,
            permutation=permutation,
        )

        original_logits = (
            full_scores.detach() + masked_scores
        ) / model.mipd_temperature
        permuted_masked_scores = torch.matmul(
            model.masked_user_view[self.users][permutation],
            model.masked_item_view.t(),
        )
        permuted_logits = (
            full_scores.detach() + permuted_masked_scores
        ) / model.mipd_temperature
        p = F.softmax(original_logits, dim=1)
        q = F.softmax(permuted_logits, dim=1)
        mixture = 0.5 * (p + q)
        expected_jsd = 0.5 * (
            torch.sum(p * (p.log() - mixture.log()), dim=1).mean()
            + torch.sum(q * (q.log() - mixture.log()), dim=1).mean()
        )

        self.assertTrue(torch.allclose(actual, -expected_jsd, atol=1e-7))

    def test_catalog_mipd_detaches_full_and_updates_masked(self):
        model = make_model()
        loss = model.calculate_catalog_mipd(
            self.users,
            permutation=torch.tensor([1, 2, 0]),
        )
        loss.backward()

        self.assertIsNone(model.full_user_view.grad)
        self.assertIsNone(model.full_item_view.grad)
        self.assertIsNotNone(model.masked_user_view.grad)
        self.assertIsNotNone(model.masked_item_view.grad)
        self.assertGreater(model.masked_user_view.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.masked_item_view.grad.abs().sum().item(), 0.0)

    def test_catalog_scores_are_reused_and_total_loss_is_correct(self):
        model = make_model()
        interaction = torch.tensor(
            [[0, 1, 2], [0, 2, 4], [3, 5, 1]]
        )
        fixed_full_scores = torch.randn(3, model.num_item)
        fixed_masked_scores = torch.randn(3, model.num_item)
        calls = {'scores': 0, 'mipd': [], 'kl': []}

        def fake_forward(this, supplied_interaction):
            return (
                torch.tensor([1.2, 0.8, 1.5], requires_grad=True),
                torch.tensor([0.1, 0.4, 0.2], requires_grad=True),
            )

        def fake_scores(this, users):
            calls['scores'] += 1
            return fixed_full_scores, fixed_masked_scores

        def fake_mipd(this, users, full_scores, masked_scores):
            calls['mipd'].append((full_scores, masked_scores))
            this.last_mipd_user_count = int(users.numel())
            this.last_mipd_candidate_count = this.num_item
            this.last_mipd_item_count = this.num_item
            loss = this.masked_user_view[users].sum() * 0.0 - 0.2
            return loss, (-loss).detach()

        def fake_kl(this, users, full_scores, masked_scores):
            calls['kl'].append((full_scores, masked_scores))
            this.last_branch_kl_user_count = int(users.numel())
            this.last_branch_kl_item_count = this.num_item
            return this.masked_user_view[users].sum() * 0.0 + 0.3

        def forbidden_builder(this, supplied_interaction):
            raise AssertionError('Catalog MIPD must not sample negatives.')

        model.forward = types.MethodType(fake_forward, model)
        model._calculate_catalog_scores = types.MethodType(fake_scores, model)
        model._calculate_catalog_mipd_from_scores = types.MethodType(
            fake_mipd,
            model,
        )
        model._calculate_branch_kl_from_scores = types.MethodType(
            fake_kl,
            model,
        )
        model._build_mipd_candidates = types.MethodType(
            forbidden_builder,
            model,
        )

        actual = model.calculate_loss(interaction)
        ranking = F.softplus(
            -(
                torch.tensor([1.2, 0.8, 1.5])
                - torch.tensor([0.1, 0.4, 0.2])
            )
        ).mean() / math.log(2.0)
        expected = ranking + model.mipd_weight * -0.2 + 0.25 * 0.3

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertEqual(calls['scores'], 1)
        self.assertEqual(len(calls['mipd']), 1)
        self.assertEqual(len(calls['kl']), 1)
        for full_scores, masked_scores in calls['mipd'] + calls['kl']:
            self.assertIs(full_scores, fixed_full_scores)
            self.assertIs(masked_scores, fixed_masked_scores)

    def test_kl_only_does_not_build_mipd_candidates(self):
        model = make_model()
        model.mipd_weight = 0.0
        interaction = torch.tensor(
            [[0, 0, 1, 2], [0, 1, 2, 4], [3, 4, 5, 0]]
        )

        def fake_forward(this, supplied_interaction):
            batch_size = supplied_interaction.size(1)
            return torch.ones(batch_size), torch.zeros(batch_size)

        def forbidden_builder(this, supplied_interaction):
            raise AssertionError('KL must not sample MIPD candidates.')

        model.forward = types.MethodType(fake_forward, model)
        model._build_mipd_candidates = types.MethodType(
            forbidden_builder,
            model,
        )

        loss = model.calculate_loss(interaction)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.last_branch_kl_user_count, 3)
        self.assertEqual(model.last_branch_kl_item_count, model.num_item)

    def test_zero_kl_weight_removes_kl_term(self):
        model = make_model()
        model.branch_kl_weight = 0.0
        users = self.users
        full_scores, masked_scores = model._calculate_catalog_scores(users)
        kl_loss = model._calculate_branch_kl_from_scores(
            users,
            full_scores,
            masked_scores,
        )

        self.assertEqual(kl_loss.item(), 0.0)
        self.assertEqual(model.last_branch_kl_user_count, 0)
        self.assertEqual(model.last_branch_kl_item_count, 0)

    def test_dynamic_model_import(self):
        self.assertIs(
            get_model('MASKED_GLORIA_MIPD_KL'),
            MASKED_GLORIA_MIPD_KL,
        )


if __name__ == '__main__':
    unittest.main()
