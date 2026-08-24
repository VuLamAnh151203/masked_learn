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
    model.last_branch_kl_candidate_count = None
    model.last_branch_kl_user_count = None
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
        self.candidates = torch.tensor(
            [[0, 2, 3], [2, 0, 5], [4, 1, 3]]
        )

    def test_matches_manual_kl(self):
        model = make_model()
        actual = model.calculate_branch_kl(self.users, self.candidates)

        full_logits = torch.sum(
            model.full_user_view[self.users, None, :]
            * model.full_item_view[self.candidates],
            dim=-1,
        ) / model.branch_kl_temperature
        masked_logits = torch.sum(
            model.masked_user_view[self.users, None, :]
            * model.masked_item_view[self.candidates],
            dim=-1,
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

        loss = model.calculate_branch_kl(self.users, self.candidates)

        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1e-7))

    def test_extreme_logits_remain_finite(self):
        model = make_model()
        model.full_user_view = model.full_user_view * 1e6
        model.masked_user_view = model.masked_user_view * -1e6

        loss = model.calculate_branch_kl(self.users, self.candidates)

        self.assertTrue(torch.isfinite(loss))

    def test_kl_updates_both_branches(self):
        model = make_model()
        loss = model.calculate_branch_kl(self.users, self.candidates)
        loss.backward()

        for view in (
            model.full_user_view,
            model.full_item_view,
            model.masked_user_view,
            model.masked_item_view,
        ):
            self.assertIsNotNone(view.grad)
            self.assertGreater(view.grad.abs().sum().item(), 0.0)

    def test_shared_candidates_and_total_loss(self):
        model = make_model()
        interaction = torch.tensor(
            [[0, 1, 2], [0, 2, 4], [3, 5, 1]]
        )
        fixed_users = torch.tensor([0, 1, 2])
        fixed_candidates = self.candidates.clone()
        calls = {'builder': 0, 'mipd_candidates': [], 'kl_candidates': []}

        def fake_forward(this, supplied_interaction):
            return (
                torch.tensor([1.2, 0.8, 1.5], requires_grad=True),
                torch.tensor([0.1, 0.4, 0.2], requires_grad=True),
            )

        def fake_builder(this, supplied_interaction):
            calls['builder'] += 1
            return fixed_users, fixed_candidates

        def fake_mipd(this, users, candidates, permutation=None):
            calls['mipd_candidates'].append(candidates)
            return this.masked_user_view[users].sum() * 0.0 - 0.2

        def fake_kl(this, users, candidates):
            calls['kl_candidates'].append(candidates)
            return this.masked_user_view[users].sum() * 0.0 + 0.3

        model.forward = types.MethodType(fake_forward, model)
        model._build_mipd_candidates = types.MethodType(fake_builder, model)
        model.calculate_listwise_mipd = types.MethodType(fake_mipd, model)
        model.calculate_branch_kl = types.MethodType(fake_kl, model)

        actual = model.calculate_loss(interaction)
        ranking = F.softplus(
            -(
                torch.tensor([1.2, 0.8, 1.5])
                - torch.tensor([0.1, 0.4, 0.2])
            )
        ).mean() / math.log(2.0)
        expected = ranking + model.mipd_weight * -0.2 + 0.25 * 0.3

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertEqual(calls['builder'], 1)
        self.assertEqual(len(calls['mipd_candidates']), 2)
        self.assertEqual(len(calls['kl_candidates']), 1)
        for candidates in (
            calls['mipd_candidates'] + calls['kl_candidates']
        ):
            self.assertIs(candidates, fixed_candidates)

    def test_zero_kl_weight_removes_kl_term(self):
        model = make_model()
        model.branch_kl_weight = 0.0
        users = self.users
        candidates = self.candidates

        kl_loss = model._calculate_branch_kl_from_candidates(
            users,
            candidates,
        )

        self.assertEqual(kl_loss.item(), 0.0)
        self.assertEqual(model.last_branch_kl_user_count, 0)
        self.assertEqual(model.last_branch_kl_candidate_count, 0)

    def test_dynamic_model_import(self):
        self.assertIs(
            get_model('MASKED_GLORIA_MIPD_KL'),
            MASKED_GLORIA_MIPD_KL,
        )


if __name__ == '__main__':
    unittest.main()
