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
    geometric_utils.add_self_loops = lambda edge_index, **kwargs: (edge_index, None)
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

from models.masked_gloria_mipd import MASKED_GLORIA_MIPD


def make_model():
    model = MASKED_GLORIA_MIPD.__new__(MASKED_GLORIA_MIPD)
    torch.nn.Module.__init__(model)
    model.num_user = 3
    model.num_item = 6
    model.mipd_weight = 0.5
    model.mipd_num_samples = 1
    model.mipd_num_negatives = 2
    model.mipd_temperature = 0.7
    model.mipd_negative_sampling = 'random'
    model.mipd_hard_pool_size = 4
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
    model.mipd_seen_items = (
        frozenset((0, 1)),
        frozenset((2, 3)),
        frozenset((4,)),
    )
    return model


class ListwiseMipdTest(unittest.TestCase):
    def test_matches_manual_jsd(self):
        model = make_model()
        users = torch.tensor([0, 1, 2])
        candidates = torch.tensor(
            [[0, 2, 3], [2, 0, 5], [4, 1, 3]]
        )
        permutation = torch.tensor([1, 2, 0])

        actual_loss = model.calculate_listwise_mipd(
            users, candidates, permutation=permutation
        )

        full_scores = torch.sum(
            model.full_user_view[users, None, :]
            * model.full_item_view[candidates],
            dim=-1,
        ).detach()
        original_mask_scores = torch.sum(
            model.masked_user_view[users, None, :]
            * model.masked_item_view[candidates],
            dim=-1,
        )
        permuted_mask_scores = torch.sum(
            model.masked_user_view[users[permutation], None, :]
            * model.masked_item_view[candidates],
            dim=-1,
        )
        p = F.softmax(
            (full_scores + original_mask_scores) / model.mipd_temperature,
            dim=1,
        )
        q = F.softmax(
            (full_scores + permuted_mask_scores) / model.mipd_temperature,
            dim=1,
        )
        mixture = 0.5 * (p + q)
        expected_jsd = 0.5 * (
            torch.sum(p * (p.log() - mixture.log()), dim=1).mean()
            + torch.sum(q * (q.log() - mixture.log()), dim=1).mean()
        )
        self.assertTrue(torch.allclose(actual_loss, -expected_jsd, atol=1e-7))

    def test_full_view_is_detached_and_mask_view_gets_gradient(self):
        model = make_model()
        users = torch.tensor([0, 1, 2])
        candidates = torch.tensor(
            [[0, 2, 3], [2, 0, 5], [4, 1, 3]]
        )
        loss = model.calculate_listwise_mipd(
            users,
            candidates,
            permutation=torch.tensor([1, 2, 0]),
        )
        loss.backward()

        self.assertIsNone(model.full_user_view.grad)
        self.assertIsNone(model.full_item_view.grad)
        self.assertIsNotNone(model.masked_user_view.grad)
        self.assertIsNotNone(model.masked_item_view.grad)
        self.assertGreater(model.masked_user_view.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.masked_item_view.grad.abs().sum().item(), 0.0)

    def test_candidates_are_unique_unseen_and_one_row_per_user(self):
        model = make_model()
        # User 0 appears twice; only the first positive is used by MIPD.
        interaction = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 1, 2, 4],
                [2, 3, 5, 0],
            ]
        )
        torch.manual_seed(123)
        users, candidates = model._build_mipd_candidates(interaction)

        self.assertEqual(users.tolist(), [0, 1, 2])
        self.assertEqual(candidates.shape, (3, 3))
        self.assertEqual(candidates[:, 0].tolist(), [0, 2, 4])
        for user_id, row in zip(users.tolist(), candidates.tolist()):
            negatives = row[1:]
            self.assertEqual(len(negatives), len(set(negatives)))
            self.assertTrue(
                set(negatives).isdisjoint(model.mipd_seen_items[user_id])
            )
            self.assertNotIn(row[0], negatives)

    def test_full_hard_mode_selects_highest_full_score_unseen_items(self):
        model = make_model()
        model.mipd_negative_sampling = 'full_hard'
        model.mipd_hard_pool_size = 6
        # Each of these users has exactly four unseen items, so the random pool
        # contains the complete eligible set before Full-score top-k selection.
        interaction = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [3, 4],
            ]
        )

        torch.manual_seed(17)
        users, candidates = model._build_mipd_candidates(interaction)

        self.assertEqual(users.tolist(), [0, 1])
        # User 0: unseen {2,3,4,5}; Full scores {1,-1,.4,.3}.
        # User 1: unseen {0,1,4,5}; Full scores {0,1,-.8,.6}.
        self.assertEqual(candidates.tolist(), [[0, 2, 4], [2, 1, 5]])
        for user_id, row in zip(users.tolist(), candidates.tolist()):
            negatives = row[1:]
            self.assertEqual(len(negatives), len(set(negatives)))
            self.assertTrue(
                set(negatives).isdisjoint(model.mipd_seen_items[user_id])
            )

    def test_derangement_has_no_fixed_points(self):
        torch.manual_seed(7)
        permutation = MASKED_GLORIA_MIPD._sample_derangement(20, 'cpu')
        self.assertEqual(sorted(permutation.tolist()), list(range(20)))
        self.assertTrue(torch.all(permutation != torch.arange(20)))

    def test_rejects_fixed_point_permutation(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, 'fixed points'):
            model.calculate_listwise_mipd(
                torch.tensor([0, 1]),
                torch.tensor([[0, 2], [2, 5]]),
                permutation=torch.tensor([0, 1]),
            )


if __name__ == '__main__':
    unittest.main()
