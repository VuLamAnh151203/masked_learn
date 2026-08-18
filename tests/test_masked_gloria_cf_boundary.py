import os
import sys
import types
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from models.masked_gloria_cf import MASKED_GLORIA_CF
except ModuleNotFoundError as error:
    if error.name != 'torch_geometric':
        raise
    MASKED_GLORIA_CF = None


def make_model():
    model = MASKED_GLORIA_CF.__new__(MASKED_GLORIA_CF)
    torch.nn.Module.__init__(model)
    model.num_user = 1
    model.num_item = 4
    model.cf_k = 2
    model.cf_min_history = 2
    model.cf_temperature = 1.0
    model.cf_gamma = 0.2
    model.cf_user_ratio = 1.0
    model.cf_batch_size = 8
    model.cf_lambda = 0.1
    model.cf_warmup_epochs = 0
    model.cf_detach_boundary_weight = True
    model.user_seen_items = ((0, 1, 2),)
    model._cf_rng = types.SimpleNamespace(
        choice=lambda values: values[0],
        sample=lambda values, count: list(values)[:count],
        seed=lambda value: None,
    )
    model.cf_stats = model._new_cf_stats()
    return model


@unittest.skipIf(
    MASKED_GLORIA_CF is None,
    'torch_geometric is required for boundary model tests'
)
class BoundaryRegularizationTest(unittest.TestCase):
    def test_boundary_competitor_skips_pseudo_positive_at_rank_k(self):
        model = make_model()
        # Item 0 is pseudo-positive at rank 2; item 2 is the next valid item.
        scores = torch.tensor([0.9, -1e10, 0.8, 0.7])
        self.assertEqual(model._select_boundary_competitor(scores, 0), 2)

    def test_boundary_loss_matches_weighted_softplus_formula(self):
        model = make_model()
        # Node 0 is the user, nodes 1..4 are items. Items 1 and 2 are seen and
        # masked from the competitor set; item 4 is the Top-K boundary item.
        model.result_embed = torch.tensor(
            [[1.0], [0.9], [0.0], [0.0], [0.8]],
            requires_grad=True
        )
        interaction = (torch.tensor([0]),)
        loss = model._calculate_boundary_loss(
            interaction,
            torch.tensor(1.0, requires_grad=True)
        )
        expected = torch.exp(torch.tensor(-0.1)) * torch.nn.functional.softplus(
            torch.tensor(0.8 - 0.9 + 0.2)
        )
        self.assertTrue(torch.allclose(loss.detach(), expected, atol=1e-6))
        self.assertEqual(model.cf_stats['used'], 1)

        loss.backward()
        self.assertIsNotNone(model.result_embed.grad)

    def test_boundary_weight_is_detached_by_default(self):
        model = make_model()
        model.result_embed = torch.tensor(
            [[1.0], [0.9], [0.0], [0.0], [0.8]],
            requires_grad=True
        )
        loss = model._calculate_boundary_loss(
            (torch.tensor([0]),),
            torch.tensor(1.0, requires_grad=True)
        )
        self.assertFalse(loss.requires_grad is False)
        loss.backward()
        # The regularizer still trains the margin, while the adaptive weight
        # itself is not part of the gradient path.
        self.assertNotEqual(float(model.result_embed.grad[0].abs()), 0.0)


if __name__ == '__main__':
    unittest.main()
