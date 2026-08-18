import os
import sys
import types
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from models.masked_gloria_cf2 import MASKED_GLORIA_CF2
except ModuleNotFoundError as error:
    if error.name != 'torch_geometric':
        raise
    MASKED_GLORIA_CF2 = None


def make_model():
    model = MASKED_GLORIA_CF2.__new__(MASKED_GLORIA_CF2)
    torch.nn.Module.__init__(model)
    model.num_user = 1
    model.num_item = 3
    model.cf2_lambda = 0.1
    model.cf2_temperature = 2.0
    model.cf2_user_ratio = 1.0
    model.cf2_batch_size = 8
    model.cf2_pair_count = 32
    model.cf2_min_history = 2
    model.cf2_similarity_eps = 1e-6
    model.user_to_edge_ids = ((0, 1, 2),)
    model.forward_edge_users = torch.tensor([0, 0, 0])
    model.forward_edge_items = torch.tensor([0, 1, 2])
    model.mask_logits = torch.nn.Parameter(torch.zeros(3))
    model.result_embed = torch.ones(4, 2, requires_grad=True)
    model.mask_rep = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
        ],
        requires_grad=True
    )
    model.cf2_rng = types.SimpleNamespace()
    model._cf2_rng = types.SimpleNamespace(
        sample=lambda values, count: list(values)[:count]
    )
    model.cf2_stats = model._new_cf2_stats()
    return model


@unittest.skipIf(
    MASKED_GLORIA_CF2 is None,
    'torch_geometric is required for CF2 tests'
)
class MaskRepresentationRegularizationTest(unittest.TestCase):
    def test_similarity_is_detached_but_mask_weights_receive_gradient(self):
        model = make_model()
        loss = model._calculate_mask_relation_loss(
            (torch.tensor([0]),),
            torch.tensor(1.0, requires_grad=True)
        )
        self.assertGreater(model.cf2_stats['pairs'], 0)
        loss.backward()
        self.assertIsNotNone(model.mask_logits.grad)
        self.assertGreater(float(model.mask_logits.grad.abs().sum()), 0.0)
        self.assertIsNone(model.mask_rep.grad)

    def test_pair_loss_is_zero_graph_when_no_similarity_gap(self):
        model = make_model()
        model.mask_rep = torch.ones(4, 2, requires_grad=True)
        loss = model._calculate_mask_relation_loss(
            (torch.tensor([0]),),
            torch.tensor(1.0, requires_grad=True)
        )
        self.assertEqual(model.cf2_stats['pairs'], 0)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_configuration_rejects_invalid_temperature(self):
        model = make_model()
        model.cf2_temperature = 0.0
        with self.assertRaises(ValueError):
            model._validate_cf2_config()


if __name__ == '__main__':
    unittest.main()
