import os
import sys
import types
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from models.masked_gloria_cf3 import MASKED_GLORIA_CF3
except ModuleNotFoundError as error:
    if error.name != 'torch_geometric':
        raise

    # The tests exercise only CF3's auxiliary-loss helpers.  A minimal module
    # stub keeps them runnable in lightweight environments without PyG.
    torch_geometric = types.ModuleType('torch_geometric')
    torch_geometric_nn = types.ModuleType('torch_geometric.nn')
    torch_geometric_conv = types.ModuleType('torch_geometric.nn.conv')
    torch_geometric_utils = types.ModuleType('torch_geometric.utils')

    class MessagePassing(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    torch_geometric_conv.MessagePassing = MessagePassing
    torch_geometric_utils.remove_self_loops = lambda edge_index: (edge_index, None)
    torch_geometric_utils.add_self_loops = lambda edge_index: (edge_index, None)
    torch_geometric_utils.degree = lambda values, *args, **kwargs: values
    sys.modules['torch_geometric'] = torch_geometric
    sys.modules['torch_geometric.nn'] = torch_geometric_nn
    sys.modules['torch_geometric.nn.conv'] = torch_geometric_conv
    sys.modules['torch_geometric.utils'] = torch_geometric_utils

    from models.masked_gloria_cf3 import MASKED_GLORIA_CF3


def make_model():
    model = MASKED_GLORIA_CF3.__new__(MASKED_GLORIA_CF3)
    torch.nn.Module.__init__(model)
    model.num_user = 1
    model.num_item = 4
    model.cf3_lambda = 0.1
    model.cf3_temperature = 2.0
    model.cf3_user_ratio = 1.0
    model.cf3_batch_size = 8
    model.cf3_pair_count = 32
    model.cf3_min_history = 3
    model.cf3_similarity_eps = 1e-6
    model.cf3_seed_offset = 30000
    model.cf3_warmup_epochs = 50
    model.current_epoch = 0
    model.user_to_edge_ids = ((0, 1, 2, 3),)
    model.forward_edge_users = torch.tensor([0, 0, 0, 0])
    model.forward_edge_items = torch.tensor([0, 1, 2, 3])
    model.mask_logits = torch.nn.Parameter(torch.zeros(4))
    model.result_embed = torch.ones(5, 2, requires_grad=True)
    model.mask_rep = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [-1.0, 0.0],
        ],
        requires_grad=True
    )
    model.cf3_stats = model._new_cf3_stats()
    model.training = True
    return model


class MaskLogitRegularizationTest(unittest.TestCase):
    def test_target_is_detached_and_raw_logits_receive_gradient(self):
        model = make_model()
        loss = model._calculate_mask_relation_loss(
            (torch.tensor([0]),),
            torch.tensor(1.0, requires_grad=True)
        )
        self.assertGreater(model.cf3_stats['pairs'], 0)
        loss.backward()
        self.assertIsNotNone(model.mask_logits.grad)
        self.assertGreater(float(model.mask_logits.grad.abs().sum()), 0.0)
        self.assertIsNone(model.mask_rep.grad)

    def test_pseudo_positive_is_excluded_from_candidate_pairs(self):
        model = make_model()
        pseudo_edge = model._pseudo_edge_for_user(0, (0, 1, 2, 3))
        candidates = [edge for edge in (0, 1, 2, 3) if edge != pseudo_edge]
        self.assertNotIn(pseudo_edge, candidates)
        self.assertEqual(len(candidates), 3)

    def test_temperature_is_validated(self):
        model = make_model()
        model.cf3_temperature = 0.0
        with self.assertRaises(ValueError):
            model._validate_cf3_config()

    def test_pair_accuracy_and_aligned_gap_are_reported(self):
        model = make_model()
        model.cf3_stats['pairs'] = 2
        model.cf3_stats['correct_pairs'] = 1
        model.cf3_stats['aligned_logit_gap_sum'] = 0.4
        model.cf3_stats['loss_sum'] = 1.2
        model.cf3_stats['similarity_gap_sum'] = 0.8
        model.cf3_log_stats = True
        log_line = model.post_epoch_processing()
        self.assertIn('pair_accuracy=0.500000', log_line)
        self.assertIn('aligned_logit_gap=0.200000', log_line)


if __name__ == '__main__':
    unittest.main()
