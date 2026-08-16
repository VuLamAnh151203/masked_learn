import os
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.trainer import CounterfactualCalibrationTrainer, Trainer
from models.masked_gloria_ex3 import MASKED_GLORIA_EX3
from utils.utils import get_model, get_trainer


def make_mask_only_model():
    model = MASKED_GLORIA_EX3.__new__(MASKED_GLORIA_EX3)
    torch.nn.Module.__init__(model)
    model.num_user = 3
    model.n_users = 3
    base_mask = torch.tensor([0.8, 0.6, 0.2])
    model.user_mask_logits = torch.nn.Parameter(torch.logit(base_mask))
    model.gamma_grid_values = [0.0, 0.5, 1.0]
    model.register_buffer('gamma_grid', torch.tensor(model.gamma_grid_values))
    model.user_feat = torch.arange(3 * 384, dtype=torch.float32).reshape(3, 384)
    model.register_buffer(
        'normalized_log_train_degree',
        torch.tensor([-1.0, 0.0, 1.0])
    )
    model.calibrator = torch.nn.Sequential(
        torch.nn.Linear(386, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 3)
    )
    return model


class CounterfactualCalibrationTest(unittest.TestCase):
    def test_effective_mask_endpoints_and_heterogeneous_gamma(self):
        model = make_mask_only_model()
        base_mask = model.get_user_mask()

        self.assertTrue(
            torch.equal(
                model.get_effective_user_mask(0.0),
                torch.ones_like(base_mask)
            )
        )
        self.assertTrue(
            torch.equal(model.get_effective_user_mask(1.0), base_mask)
        )

        gamma = torch.tensor([0.0, 0.5, 1.0])
        expected = 1.0 - gamma * (1.0 - base_mask)
        self.assertTrue(
            torch.allclose(model.get_effective_user_mask(gamma), expected)
        )

    def test_calibrator_probabilities_and_expected_strength(self):
        model = make_mask_only_model()
        model.eval()
        features = model.get_calibrator_features()
        probabilities = model.get_gamma_probabilities()
        gamma = model.get_user_gamma()

        self.assertEqual(tuple(features.shape), (3, 386))
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(3),
                atol=1e-6
            )
        )
        self.assertTrue(torch.all(gamma >= 0.0))
        self.assertTrue(torch.all(gamma <= 1.0))
        self.assertTrue(
            torch.allclose(
                gamma,
                (probabilities * model.gamma_grid).sum(dim=1)
            )
        )

    def test_legacy_checkpoint_mapping_and_base_freezing(self):
        model = MASKED_GLORIA_EX3.__new__(MASKED_GLORIA_EX3)
        torch.nn.Module.__init__(model)
        model.user_mask_logits = torch.nn.Parameter(torch.zeros(2))
        model.full_gcn = torch.nn.Module()
        model.full_gcn.preference = torch.nn.Parameter(torch.zeros(2, 2))
        model.mask_gcn = torch.nn.Module()
        model.mask_gcn.preference = torch.nn.Parameter(torch.zeros(2, 2))
        model.id_embedding_full = torch.nn.Embedding(3, 2)
        model.id_embedding_masked = torch.nn.Embedding(3, 2)
        model.mlp_item = torch.nn.Linear(4, 2, bias=False)
        model.mlp_user = torch.nn.Linear(4, 2, bias=False)
        model.calibrator = torch.nn.Linear(6, 2)

        expected_mask = torch.tensor([1.0, -1.0])
        checkpoint_state = {
            'user_mask_logits': expected_mask,
            'full_preference': torch.full((2, 2), 2.0),
            'mask_preference': torch.full((2, 2), 3.0),
            'id_embedding_full.weight': torch.full((3, 2), 4.0),
            'id_embedding_masked.weight': torch.full((3, 2), 5.0),
            'mlp_item.weight': torch.full((2, 4), 6.0),
            'mlp_user.weight': torch.full((2, 4), 7.0),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = os.path.join(temporary_directory, 'base.pth')
            torch.save({'state_dict': checkpoint_state}, checkpoint_path)
            model.cf_base_checkpoint = checkpoint_path
            model.cf_base_checkpoint_sha256 = 'test-sha256'
            model._load_and_freeze_base_checkpoint()

        self.assertTrue(torch.equal(model.user_mask_logits, expected_mask))
        self.assertTrue(
            torch.equal(
                model.full_gcn.preference,
                checkpoint_state['full_preference']
            )
        )
        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(trainable, ['calibrator.weight', 'calibrator.bias'])

    def test_fixed_negatives_are_deterministic_and_exclude_seen_items(self):
        train = pd.DataFrame({
            'userID': [0, 0, 1],
            'itemID': [0, 1, 2],
        })
        validation = pd.DataFrame({
            'userID': [0, 0, 1],
            'itemID': [2, 3, 3],
        })

        first = MASKED_GLORIA_EX3._sample_fixed_negatives(
            train, validation, 'userID', 'itemID', 6, 2, 17
        )
        second = MASKED_GLORIA_EX3._sample_fixed_negatives(
            train, validation, 'userID', 'itemID', 6, 2, 17
        )
        for first_array, second_array in zip(first, second):
            self.assertTrue(np.array_equal(first_array, second_array))

        users, _, negatives = first
        forbidden = {0: {0, 1, 2, 3}, 1: {2, 3}}
        for user, negative in zip(users, negatives):
            self.assertNotIn(int(negative), forbidden[int(user)])

    def test_per_user_loss_matrix_uses_one_world_per_gamma(self):
        model = make_mask_only_model()
        model.num_user = 2
        model.gamma_grid_values = [0.0, 1.0]
        representations = {
            0.0: torch.tensor([[1.0], [1.0], [2.0], [0.0], [-1.0]]),
            1.0: torch.tensor([[1.0], [1.0], [0.0], [2.0], [1.0]]),
        }
        calls = []

        def fake_compute_result_embedding(self, gamma):
            calls.append(float(gamma))
            return representations[float(gamma)]

        def fake_invalidate(self):
            return None

        model.compute_result_embedding = types.MethodType(
            fake_compute_result_embedding,
            model
        )
        model.invalidate_inference_cache = types.MethodType(
            fake_invalidate,
            model
        )
        model.cf_score_batch_size = 2
        pair_users = np.asarray([0, 0, 1], dtype=np.int64)
        positives = np.asarray([0, 0, 1], dtype=np.int64)
        negatives = np.asarray([1, 2, 0], dtype=np.int64)
        losses = model._counterfactual_user_losses(
            pair_users,
            positives,
            negatives,
            np.asarray([0, 1], dtype=np.int64)
        )

        expected = torch.tensor([
            [
                (torch.nn.functional.softplus(torch.tensor(-2.0))
                 + torch.nn.functional.softplus(torch.tensor(-3.0))) / 2.0,
                (torch.nn.functional.softplus(torch.tensor(2.0))
                 + torch.nn.functional.softplus(torch.tensor(1.0))) / 2.0,
            ],
            [
                torch.nn.functional.softplus(torch.tensor(2.0)),
                torch.nn.functional.softplus(torch.tensor(-2.0)),
            ],
        ])
        self.assertEqual(calls, [0.0, 1.0])
        self.assertTrue(torch.allclose(losses, expected))

    def test_tie_breaking_prefers_smaller_gamma(self):
        model = make_mask_only_model()
        model.cf_label_tolerance = 1e-6
        losses = torch.tensor([
            [0.5, 0.5000005, 0.8],
            [0.5, 0.4999995, 0.8],
            [0.5, 0.4990, 0.8],
        ])
        labels = model._select_counterfactual_labels(losses)
        self.assertTrue(torch.equal(labels, torch.tensor([0, 0, 1])))

    def test_stratified_split_is_disjoint_and_reproducible(self):
        model = make_mask_only_model()
        model.cf_negative_seed = 9
        model.cf_calibration_train_ratio = 0.8
        users = torch.arange(11)
        labels = torch.tensor([0] * 5 + [1] * 5 + [2])

        first = model._stratified_user_split(users, labels)
        second = model._stratified_user_split(users, labels)
        train_users = set(first['train_user_ids'].tolist())
        validation_users = set(first['validation_user_ids'].tolist())

        self.assertTrue(train_users.isdisjoint(validation_users))
        self.assertEqual(train_users | validation_users, set(users.tolist()))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_label_artifact_is_cached_with_reproducibility_metadata(self):
        model = make_mask_only_model()
        model.config = {'dataset': 'tiny'}
        model.num_item = 8
        model.cf_base_checkpoint_sha256 = 'checkpoint-id'
        model.cf_negatives_per_positive = 1
        model.cf_negative_seed = 11
        model.cf_label_tolerance = 1e-6
        model.cf_calibration_train_ratio = 0.5
        model.calibration_metadata = {}

        train_frame = pd.DataFrame({
            'userID': [0, 1, 2, 3],
            'itemID': [0, 1, 2, 3],
        })
        validation_frame = pd.DataFrame({
            'userID': [0, 1, 2, 3],
            'itemID': [4, 5, 6, 7],
        })
        train_data = types.SimpleNamespace(
            dataset=types.SimpleNamespace(
                df=train_frame,
                uid_field='userID',
                iid_field='itemID'
            )
        )
        validation_data = types.SimpleNamespace(
            dataset=types.SimpleNamespace(
                df=validation_frame,
                uid_field='userID',
                iid_field='itemID'
            )
        )
        generated = {'count': 0}

        def fake_losses(self, pair_users, positives, negatives, labeled_users):
            generated['count'] += 1
            return torch.tensor([
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.2, 0.1, 0.3],
                [0.2, 0.1, 0.3],
            ])

        model._counterfactual_user_losses = types.MethodType(fake_losses, model)
        with tempfile.TemporaryDirectory() as temporary_directory:
            model.cf_label_cache_dir = temporary_directory
            first = model.prepare_counterfactual_calibration(
                train_data,
                validation_data
            )
            second = model.prepare_counterfactual_calibration(
                train_data,
                validation_data
            )

        self.assertEqual(generated['count'], 1)
        self.assertTrue(torch.equal(first['labels'], second['labels']))
        self.assertEqual(first['base_checkpoint_sha256'], 'checkpoint-id')
        self.assertEqual(first['negative_seed'], 11)
        self.assertEqual(first['split_seed'], 11)
        self.assertTrue(
            torch.equal(first['gamma_grid'], torch.tensor([0.0, 0.5, 1.0]))
        )

    def test_dynamic_model_and_trainer_routing(self):
        self.assertIs(get_model('MASKED_GLORIA_EX3'), MASKED_GLORIA_EX3)
        self.assertIs(
            get_trainer('MASKED_GLORIA_EX3'),
            CounterfactualCalibrationTrainer
        )
        self.assertIs(get_trainer('MASKED_GLORIA_EX'), Trainer)
        self.assertIs(get_trainer(), Trainer)


if __name__ == '__main__':
    unittest.main()
