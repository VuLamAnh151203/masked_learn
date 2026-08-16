import math
import os
import sys
import types
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.trainer import Trainer
from models.masked_gloria_ex2 import MASKED_GLORIA_EX2
from utils.utils import get_model


def make_schedule_only_model(discovery_epochs=60, anneal_epochs=40):
    """Build the schedule portion of the model without loading a dataset."""
    model = MASKED_GLORIA_EX2.__new__(MASKED_GLORIA_EX2)
    torch.nn.Module.__init__(model)
    model.mask_discovery_epochs = discovery_epochs
    model.mask_anneal_epochs = anneal_epochs
    model.current_epoch = 0
    model.mask_alpha = 1.0
    model.mask_stage = 'discovery'
    model._inference_cache_valid = False
    model.user_mask_logits = torch.nn.Parameter(torch.zeros(3))
    return model


class AnnealedMaskScheduleTest(unittest.TestCase):
    def test_schedule_boundaries(self):
        model = make_schedule_only_model()
        expected = {
            1: ('discovery', 1.0),
            60: ('discovery', 1.0),
            61: ('annealing', 0.975),
            100: ('annealing', 0.0),
            101: ('identity_finetune', 0.0),
            500: ('identity_finetune', 0.0),
        }

        for epoch, (expected_stage, expected_alpha) in expected.items():
            stage, alpha = model._schedule_at_epoch(epoch)
            self.assertEqual(stage, expected_stage)
            self.assertTrue(
                math.isclose(alpha, expected_alpha, abs_tol=1e-12),
                msg='epoch {} produced alpha {}'.format(epoch, alpha)
            )

    def test_effective_mask_formula_and_exact_identity(self):
        model = make_schedule_only_model()
        base_mask = model.get_user_mask()

        self.assertTrue(torch.equal(model.get_effective_user_mask(1.0), base_mask))
        self.assertTrue(
            torch.allclose(
                model.get_effective_user_mask(0.5),
                1.0 - 0.5 * (1.0 - base_mask)
            )
        )
        self.assertTrue(
            torch.equal(
                model.get_effective_user_mask(0.0),
                torch.ones_like(base_mask)
            )
        )
        self.assertTrue(
            torch.equal(
                model.get_inference_user_mask(),
                torch.ones_like(base_mask)
            )
        )

    def test_mask_learns_then_freezes_before_annealing(self):
        model = make_schedule_only_model()
        optimizer = torch.optim.Adam([model.user_mask_logits], lr=0.1)

        model.set_training_epoch(59)
        optimizer.zero_grad()
        model.get_effective_user_mask().sum().backward()
        self.assertIsNotNone(model.user_mask_logits.grad)
        optimizer.step()

        frozen_value = model.user_mask_logits.detach().clone()
        model.set_training_epoch(60)
        optimizer.zero_grad()
        self.assertFalse(model.user_mask_logits.requires_grad)

        # An optimizer containing the now-frozen parameter must not update it.
        optimizer.step()
        self.assertTrue(torch.equal(model.user_mask_logits, frozen_value))

    def test_model_selection_starts_in_identity_phase(self):
        model = make_schedule_only_model()
        self.assertFalse(model.is_model_selection_epoch(99))
        self.assertTrue(model.is_model_selection_epoch(100))

        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        self.assertFalse(trainer._is_model_selection_epoch(99))
        self.assertTrue(trainer._is_model_selection_epoch(100))

    def test_trainer_defers_best_model_and_checkpoint_updates(self):
        class FakeStagedModel:
            def __init__(self):
                self.current_epoch = 0

            def set_training_epoch(self, epoch_idx):
                self.current_epoch = epoch_idx + 1

            def is_model_selection_epoch(self, epoch_idx):
                return epoch_idx >= 4

            def pre_epoch_processing(self):
                return None

            def post_epoch_processing(self):
                return None

        class FakeScheduler:
            def step(self):
                return None

        model = FakeStagedModel()
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.config = {'model': 'FAKE'}
        trainer.start_epoch = 0
        trainer.epochs = 5
        trainer.eval_step = 1
        trainer.stopping_step = 30
        trainer.valid_metric_bigger = True
        trainer.best_valid_score = -1
        trainer.cur_step = 0
        trainer.best_valid_result = None
        trainer.best_test_upon_valid = None
        trainer.train_loss_dict = {}
        trainer.lr_scheduler = FakeScheduler()

        trainer._train_epoch = types.MethodType(
            lambda self, train_data, epoch_idx: (0.0, []),
            trainer
        )

        preselection_scores = {1: 100.0, 2: 90.0, 3: 80.0, 4: 70.0}

        def fake_valid_epoch(self, data, is_test=False, idx=0):
            if data == 'valid':
                score = preselection_scores.get(model.current_epoch, 1.0)
            else:
                score = float(model.current_epoch)
            return score, {'recall@20': score}

        trainer._valid_epoch = types.MethodType(fake_valid_epoch, trainer)
        saved_epochs = []
        trainer._save_checkpoint = types.MethodType(
            lambda self: saved_epochs.append(model.current_epoch),
            trainer
        )

        trainer.fit(
            train_data=[],
            valid_data='valid',
            test_data='test',
            saved=True,
            verbose=False
        )

        self.assertEqual(trainer.best_valid_score, 1.0)
        self.assertEqual(trainer.best_valid_result, {'recall@20': 1.0})
        self.assertEqual(saved_epochs, [5])

    def test_inference_embedding_is_cached_and_uses_ones(self):
        model = make_schedule_only_model(discovery_epochs=2, anneal_epochs=2)
        model.n_users = 1
        model.num_user = 1
        calls = []

        def fake_compute_result_embedding(self, user_mask):
            calls.append(user_mask.detach().clone())
            self.result_embed = torch.tensor([[1.0], [2.0]])
            return self.result_embed

        model.compute_result_embedding = types.MethodType(
            fake_compute_result_embedding,
            model
        )
        interaction = [torch.tensor([0])]

        model.full_sort_predict(interaction)
        model.full_sort_predict(interaction)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0], torch.ones_like(calls[0])))

        model.set_training_epoch(0)
        model.full_sort_predict(interaction)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[1], torch.ones_like(calls[1])))

    def test_dynamic_model_loading(self):
        self.assertIs(get_model('MASKED_GLORIA_EX2'), MASKED_GLORIA_EX2)


if __name__ == '__main__':
    unittest.main()
