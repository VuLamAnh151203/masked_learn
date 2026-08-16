# coding: utf-8


r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.best_test_upon_valid = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        checkpoint_dir = config['checkpoint_dir'] or 'saved'
        self.checkpoint_dir = os.path.abspath(
            os.path.expanduser(str(checkpoint_dir))
        )
        self.saved_model_file = self._build_saved_model_file()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def _safe_filename_part(value):
        """Return a filesystem-friendly representation of a config value."""
        value = str(value)
        return ''.join(
            character if character.isalnum() or character in ('-', '_') else '-'
            for character in value
        )

    def _build_saved_model_file(self):
        """Build a unique checkpoint filename for this hyperparameter run."""
        model_name = self._safe_filename_part(self.config['model'])
        dataset_name = self._safe_filename_part(self.config['dataset'])
        seed = self._safe_filename_part(self.config['seed'])
        timestamp = get_local_time()
        filename = '{}-{}-seed{}-{}.pth'.format(
            model_name,
            dataset_name,
            seed,
            timestamp
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Hyperparameter runs can start within the same second. Avoid replacing
        # a checkpoint produced by an earlier run.
        suffix = 1
        while os.path.exists(checkpoint_path):
            filename = '{}-{}-seed{}-{}-{}.pth'.format(
                model_name,
                dataset_name,
                seed,
                timestamp,
                suffix
            )
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            suffix += 1
        return checkpoint_path

    def _save_checkpoint(self):
        """Atomically save model parameters and optional model metadata."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        parameter_state = {
            name: parameter.detach()
            for name, parameter in self.model.named_parameters()
        }
        checkpoint = {'state_dict': parameter_state}

        metadata_hook = getattr(self.model, 'get_checkpoint_metadata', None)
        if callable(metadata_hook):
            checkpoint['metadata'] = metadata_hook()

        temporary_file = self.saved_model_file + '.tmp'
        torch.save(checkpoint, temporary_file)
        os.replace(temporary_file, self.saved_model_file)
        self.logger.info(
            'Saved best model checkpoint to: {}'.format(
                self.saved_model_file
            )
        )

    def _is_model_selection_epoch(self, epoch_idx):
        """Allow models with staged training to delay early stopping."""
        selection_hook = getattr(
            self.model,
            'is_model_selection_epoch',
            None
        )
        if callable(selection_hook):
            return bool(selection_hook(epoch_idx))
        return True

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, is_test=False, idx=0):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data,is_test,idx)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()

            epoch_hook = getattr(self.model, 'set_training_epoch', None)
            if callable(epoch_hook):
                epoch_hook(epoch_idx)

            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data, False, epoch_idx)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))

                if not self._is_model_selection_epoch(epoch_idx):
                    if verbose:
                        self.logger.info(
                            'Model selection deferred until the staged '
                            'training schedule is complete.'
                        )
                    continue

                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result
                    if saved:
                        self._save_checkpoint()

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


class CounterfactualCalibrationTrainer(Trainer):
    """Train only the MASKED_GLORIA_EX3 per-user gamma calibrator."""

    def __init__(self, config, model):
        super(CounterfactualCalibrationTrainer, self).__init__(config, model)
        epochs = config['cf_calibrator_epochs']
        patience = config['cf_calibrator_patience']
        batch_size = config['cf_calibrator_batch_size']
        self.calibrator_epochs = int(
            200 if epochs is None else epochs
        )
        self.calibrator_patience = int(
            20 if patience is None else patience
        )
        self.calibrator_batch_size = int(
            512 if batch_size is None else batch_size
        )

        if self.calibrator_epochs <= 0:
            raise ValueError('cf_calibrator_epochs must be positive.')
        if self.calibrator_patience <= 0:
            raise ValueError('cf_calibrator_patience must be positive.')
        if self.calibrator_batch_size <= 0:
            raise ValueError('cf_calibrator_batch_size must be positive.')

    def _build_optimizer(self):
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError('The EX3 calibrator has no trainable parameters.')

        unexpected_trainable = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and not name.startswith('calibrator.')
        ]
        if unexpected_trainable:
            raise RuntimeError(
                'Only calibrator parameters may be trainable; found {}.'.format(
                    unexpected_trainable
                )
            )

        configured_learning_rate = self.config['cf_calibrator_lr']
        learning_rate = float(
            1e-3
            if configured_learning_rate is None
            else configured_learning_rate
        )
        if learning_rate <= 0.0:
            raise ValueError('cf_calibrator_lr must be positive.')
        return optim.Adam(trainable_parameters, lr=learning_rate)

    @torch.no_grad()
    def _evaluate_calibrator(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_users = 0
        total_gamma_error = 0.0

        for users, labels in data_loader:
            users = users.to(self.device)
            labels = labels.to(self.device)
            logits = self.model.get_calibrator_logits(users)
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                reduction='sum'
            )
            probabilities = torch.softmax(logits, dim=-1)
            predicted_gamma = torch.sum(
                probabilities * self.model.gamma_grid,
                dim=-1
            )
            target_gamma = self.model.gamma_grid[labels]

            total_loss += float(loss.cpu())
            total_correct += int(
                (logits.argmax(dim=-1) == labels).sum().cpu()
            )
            total_gamma_error += float(
                torch.abs(predicted_gamma - target_gamma).sum().cpu()
            )
            total_users += int(users.numel())

        if total_users == 0:
            raise RuntimeError('Calibration validation loader is empty.')
        return {
            'ce': total_loss / total_users,
            'accuracy': total_correct / total_users,
            'gamma_mae': total_gamma_error / total_users,
        }

    def fit(
        self,
        train_data,
        valid_data=None,
        test_data=None,
        saved=False,
        verbose=True
    ):
        if valid_data is None or test_data is None:
            raise ValueError(
                'CounterfactualCalibrationTrainer requires validation and '
                'test dataloaders.'
            )

        artifact = self.model.prepare_counterfactual_calibration(
            train_data,
            valid_data
        )
        training_dataset = TensorDataset(
            artifact['train_user_ids'].long(),
            artifact['train_labels'].long()
        )
        validation_dataset = TensorDataset(
            artifact['validation_user_ids'].long(),
            artifact['validation_labels'].long()
        )

        generator = torch.Generator()
        generator.manual_seed(int(self.config['seed']))
        training_loader = DataLoader(
            training_dataset,
            batch_size=self.calibrator_batch_size,
            shuffle=True,
            generator=generator
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.calibrator_batch_size,
            shuffle=False
        )

        best_validation_ce = float('inf')
        best_epoch = None
        best_calibrator_state = None
        epochs_without_improvement = 0

        for epoch_idx in range(self.calibrator_epochs):
            self.model.train()
            total_training_loss = 0.0
            total_training_users = 0
            start_time = time()

            for users, labels in training_loader:
                users = users.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model.get_calibrator_logits(users)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                self._check_nan(loss)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(
                        self.model.calibrator.parameters(),
                        **self.clip_grad_norm
                    )
                self.optimizer.step()
                self.model.invalidate_inference_cache()

                batch_size = int(users.numel())
                total_training_loss += float(loss.detach().cpu()) * batch_size
                total_training_users += batch_size

            validation_statistics = self._evaluate_calibrator(
                validation_loader
            )
            training_ce = total_training_loss / max(total_training_users, 1)
            elapsed = time() - start_time

            if verbose:
                self.logger.info(
                    'calibrator epoch {} [time: {:.2f}s, train_ce: {:.6f}, '
                    'valid_ce: {:.6f}, valid_accuracy: {:.6f}, '
                    'valid_gamma_mae: {:.6f}]'.format(
                        epoch_idx + 1,
                        elapsed,
                        training_ce,
                        validation_statistics['ce'],
                        validation_statistics['accuracy'],
                        validation_statistics['gamma_mae']
                    )
                )

            if validation_statistics['ce'] < best_validation_ce - 1e-12:
                best_validation_ce = validation_statistics['ce']
                best_epoch = epoch_idx + 1
                epochs_without_improvement = 0
                best_calibrator_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.calibrator.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.calibrator_patience:
                if verbose:
                    self.logger.info(
                        'Calibrator early stopping at epoch {}; best epoch '
                        'was {} with CE {:.6f}.'.format(
                            epoch_idx + 1,
                            best_epoch,
                            best_validation_ce
                        )
                    )
                break

        if best_calibrator_state is None:
            raise RuntimeError('Calibrator training did not produce a checkpoint.')

        self.model.calibrator.load_state_dict(best_calibrator_state)
        self.model.calibration_metadata.update({
            'best_calibration_epoch': int(best_epoch),
            'best_calibration_ce': float(best_validation_ce),
        })
        self.model.invalidate_inference_cache()

        valid_score, valid_result = self._valid_epoch(valid_data)
        _, test_result = self._valid_epoch(
            test_data,
            is_test=True,
            idx=best_epoch
        )

        self.best_valid_score = valid_score
        self.best_valid_result = valid_result
        self.best_test_upon_valid = test_result

        if saved:
            self._save_checkpoint()

        if verbose:
            self.logger.info(
                'Best calibrator validation CE: {:.6f} at epoch {}'.format(
                    best_validation_ce,
                    best_epoch
                )
            )
            self.logger.info('validation result: \n' + dict2str(valid_result))
            self.logger.info('test result: \n' + dict2str(test_result))
            self.logger.info(
                'calibration statistics: {}'.format(
                    self.model.get_calibration_statistics()
                )
            )

        return valid_score, valid_result, test_result

