# coding: utf-8
"""Exact per-edge counterfactual analysis for a trained MASKED_GLORIA model."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.configurator import Config
from utils.dataloader import TrainDataLoader
from utils.dataset import RecDataset
from utils.utils import get_model, init_seed


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'saved'
    / 'MASKED_GLORIA-book-seed999-Aug-12-2026-13-46-25.pth'
)
METRIC_NAMES = (
    'recall_at_5',
    'recall_at_20',
    'ndcg_at_5',
    'ndcg_at_20',
)
SCOPES = ('user', 'overall')
MAX_TOP_K = 20
RESULT_VERSION = 1


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Zero every train edge incident to selected test users, one at a '
            'time, and measure target-user plus exact full-test performance.'
        )
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help='trained MASKED_GLORIA checkpoint'
    )
    parser.add_argument(
        '--number_of_user',
        '--number-of-user',
        dest='number_of_user',
        type=int,
        default=None,
        help='number of eligible test users to analyze; omitted means all'
    )
    parser.add_argument(
        '--selection_seed',
        '--selection-seed',
        dest='selection_seed',
        type=int,
        default=999,
        help='seed for reproducible random user sampling'
    )
    parser.add_argument(
        '--user_selection',
        '--user-selection',
        dest='user_selection',
        choices=('random', 'recall_desc'),
        default='random',
        help=(
            'random sampling, or baseline Recall@20 from highest to lowest '
            '(ties use ascending user ID)'
        )
    )
    parser.add_argument(
        '--gpu_id',
        '--gpu-id',
        dest='gpu_id',
        type=int,
        default=0,
        help='CUDA device ID; pass a negative value to force CPU'
    )
    parser.add_argument(
        '--eval_batch_size',
        '--eval-batch-size',
        dest='eval_batch_size',
        type=int,
        default=1024,
        help='number of test users scored at once'
    )
    parser.add_argument(
        '--output_dir',
        '--output-dir',
        dest='output_dir',
        type=str,
        default=str(PROJECT_ROOT / 'counterfactual_results'),
        help='root directory for CSV, metadata, progress, and summary files'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume a matching run and skip completed edge IDs'
    )
    args = parser.parse_args(argv)

    if args.number_of_user is not None and args.number_of_user <= 0:
        parser.error('--number_of_user must be a positive integer.')
    if args.eval_batch_size <= 0:
        parser.error('--eval_batch_size must be a positive integer.')
    return args


def build_analysis_config(args):
    """Recreate the scalar MASKED_GLORIA/book configuration used by main.py."""
    use_gpu = args.gpu_id >= 0
    config_dict = {
        'gpu_id': max(args.gpu_id, 0),
        'use_gpu': use_gpu,
        'fusion': 'add',
        'dropout': 0.2,
        'reg_weight': 0.001,
        'learning_rate': 0.003,
        'seed': args.selection_seed,
        'mm_image_weight': 0.5,
        'k': 40,
        'n_mm_layers': 1,
        'knn_k': 10,
        'aggr_mode': 'add',
        'eval_batch_size': args.eval_batch_size,
        'metrics': ['Recall', 'NDCG'],
        'topk': [5, 20],
    }
    return Config('MASKED_GLORIA', 'book', config_dict)


def load_checkpoint_strict(model, checkpoint_path):
    """Load a tensor checkpoint and report key/shape mismatches clearly."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Checkpoint does not exist: {}'.format(checkpoint_path)
        )

    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location='cpu',
            weights_only=True
        )
    except TypeError:
        # Compatibility with PyTorch versions that predate weights_only.
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')

    if not isinstance(checkpoint, dict):
        raise RuntimeError('Checkpoint must contain a dictionary.')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError('Checkpoint state_dict must be a dictionary.')

    # Historical MASKED_GLORIA checkpoints were saved after forward() assigned
    # the two GCN preference Parameters to top-level attributes.  Trainer used
    # named_parameters(), whose duplicate removal retained these alias names.
    # Canonicalize only those known aliases, then keep validation fully strict.
    legacy_aliases = {
        'full_preference': 'full_gcn.preference',
        'mask_preference': 'mask_gcn.preference',
    }
    state_dict = dict(state_dict)
    for legacy_name, canonical_name in legacy_aliases.items():
        if legacy_name not in state_dict:
            continue
        if canonical_name in state_dict:
            raise RuntimeError(
                'Checkpoint contains both legacy key {} and canonical key {}.'
                .format(legacy_name, canonical_name)
            )
        state_dict[canonical_name] = state_dict.pop(legacy_name)

    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    shape_mismatches = []
    for name in sorted(set(expected) & set(state_dict)):
        if tuple(expected[name].shape) != tuple(state_dict[name].shape):
            shape_mismatches.append(
                '{}: checkpoint {} != model {}'.format(
                    name,
                    tuple(state_dict[name].shape),
                    tuple(expected[name].shape)
                )
            )

    problems = []
    if missing:
        problems.append('missing keys: {}'.format(', '.join(missing)))
    if unexpected:
        problems.append('unexpected keys: {}'.format(', '.join(unexpected)))
    if shape_mismatches:
        problems.append('shape mismatches: {}'.format('; '.join(shape_mismatches)))
    if problems:
        raise RuntimeError(
            'Checkpoint is incompatible with this model: {}'.format(
                ' | '.join(problems)
            )
        )

    model.load_state_dict(state_dict, strict=True)
    return checkpoint.get('metadata')


def select_target_users(
    eval_users,
    edge_users,
    number_of_user,
    seed,
    strategy='random',
    user_scores=None,
    exclude_zero_scores=False
):
    """Select eligible test users randomly or by descending baseline score."""
    eval_users = np.asarray(eval_users, dtype=np.int64)
    edge_users = np.asarray(edge_users, dtype=np.int64)
    eligible = np.intersect1d(
        np.unique(eval_users),
        np.unique(edge_users),
        assume_unique=False
    )
    if eligible.size == 0:
        raise ValueError('No test user has an incident train edge.')

    if exclude_zero_scores:
        if user_scores is None:
            raise ValueError(
                'user_scores are required when excluding zero-score users.'
            )
        try:
            eligible_scores = np.asarray(
                [user_scores[int(user_id)] for user_id in eligible],
                dtype=np.float64
            )
        except KeyError as error:
            raise ValueError(
                'Missing baseline score for eligible user {}.'.format(
                    error.args[0]
                )
            )
        if not np.isfinite(eligible_scores).all():
            raise ValueError('Baseline user scores must all be finite.')
        eligible = eligible[eligible_scores > 0.0]
        if eligible.size == 0:
            raise ValueError(
                'No eligible test user has baseline Recall@20 greater than 0.'
            )

    if number_of_user is not None and number_of_user > eligible.size:
        raise ValueError(
            '--number_of_user={} exceeds the {} eligible test users.'.format(
                number_of_user,
                eligible.size
            )
        )

    if strategy == 'recall_desc':
        if user_scores is None:
            raise ValueError(
                'user_scores are required for recall_desc selection.'
            )
        try:
            scores = np.asarray(
                [user_scores[int(user_id)] for user_id in eligible],
                dtype=np.float64
            )
        except KeyError as error:
            raise ValueError(
                'Missing baseline score for eligible user {}.'.format(
                    error.args[0]
                )
            )
        if not np.isfinite(scores).all():
            raise ValueError('Baseline user scores must all be finite.')
        # np.lexsort uses the last key as primary: descending score first,
        # followed by ascending user ID for deterministic ties.
        ranking = np.lexsort((eligible, -scores))
        ranked_users = eligible[ranking]
        if number_of_user is None:
            return ranked_users
        return ranked_users[:number_of_user]

    if strategy != 'random':
        raise ValueError(
            'Unknown user selection strategy: {}'.format(strategy)
        )
    if number_of_user is None:
        return eligible
    rng = np.random.RandomState(seed)
    selected = rng.choice(eligible, size=number_of_user, replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def build_user_edge_map(edge_users, selected_users=None):
    selected_set = None
    if selected_users is not None:
        selected_set = set(int(user_id) for user_id in selected_users)

    user_edges = defaultdict(list)
    for edge_id, user_id in enumerate(np.asarray(edge_users, dtype=np.int64)):
        user_id = int(user_id)
        if selected_set is None or user_id in selected_set:
            user_edges[user_id].append(int(edge_id))
    return dict(user_edges)


def metric_field(scope, metric, value_type):
    return '{}_{}_{}'.format(scope, metric, value_type)


def result_fieldnames():
    fields = [
        'user_id',
        'edge_id',
        'item_id',
        'user_train_degree',
        'original_edge_weight',
        'intervened_weight',
    ]
    for scope in SCOPES:
        for metric in METRIC_NAMES:
            for value_type in ('baseline', 'counterfactual', 'drop'):
                fields.append(metric_field(scope, metric, value_type))
    return fields


def file_fingerprint(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    stat = path.stat()
    return {
        'path': str(path),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
        'sha256': digest.hexdigest(),
    }


def atomic_write_json(path, payload):
    path = Path(path)
    temporary_path = Path(str(path) + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    os.replace(str(temporary_path), str(path))


def run_signature(metadata):
    signature_payload = {
        key: metadata[key]
        for key in (
            'version',
            'checkpoint',
            'dataset_file',
            'selected_users',
            'eligible_test_users',
            'total_interventions',
            'metrics',
            'intervention',
            'selection_seed',
            'number_of_user',
            'eval_batch_size',
        )
    }
    encoded = json.dumps(
        signature_payload,
        sort_keys=True,
        separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def prepare_run_directory(output_root, run_name, metadata, resume):
    run_dir = Path(output_root).expanduser()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir.resolve() / run_name
    metadata_path = run_dir / 'metadata.json'
    results_path = run_dir / 'edge_results.csv'
    progress_path = run_dir / 'progress.json'
    summary_path = run_dir / 'summary.json'

    if run_dir.exists() and not resume:
        existing = [
            path for path in (
                metadata_path,
                results_path,
                progress_path,
                summary_path,
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                'Run output already exists at {}. Use --resume or choose a '
                'different --output_dir.'.format(run_dir)
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    expected_signature = run_signature(metadata)
    metadata['run_signature'] = expected_signature

    if metadata_path.exists():
        with metadata_path.open('r', encoding='utf-8') as handle:
            previous = json.load(handle)
        if previous.get('run_signature') != expected_signature:
            raise RuntimeError(
                'Cannot resume: metadata does not match checkpoint, dataset, '
                'selected users, metrics, or evaluation settings.'
            )
        metadata_updated = False
        for key in (
            'user_selection',
            'selection_metric',
            'selection_order',
            'zero_recall_users_excluded',
            'positive_recall_eligible_users',
        ):
            if key not in previous and key in metadata:
                previous[key] = metadata[key]
                metadata_updated = True
        if metadata_updated:
            atomic_write_json(metadata_path, previous)
        metadata = previous
    else:
        atomic_write_json(metadata_path, metadata)

    return {
        'run_dir': run_dir,
        'metadata': metadata,
        'metadata_path': metadata_path,
        'results_path': results_path,
        'progress_path': progress_path,
        'summary_path': summary_path,
    }


def load_completed_edge_ids(results_path, fieldnames, repair=True):
    """Load valid completed rows and optionally discard an interrupted tail."""
    results_path = Path(results_path)
    if not results_path.exists():
        return set()

    valid_rows = []
    invalid_rows = 0
    with results_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        if not header and repair:
            with results_path.open(
                'w', encoding='utf-8', newline=''
            ) as repaired_handle:
                csv.writer(repaired_handle).writerow(fieldnames)
            return set()
        if header != fieldnames:
            raise RuntimeError(
                'Existing CSV header does not match this analysis version.'
            )
        edge_position = fieldnames.index('edge_id')
        for row in reader:
            if len(row) != len(fieldnames) or not row[edge_position]:
                invalid_rows += 1
                continue
            try:
                for position in range(4):
                    int(row[position])
                for position in range(4, len(fieldnames)):
                    float(row[position])
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            valid_rows.append(row)

    completed = set()
    for row in valid_rows:
        edge_id = int(row[edge_position])
        if edge_id in completed:
            raise RuntimeError(
                'Existing CSV contains duplicate edge_id {}.'.format(edge_id)
            )
        completed.add(edge_id)

    if invalid_rows and repair:
        temporary_path = Path(str(results_path) + '.repair.tmp')
        with temporary_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            writer.writerows(valid_rows)
        os.replace(str(temporary_path), str(results_path))
    return completed


def compute_metrics_from_topk(topk_items, positive_items):
    """Return per-user Recall@5/20 and NDCG@5/20 without rounding."""
    topk_items = np.asarray(topk_items, dtype=np.int64)
    if topk_items.ndim != 2 or topk_items.shape[1] < MAX_TOP_K:
        raise ValueError('topk_items must have shape [num_users, >=20].')
    if len(positive_items) != topk_items.shape[0]:
        raise ValueError('positive_items length must match topk_items rows.')

    num_users = topk_items.shape[0]
    hits = np.zeros((num_users, MAX_TOP_K), dtype=np.float64)
    positive_lengths = np.empty(num_users, dtype=np.int64)
    for row_index, positives in enumerate(positive_items):
        positives = np.asarray(positives, dtype=np.int64)
        if positives.size == 0:
            raise ValueError('Every evaluated user must have a test positive.')
        positive_lengths[row_index] = positives.size
        hits[row_index] = np.isin(
            topk_items[row_index, :MAX_TOP_K],
            positives,
            assume_unique=False
        )

    cumulative_hits = np.cumsum(hits, axis=1)
    recall_at_5 = cumulative_hits[:, 4] / positive_lengths
    recall_at_20 = cumulative_hits[:, 19] / positive_lengths

    discounts = 1.0 / np.log2(np.arange(2, MAX_TOP_K + 2))
    cumulative_dcg = np.cumsum(hits * discounts.reshape(1, -1), axis=1)
    discount_cumulative = np.cumsum(discounts)
    idcg_at_5 = discount_cumulative[
        np.minimum(positive_lengths, 5) - 1
    ]
    idcg_at_20 = discount_cumulative[
        np.minimum(positive_lengths, 20) - 1
    ]
    ndcg_at_5 = cumulative_dcg[:, 4] / idcg_at_5
    ndcg_at_20 = cumulative_dcg[:, 19] / idcg_at_20

    return np.column_stack(
        [recall_at_5, recall_at_20, ndcg_at_5, ndcg_at_20]
    )


class ExactFullTestEvaluator(object):
    """Reusable exact full-sort evaluator with cached users/history masks."""

    def __init__(
        self,
        train_dataset,
        test_dataset,
        num_users,
        num_items,
        device,
        batch_size
    ):
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.device = device
        self.batch_size = int(batch_size)
        uid_field = test_dataset.uid_field
        iid_field = test_dataset.iid_field

        self.eval_users = test_dataset.df[uid_field].unique().astype(np.int64)
        train_groups = {
            int(user_id): group[iid_field].to_numpy(dtype=np.int64)
            for user_id, group in train_dataset.df.groupby(uid_field, sort=False)
        }
        test_groups = {
            int(user_id): group[iid_field].to_numpy(dtype=np.int64)
            for user_id, group in test_dataset.df.groupby(uid_field, sort=False)
        }

        self.positive_items = []
        self.history_items = []
        for user_id in self.eval_users:
            user_id = int(user_id)
            if user_id not in train_groups:
                raise ValueError(
                    'Test user {} has no train history.'.format(user_id)
                )
            self.history_items.append(train_groups[user_id])
            self.positive_items.append(test_groups[user_id])

        self.user_to_position = {
            int(user_id): position
            for position, user_id in enumerate(self.eval_users)
        }
        self.batches = self._build_batches()

    def _build_batches(self):
        batches = []
        for start in range(0, len(self.eval_users), self.batch_size):
            end = min(start + self.batch_size, len(self.eval_users))
            user_ids = torch.as_tensor(
                self.eval_users[start:end],
                dtype=torch.long,
                device=self.device
            )
            histories = self.history_items[start:end]
            lengths = np.asarray(
                [len(items) for items in histories],
                dtype=np.int64
            )
            if lengths.sum() > 0:
                mask_rows = np.repeat(
                    np.arange(end - start, dtype=np.int64),
                    lengths
                )
                mask_items = np.concatenate(histories).astype(
                    np.int64,
                    copy=False
                )
            else:
                mask_rows = np.empty(0, dtype=np.int64)
                mask_items = np.empty(0, dtype=np.int64)
            batches.append((
                start,
                end,
                user_ids,
                torch.as_tensor(mask_rows, dtype=torch.long, device=self.device),
                torch.as_tensor(mask_items, dtype=torch.long, device=self.device),
            ))
        return batches

    def evaluate(self, result_embedding):
        if result_embedding.shape[0] != self.num_users + self.num_items:
            raise ValueError('result_embedding has an invalid node count.')

        item_embedding = result_embedding[self.num_users:]
        topk_batches = []
        for _, _, user_ids, mask_rows, mask_items in self.batches:
            user_embedding = result_embedding[user_ids]
            scores = torch.matmul(user_embedding, item_embedding.t())
            if mask_items.numel() > 0:
                scores[mask_rows, mask_items] = -1e10
            topk = torch.topk(scores, MAX_TOP_K, dim=1).indices
            topk_batches.append(topk.cpu().numpy())

        topk_items = np.concatenate(topk_batches, axis=0)
        per_user = compute_metrics_from_topk(
            topk_items,
            self.positive_items
        )
        overall = {
            metric: float(per_user[:, metric_index].mean())
            for metric_index, metric in enumerate(METRIC_NAMES)
        }
        return per_user, overall


def build_result_row(
    user_id,
    edge_id,
    item_id,
    train_degree,
    original_weight,
    target_position,
    baseline_per_user,
    baseline_overall,
    counterfactual_per_user,
    counterfactual_overall
):
    row = {
        'user_id': int(user_id),
        'edge_id': int(edge_id),
        'item_id': int(item_id),
        'user_train_degree': int(train_degree),
        'original_edge_weight': float(original_weight),
        'intervened_weight': 0.0,
    }
    for metric_index, metric in enumerate(METRIC_NAMES):
        user_baseline = float(baseline_per_user[target_position, metric_index])
        user_counterfactual = float(
            counterfactual_per_user[target_position, metric_index]
        )
        overall_baseline = float(baseline_overall[metric])
        overall_counterfactual = float(counterfactual_overall[metric])
        row[metric_field('user', metric, 'baseline')] = user_baseline
        row[metric_field('user', metric, 'counterfactual')] = (
            user_counterfactual
        )
        row[metric_field('user', metric, 'drop')] = (
            user_baseline - user_counterfactual
        )
        row[metric_field('overall', metric, 'baseline')] = overall_baseline
        row[metric_field('overall', metric, 'counterfactual')] = (
            overall_counterfactual
        )
        row[metric_field('overall', metric, 'drop')] = (
            overall_baseline - overall_counterfactual
        )
    return row


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def safe_correlation(x_values, y_values, method='pearson'):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 2:
        return None
    if method == 'spearman':
        x_values = average_ranks(x_values)
        y_values = average_ranks(y_values)
    elif method != 'pearson':
        raise ValueError('Unknown correlation method: {}'.format(method))
    if np.ptp(x_values) == 0.0 or np.ptp(y_values) == 0.0:
        return None
    value = float(np.corrcoef(x_values, y_values)[0, 1])
    return value if math.isfinite(value) else None


def describe_values(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            'mean': None,
            'min': None,
            'max': None,
            'decreased_count': 0,
            'increased_count': 0,
            'unchanged_count': 0,
        }
    tolerance = 1e-12
    return {
        'mean': float(values.mean()),
        'min': float(values.min()),
        'max': float(values.max()),
        'decreased_count': int((values > tolerance).sum()),
        'increased_count': int((values < -tolerance).sum()),
        'unchanged_count': int((np.abs(values) <= tolerance).sum()),
    }


def build_summary(results_path, metadata, baseline_overall):
    weights = []
    drops = {
        scope: {metric: [] for metric in METRIC_NAMES}
        for scope in SCOPES
    }
    counterfactual_values = {
        scope: {metric: [] for metric in METRIC_NAMES}
        for scope in SCOPES
    }

    with Path(results_path).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            weights.append(float(row['original_edge_weight']))
            for scope in SCOPES:
                for metric in METRIC_NAMES:
                    drops[scope][metric].append(
                        float(row[metric_field(scope, metric, 'drop')])
                    )
                    counterfactual_values[scope][metric].append(
                        float(
                            row[metric_field(
                                scope,
                                metric,
                                'counterfactual'
                            )]
                        )
                    )

    summary = {
        'version': RESULT_VERSION,
        'completed_at_utc': utc_now(),
        'run_signature': metadata['run_signature'],
        'checkpoint': metadata['checkpoint'],
        'dataset_file': metadata['dataset_file'],
        'user_selection': metadata.get('user_selection', 'random'),
        'selection_metric': metadata.get('selection_metric'),
        'selection_order': metadata.get('selection_order'),
        'zero_recall_users_excluded': metadata.get(
            'zero_recall_users_excluded',
            False
        ),
        'positive_recall_eligible_users': metadata.get(
            'positive_recall_eligible_users'
        ),
        'selection_seed': metadata['selection_seed'],
        'number_of_user': metadata['number_of_user'],
        'selected_user_count': len(metadata['selected_users']),
        'completed_interventions': len(weights),
        'baseline_overall': baseline_overall,
        'statistics': {},
    }
    for scope in SCOPES:
        summary['statistics'][scope] = {}
        for metric in METRIC_NAMES:
            metric_drops = np.asarray(drops[scope][metric], dtype=np.float64)
            metric_summary = describe_values(metric_drops)
            counterfactual_array = np.asarray(
                counterfactual_values[scope][metric],
                dtype=np.float64
            )
            metric_summary['mean_counterfactual'] = (
                float(counterfactual_array.mean())
                if counterfactual_array.size
                else None
            )
            metric_summary['weight_drop_pearson'] = safe_correlation(
                weights,
                metric_drops,
                method='pearson'
            )
            metric_summary['weight_drop_spearman'] = safe_correlation(
                weights,
                metric_drops,
                method='spearman'
            )
            summary['statistics'][scope][metric] = metric_summary
    return summary


def progress_payload(
    status,
    completed,
    total,
    started_at,
    last_user=None,
    last_edge=None
):
    elapsed = max(0.0, time.time() - started_at)
    processed_this_session = completed[1]
    average_seconds = (
        elapsed / processed_this_session
        if processed_this_session > 0
        else None
    )
    remaining = max(0, total - completed[0])
    eta_seconds = (
        average_seconds * remaining
        if average_seconds is not None
        else None
    )
    return {
        'status': status,
        'updated_at_utc': utc_now(),
        'completed_interventions': int(completed[0]),
        'processed_this_session': int(processed_this_session),
        'total_interventions': int(total),
        'percent_complete': (
            100.0 * completed[0] / total if total else 100.0
        ),
        'elapsed_seconds_this_session': elapsed,
        'average_seconds_per_intervention': average_seconds,
        'eta_seconds': eta_seconds,
        'last_user_id': last_user,
        'last_edge_id': last_edge,
    }


def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return 'unknown'
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return '{}d {:02d}h'.format(days, hours)
    if hours:
        return '{}h {:02d}m'.format(hours, minutes)
    if minutes:
        return '{}m {:02d}s'.format(minutes, seconds)
    return '{}s'.format(seconds)


def run_analysis(args):
    os.chdir(str(PROJECT_ROOT))
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    config = build_analysis_config(args)
    init_seed(args.selection_seed)
    print('Loading dataset and reconstructing train/test split...')
    dataset = RecDataset(config)
    train_dataset, _, test_dataset = dataset.split()
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True
    )

    print('Building MASKED_GLORIA on {}...'.format(config['device']))
    model = get_model('MASKED_GLORIA')(config, train_data).to(config['device'])
    checkpoint_metadata = load_checkpoint_strict(model, checkpoint_path)
    model.eval()
    if model.mask_logits.numel() != model.num_interactions:
        raise RuntimeError(
            'mask_logits has {} values but the train graph has {} edges.'
            .format(model.mask_logits.numel(), model.num_interactions)
        )

    edge_users = model.forward_edge_users.detach().cpu().numpy()
    edge_items = model.forward_edge_items.detach().cpu().numpy()
    eval_users = test_dataset.df[test_dataset.uid_field].unique()

    evaluator = ExactFullTestEvaluator(
        train_dataset,
        test_dataset,
        model.num_user,
        model.num_item,
        config['device'],
        args.eval_batch_size
    )
    print('Computing baseline metrics for user selection...')
    mask_logits_before = model.mask_logits.detach().clone()
    with torch.inference_mode():
        full_view = model.compute_full_view()
        base_mask = model.get_forward_edge_mask().detach()
        baseline_embedding = model.compute_result_embedding(
            base_mask,
            full_view=full_view
        )
        baseline_per_user, baseline_overall = evaluator.evaluate(
            baseline_embedding
        )
    recall_at_20_index = METRIC_NAMES.index('recall_at_20')
    baseline_recall_at_20 = {
        int(user_id): float(baseline_per_user[position, recall_at_20_index])
        for position, user_id in enumerate(evaluator.eval_users)
    }
    selected_users = select_target_users(
        eval_users,
        edge_users,
        args.number_of_user,
        args.selection_seed,
        strategy=args.user_selection,
        user_scores=baseline_recall_at_20,
        exclude_zero_scores=True
    )
    user_edges = build_user_edge_map(edge_users, selected_users)
    total_interventions = sum(
        len(user_edges[int(user_id)]) for user_id in selected_users
    )

    dataset_file = (
        PROJECT_ROOT
        / config['data_path']
        / config['dataset']
        / config['inter_file_name']
    ).resolve()
    metadata = {
        'version': RESULT_VERSION,
        'created_at_utc': utc_now(),
        'checkpoint': file_fingerprint(checkpoint_path),
        'checkpoint_metadata_present': checkpoint_metadata is not None,
        'dataset_file': file_fingerprint(dataset_file),
        'model': 'MASKED_GLORIA',
        'dataset': 'book',
        'device': str(config['device']),
        'metrics': list(METRIC_NAMES),
        'intervention': {
            'branch': 'masked_gcn_only',
            'unit': 'one_incident_train_edge_at_a_time',
            'forward_weight_after': 0.0,
            'reverse_weight_after': 0.0,
            'overall_scope': 'all_test_users_per_intervention',
            'drop_definition': 'baseline_minus_counterfactual',
        },
        'user_selection': args.user_selection,
        'selection_metric': (
            'recall_at_20' if args.user_selection == 'recall_desc' else None
        ),
        'selection_order': (
            'descending_score_then_ascending_user_id'
            if args.user_selection == 'recall_desc'
            else 'seeded_random_without_replacement'
        ),
        'zero_recall_users_excluded': True,
        'positive_recall_eligible_users': int(sum(
            baseline_recall_at_20[int(user_id)] > 0.0
            for user_id in np.intersect1d(
                np.unique(eval_users),
                np.unique(edge_users)
            )
        )),
        'selection_seed': int(args.selection_seed),
        'number_of_user': args.number_of_user,
        'eligible_test_users': int(
            np.intersect1d(np.unique(eval_users), np.unique(edge_users)).size
        ),
        'selected_users': [int(user_id) for user_id in selected_users],
        'eval_batch_size': int(args.eval_batch_size),
        'train_edge_count': int(model.num_interactions),
        'total_interventions': int(total_interventions),
    }

    selected_label = (
        'all' if args.number_of_user is None else str(args.number_of_user)
    )
    if args.user_selection == 'recall_desc':
        run_name = '{}-users{}-recall20-desc-seed{}'.format(
            checkpoint_path.stem,
            selected_label,
            args.selection_seed
        )
    else:
        run_name = '{}-users{}-seed{}'.format(
            checkpoint_path.stem,
            selected_label,
            args.selection_seed
        )
    paths = prepare_run_directory(
        args.output_dir,
        run_name,
        metadata,
        args.resume
    )
    metadata = paths['metadata']
    fields = result_fieldnames()
    completed_edges = load_completed_edge_ids(
        paths['results_path'],
        fields,
        repair=True
    )
    invalid_completed = completed_edges.difference(
        edge_id
        for user_id in selected_users
        for edge_id in user_edges[int(user_id)]
    )
    if invalid_completed:
        raise RuntimeError(
            'Existing CSV contains edge IDs outside the selected-user set.'
        )

    for user_id in selected_users:
        if int(user_id) not in evaluator.user_to_position:
            raise RuntimeError(
                'Selected user {} is absent from test evaluation.'.format(
                    int(user_id)
                )
            )

    print('Selected test users : {}'.format(len(selected_users)))
    print('User selection      : {}'.format(args.user_selection))
    print('Recall@20 = 0 users : {} excluded'.format(
        metadata['eligible_test_users']
        - metadata['positive_recall_eligible_users']
    ))
    if args.user_selection == 'recall_desc' and len(selected_users):
        first_user = int(selected_users[0])
        last_user = int(selected_users[-1])
        print('Recall@20 range     : {:.6f} -> {:.6f}'.format(
            baseline_recall_at_20[first_user],
            baseline_recall_at_20[last_user]
        ))
    print('Incident edges/tests: {}'.format(total_interventions))
    print('Exact overall scope : {} test users per edge'.format(
        len(evaluator.eval_users)
    ))
    print('Output directory    : {}'.format(paths['run_dir']))
    if completed_edges:
        print('Resume: skipping {} completed edges.'.format(
            len(completed_edges)
        ))

    started_at = time.time()
    processed_this_session = 0

    csv_exists = paths['results_path'].exists()
    csv_mode = 'a' if csv_exists else 'w'
    interrupted = False
    last_user = None
    last_edge = None
    try:
        with torch.inference_mode():
            with paths['results_path'].open(
                csv_mode,
                encoding='utf-8',
                newline=''
            ) as csv_handle:
                writer = csv.DictWriter(csv_handle, fieldnames=fields)
                if not csv_exists:
                    writer.writeheader()
                    csv_handle.flush()

                for user_id_value in selected_users:
                    user_id = int(user_id_value)
                    incident_edges = user_edges[user_id]
                    target_position = evaluator.user_to_position[user_id]
                    for edge_id in incident_edges:
                        if edge_id in completed_edges:
                            continue

                        counterfactual_mask = (
                            model.get_counterfactual_forward_mask(
                                edge_id,
                                base_mask=base_mask
                            )
                        )
                        counterfactual_embedding = (
                            model.compute_result_embedding(
                                counterfactual_mask,
                                full_view=full_view
                            )
                        )
                        counterfactual_per_user, counterfactual_overall = (
                            evaluator.evaluate(counterfactual_embedding)
                        )
                        row = build_result_row(
                            user_id=user_id,
                            edge_id=edge_id,
                            item_id=int(edge_items[edge_id]),
                            train_degree=len(incident_edges),
                            original_weight=float(base_mask[edge_id].item()),
                            target_position=target_position,
                            baseline_per_user=baseline_per_user,
                            baseline_overall=baseline_overall,
                            counterfactual_per_user=counterfactual_per_user,
                            counterfactual_overall=counterfactual_overall
                        )
                        writer.writerow(row)
                        csv_handle.flush()

                        completed_edges.add(edge_id)
                        processed_this_session += 1
                        last_user = user_id
                        last_edge = edge_id
                        completed_state = (
                            len(completed_edges),
                            processed_this_session
                        )
                        progress = progress_payload(
                            'running',
                            completed_state,
                            total_interventions,
                            started_at,
                            last_user,
                            last_edge
                        )
                        atomic_write_json(paths['progress_path'], progress)
                        print(
                            '[{}/{} | {:.2f}%] user={} edge={} item={} '
                            'weight={:.6f} ETA={}'.format(
                                len(completed_edges),
                                total_interventions,
                                progress['percent_complete'],
                                user_id,
                                edge_id,
                                int(edge_items[edge_id]),
                                float(base_mask[edge_id].item()),
                                format_duration(progress['eta_seconds'])
                            ),
                            flush=True
                        )
    except KeyboardInterrupt:
        interrupted = True
        progress = progress_payload(
            'interrupted',
            (len(completed_edges), processed_this_session),
            total_interventions,
            started_at,
            last_user,
            last_edge
        )
        atomic_write_json(paths['progress_path'], progress)
        print(
            '\nInterrupted. Progress is safe; rerun the same command with '
            '--resume.',
            file=sys.stderr
        )
    except Exception:
        progress = progress_payload(
            'failed',
            (len(completed_edges), processed_this_session),
            total_interventions,
            started_at,
            last_user,
            last_edge
        )
        atomic_write_json(paths['progress_path'], progress)
        raise
    finally:
        if base_mask is not None and full_view is not None:
            with torch.inference_mode():
                model.compute_result_embedding(base_mask, full_view=full_view)
        if not torch.equal(model.mask_logits.detach(), mask_logits_before):
            raise RuntimeError('Analysis unexpectedly modified mask_logits.')

    if interrupted:
        return 130

    if len(completed_edges) != total_interventions:
        raise RuntimeError(
            'Analysis ended with {}/{} completed interventions.'.format(
                len(completed_edges),
                total_interventions
            )
        )

    summary = build_summary(
        paths['results_path'],
        metadata,
        baseline_overall
    )
    atomic_write_json(paths['summary_path'], summary)
    progress = progress_payload(
        'complete',
        (len(completed_edges), processed_this_session),
        total_interventions,
        started_at,
        last_user,
        last_edge
    )
    atomic_write_json(paths['progress_path'], progress)
    print('Analysis complete.')
    print('CSV    : {}'.format(paths['results_path']))
    print('Summary: {}'.format(paths['summary_path']))
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_analysis(args)
    except Exception as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
