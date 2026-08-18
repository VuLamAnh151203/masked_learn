# coding: utf-8
"""Targeted rank-transition analysis for recall-decreasing edge removals.

The source edge_results.csv selects edges whose stored target-user Recall@5
or Recall@20 drop is positive.  This script reruns only those interventions,
records exact target-user rankings and score margins, and keeps stored and
locally reproduced effects side by side.  Baseline mismatches are rejected by
default so results from incompatible experimental contexts cannot be mixed.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import counterfactual_edge_analysis as cea


METRICS = cea.METRIC_NAMES
CUTOFFS = (5, 20)
EPSILON = 1e-12
DEFAULT_SOURCE = (
    PROJECT_ROOT / 'counterfactual_results' / 'CF_res' / 'edge_results.csv'
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'saved'
    / 'MASKED_GLORIA-book-seed999-Aug-12-2026-13-46-25.pth'
)
DEFAULT_DECREASING_OUTPUT = (
    PROJECT_ROOT
    / 'counterfactual_results'
    / 'CF_res'
    / 'rank_margin_495'
)
DEFAULT_AFFECTED_ALL_OUTPUT = (
    PROJECT_ROOT
    / 'counterfactual_results'
    / 'CF_res'
    / 'rank_margin_affected_1689'
)


def canonicalize_train_edge_order(train_dataset):
    """Recreate the canonical edge order used by the saved checkpoint.

    The source run converted the training interactions to a canonical COO
    order: user ID first, then item ID.  Newer SciPy versions preserve the
    input row order when a COO matrix is constructed, so sorting explicitly
    keeps edge IDs and edge-indexed checkpoint tensors portable.
    """
    train_dataset.df = train_dataset.df.sort_values(
        [train_dataset.uid_field, train_dataset.iid_field],
        kind='mergesort'
    ).reset_index(drop=True)
    train_dataset.inter_num = len(train_dataset.df)


class CanonicalEdgeTrainDataLoader(cea.TrainDataLoader):
    """Build float32 COO directly so SciPy cannot reorder during casting."""

    def inter_matrix(self, form='coo', value_field=None):
        frame = self.dataset.df
        source = frame[self.dataset.uid_field].to_numpy()
        target = frame[self.dataset.iid_field].to_numpy()
        if value_field is None:
            values = np.ones(len(frame), dtype=np.float32)
        else:
            if value_field not in frame.columns:
                raise ValueError(
                    'value_field [{}] is absent from train data'.format(
                        value_field
                    )
                )
            values = frame[value_field].to_numpy(dtype=np.float32)
        matrix = coo_matrix(
            (values, (source, target)),
            shape=(self.dataset.user_num, self.dataset.item_num),
            dtype=np.float32
        )
        if form == 'coo':
            return matrix
        if form == 'csr':
            return matrix.tocsr()
        raise NotImplementedError(
            'sparse matrix format [{}] has not been implemented'.format(form)
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Rerun stored edge interventions and record target-user rank '
            'transitions, relevant-item ranks, and top-K score margins.'
        )
    )
    parser.add_argument('--source-results', default=str(DEFAULT_SOURCE))
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument(
        '--selection-scope',
        choices=('recall-decreasing', 'all-edges-of-affected-users'),
        default='recall-decreasing',
        help=(
            'analyze only the 495 Recall-decreasing edges, or all 1,689 '
            'edges incident to the same 214 affected users'
        )
    )
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--allow-baseline-mismatch',
        action='store_true',
        help=(
            'run as a labelled local replication when target-user baselines '
            'do not exactly match the source CSV'
        )
    )
    parser.add_argument(
        '--max-edges',
        type=int,
        default=None,
        help='optional leading-edge limit for a smoke test'
    )
    args = parser.parse_args(argv)
    if args.max_edges is not None and args.max_edges <= 0:
        parser.error('--max-edges must be positive.')
    if args.eval_batch_size <= 0:
        parser.error('--eval-batch-size must be positive.')
    return args


def result_fieldnames():
    fields = [
        'user_id',
        'edge_id',
        'item_id',
        'user_train_degree',
        'original_edge_weight',
        'item_train_popularity',
        'source_positive_recall_at_5',
        'source_positive_recall_at_20',
        'source_effect_label_at_5',
        'source_effect_label_at_20',
        'local_effect_label_at_5',
        'local_effect_label_at_20',
        'test_positive_count',
    ]
    for metric in METRICS:
        fields.extend([
            'source_user_{}_baseline'.format(metric),
            'source_user_{}_counterfactual'.format(metric),
            'source_user_{}_drop'.format(metric),
            'local_user_{}_baseline'.format(metric),
            'local_user_{}_counterfactual'.format(metric),
            'local_user_{}_drop'.format(metric),
            'baseline_matches_source_{}'.format(metric),
            'source_positive_effect_reproduced_{}'.format(metric),
        ])
    fields.extend([
        'baseline_relevant_ranks',
        'counterfactual_relevant_ranks',
        'mean_relevant_rank_shift',
        'median_relevant_rank_shift',
        'max_relevant_rank_worsening',
        'max_relevant_rank_improvement',
    ])
    for cutoff in CUTOFFS:
        fields.extend([
            'baseline_top_{}_items'.format(cutoff),
            'counterfactual_top_{}_items'.format(cutoff),
            'all_items_left_top_{}'.format(cutoff),
            'all_items_entered_top_{}'.format(cutoff),
            'all_items_left_count_at_{}'.format(cutoff),
            'all_items_entered_count_at_{}'.format(cutoff),
            'relevant_items_left_top_{}'.format(cutoff),
            'relevant_items_entered_top_{}'.format(cutoff),
            'relevant_left_count_at_{}'.format(cutoff),
            'relevant_entered_count_at_{}'.format(cutoff),
            'baseline_score_at_{}'.format(cutoff),
            'baseline_score_at_{}'.format(cutoff + 1),
            'baseline_boundary_gap_at_{}'.format(cutoff),
            'counterfactual_score_at_{}'.format(cutoff),
            'counterfactual_score_at_{}'.format(cutoff + 1),
            'counterfactual_boundary_gap_at_{}'.format(cutoff),
            'boundary_gap_change_at_{}'.format(cutoff),
            'baseline_nearest_relevant_distance_to_{}'.format(cutoff),
            'counterfactual_nearest_relevant_distance_to_{}'.format(cutoff),
            'tracked_relevant_selection_at_{}'.format(cutoff),
            'tracked_relevant_item_at_{}'.format(cutoff),
            'baseline_tracked_relevant_rank_at_{}'.format(cutoff),
            'counterfactual_tracked_relevant_rank_at_{}'.format(cutoff),
            'tracked_relevant_rank_shift_at_{}'.format(cutoff),
            'baseline_tracked_relevant_score_at_{}'.format(cutoff),
            'counterfactual_tracked_relevant_score_at_{}'.format(cutoff),
            'tracked_relevant_score_change_at_{}'.format(cutoff),
            'baseline_competitor_item_at_{}'.format(cutoff),
            'counterfactual_competitor_item_at_{}'.format(cutoff),
            'baseline_tracked_margin_at_{}'.format(cutoff),
            'counterfactual_tracked_margin_at_{}'.format(cutoff),
            'tracked_margin_change_at_{}'.format(cutoff),
        ])
    fields.extend([
        'stored_recall_decrease_reproduced_at_any_selected_cutoff',
        'seconds',
    ])
    return fields


def effect_label(drop):
    """Label the effect of removing an edge using baseline - counterfactual."""
    drop = float(drop)
    if drop > EPSILON:
        return 'decrease'
    if drop < -EPSILON:
        return 'increase'
    return 'neutral'


def json_int_list(values):
    return json.dumps([int(value) for value in values], separators=(',', ':'))


def atomic_write_json_retry(path, payload, attempts=20):
    """Atomically write JSON, tolerating brief Windows file-indexer locks."""
    path = Path(path)
    temporary_path = Path(
        '{}.{}.tmp'.format(path, os.getpid())
    )
    with temporary_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    try:
        for attempt in range(attempts):
            try:
                os.replace(str(temporary_path), str(path))
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except PermissionError:
                pass


def load_completed_rank_edges(results_path, fieldnames, repair=True):
    """Load resumable rows containing JSON/list and boolean fields safely."""
    results_path = Path(results_path)
    if not results_path.exists():
        return set()
    valid_rows = []
    invalid_rows = 0
    edge_position = fieldnames.index('edge_id')
    with results_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header != fieldnames:
            raise RuntimeError(
                'Existing CSV header does not match this analysis version.'
            )
        for row in reader:
            try:
                valid = len(row) == len(fieldnames) and bool(row[edge_position])
                if valid:
                    int(row[edge_position])
            except (TypeError, ValueError):
                valid = False
            if valid:
                valid_rows.append(row)
            else:
                invalid_rows += 1

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


def ranking_snapshot(result_embedding, model, evaluator, user_id):
    """Return target-user metrics using the source evaluator's exact GEMM path."""
    user_id = int(user_id)
    position = evaluator.user_to_position[user_id]
    item_embedding = result_embedding[model.num_user:]
    batch_index = position // evaluator.batch_size
    start, _, user_ids, mask_rows, mask_items = evaluator.batches[batch_index]
    scores_batch = torch.matmul(
        result_embedding[user_ids],
        item_embedding.t()
    )
    if mask_items.numel() > 0:
        scores_batch[mask_rows, mask_items] = -1e10
    score_tensor = scores_batch[position - start]
    top_20 = torch.topk(score_tensor, 20).indices.detach().cpu().numpy()
    scores = np.asarray(
        score_tensor.detach().cpu().numpy(),
        dtype=np.float64
    )

    # Preserve torch.topk's exact first 20 positions (the source metric path),
    # then deterministically order the remaining items for full-rank analysis.
    stable_ranking = np.argsort(-scores, kind='mergesort')
    top_20_set = set(int(item_id) for item_id in top_20)
    remainder = np.asarray(
        [item_id for item_id in stable_ranking if int(item_id) not in top_20_set],
        dtype=np.int64
    )
    ranking = np.concatenate((top_20.astype(np.int64), remainder))
    metrics = cea.compute_metrics_from_topk(
        top_20.reshape(1, -1),
        [evaluator.positive_items[position]]
    )[0]
    metric_values = {
        metric: float(metrics[index])
        for index, metric in enumerate(METRICS)
    }

    inverse_rank = np.empty(model.num_item, dtype=np.int64)
    inverse_rank[ranking] = np.arange(1, model.num_item + 1, dtype=np.int64)
    positives = np.asarray(evaluator.positive_items[position], dtype=np.int64)
    positive_ranks = {
        int(item_id): int(inverse_rank[int(item_id)])
        for item_id in positives
    }

    snapshot = {
        'metrics': metric_values,
        'ranking': ranking,
        'scores': scores,
        'positive_items': set(int(item_id) for item_id in positives),
        'positive_ranks': positive_ranks,
    }
    return snapshot


def transition_fields(baseline, counterfactual):
    row = {}
    positive_ids = sorted(baseline['positive_ranks'])
    baseline_ranks = [baseline['positive_ranks'][item] for item in positive_ids]
    counterfactual_ranks = [
        counterfactual['positive_ranks'][item] for item in positive_ids
    ]
    rank_shifts = np.asarray(counterfactual_ranks) - np.asarray(baseline_ranks)
    row['baseline_relevant_ranks'] = json_int_list(baseline_ranks)
    row['counterfactual_relevant_ranks'] = json_int_list(counterfactual_ranks)
    row['mean_relevant_rank_shift'] = float(rank_shifts.mean())
    row['median_relevant_rank_shift'] = float(np.median(rank_shifts))
    row['max_relevant_rank_worsening'] = int(max(rank_shifts.max(), 0))
    row['max_relevant_rank_improvement'] = int(max((-rank_shifts).max(), 0))

    for cutoff in CUTOFFS:
        base_top = [int(item) for item in baseline['ranking'][:cutoff]]
        cf_top = [int(item) for item in counterfactual['ranking'][:cutoff]]
        base_top_set = set(base_top)
        cf_top_set = set(cf_top)
        base_relevant = base_top_set & baseline['positive_items']
        cf_relevant = cf_top_set & baseline['positive_items']
        relevant_left = sorted(base_relevant - cf_relevant)
        relevant_entered = sorted(cf_relevant - base_relevant)
        all_left = sorted(base_top_set - cf_top_set)
        all_entered = sorted(cf_top_set - base_top_set)

        if relevant_left:
            tracked_item = max(
                relevant_left,
                key=lambda item: baseline['positive_ranks'][item]
            )
            tracked_selection = 'left_top_{}'.format(cutoff)
        elif relevant_entered:
            tracked_item = max(
                relevant_entered,
                key=lambda item: counterfactual['positive_ranks'][item]
            )
            tracked_selection = 'entered_top_{}'.format(cutoff)
        else:
            tracked_item = min(
                positive_ids,
                key=lambda item: (
                    abs(baseline['positive_ranks'][item] - cutoff),
                    baseline['positive_ranks'][item],
                    item
                )
            )
            tracked_selection = 'nearest_baseline_boundary'

        base_tracked_rank = baseline['positive_ranks'][tracked_item]
        cf_tracked_rank = counterfactual['positive_ranks'][tracked_item]
        base_competitor_position = (
            cutoff if base_tracked_rank <= cutoff else cutoff - 1
        )
        cf_competitor_position = (
            cutoff if cf_tracked_rank <= cutoff else cutoff - 1
        )
        base_competitor = int(
            baseline['ranking'][base_competitor_position]
        )
        cf_competitor = int(
            counterfactual['ranking'][cf_competitor_position]
        )
        base_tracked_score = float(baseline['scores'][tracked_item])
        cf_tracked_score = float(counterfactual['scores'][tracked_item])
        base_tracked_margin = (
            base_tracked_score - float(baseline['scores'][base_competitor])
        )
        cf_tracked_margin = (
            cf_tracked_score
            - float(counterfactual['scores'][cf_competitor])
        )

        base_score_k = float(baseline['scores'][baseline['ranking'][cutoff - 1]])
        base_score_next = float(baseline['scores'][baseline['ranking'][cutoff]])
        cf_score_k = float(
            counterfactual['scores'][counterfactual['ranking'][cutoff - 1]]
        )
        cf_score_next = float(
            counterfactual['scores'][counterfactual['ranking'][cutoff]]
        )
        base_gap = base_score_k - base_score_next
        cf_gap = cf_score_k - cf_score_next

        row['baseline_top_{}_items'.format(cutoff)] = json_int_list(base_top)
        row['counterfactual_top_{}_items'.format(cutoff)] = json_int_list(cf_top)
        row['all_items_left_top_{}'.format(cutoff)] = json_int_list(
            all_left
        )
        row['all_items_entered_top_{}'.format(cutoff)] = json_int_list(
            all_entered
        )
        row['all_items_left_count_at_{}'.format(cutoff)] = len(all_left)
        row['all_items_entered_count_at_{}'.format(cutoff)] = len(all_entered)
        row['relevant_items_left_top_{}'.format(cutoff)] = json_int_list(
            relevant_left
        )
        row['relevant_items_entered_top_{}'.format(cutoff)] = json_int_list(
            relevant_entered
        )
        row['relevant_left_count_at_{}'.format(cutoff)] = len(relevant_left)
        row['relevant_entered_count_at_{}'.format(cutoff)] = len(relevant_entered)
        row['baseline_score_at_{}'.format(cutoff)] = base_score_k
        row['baseline_score_at_{}'.format(cutoff + 1)] = base_score_next
        row['baseline_boundary_gap_at_{}'.format(cutoff)] = base_gap
        row['counterfactual_score_at_{}'.format(cutoff)] = cf_score_k
        row['counterfactual_score_at_{}'.format(cutoff + 1)] = cf_score_next
        row['counterfactual_boundary_gap_at_{}'.format(cutoff)] = cf_gap
        row['boundary_gap_change_at_{}'.format(cutoff)] = cf_gap - base_gap
        row['baseline_nearest_relevant_distance_to_{}'.format(cutoff)] = int(
            min(abs(rank - cutoff) for rank in baseline_ranks)
        )
        row['counterfactual_nearest_relevant_distance_to_{}'.format(cutoff)] = int(
            min(abs(rank - cutoff) for rank in counterfactual_ranks)
        )
        row['tracked_relevant_selection_at_{}'.format(cutoff)] = (
            tracked_selection
        )
        row['tracked_relevant_item_at_{}'.format(cutoff)] = int(tracked_item)
        row['baseline_tracked_relevant_rank_at_{}'.format(cutoff)] = int(
            base_tracked_rank
        )
        row['counterfactual_tracked_relevant_rank_at_{}'.format(cutoff)] = int(
            cf_tracked_rank
        )
        row['tracked_relevant_rank_shift_at_{}'.format(cutoff)] = int(
            cf_tracked_rank - base_tracked_rank
        )
        row['baseline_tracked_relevant_score_at_{}'.format(cutoff)] = (
            base_tracked_score
        )
        row['counterfactual_tracked_relevant_score_at_{}'.format(cutoff)] = (
            cf_tracked_score
        )
        row['tracked_relevant_score_change_at_{}'.format(cutoff)] = (
            cf_tracked_score - base_tracked_score
        )
        row['baseline_competitor_item_at_{}'.format(cutoff)] = (
            base_competitor
        )
        row['counterfactual_competitor_item_at_{}'.format(cutoff)] = (
            cf_competitor
        )
        row['baseline_tracked_margin_at_{}'.format(cutoff)] = (
            base_tracked_margin
        )
        row['counterfactual_tracked_margin_at_{}'.format(cutoff)] = (
            cf_tracked_margin
        )
        row['tracked_margin_change_at_{}'.format(cutoff)] = (
            cf_tracked_margin - base_tracked_margin
        )
    return row


def select_source_edges(source_results, selection_scope, max_edges=None):
    source = pd.read_csv(source_results)
    required = {
        'user_id', 'edge_id', 'item_id', 'user_train_degree',
        'original_edge_weight',
    }
    for metric in METRICS:
        for kind in ('baseline', 'counterfactual', 'drop'):
            required.add('user_{}_{}'.format(metric, kind))
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError('Source CSV is missing fields: {}'.format(missing))

    decreasing_mask = (
        (source['user_recall_at_5_drop'] > EPSILON)
        | (source['user_recall_at_20_drop'] > EPSILON)
    )
    decreasing = source.loc[decreasing_mask].copy()
    if len(decreasing) != 495:
        raise RuntimeError(
            'Expected 495 Recall-decreasing source edges, found {}.'.format(
                len(decreasing)
            )
        )
    affected_user_ids = decreasing['user_id'].drop_duplicates()
    if len(affected_user_ids) != 214:
        raise RuntimeError(
            'Expected 214 affected users, found {}.'.format(
                len(affected_user_ids)
            )
        )
    if selection_scope == 'recall-decreasing':
        selected = decreasing
    elif selection_scope == 'all-edges-of-affected-users':
        selected = source.loc[
            source['user_id'].isin(affected_user_ids)
        ].copy()
        if len(selected) != 1689:
            raise RuntimeError(
                'Expected 1,689 edges for affected users, found {}.'.format(
                    len(selected)
                )
            )
    else:
        raise ValueError('Unknown selection scope: {}'.format(selection_scope))
    if max_edges is not None:
        selected = selected.head(max_edges).copy()
    return source, selected


def baseline_match_summary(selected, baseline_snapshots):
    source_users = selected.drop_duplicates('user_id')
    summary = {'users': int(len(source_users)), 'metrics': {}}
    all_metric_matches = np.ones(len(source_users), dtype=bool)
    for metric in METRICS:
        matches = []
        differences = []
        for row in source_users.itertuples(index=False):
            source_value = float(
                getattr(row, 'user_{}_baseline'.format(metric))
            )
            local_value = baseline_snapshots[int(row.user_id)]['metrics'][metric]
            difference = abs(source_value - local_value)
            differences.append(difference)
            matches.append(difference <= EPSILON)
        matches = np.asarray(matches, dtype=bool)
        all_metric_matches &= matches
        summary['metrics'][metric] = {
            'exact_match_count': int(matches.sum()),
            'exact_match_rate': float(matches.mean()),
            'mean_absolute_difference': float(np.mean(differences)),
            'max_absolute_difference': float(np.max(differences)),
        }
    summary['all_metrics_exact_match_count'] = int(all_metric_matches.sum())
    summary['all_metrics_exact_match_rate'] = float(all_metric_matches.mean())
    summary['context_exactly_reproduced'] = bool(all_metric_matches.all())
    return summary


def build_row(
    source_row,
    baseline,
    counterfactual,
    item_popularity,
    elapsed_seconds
):
    row = {
        'user_id': int(source_row.user_id),
        'edge_id': int(source_row.edge_id),
        'item_id': int(source_row.item_id),
        'user_train_degree': int(source_row.user_train_degree),
        'original_edge_weight': float(source_row.original_edge_weight),
        'item_train_popularity': int(
            item_popularity.get(int(source_row.item_id), 0)
        ),
        'source_positive_recall_at_5': bool(
            source_row.user_recall_at_5_drop > EPSILON
        ),
        'source_positive_recall_at_20': bool(
            source_row.user_recall_at_20_drop > EPSILON
        ),
        'source_effect_label_at_5': effect_label(
            source_row.user_recall_at_5_drop
        ),
        'source_effect_label_at_20': effect_label(
            source_row.user_recall_at_20_drop
        ),
        'test_positive_count': len(baseline['positive_items']),
    }
    reproduced_selected_effect = False
    for metric in METRICS:
        source_baseline = float(
            getattr(source_row, 'user_{}_baseline'.format(metric))
        )
        source_counterfactual = float(
            getattr(source_row, 'user_{}_counterfactual'.format(metric))
        )
        source_drop = float(
            getattr(source_row, 'user_{}_drop'.format(metric))
        )
        local_baseline = baseline['metrics'][metric]
        local_counterfactual = counterfactual['metrics'][metric]
        local_drop = local_baseline - local_counterfactual
        source_positive = source_drop > EPSILON
        reproduced = (local_drop > EPSILON) if source_positive else False

        row['source_user_{}_baseline'.format(metric)] = source_baseline
        row['source_user_{}_counterfactual'.format(metric)] = source_counterfactual
        row['source_user_{}_drop'.format(metric)] = source_drop
        row['local_user_{}_baseline'.format(metric)] = local_baseline
        row['local_user_{}_counterfactual'.format(metric)] = local_counterfactual
        row['local_user_{}_drop'.format(metric)] = local_drop
        row['baseline_matches_source_{}'.format(metric)] = bool(
            abs(source_baseline - local_baseline) <= EPSILON
        )
        row['source_positive_effect_reproduced_{}'.format(metric)] = bool(
            reproduced
        )
        if metric in ('recall_at_5', 'recall_at_20'):
            cutoff = int(metric.rsplit('_', 1)[1])
            row['local_effect_label_at_{}'.format(cutoff)] = effect_label(
                local_drop
            )
        if metric in ('recall_at_5', 'recall_at_20') and source_positive:
            reproduced_selected_effect = reproduced_selected_effect or reproduced

    row.update(transition_fields(baseline, counterfactual))
    row['stored_recall_decrease_reproduced_at_any_selected_cutoff'] = bool(
        reproduced_selected_effect
    )
    row['seconds'] = float(elapsed_seconds)
    return row


def build_affected_user_summary(results):
    """Collapse edge-level diagnostics into one row per affected user."""
    rows = []
    for user_id, group in results.groupby('user_id', sort=True):
        first = group.iloc[0]
        flag_5 = group['source_positive_recall_at_5'].astype(bool)
        flag_20 = group['source_positive_recall_at_20'].astype(bool)
        row = {
            'user_id': int(user_id),
            'user_train_degree': int(first['user_train_degree']),
            'test_positive_count': int(first['test_positive_count']),
            'analyzed_edge_count': int(len(group)),
            'analyzed_edge_ratio': float(
                len(group) / first['user_train_degree']
            ),
            'recall_at_5_decrease_edge_count': int(flag_5.sum()),
            'recall_at_20_decrease_edge_count': int(flag_20.sum()),
            'both_cutoffs_decrease_edge_count': int((flag_5 & flag_20).sum()),
            'mean_analyzed_edge_weight': float(
                group['original_edge_weight'].mean()
            ),
        }
        for metric in METRICS:
            row['baseline_{}'.format(metric)] = float(
                first['local_user_{}_baseline'.format(metric)]
            )
        for cutoff, flag in ((5, flag_5), (20, flag_20)):
            relevant = group.loc[flag]
            prefix = 'recall_at_{}'.format(cutoff)
            labels = group['source_effect_label_at_{}'.format(cutoff)]
            increasing = group.loc[labels.eq('increase')]
            row['recall_at_{}_neutral_edge_count'.format(cutoff)] = int(
                labels.eq('neutral').sum()
            )
            row['recall_at_{}_increase_edge_count'.format(cutoff)] = int(
                labels.eq('increase').sum()
            )
            row['mean_{}_drop'.format(prefix)] = float(
                relevant['local_user_{}_drop'.format(prefix)].mean()
            ) if len(relevant) else np.nan
            row['max_{}_drop'.format(prefix)] = float(
                relevant['local_user_{}_drop'.format(prefix)].max()
            ) if len(relevant) else np.nan
            row['mean_boundary_gap_at_{}'.format(cutoff)] = float(
                relevant['baseline_boundary_gap_at_{}'.format(cutoff)].mean()
            ) if len(relevant) else np.nan
            row['mean_max_rank_worsening_at_{}'.format(cutoff)] = float(
                relevant['max_relevant_rank_worsening'].mean()
            ) if len(relevant) else np.nan
            row['mean_recall_at_{}_gain'.format(cutoff)] = float(
                -increasing['local_user_recall_at_{}_drop'.format(cutoff)].mean()
            ) if len(increasing) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_effect_group_summary(results):
    """Summarize decrease/neutral/increase edges at each Recall cutoff."""
    frame = results.copy()
    frame['weight_centered_within_user'] = (
        frame['original_edge_weight']
        - frame.groupby('user_id')['original_edge_weight'].transform('mean')
    )
    frame['popularity_centered_within_user'] = (
        frame['item_train_popularity']
        - frame.groupby('user_id')['item_train_popularity'].transform('mean')
    )
    rows = []
    for cutoff in CUTOFFS:
        source_label = 'source_effect_label_at_{}'.format(cutoff)
        local_label = 'local_effect_label_at_{}'.format(cutoff)
        for label in ('decrease', 'neutral', 'increase'):
            group = frame.loc[frame[source_label].eq(label)]
            if not len(group):
                continue
            rows.append({
                'cutoff': cutoff,
                'effect_label_when_edge_removed': label,
                'edge_count': int(len(group)),
                'exact_local_label_match_count': int(
                    group[local_label].eq(label).sum()
                ),
                'exact_local_label_match_rate': float(
                    group[local_label].eq(label).mean()
                ),
                'mean_raw_recall_drop': float(
                    group['local_user_recall_at_{}_drop'.format(cutoff)].mean()
                ),
                'median_raw_recall_drop': float(
                    group['local_user_recall_at_{}_drop'.format(cutoff)].median()
                ),
                'mean_edge_weight': float(group['original_edge_weight'].mean()),
                'median_edge_weight': float(
                    group['original_edge_weight'].median()
                ),
                'mean_weight_centered_within_user': float(
                    group['weight_centered_within_user'].mean()
                ),
                'mean_item_popularity': float(
                    group['item_train_popularity'].mean()
                ),
                'median_item_popularity': float(
                    group['item_train_popularity'].median()
                ),
                'mean_popularity_centered_within_user': float(
                    group['popularity_centered_within_user'].mean()
                ),
                'mean_tracked_relevant_rank_shift': float(
                    group[
                        'tracked_relevant_rank_shift_at_{}'.format(cutoff)
                    ].mean()
                ),
                'median_tracked_relevant_rank_shift': float(
                    group[
                        'tracked_relevant_rank_shift_at_{}'.format(cutoff)
                    ].median()
                ),
                'mean_tracked_relevant_score_change': float(
                    group[
                        'tracked_relevant_score_change_at_{}'.format(cutoff)
                    ].mean()
                ),
                'mean_tracked_margin_change': float(
                    group['tracked_margin_change_at_{}'.format(cutoff)].mean()
                ),
                'median_tracked_margin_change': float(
                    group['tracked_margin_change_at_{}'.format(cutoff)].median()
                ),
                'mean_top_k_item_churn': float(
                    group['all_items_left_count_at_{}'.format(cutoff)].mean()
                ),
            })
    return pd.DataFrame(rows)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_similarity_features(
    results,
    baseline_embedding,
    model,
    full_view,
    baseline_snapshots,
    text_feature_path,
    user_feature_path
):
    """Measure how each train-edge item aligns with the user's test targets."""
    final = baseline_embedding.detach().cpu().numpy()
    final_user = final[:model.num_user]
    final_item = final[model.num_user:]
    masked = model.mask_rep.detach().cpu().numpy()
    masked_user = masked[:model.num_user]
    masked_item = masked[model.num_user:]
    full_user = full_view[0].detach().cpu().numpy()
    full_item = full_view[1].detach().cpu().numpy()
    text_item = np.load(text_feature_path)
    text_user = np.load(user_feature_path)

    spaces = {
        'final': (normalize_rows(final_user), normalize_rows(final_item)),
        'masked': (normalize_rows(masked_user), normalize_rows(masked_item)),
        'full': (normalize_rows(full_user), normalize_rows(full_item)),
        'text': (normalize_rows(text_user), normalize_rows(text_item)),
    }
    feature_columns = [
        'user_id', 'edge_id', 'item_id', 'original_edge_weight',
        'item_train_popularity', 'source_effect_label_at_5',
        'source_effect_label_at_20', 'tracked_relevant_item_at_5',
        'tracked_relevant_item_at_20', 'tracked_margin_change_at_5',
        'tracked_margin_change_at_20',
    ]
    features = results[feature_columns].copy()
    user_ids = results['user_id'].to_numpy(dtype=np.int64)
    edge_item_ids = results['item_id'].to_numpy(dtype=np.int64)

    positive_ids_by_user = {
        int(user_id): np.asarray(
            sorted(baseline_snapshots[int(user_id)]['positive_items']),
            dtype=np.int64
        )
        for user_id in np.unique(user_ids)
    }
    for name, (user_vectors, item_vectors) in spaces.items():
        positive_centroids = {
            user_id: normalize_rows(
                item_vectors[item_ids].mean(axis=0, keepdims=True)
            )[0]
            for user_id, item_ids in positive_ids_by_user.items()
        }
        centroid_rows = np.stack(
            [positive_centroids[int(user_id)] for user_id in user_ids]
        )
        edge_vectors = item_vectors[edge_item_ids]
        features['{}_edge_to_user_cosine'.format(name)] = np.sum(
            edge_vectors * user_vectors[user_ids], axis=1
        )
        features[
            '{}_edge_to_test_positive_centroid_cosine'.format(name)
        ] = np.sum(edge_vectors * centroid_rows, axis=1)
        for cutoff in CUTOFFS:
            tracked_ids = results[
                'tracked_relevant_item_at_{}'.format(cutoff)
            ].to_numpy(dtype=np.int64)
            features[
                '{}_edge_to_tracked_relevant_at_{}_cosine'.format(
                    name, cutoff
                )
            ] = np.sum(edge_vectors * item_vectors[tracked_ids], axis=1)
    return features


def build_similarity_group_summary(similarity_features):
    similarity_columns = [
        column for column in similarity_features.columns
        if column.endswith('_cosine')
    ]
    rows = []
    for cutoff in CUTOFFS:
        label_column = 'source_effect_label_at_{}'.format(cutoff)
        for label in ('decrease', 'neutral', 'increase'):
            group = similarity_features.loc[
                similarity_features[label_column].eq(label)
            ]
            if not len(group):
                continue
            row = {
                'cutoff': cutoff,
                'effect_label_when_edge_removed': label,
                'edge_count': int(len(group)),
            }
            for column in similarity_columns:
                row['mean_{}'.format(column)] = float(group[column].mean())
                row['median_{}'.format(column)] = float(group[column].median())
            rows.append(row)
    return pd.DataFrame(rows)


def build_feature_contrast_summary(results, similarity_features):
    """Quantify raw and within-user feature separation between effect groups."""
    mechanism_columns = ['edge_id']
    for cutoff in CUTOFFS:
        mechanism_columns.extend([
            'tracked_relevant_score_change_at_{}'.format(cutoff),
            'tracked_relevant_rank_shift_at_{}'.format(cutoff),
        ])
    frame = similarity_features.merge(
        results[mechanism_columns], on='edge_id', validate='one_to_one'
    )
    rows = []
    for cutoff in CUTOFFS:
        label_column = 'source_effect_label_at_{}'.format(cutoff)
        feature_specs = [
            ('pre_intervention', 'original_edge_weight'),
            ('pre_intervention', 'item_train_popularity'),
            (
                'pre_intervention',
                'masked_edge_to_test_positive_centroid_cosine'
            ),
            (
                'pre_intervention',
                'final_edge_to_test_positive_centroid_cosine'
            ),
            (
                'pre_intervention',
                'text_edge_to_test_positive_centroid_cosine'
            ),
            (
                'outcome_tracked_explanation',
                'masked_edge_to_tracked_relevant_at_{}_cosine'.format(cutoff)
            ),
            (
                'outcome_tracked_explanation',
                'final_edge_to_tracked_relevant_at_{}_cosine'.format(cutoff)
            ),
            (
                'outcome_tracked_explanation',
                'text_edge_to_tracked_relevant_at_{}_cosine'.format(cutoff)
            ),
            (
                'post_intervention_mechanism',
                'tracked_relevant_score_change_at_{}'.format(cutoff)
            ),
            (
                'post_intervention_mechanism',
                'tracked_margin_change_at_{}'.format(cutoff)
            ),
            (
                'post_intervention_mechanism',
                'tracked_relevant_rank_shift_at_{}'.format(cutoff)
            ),
        ]
        for feature_role, feature in feature_specs:
            centered = (
                frame[feature]
                - frame.groupby('user_id')[feature].transform('mean')
            )
            for first_label, second_label in (
                ('decrease', 'neutral'),
                ('decrease', 'increase'),
            ):
                first_mask = frame[label_column].eq(first_label)
                second_mask = frame[label_column].eq(second_label)
                first = frame.loc[first_mask, feature]
                second = frame.loc[second_mask, feature]
                if not len(first) or not len(second):
                    continue
                statistic, p_value = mannwhitneyu(
                    first, second, alternative='two-sided'
                )
                first_centered = centered.loc[first_mask]
                second_centered = centered.loc[second_mask]
                centered_statistic, centered_p_value = mannwhitneyu(
                    first_centered,
                    second_centered,
                    alternative='two-sided'
                )
                rows.append({
                    'cutoff': cutoff,
                    'feature_role': feature_role,
                    'feature': feature,
                    'first_effect_label': first_label,
                    'second_effect_label': second_label,
                    'first_count': int(len(first)),
                    'second_count': int(len(second)),
                    'first_mean': float(first.mean()),
                    'second_mean': float(second.mean()),
                    'auc_probability_first_higher': float(
                        statistic / (len(first) * len(second))
                    ),
                    'mann_whitney_p_value': float(p_value),
                    'first_within_user_centered_mean': float(
                        first_centered.mean()
                    ),
                    'second_within_user_centered_mean': float(
                        second_centered.mean()
                    ),
                    'within_user_centered_auc': float(
                        centered_statistic / (len(first) * len(second))
                    ),
                    'within_user_centered_p_value': float(centered_p_value),
                })
    return pd.DataFrame(rows)


def run(args):
    source_results = Path(args.source_results).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    default_output = (
        DEFAULT_AFFECTED_ALL_OUTPUT
        if args.selection_scope == 'all-edges-of-affected-users'
        else DEFAULT_DECREASING_OUTPUT
    )
    output_dir = Path(args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'edge_rank_margin_results.csv'
    metadata_path = output_dir / 'metadata.json'
    progress_path = output_dir / 'progress.json'
    summary_path = output_dir / 'summary.json'
    user_summary_path = output_dir / 'affected_user_summary.csv'
    effect_summary_path = output_dir / 'effect_group_summary.csv'
    similarity_path = output_dir / 'edge_similarity_features.csv'
    similarity_summary_path = output_dir / 'similarity_group_summary.csv'
    contrast_summary_path = output_dir / 'feature_contrast_summary.csv'

    source, selected = select_source_edges(
        source_results,
        args.selection_scope,
        args.max_edges
    )
    target_user_ids = selected['user_id'].astype(int).drop_duplicates().tolist()
    print('Selection scope: {}'.format(args.selection_scope))
    print('Selected edges : {}'.format(len(selected)))
    print('Target users   : {}'.format(len(target_user_ids)))

    config_args = argparse.Namespace(
        gpu_id=args.gpu_id,
        selection_seed=args.seed,
        eval_batch_size=args.eval_batch_size,
    )
    original_cwd = Path.cwd()
    os.chdir(str(PROJECT_ROOT))
    try:
        config = cea.build_analysis_config(config_args)
        cea.init_seed(args.seed)
        dataset = cea.RecDataset(config)
        train_dataset, _, test_dataset = dataset.split()
        canonicalize_train_edge_order(train_dataset)
        train_data = CanonicalEdgeTrainDataLoader(
            config,
            train_dataset,
            batch_size=config['train_batch_size'],
            shuffle=True
        )
        model = cea.get_model('MASKED_GLORIA')(config, train_data).to(
            config['device']
        )
        cea.load_checkpoint_strict(model, checkpoint_path)
        model.eval()
        evaluator = cea.ExactFullTestEvaluator(
            train_dataset,
            test_dataset,
            model.num_user,
            model.num_item,
            config['device'],
            args.eval_batch_size
        )

        edge_users = model.forward_edge_users.detach().cpu().numpy()
        edge_items = model.forward_edge_items.detach().cpu().numpy()
        mask_logits_before = model.mask_logits.detach().clone()
        with torch.inference_mode():
            full_view = model.compute_full_view()
            base_mask = model.get_forward_edge_mask().detach()
            baseline_embedding = model.compute_result_embedding(
                base_mask,
                full_view=full_view
            )
            baseline_snapshots = {
                user_id: ranking_snapshot(
                    baseline_embedding, model, evaluator, user_id
                )
                for user_id in target_user_ids
            }

        match_summary = baseline_match_summary(selected, baseline_snapshots)
        selection_rule = (
            'all source edges belonging to users with at least one edge where '
            'user_recall_at_5_drop > 1e-12 OR '
            'user_recall_at_20_drop > 1e-12'
            if args.selection_scope == 'all-edges-of-affected-users'
            else (
                'source user_recall_at_5_drop > 1e-12 OR '
                'source user_recall_at_20_drop > 1e-12'
            )
        )
        metadata = {
            'created_at_utc': cea.utc_now(),
            'experiment': 'targeted_edge_effect_rank_margin_comparison',
            'selection_scope': args.selection_scope,
            'selection_rule': selection_rule,
            'source_results': cea.file_fingerprint(source_results),
            'checkpoint': cea.file_fingerprint(checkpoint_path),
            'dataset_file': cea.file_fingerprint(
                PROJECT_ROOT / 'data' / 'book' / 'book.inter'
            ),
            'text_feature_file': cea.file_fingerprint(
                PROJECT_ROOT / 'data' / 'book' / 'text_feat.npy'
            ),
            'user_feature_file': cea.file_fingerprint(
                PROJECT_ROOT / 'data' / 'book' / 'user_feat.npy'
            ),
            'device': str(config['device']),
            'seed': int(args.seed),
            'train_edge_order': (
                'user_id_then_item_id_ascending_stable_float32_coo'
            ),
            'target_edges': int(len(selected)),
            'target_users': int(len(target_user_ids)),
            'baseline_match': match_summary,
            'allow_baseline_mismatch': bool(args.allow_baseline_mismatch),
            'interpretation': (
                'direct_extension' if match_summary['context_exactly_reproduced']
                else 'local_replication_with_context_mismatch'
            ),
        }
        atomic_write_json_retry(metadata_path, metadata)
        if (
            not match_summary['context_exactly_reproduced']
            and not args.allow_baseline_mismatch
        ):
            raise RuntimeError(
                'Target-user baselines do not exactly reproduce the source '
                'CSV. See {}. Pass --allow-baseline-mismatch only to run a '
                'labelled local replication.'.format(metadata_path)
            )

        target_edge_ids = selected['edge_id'].astype(int).to_numpy()
        if target_edge_ids.min() < 0 or target_edge_ids.max() >= len(edge_users):
            raise IndexError('Source edge IDs are outside the model edge mask.')
        selected_users = selected['user_id'].astype(int).to_numpy()
        selected_items = selected['item_id'].astype(int).to_numpy()
        if not np.array_equal(edge_users[target_edge_ids], selected_users):
            raise RuntimeError('Source edge IDs do not match model user IDs.')
        if not np.array_equal(edge_items[target_edge_ids], selected_items):
            raise RuntimeError('Source edge IDs do not match model item IDs.')
        source_weights = selected['original_edge_weight'].to_numpy(dtype=float)
        local_weights = base_mask[
            torch.as_tensor(
                target_edge_ids,
                dtype=torch.long,
                device=base_mask.device
            )
        ].detach().cpu().numpy()
        if not np.allclose(source_weights, local_weights, atol=1e-6, rtol=0.0):
            raise RuntimeError('Source weights do not match the model mask.')

        item_popularity = (
            train_dataset.df.groupby(train_dataset.iid_field).size().to_dict()
        )
        fields = result_fieldnames()
        completed = load_completed_rank_edges(
            results_path,
            fields,
            repair=True
        ) if args.resume else set()
        if results_path.exists() and not args.resume:
            raise FileExistsError(
                'Output exists; pass --resume or choose another --output-dir: '
                '{}'.format(results_path)
            )

        csv_exists = results_path.exists()
        mode = 'a' if csv_exists else 'w'
        started_at = time.time()
        processed = 0
        selected_by_edge = selected.set_index('edge_id', drop=False)
        with results_path.open(mode, encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not csv_exists:
                writer.writeheader()
                handle.flush()

            for position, edge_id in enumerate(target_edge_ids, start=1):
                edge_id = int(edge_id)
                if edge_id in completed:
                    continue
                source_row = selected_by_edge.loc[edge_id]
                user_id = int(source_row['user_id'])
                edge_started = time.time()
                with torch.inference_mode():
                    counterfactual_mask = model.get_counterfactual_forward_mask(
                        edge_id,
                        base_mask=base_mask
                    )
                    counterfactual_embedding = model.compute_result_embedding(
                        counterfactual_mask,
                        full_view=full_view
                    )
                    counterfactual_snapshot = ranking_snapshot(
                        counterfactual_embedding,
                        model,
                        evaluator,
                        user_id
                    )
                source_namespace = argparse.Namespace(**source_row.to_dict())
                row = build_row(
                    source_namespace,
                    baseline_snapshots[user_id],
                    counterfactual_snapshot,
                    item_popularity,
                    time.time() - edge_started
                )
                writer.writerow(row)
                handle.flush()
                completed.add(edge_id)
                processed += 1

                elapsed = time.time() - started_at
                average = elapsed / processed
                remaining = len(selected) - len(completed)
                progress = {
                    'status': 'running',
                    'completed': int(len(completed)),
                    'total': int(len(selected)),
                    'percent': 100.0 * len(completed) / len(selected),
                    'average_seconds': average,
                    'eta_seconds': average * remaining,
                    'last_edge_id': edge_id,
                    'updated_at_utc': cea.utc_now(),
                }
                atomic_write_json_retry(progress_path, progress)
                if position == 1 or position % 25 == 0 or position == len(selected):
                    print(
                        '[{}/{}] edge={} user={} ETA={}'.format(
                            len(completed),
                            len(selected),
                            edge_id,
                            user_id,
                            cea.format_duration(progress['eta_seconds'])
                        ),
                        flush=True
                    )

        with torch.inference_mode():
            model.compute_result_embedding(base_mask, full_view=full_view)
        if not torch.equal(model.mask_logits.detach(), mask_logits_before):
            raise RuntimeError('Analysis unexpectedly modified mask_logits.')

        results = pd.read_csv(results_path)
        user_summary = build_affected_user_summary(results)
        user_summary.to_csv(user_summary_path, index=False)
        effect_summary = build_effect_group_summary(results)
        effect_summary.to_csv(effect_summary_path, index=False)
        similarity_features = build_similarity_features(
            results,
            baseline_embedding,
            model,
            full_view,
            baseline_snapshots,
            PROJECT_ROOT / 'data' / 'book' / 'text_feat.npy',
            PROJECT_ROOT / 'data' / 'book' / 'user_feat.npy'
        )
        similarity_features.to_csv(similarity_path, index=False)
        similarity_summary = build_similarity_group_summary(
            similarity_features
        )
        similarity_summary.to_csv(similarity_summary_path, index=False)
        contrast_summary = build_feature_contrast_summary(
            results, similarity_features
        )
        contrast_summary.to_csv(contrast_summary_path, index=False)
        recall_rows = []
        for cutoff in CUTOFFS:
            source_flag = results['source_positive_recall_at_{}'.format(cutoff)]
            reproduced = results[
                'source_positive_effect_reproduced_recall_at_{}'.format(cutoff)
            ]
            relevant = results.loc[source_flag]
            reproduced_relevant = reproduced.loc[source_flag]
            has_relevant = len(relevant) > 0
            recall_rows.append({
                'cutoff': cutoff,
                'source_positive_edges': int(source_flag.sum()),
                'locally_reproduced_positive_edges': int(
                    reproduced_relevant.sum()
                ),
                'reproduction_rate': (
                    float(reproduced_relevant.mean()) if has_relevant else None
                ),
                'mean_local_drop': (
                    float(relevant[
                        'local_user_recall_at_{}_drop'.format(cutoff)
                    ].mean()) if has_relevant else None
                ),
                'mean_relevant_left_count': (
                    float(relevant[
                        'relevant_left_count_at_{}'.format(cutoff)
                    ].mean()) if has_relevant else None
                ),
            })
        summary = {
            'completed_at_utc': cea.utc_now(),
            'interpretation': metadata['interpretation'],
            'completed_edges': int(len(results)),
            'baseline_match': match_summary,
            'recall_reproduction': recall_rows,
            'effect_group_summary': effect_summary.to_dict(orient='records'),
            'mean_seconds_per_edge': float(results['seconds'].mean()),
        }
        atomic_write_json_retry(summary_path, summary)
        atomic_write_json_retry(progress_path, {
            'status': 'complete',
            'completed': int(len(results)),
            'total': int(len(selected)),
            'percent': 100.0,
            'updated_at_utc': cea.utc_now(),
        })
        print('Analysis complete: {}'.format(results_path))
        print('User summary     : {}'.format(user_summary_path))
        print('Effect summary   : {}'.format(effect_summary_path))
        print('Similarity       : {}'.format(similarity_path))
        print('Similarity groups: {}'.format(similarity_summary_path))
        print('Feature contrasts: {}'.format(contrast_summary_path))
        print('Summary          : {}'.format(summary_path))
        return 0
    finally:
        os.chdir(str(original_cwd))


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
