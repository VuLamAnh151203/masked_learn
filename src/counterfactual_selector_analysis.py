# coding: utf-8
"""Offline validation for counterfactual edge selectors.

The benchmark deliberately performs exhaustive actual drops offline.  It uses
one seeded train-history holdout per user, which matches the pseudo-positive
protocol used by MASKED_GLORIA_CF without using test labels.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

SELECTORS = ('representation', 'gradient', 'random')
TOP_NS = (1, 3, 5)


def recall_at_n(selector_edges, harmful_edges, n):
    """Return edge recall for a ranked candidate list."""
    harmful_edges = set(int(edge_id) for edge_id in harmful_edges)
    if not harmful_edges:
        return None
    selected = set(int(edge_id) for edge_id in selector_edges[:int(n)])
    return float(len(selected & harmful_edges)) / float(len(harmful_edges))


def user_hit_at_n(selector_edges, harmful_edges, n):
    harmful_edges = set(int(edge_id) for edge_id in harmful_edges)
    if not harmful_edges:
        return None
    return float(bool(set(selector_edges[:int(n)]) & harmful_edges))


def summarize_rows(rows):
    """Aggregate per-user selector rows into JSON-serializable summaries."""
    summaries = []
    for selector in SELECTORS:
        selector_rows = [row for row in rows if row['selector'] == selector]
        for n in TOP_NS:
            eligible = [
                row for row in selector_rows
                if row['harmful_count'] > 0
            ]
            recalls = [
                row['recall_at_{}'.format(n)] for row in eligible
                if row['recall_at_{}'.format(n)] is not None
            ]
            hits = [
                row['hit_at_{}'.format(n)] for row in eligible
                if row['hit_at_{}'.format(n)] is not None
            ]
            harmful_total = sum(row['harmful_count'] for row in eligible)
            captured_total = sum(
                row['captured_at_{}'.format(n)] for row in eligible
            )
            summaries.append({
                'selector': selector,
                'top_n': int(n),
                'pseudo_count': len(selector_rows),
                'eligible_pseudo_count': len(eligible),
                'pseudo_without_harmful_count': (
                    len(selector_rows) - len(eligible)
                ),
                'mean_candidate_count': (
                    float(sum(row['candidate_count'] for row in selector_rows))
                    / float(len(selector_rows))
                    if selector_rows else None
                ),
                'harmful_edge_count': int(harmful_total),
                'captured_harmful_edge_count': int(captured_total),
                'edge_recall_micro': (
                    float(captured_total) / float(harmful_total)
                    if harmful_total else None
                ),
                'edge_recall_macro': (
                    float(sum(recalls)) / float(len(recalls))
                    if recalls else None
                ),
                'user_hit_rate': (
                    float(sum(hits)) / float(len(hits))
                    if hits else None
                ),
            })
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Benchmark gradient, representation, and random CF selectors.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default='book')
    parser.add_argument('--gpu_id', '--gpu-id', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--number_of_user', '--number-of-user', dest='number_of_user', type=int, default=None)
    parser.add_argument('--output_dir', '--output-dir', dest='output_dir', default='counterfactual_selector_results')
    parser.add_argument('--cf_k', '--cf-k', dest='cf_k', type=int, default=20)
    parser.add_argument('--cf_boundary_width', '--cf-boundary-width', dest='cf_boundary_width', type=int, default=5)
    parser.add_argument('--cf_min_history', '--cf-min-history', dest='cf_min_history', type=int, default=2)
    parser.add_argument('--damage_eps', '--damage-eps', dest='damage_eps', type=float, default=1e-8)
    return parser.parse_args(argv)


def build_config(args):
    from utils.configurator import Config

    use_gpu = int(args.gpu_id) >= 0
    return Config(
        'MASKED_GLORIA_CF',
        args.dataset,
        {
            'gpu_id': max(int(args.gpu_id), 0),
            'use_gpu': use_gpu,
            'fusion': 'add',
            'dropout': 0.2,
            'reg_weight': 0.001,
            'learning_rate': 0.003,
            'seed': int(args.seed),
            'cf_lambda': 0.0,
            'cf_warmup_ratio': 0.0,
            'cf_warmup_epochs': 1,
            'cf_user_ratio': 1.0,
            'cf_batch_size': 1024,
            'cf_k': int(args.cf_k),
            'cf_boundary_width': int(args.cf_boundary_width),
            'cf_min_history': int(args.cf_min_history),
            'cf_edge_selector': 'representation',
            'cf_selector_top_n': 3,
            'cf_selector_damage_eps': float(args.damage_eps),
            'cf_log_stats': False,
        }
    )


def _load_model(args, config, train_data):
    from counterfactual_edge_analysis import load_checkpoint_strict
    from utils.utils import get_model

    model_class = get_model('MASKED_GLORIA_CF')
    model = model_class(config, train_data).to(config['device'])
    load_checkpoint_strict(model, Path(args.checkpoint).expanduser().resolve())
    model.eval()
    return model


def _probe_user(model, full_view, base_mask, user_id, pseudo_edge_id):
    pseudo_item_id = int(model.forward_edge_items[pseudo_edge_id].item())
    probe_mask = base_mask.clone()
    probe_mask[pseudo_edge_id] = 0.0
    with torch.no_grad():
        probe_embed = model.compute_result_embedding(
            forward_edge_mask=probe_mask,
            full_view=full_view
        )
        probe_scores = model._score_user_items(probe_embed, user_id)
        model._mask_remaining_history(
            probe_scores,
            user_id,
            pseudo_item_id
        )
        rank = model._pseudo_positive_rank(probe_scores, pseudo_item_id)
        boundary_item = model._select_fixed_boundary_item(
            probe_scores,
            pseudo_item_id
        )
    if boundary_item is None or not model._is_fragile_rank(rank):
        return None
    margin = float(
        (probe_scores[pseudo_item_id] - probe_scores[boundary_item]).cpu()
    )
    return {
        'pseudo_edge_id': int(pseudo_edge_id),
        'pseudo_item_id': pseudo_item_id,
        'probe_mask': probe_mask,
        'probe_margin': margin,
        'boundary_item_id': int(boundary_item),
    }


def _actual_damages(model, full_view, base_mask, user_id, probe, candidates):
    damages = {}
    with torch.no_grad():
        for edge_id in candidates:
            cf_mask = base_mask.clone()
            cf_mask[probe['pseudo_edge_id']] = 0.0
            cf_mask[int(edge_id)] = 0.0
            cf_embed = model.compute_result_embedding(
                forward_edge_mask=cf_mask,
                full_view=full_view
            )
            scores = model._score_user_items(cf_embed, user_id)
            model._mask_remaining_history(
                scores,
                user_id,
                probe['pseudo_item_id']
            )
            cf_margin = (
                scores[probe['pseudo_item_id']]
                - scores[probe['boundary_item_id']]
            )
            damages[int(edge_id)] = probe['probe_margin'] - float(
                cf_margin.cpu()
            )
    return damages


def run(args):
    from utils.dataloader import TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed

    os.chdir(str(PROJECT_ROOT))
    init_seed(args.seed)
    config = build_config(args)
    dataset = RecDataset(config)
    train_dataset, _, _ = dataset.split()
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False
    )
    model = _load_model(args, config, train_data)

    with torch.no_grad():
        full_view = model.compute_full_view()
        base_mask = model.get_forward_edge_mask().detach()

    rng = random.Random(int(args.seed))
    user_ids = [
        user_id for user_id, edges in enumerate(model.user_to_edge_ids)
        if len(edges) >= int(args.cf_min_history)
    ]
    if args.number_of_user is not None:
        if args.number_of_user <= 0:
            raise ValueError('number_of_user must be positive.')
        user_ids = rng.sample(user_ids, min(args.number_of_user, len(user_ids)))

    rows = []
    candidate_rows = []
    pseudo_by_user = {
        int(user_id): int(rng.choice(model.user_to_edge_ids[int(user_id)]))
        for user_id in user_ids
    }
    for selector in SELECTORS:
        model.cf_edge_selector = selector
        for user_id in user_ids:
            history_edges = model.user_to_edge_ids[int(user_id)]
            pseudo_edge_id = pseudo_by_user[int(user_id)]
            probe = _probe_user(
                model, full_view, base_mask, user_id, pseudo_edge_id
            )
            if probe is None:
                continue
            candidates = [
                int(edge_id) for edge_id in history_edges
                if int(edge_id) != int(pseudo_edge_id)
            ]
            model._cf_rng.seed(int(args.seed) + int(user_id) * 1009)
            selector_scores = model._score_cf_candidates(
                base_mask=base_mask,
                full_view=full_view,
                probe_mask=probe['probe_mask'],
                user_id=user_id,
                pseudo_item_id=probe['pseudo_item_id'],
                boundary_item_id=probe['boundary_item_id'],
                candidate_edges=candidates
            )
            ranked = model._rank_cf_candidates(candidates, selector_scores)
            damages = _actual_damages(
                model, full_view, base_mask, user_id, probe, candidates
            )
            harmful = {
                edge_id for edge_id, damage in damages.items()
                if damage > float(args.damage_eps)
            }
            rank_by_edge = {
                int(edge_id): index + 1
                for index, edge_id in enumerate(ranked)
            }
            for edge_id in candidates:
                candidate_rows.append({
                    'selector': selector,
                    'user_id': int(user_id),
                    'pseudo_edge_id': int(pseudo_edge_id),
                    'pseudo_item_id': int(probe['pseudo_item_id']),
                    'candidate_edge_id': int(edge_id),
                    'candidate_item_id': int(
                        model.forward_edge_items[int(edge_id)].item()
                    ),
                    'selector_score': float(selector_scores[int(edge_id)]),
                    'selector_rank': int(rank_by_edge[int(edge_id)]),
                    'actual_damage': float(damages[int(edge_id)]),
                    'harmful': int(int(edge_id) in harmful),
                })
            row = {
                'selector': selector,
                'user_id': int(user_id),
                'pseudo_edge_id': int(pseudo_edge_id),
                'pseudo_item_id': int(probe['pseudo_item_id']),
                'boundary_item_id': int(probe['boundary_item_id']),
                'probe_margin': float(probe['probe_margin']),
                'candidate_count': len(candidates),
                'harmful_count': len(harmful),
            }
            for n in TOP_NS:
                selected = ranked[:n]
                row['recall_at_{}'.format(n)] = recall_at_n(
                    selected, harmful, n
                )
                row['hit_at_{}'.format(n)] = user_hit_at_n(
                    selected, harmful, n
                )
                row['captured_at_{}'.format(n)] = len(
                    set(selected) & harmful
                )
            rows.append(row)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / 'selector_user_results.csv'
    with row_path.open('w', encoding='utf-8', newline='') as handle:
        fields = list(rows[0].keys()) if rows else [
            'selector', 'user_id', 'pseudo_edge_id'
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    candidate_path = output_dir / 'selector_candidate_results.csv'
    with candidate_path.open('w', encoding='utf-8', newline='') as handle:
        fields = list(candidate_rows[0].keys()) if candidate_rows else [
            'selector', 'user_id', 'candidate_edge_id'
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    summary_rows = summarize_rows(rows)
    summary_csv_path = output_dir / 'selector_summary.csv'
    with summary_csv_path.open('w', encoding='utf-8', newline='') as handle:
        fields = list(summary_rows[0].keys()) if summary_rows else [
            'selector', 'top_n', 'edge_recall_micro'
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {
        'seed': int(args.seed),
        'checkpoint': str(Path(args.checkpoint).expanduser().resolve()),
        'dataset': args.dataset,
        'damage_eps': float(args.damage_eps),
        'cf_k': int(args.cf_k),
        'cf_boundary_width': int(args.cf_boundary_width),
        'cf_min_history': int(args.cf_min_history),
        'selected_user_count': len(user_ids),
        'fragile_pseudo_count': int(
            len({(row['user_id'], row['pseudo_edge_id']) for row in rows})
        ),
        'top_n': list(TOP_NS),
        'pseudo_protocol': 'one_seeded_train_history_holdout_per_user',
        'rows': summary_rows,
    }
    summary_path = output_dir / 'selector_summary.json'
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8'
    )
    print('Selector rows: {}'.format(row_path))
    print('Selector candidates: {}'.format(candidate_path))
    print('Selector summary CSV: {}'.format(summary_csv_path))
    print('Selector summary: {}'.format(summary_path))


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
