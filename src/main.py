import os
import argparse
from utils.quick_start import quick_start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CAMU', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='GPU ID to use')
    parser.add_argument(
        '--mask_discovery_epochs',
        '--mask-discovery-epochs',
        dest='mask_discovery_epochs',
        type=int,
        default=60,
        help='epochs that learn the user mask at alpha=1'
    )
    parser.add_argument(
        '--mask_anneal_epochs',
        '--mask-anneal-epochs',
        dest='mask_anneal_epochs',
        type=int,
        default=40,
        help='epochs that linearly anneal alpha from 1 toward 0'
    )
    parser.add_argument('--cf_lambda', '--cf-lambda', dest='cf_lambda', type=float, default=0.1)
    parser.add_argument('--cf_warmup_ratio', '--cf-warmup-ratio', dest='cf_warmup_ratio', type=float, default=0.05)
    parser.add_argument('--cf_warmup_epochs', '--cf-warmup-epochs', dest='cf_warmup_epochs', type=int, default=-1)
    parser.add_argument('--cf_user_ratio', '--cf-user-ratio', dest='cf_user_ratio', type=float, default=0.10)
    parser.add_argument('--cf_batch_size', '--cf-batch-size', dest='cf_batch_size', type=int, default=8)
    parser.add_argument('--cf_k', '--cf-k', dest='cf_k', type=int, default=20)
    parser.add_argument('--cf_boundary_width', '--cf-boundary-width', dest='cf_boundary_width', type=int, default=20)
    parser.add_argument('--cf_boundary_q', '--cf-boundary-q', dest='cf_boundary_q', type=int, default=3)
    parser.add_argument('--cf_temperature', '--cf-temperature', dest='cf_temperature', type=float, default=1.0)
    parser.add_argument('--cf_min_history', '--cf-min-history', dest='cf_min_history', type=int, default=2)
    parser.add_argument('--cf_seed_offset', '--cf-seed-offset', dest='cf_seed_offset', type=int, default=10000)
    parser.add_argument('--cf_log_stats', '--cf-log-stats', dest='cf_log_stats', action='store_true', default=True)
    parser.add_argument('--no_cf_log_stats', '--no-cf-log-stats', dest='cf_log_stats', action='store_false')
    parser.add_argument(
        '--cf_base_checkpoint',
        '--cf-base-checkpoint',
        dest='cf_base_checkpoint',
        type=str,
        default=None,
        help='trained MASKED_GLORIA_EX checkpoint required by EX3'
    )
    parser.add_argument(
        '--cf_gamma_grid',
        '--cf-gamma-grid',
        dest='cf_gamma_grid',
        type=str,
        default='0,0.25,0.5,0.75,1',
        help='comma-separated counterfactual strength grid for EX3'
    )
    parser.add_argument(
        '--cf_negatives_per_positive',
        '--cf-negatives-per-positive',
        dest='cf_negatives_per_positive',
        type=int,
        default=20
    )
    parser.add_argument(
        '--cf_calibration_train_ratio',
        '--cf-calibration-train-ratio',
        dest='cf_calibration_train_ratio',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--cf_calibrator_hidden_dim',
        '--cf-calibrator-hidden-dim',
        dest='cf_calibrator_hidden_dim',
        type=int,
        default=64
    )
    parser.add_argument(
        '--cf_calibrator_lr',
        '--cf-calibrator-lr',
        dest='cf_calibrator_lr',
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--cf_calibrator_batch_size',
        '--cf-calibrator-batch-size',
        dest='cf_calibrator_batch_size',
        type=int,
        default=512
    )
    parser.add_argument(
        '--cf_calibrator_epochs',
        '--cf-calibrator-epochs',
        dest='cf_calibrator_epochs',
        type=int,
        default=200
    )
    parser.add_argument(
        '--cf_calibrator_patience',
        '--cf-calibrator-patience',
        dest='cf_calibrator_patience',
        type=int,
        default=20
    )

    args, _ = parser.parse_known_args()
    config_dict = {
        'dropout': [0.2],
        'reg_weight': [0.001],
        'learning_rate': [0.003],
        'gpu_id': args.gpu_id,
        'fusion': 'add'
    }

    if args.model.upper() == 'MASKED_GLORIA_CF':
        config_dict.update({
            'cf_lambda': args.cf_lambda,
            'cf_warmup_ratio': args.cf_warmup_ratio,
            'cf_warmup_epochs': args.cf_warmup_epochs,
            'cf_user_ratio': args.cf_user_ratio,
            'cf_batch_size': args.cf_batch_size,
            'cf_k': args.cf_k,
            'cf_boundary_width': args.cf_boundary_width,
            'cf_boundary_q': args.cf_boundary_q,
            'cf_temperature': args.cf_temperature,
            'cf_min_history': args.cf_min_history,
            'cf_seed_offset': args.cf_seed_offset,
            'cf_log_stats': args.cf_log_stats,
        })
    elif args.model.upper() == 'MASKED_GLORIA_EX2':
        config_dict.update({
            'mask_discovery_epochs': args.mask_discovery_epochs,
            'mask_anneal_epochs': args.mask_anneal_epochs,
        })
    elif args.model.upper() == 'MASKED_GLORIA_EX3':
        if args.cf_base_checkpoint is None:
            parser.error('--cf_base_checkpoint is required for MASKED_GLORIA_EX3')
        config_dict.update({
            'cf_base_checkpoint': args.cf_base_checkpoint,
            'cf_gamma_grid': args.cf_gamma_grid,
            'cf_negatives_per_positive': args.cf_negatives_per_positive,
            'cf_calibration_train_ratio': args.cf_calibration_train_ratio,
            'cf_calibrator_hidden_dim': args.cf_calibrator_hidden_dim,
            'cf_calibrator_lr': args.cf_calibrator_lr,
            'cf_calibrator_batch_size': args.cf_calibrator_batch_size,
            'cf_calibrator_epochs': args.cf_calibrator_epochs,
            'cf_calibrator_patience': args.cf_calibrator_patience,
        })

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

