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

    args, _ = parser.parse_known_args()
    config_dict = {
        'dropout': [0.2],
        'reg_weight': [0.001],
        'learning_rate': [0.003],
        'gpu_id': args.gpu_id,
        'fusion': 'add'
    }

    if args.model.upper() == 'MASKED_GLORIA_EX2':
        config_dict.update({
            'mask_discovery_epochs': args.mask_discovery_epochs,
            'mask_anneal_epochs': args.mask_anneal_epochs,
        })

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

