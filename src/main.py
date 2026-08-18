import os
import argparse
from utils.quick_start import quick_start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CAMU', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='GPU ID to use')

    args, _ = parser.parse_known_args()
    config_dict = {
        'dropout': [0.2],
        'reg_weight': [0.001],
        'learning_rate': [0.003],
        'gpu_id': args.gpu_id,
    }


    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
