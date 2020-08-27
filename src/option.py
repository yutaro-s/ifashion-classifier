import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    # util
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--disable_data_parallel', action='store_true')
    parser.add_argument('--disable_load_optimizer', action='store_true')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--verbose', action='store_false')

    # files
    parser.add_argument('--train_dir',
                        type=str,
                        default='./data/iFashion/img/train')
    parser.add_argument('--train_file',
                        type=str,
                        default='./data/iFashion/json/tweak/train.json')
    parser.add_argument('--eval_dir',
                        type=str,
                        default='./data/iFashion/img/validation')
    parser.add_argument('--eval_file',
                        type=str,
                        default='./data/iFashion/json/tweak/validation.json')
    parser.add_argument('--cfg_file', type=str, default='./cfg/cfg.yaml')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--checkpoint', type=str, default=None)

    # epoch
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch',
                        type=int,
                        default=1000,
                        help='max epoch')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--disable_shuffle',
                        action='store_true',
                        help='Disable train_loader shuffle')

    # log
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    #parser.add_argument('--debug', action='store_true')
    parser.add_argument('--note', type=str, default='')
    args = parser.parse_args()

    # data parallel
    args.data_parallel = not args.disable_data_parallel
    args.load_optimizer = not args.disable_load_optimizer

    # for cuda
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # for shuffle
    args.loader_shuffle = not args.disable_shuffle

    return args
