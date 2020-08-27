import argparse
import time

import yaml
import numpy as np
import torch
import torch.nn as nn

import src

##############################################

np.random.seed(0)
torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


def main(args, cfg):
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint + '/checkpoint_best.pt')
    else:
        checkpoint = None

    # set dataloader
    train_loader, eval_loader = src.set_loader(args, cfg)
    # set model
    model, optimizer = src.set_model(args, cfg, checkpoint)
    # set loss function
    loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    # set logger and tensorboard writer
    logger, writer = src.set_logger(args, cfg)

    if args.evaluation:
        res = src.evaluator(args, model, eval_loader, loss_fn)
        src.save_result(args, 0, res, filename='result_eval')
        logger.info('[val] epoch-%d: %ds, %f', 0, res['time'], res['loss'])
        return 0, args

    # initialize
    score = src.Score(best=True)

    for epoch in range(args.start_epoch, args.max_epoch):
        # train
        res = src.trainer(args, model, train_loader, loss_fn, optimizer)
        src.save_checkpoint(args, cfg, epoch, model, optimizer)
        logger.info('[tra] epoch-%d: %ds, %f', epoch, res['time'], res['loss'])

        # eval
        res = src.evaluator(args, model, eval_loader, loss_fn)
        src.save_result(args, epoch, res)
        logger.info('[val] epoch-%d: %ds, %f', epoch, res['time'], res['loss'])
        score.update(args, res)

        # check convergence
        if epoch > 50 and score.cnt > 10:
            break

    return epoch, args


if __name__ == '__main__':
    args = src.get_args()

    # load config file
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    start_time = time.time()
    epoch, args = main(args, cfg)
    print('%d epochs (%ds): %s' %
          (epoch, round(time.time() - start_time), args.output_dir))
