import json
from pathlib import Path
import torch


def save_checkpoint(args, cfg, epoch, model, optimizer):
    if args.data_parallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(
        {
            'args': args,
            'cfg': cfg,
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict()
        }, Path(args.output_dir, 'checkpoint_latest.pt'))


def save_result(args, epoch, res, filename=None):
    if filename == None:
        filename = 'result_latest'

    with Path(args.output_dir, filename + '.json').open(mode='w') as f:
        json.dump(
            {
                'eval_dir': args.eval_dir,
                'eval_file': args.eval_file,
                'epoch': epoch,
                'res': res,
            },
            f,
            indent=2)
