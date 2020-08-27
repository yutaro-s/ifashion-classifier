## Usage: python3 make_config.py

from pathlib import Path
import argparse
import yaml
import itertools

##############

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./output')
args = parser.parse_args()

# python command
cmd = ' python main.py --log'

########################################

cfg = {}
# fixed parameters
cfg['batch_size'] = 128
cfg['learning_rate'] = 0.001

# target parameters
OPT = ['adam', 'adabound']
WD = [0.000001, 0]  # weight decay

########################################


def save_files():
    # make output dir
    target_path.mkdir(parents=True, exist_ok=True)
    # dump yaml
    cfg_path = target_path.joinpath('cfg.yaml')
    with cfg_path.open(mode='w') as f:
        yaml.dump(cfg, f)
    cmd_opt = ' --output_dir ' + str(target_path) + ' --cfg_file ' + str(
        cfg_path)
    print(cmd + cmd_opt)


########################################

for opt, wd in itertools.product(OPT, WD):
    # set parameters
    cfg['optimizer'] = opt
    cfg['weight_decay'] = wd

    # make output path
    params = '_'.join([
        opt,
        'wd' + str(wd),
    ])
    target_path = Path(args.output, params)

    # save files
    save_files()
