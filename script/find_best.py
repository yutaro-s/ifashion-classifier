# Usage: python find_cands.py --output_dir []

from pathlib import Path
import argparse, json


def get_score(args, filename):
    with filename.open('r') as f:
        result = json.load(f)
    return {'score': result['res'][args.metric_base], 'epoch': result['epoch']}


def find_cands(args):
    scores = {}
    for filename in Path(args.output_dir).glob('**/result_best.json'):
        scores[filename] = get_score(args, filename)

    cands = []
    for filename, score in sorted(scores.items(),
                                  key=lambda x: x[1]['score'],
                                  reverse=args.reverse):
        target = str(filename.parent)
        cands.append(target)
        if args.verbose:
            print('%s:\t%.4f\t%d' % (target, score['score'], score['epoch']))

    return cands


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--metric_base',
                        type=str,
                        choices=[
                            'roc_auc', 'converage_error',
                            'rank_average_precision', 'zero_one', 'f1'
                        ],
                        default='f1')
    parser.add_argument('--verbose', action='store_false')
    args = parser.parse_args()

    if args.metric_base in ['roc_auc', 'rank_average_precision', 'f1']:
        args.reverse = True
    else:
        args.reverse = False

    print('Target:', args.output_dir)
    print('score:', args.metric_base)
    cands = find_cands(args)
