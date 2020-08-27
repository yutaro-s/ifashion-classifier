import shutil
import math
from pathlib import Path


class Score:
    def __init__(self, base='loss', best=False):
        self.value = math.inf
        self.base = base
        self.best = best
        self.cnt = 0

    def update(self, args, res):
        if self.value > res[self.base]:
            self.value = res[self.base]

            # reset cnt
            self.cnt = 0

            # rename
            if self.best:
                shutil.move(Path(args.output_dir, 'result_latest.json'),
                            Path(args.output_dir, 'result_best.json'))
                shutil.move(Path(args.output_dir, 'checkpoint_latest.pt'),
                            Path(args.output_dir, 'checkpoint_best.pt'))
        else:
            self.cnt += 1
