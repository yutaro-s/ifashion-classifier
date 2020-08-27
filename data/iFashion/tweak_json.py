import json
from pathlib import Path
import collections
import random

random.seed(0)

#list_subset = ['train', 'validation', 'test']
list_subset = ['train', 'validation']

for subset in list_subset:
    # load json
    with Path('json', 'raw', subset + '.json').open(mode='r') as f:
        data = json.load(f)
        filenames = {
            img['imageId']: Path(img['url']).name
            for img in data['images']
        }  # drop prefix
        annotations = {
            img['imageId']: img['labelId']
            for img in data['annotations']
        }

    # merge dics
    info = {}
    cnt_label = []
    img_dir = Path('img', subset)
    for img_id, filename in filenames.items():
        # check if a file exists
        if img_dir.joinpath(filename).exists():
            label = [int(i) - 1 for i in annotations[img_id]
                     ]  # list of str > list of int, 0 to n
            info[img_id] = {'filename': filename, 'label': label}

            assert len(label) > 0, '{}'.format(ima_id)
            cnt_label += label

    # show empirical distribution
    c = collections.Counter(cnt_label)
    print(len(c), c)
    if subset == 'train' and len(c) < 228:
        print('warining')

    # save json
    with Path('json', 'tweak', subset + '.json').open(mode='w') as f:
        json.dump(info, f)

    # for debugging program, I extract a small number of samples from training set
    if subset == 'train':
        debug_id = random.sample(info.keys(), 256)
        debug_info = {img_id: info[img_id] for img_id in debug_id}
        with Path('json', 'tweak', 'debug.json').open(mode='w') as f:
            json.dump(debug_info, f)
