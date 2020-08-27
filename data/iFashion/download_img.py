import json
from pathlib import Path
import urllib.error
import urllib.request
import time

DOWNLOAD_DELAY = # set the amount of time (in seconds) that the downloader should wait after downloading an image
# e.g., DOWNLOAD_DELAY = 1000000000
print('DOWNLOAD_DELAY={} sec.'.format(DOWNLOAD_DELAY))

#list_subset = ['train', 'validation', 'test']
list_subset = ['train', 'validation']

for subset in list_subset:
    data = json.load(Path('json', 'raw', subset + '.json').open(mode='r'))
    path = Path('img', subset)

    for info in data['images']:
        url = info['url']
        filename = Path(url).name
        trg_path = path.joinpath(filename)

        if not trg_path.exists():
            try:
                with urllib.request.urlopen(url) as web_file:
                    img = web_file.read()
                    with trg_path.open(mode='wb') as local_file:
                        local_file.write(img)
            except urllib.error.URLError as e:
                print(e)
                print(url)

            time.sleep(DOWNLOAD_DELAY)
