VER='0.1'
DOCKER_IMAGE='shigeto/ifashion:'$(VER)

################

docker-build:
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile .

docker-run-cpu:
	docker run -it --rm -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE) bash

docker-run-gpu:
	docker run --gpus all -it --rm --shm-size 100G -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE) bash

################

run-experiment:
	python3 script/make_config.py > train_list.sh
	/bin/bash ./train_list.sh
	python script/find_best.py

debug:
	python main.py --max_epoch 5 --train_dir data/iFashion/img/train --train_file data/iFashion/json/tweak/debug.json

train-model:
	mkdir -p output
	python main.py --train_dir data/iFashion/img/train --train_file data/iFashion/json/tweak/train.json --cfg_file ./cfg/cfg.yaml

eval-model:
	python main.py --eval_dir data/iFashion/img/validation --eval_file data/iFashion/json/tweak/validation.json --evaluation --checkpoint output --output_dir output

clean:
	rm -rf output train_list.sh
