

# A classifier for iFashion Attribute 

## Requirements
- [Docker](https://docs.docker.com/) >= 19.03
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)


## Setup
Build a docker image from Dockerfile

```
make docker-build
```


## Dataset construction

See [this page](./data/iFashion/README.md).


## Training

Launch a docker container and then run a training command in this container

```
make train-model
```


## Trained model
The trained model is available at [google drive](https://drive.google.com/file/d/1jiwvzFEEMlXaZ0HyxmWXYChre9KfzDDY/view?usp=sharing). 
The validation score (F1) of this trained model is 60.3.
To get the validation scores with this trained model, you can specify the directory as follows:

```
# download zip file from my google drive.
unzip trained_model.zip -d output
python main.py --eval_dir data/iFashion/img/validation --eval_file data/iFashion/json/tweak/validation.json --evaluation --checkpoint output/adabound_wd1e-06 --output_dir output/adabound_wd1e-06
```


## Difference 
This implementation is not intended to reproduce the results reported in [Guo et al.'s paper](https://arxiv.org/abs/1906.05750).
There are some differences between the configuration in the paper and this implementation. 

| | reported | this repo. |
|--- | --- | --- |
| base model | Inception-{V1,BN,V3}, ResNet101 | ResNet152 |
| loss | weighted BCE | unweighted BCE |
| optimizer | RMSProp | Adabound | 
