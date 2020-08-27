FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    curl \
    python3-pip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools

# pytorch 1.2 with CUDA 10.0
RUN pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

# util
RUN pip3 install \ 
    adabound \ 
    logzero \
    pyyaml \
    scikit-learn

RUN ln -s /usr/bin/python3 /usr/bin/python
ENV LANG C.UTF-8
