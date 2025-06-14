# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Other version: https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH='2.0.0'
ARG CUDA='cu118'
ARG SHELL='/bin/bash'
ARG MINICONDA='Miniconda3-py39_23.3.1-0-Linux-x86_64.sh'

ENV LANG=en_US.UTF-8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 CUDA_HOME=/usr/local/cuda CONDA_HOME=/opt/conda SHELL=${SHELL}
ENV PATH=$CONDA_HOME/bin:$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
    LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH \
    CONDA_PREFIX=$CONDA_HOME \
    NCCL_HOME=$CUDA_HOME

# Install ubuntu packages
RUN apt-get update \
    && apt-get -y install \
    python3-pip ffmpeg git less wget libsm6 libxext6 libxrender-dev \
    build-essential cmake pkg-config libx11-dev libatlas-base-dev \
    libgtk-3-dev libboost-python-dev vim libgl1-mesa-glx \
    libaio-dev software-properties-common tmux \
    espeak-ng

# Install miniconda with python 3.9
USER root
# COPY Miniconda3-py39_23.3.1-0-Linux-x86_64.sh /root/anaconda.sh
RUN wget -t 0 -c -O /tmp/anaconda.sh https://repo.anaconda.com/miniconda/${MINICONDA} \
    && mv /tmp/anaconda.sh /root/anaconda.sh \
    && ${SHELL} /root/anaconda.sh -b -p $CONDA_HOME \
    && rm /root/anaconda.sh

RUN conda create -y --name amphion python=3.9.15

WORKDIR /app
COPY requirements.txt requirements.txt

# If other source files are needed for any package in requirements.txt to build,
# they might need to be copied earlier. For now, assume requirements.txt is self-contained
# or dependencies are fetched from PyPI.
# Consider adding COPY . /app if full context is needed for pip install.

# Install Python packages from requirements.txt into the 'amphion' conda environment
RUN conda run -n amphion pip install -vvv --no-cache-dir -r requirements.txt

RUN conda init \
    && echo "\nconda activate amphion\n" >> ~/.bashrc

CMD ["conda", "run", "-n", "amphion", "python", "handler.py"]

# *** Build ***
# docker build -t realamphion/amphion .

# *** Run ***
# cd Amphion
# docker run --runtime=nvidia --gpus all -it -v .:/app -v /mnt:/mnt_host realamphion/amphion

# *** Push and release ***
# docker login
# docker push realamphion/amphion
