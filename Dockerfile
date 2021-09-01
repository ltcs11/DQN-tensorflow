# Dockerfile
# build docker image for DQN-tensorflow with cuda9.0
# [Notice]:
#  python : py3(python35)

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /home/work/
ENV DOCKER_PIP_OPTION="-i https://pypi.douban.com/simple/ --no-cache-dir"

RUN echo "#0. prepare apt-get" && \
        apt-get update && \
        apt-get install -y vim cmake curl wget python3-dev && \
        apt-get install -y zlib1g-dev libglib2.0-dev libx11-dev libxext-dev && \
        rm -rf /home/work/* && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "#1. prepare python pkg" && \
        cd /home/work/ && \
        curl "https://bootstrap.pypa.io/pip/3.5/get-pip.py" -o get-pip.py35.py && \
        python3 get-pip.py35.py && \
    echo "#2.pip install" && \
        pip3 install tensorflow-gpu==1.10 ${DOCKER_PIP_OPTION} && \
        pip3 install tqdm ${DOCKER_PIP_OPTION} && \
        pip3 install gym ${DOCKER_PIP_OPTION} && \
        pip3 install atari-py==0.2.5 ${DOCKER_PIP_OPTION} && \
        pip3 install opencv-python ${DOCKER_PIP_OPTION} && \
