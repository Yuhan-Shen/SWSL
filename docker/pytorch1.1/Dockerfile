FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# install some dependencies on ubuntu18, not sure if really needed!
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# install anaconda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=3.7 numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# install pytorch
RUN conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# install python dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Mounting Points
VOLUME /app
VOLUME /data/root
VOLUME /data/datasets

# Set the working directory
WORKDIR /app
