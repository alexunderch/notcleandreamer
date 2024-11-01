FROM ubuntu:22.04

# install python
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  software-properties-common \
  build-essential \
  curl \
  ffmpeg \
  git \
  htop \
  swig \
  vim \
  nano \
  rsync \
  tmux \
  wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

  RUN add-apt-repository ppa:deadsnakes/ppa
  RUN apt-get update && apt-get install -y -qq python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-distutils

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

EXPOSE 9999

# default workdir
WORKDIR /home/workdir
ADD ./requirements.txt .

RUN pip install -r requirements.txt --ignore-installed && \
    # pip install -r requirements_gym.txt && \
    # pip install tensorflow[and-cuda]==2.14 && \
    pip install --upgrade jax jaxlib

RUN git config --global --add safe.directory /home/workdir
CMD ["/bin/bash"]