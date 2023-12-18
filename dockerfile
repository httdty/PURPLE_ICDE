FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY saved_models /saved_models
COPY requirements.txt /workspace/
COPY env.sh /workspace/
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y -q curl unzip wget vim tree g++ \
    && rm -rf /var/lib/apt/lists/*

RUN bash env.sh \
    && rm -rf /workspace/* \
    && rm -rf /root/.cache/pip/*

# docker build