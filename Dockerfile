ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN cd /tmp/ && git clone https://github.com/athaddius/MFT_STIR.git && cd MFT_STIR && git checkout develop \
    && pip install .
    #&& cd MFT_STIR && pip install .

# For STIRLoader
RUN mkdir -p /tmp \
    && cd /tmp/ && git clone https://github.com/athaddius/STIRLoader.git && cd STIRLoader && git checkout develop \
    && pip install .
    #&& cd STIRLoader && pip install .
RUN pip install torchvision onnxruntime-gpu
RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6

RUN git clone https://github.com/athaddius/STIRMetrics.git
