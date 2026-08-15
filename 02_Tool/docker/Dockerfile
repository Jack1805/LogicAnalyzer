FROM ubuntu:24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        gcc-arm-none-eabi \
        cmake \
        ninja-build \
        git \
        make \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace