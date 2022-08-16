# syntax=docker/dockerfile:1
ARG BASE_REGISTRY=mcr.microsoft.com \
    BASE_IMAGE=vscode/devcontainers/base \
    BASE_TAG=ubuntu-22.04

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} as base
USER root
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash","-c"]
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    # [VENV && PIP]
    python3.10-venv python3-pip=22.0.2+dfsg-1 \
    # [GEOS && PROJ && GDAL] needed for cartopy
    libgeos-dev=3.10.2-1 libproj-dev=8.2.1-1 libgdal-dev=3.4.1+dfsg-1build4 \
    && rm -rf /var/lib/apt/lists/*
# 
# 
# 
FROM base as builder
USER root
ENV PATH="/opt/venv/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash","-c"]
# tools needed to compile cartopy
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    proj-bin=8.2.1-1 \
    gdal-bin=3.4.1+dfsg-1build4 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-dev.txt .
RUN python3 -m venv /opt/venv && pip install --upgrade pip && pip install -r requirements.txt && pip install -r requirements-dev.txt
# 
# 
# 
FROM base as final
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV PATH="/opt/venv/bin:$PATH"
COPY --chown=$USERNAME:$USER_GID --from=builder /opt/venv /opt/venv

USER $USERNAME

