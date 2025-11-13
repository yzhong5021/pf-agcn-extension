# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch \
    HF_HOME=/opt/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libopenblas-dev \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install micromamba for environment management.
RUN set -eux \
    && mkdir -p /tmp/micromamba \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba/micromamba.tar.bz2 \
    && tar -xjf /tmp/micromamba/micromamba.tar.bz2 -C /tmp/micromamba \
    && mv /tmp/micromamba/bin/micromamba /usr/local/bin/micromamba \
    && rm -rf /tmp/micromamba

# Install Google Cloud CLI for GCS interactions.
RUN mkdir -p /usr/share/keyrings \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" pfagcn && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash pfagcn

ENV MAMBA_ROOT_PREFIX=/home/pfagcn/micromamba

WORKDIR /workspace/pf_agcn
RUN mkdir -p "${MAMBA_ROOT_PREFIX}" \
    && chown -R pfagcn:pfagcn /workspace \
    && chown -R pfagcn:pfagcn "${MAMBA_ROOT_PREFIX}"

USER pfagcn

COPY --chown=pfagcn:pfagcn environment.yml ./environment.yml
RUN micromamba env create -y -f environment.yml \
    && micromamba clean --all --yes
ENV PATH="${MAMBA_ROOT_PREFIX}/envs/pf-agcn/bin:${PATH}"
SHELL ["micromamba", "run", "-n", "pf-agcn", "/bin/bash", "-c"]

COPY --chown=pfagcn:pfagcn . .

# Install BLAST+ toolkit (blastp) into repository path for config compatibility.
RUN set -euo pipefail \
 && mkdir -p /workspace/pf_agcn/ncbi-blast-2.17.0+ \
 && tmpdir="$(mktemp -d)" \
 && curl -L -o "${tmpdir}/ncbi-blast.tar.gz" \
      "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-linux.tar.gz" \
 && tar -xzf "${tmpdir}/ncbi-blast.tar.gz" -C "${tmpdir}" \
 && cp -r "${tmpdir}/ncbi-blast-2.17.0+/bin" /workspace/pf_agcn/ncbi-blast-2.17.0+/ \
 && cp -r "${tmpdir}/ncbi-blast-2.17.0+/lib" /workspace/pf_agcn/ncbi-blast-2.17.0+/ \
 && cp -r "${tmpdir}/ncbi-blast-2.17.0+/data" /workspace/pf_agcn/ncbi-blast-2.17.0+/ \
 && ln -sf /workspace/pf_agcn/ncbi-blast-2.17.0+/bin/blastp /workspace/pf_agcn/ncbi-blast-2.17.0+/bin/blastp.exe \
 && rm -rf "${tmpdir}"

ENV PATH="/workspace/pf_agcn/ncbi-blast-2.17.0+/bin:/home/pfagcn/.local/bin:${PATH}"

ENTRYPOINT ["micromamba", "run", "-n", "pf-agcn", "/bin/bash"]
