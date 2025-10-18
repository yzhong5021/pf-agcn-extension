# PF-AGCN: Reproduction and Extension (WIP)

This repo is a work in progress - current contents include toy data, mock modules, and test scripts.

This project aims to:
- Create from scratch a reimplementation of PF-AGCN for protein function prediction with a clean, testable codebase
- Extend the model by integrating multi-omics data (e.g., RNA-seq via WGCNA prior graph) and optimizing adaptive attention components to improve interpretability and efficiency

--- 

## Current Progress
- Repo skeleton
- Toy data generator (sequences + GO terms)
- Mock ESM (sequence embedding + linear)
- Dilated Causal CNN + gating
- Merging sequence embeddings for protein & GO-centric representations + adaptive pooling
- Adaptive function/protein attention
- GO DAG prior generator
- Loss/training loop
- Evaluation metrics (CAFA)
- Testing notebook

## To-do
- Ablation analysis

## Low-Compute Local Stack
- Run `python -m src.train.training --config-name default_config_low_compute` after activating the env to smoke-test the full pipeline within 12 GB RAM and two loader workers.
- Hydra logs and artifacts land under `results/local_low/` so multiple dry runs stay organised.
- `python scripts/main.py train` auto-builds a manifest from CAFA URLs when none are configured.
- Provide an aspect flag (MF/BP/CC) via --aspect to train or predict dedicated models; manifests land under data/manifests/{train,val,test} with aspect-specific priors.

## Reference
Original paper: PF-AGCN: An Adaptive Graph Convolutional Network for Protein Function Prediction
https://doi.org/10.1093/bioinformatics/btaf473
