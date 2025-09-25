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
- Testing notebook

## To-do
- GO DAG prior generator
- Adaptive function/protein attention
- Training loop + Loss
- Evaluation metrics
- Ablation analysis

## Reference
Original paper: PF-AGCN: An Adaptive Graph Convolutional Network for Protein Function Prediction
https://doi.org/10.1093/bioinformatics/btaf473
