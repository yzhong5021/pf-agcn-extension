# PF-AGCN: Reproduction and Extension (WIP)

This project aims to:
- Create from scratch a reimplementation of PF-AGCN for protein function prediction with a clean, testable codebase
- Extend the model by integrating multi-omics data (ex. RNA-seq WCGNA prior graph) and optimizing adaptive attention component to improve interpretability and efficiency

--- 
## Current Progress
- Repo skeleton
- Toy data generator (sequences + GO terms)
- Mock ESM (sequence embedding + linear)
- Dilated Causal CNN + gating
- Testing notebook

## To do
- GO DAG prior generator
- Adaptive function/protein attention
- Training loop + Loss
- Eval metrics
- Ablation analysis

Original paper: PF-AGCN: An Adaptive Graph Convolutional Network for Protein Function Prediction
https://doi.org/10.1093/bioinformatics/btaf473