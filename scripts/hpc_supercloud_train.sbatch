#!/bin/bash
# Reference: https://mit-supercloud.github.io/supercloud-docs/submitting-jobs/
#
# Submit with: sbatch scripts/hpc_supercloud_train.sbatch

#SBATCH -J pfagcn
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:volta:1
#SBATCH -p normal
#SBATCH --qos=high
#SBATCH -t 24:00:00
#SBATCH -o logs/hpc/pfagcn_%j.log
#SBATCH -e logs/hpc/pfagcn_%j.err

set -euo pipefail

module load anaconda/2023a
module load cuda/12.1

CONDA_ENV=${CONDA_ENV:-pf-agcn}
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/hpc

export ASPECT=${ASPECT:-MF}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-32}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-32}
export HYDRA_FULL_ERROR=1

srun python -u scripts/main.py train --config-path configs --config-name hpc_config --aspect "${ASPECT}"                                                       