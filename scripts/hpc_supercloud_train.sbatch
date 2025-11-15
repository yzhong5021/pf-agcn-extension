#!/bin/bash
# Reference: https://mit-supercloud.github.io/supercloud-docs/submitting-jobs/
#
# Submit with: sbatch scripts/hpc_supercloud_train.sbatch
#
# IMPORTANT: Remember to run 'mkdir -p logs/hpc' once in your terminal
#            before submitting this script for the first time.

#SBATCH -J pfagcn
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -o logs/hpc/pfagcn_%j.log
#SBATCH -e logs/hpc/pfagcn_%j.err

set -euo pipefail

module load deprecated-modules
module load python/3.10.8-x86_64

source pfagcn_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

export ASPECT=${ASPECT:-MF}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-20}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-20}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-20}
export HYDRA_FULL_ERROR=1

srun python -u scripts/main.py train --config-path configs --config-name hpc_full_config --aspect "${ASPECT}"

export ASPECT=${ASPECT:-BP}
srun python -u scripts/main.py train --config-path configs --config-name hpc_full_config --aspect "${ASPECT}"

export ASPECT=${ASPECT:-CC}
srun python -u scripts/main.py train --config-path configs --config-name hpc_full_config --aspect "${ASPECT}"