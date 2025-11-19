#!/bin/bash

# run mkdir -p logs/hpc to get the log info.

#SBATCH -J pfagcn_smoke
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p mit_normal_gpu
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o logs/hpc/pfagcn_smoke_%j.log
#SBATCH -e logs/hpc/pfagcn_smoke_%j.err

set -euo pipefail

module load deprecated-modules
module load python/3.10.8-x86_64

source pfagcn_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-12}
export HYDRA_FULL_ERROR=1

  
srun python -u scripts/main.py train \
    --config-path configs \
    --config-name hpc_smoke_config \
    --aspect BP \
