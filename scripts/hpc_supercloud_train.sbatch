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
#SBATCH --cpus-per-task=16
#SBATCH -p mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -o logs/hpc/pfagcn_%j.log
#SBATCH -e logs/hpc/pfagcn_%j.err

set -euo pipefail

module load deprecated-modules
module load python/3.10.8-x86_64

source /orcd/home/002/lerchen/code/pfagcn_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-16}
export HYDRA_FULL_ERROR=1

for ASPECT in MF BP CC; do
  echo "=== Starting training for ASPECT=${ASPECT} ==="
  srun python -u scripts/main.py train \
    --config-path configs \
    --config-name hpc_full_config \
    --aspect "${ASPECT}" \
    --experiment_name="pfagcn_supercloud_test" \
    --run_name="pfagcn_supercloud_test_{$ASPECT}"
done