#!/bin/bash
# slurm
#SBATCH --job-name=sfl
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/%j_out.log # %j will be the Job ID
#SBATCH --error=slurm_logs/%j_err.log

mkdir -p slurm_logs
cd ~/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability
source .venv/bin/activate

python -c "print('net_random')"
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py
unset TF_GPU_ALLOCATOR
