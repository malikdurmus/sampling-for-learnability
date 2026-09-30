#!/bin/bash
#SBATCH --job-name=cvar-rollout
#SBATCH --partition=NvidiaAll
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/cvar_%A_%a.out
#SBATCH --array=0-1
# array index = rollout seed (0 and 1)
cd "$SLURM_SUBMIT_DIR"
TF_GPU_ALLOCATOR=cuda_malloc_async .venv/bin/python cvar_eval/cvar_1_rollout.py ${SLURM_ARRAY_TASK_ID}
