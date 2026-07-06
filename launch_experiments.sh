#!/bin/bash
# =============================================================
# Batch Experiment Launcher
# Generates individual SLURM scripts and submits them.
# =============================================================

set -e
mkdir -p slurm_logs
mkdir -p slurm_scripts

# ---- Experiment matrix ----
METHODS=(
    "standard"
    "hybrid_linear"
    "hybrid_soft_handoff"
    "hybrid_learnability_weighted"
    "hybrid_multiplicative"
)
SEEDS=(1 2 3 4)

PROJECT="jaxnav_sfl_experiments_performance_adaptive_new"
GROUP="full_comparison"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${METHOD}_sigmoid"
        JOB_NAME="${METHOD}_s${SEED}"
        SCRIPT_PATH="slurm_scripts/${RUN_NAME}.sh"

        cat > "${SCRIPT_PATH}" << 'HEADER'
#!/bin/bash
#SBATCH --partition=NvidiaAll
HEADER

        cat >> "${SCRIPT_PATH}" << BODY
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/%j_out.log
#SBATCH --error=slurm_logs/%j_err.log

cd ${WORKDIR}
source .venv/bin/activate
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py LEARN_METHOD=${METHOD} SEED=${SEED} RUN_NAME=${RUN_NAME} PROJECT=${PROJECT} GROUP_NAME=${GROUP}
unset TF_GPU_ALLOCATOR
BODY

        echo "Submitting: ${RUN_NAME}"
        sbatch "${SCRIPT_PATH}"
    done
done

echo ""
echo "=========================================="
echo "Submitted $((${#METHODS[@]} * ${#SEEDS[@]})) jobs total."
echo "Methods: ${#METHODS[@]}"
echo "Seeds per method: ${#SEEDS[@]}"
echo "=========================================="
