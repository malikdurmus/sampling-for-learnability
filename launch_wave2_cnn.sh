#!/bin/bash
# =============================================================
# Wave 2 — xland difficult CNN arms (post-wave1fix arm selection,
# approved by Malik 2026-08-30): cnn, hybrid_linear,
# hybrid_soft_handoff x seeds 1-4. Justification for the arm set
# (and for excluding learnability_weighted / multiplicative):
# THESIS_FINDINGS.md, Finding 6 addendum.
#
# All Abaki nodes usable per Malik (no exclusion). Project:
# xland-sfl-campaign, groups match the running baselines.
# Default: GENERATE only; submit with --submit.
# =============================================================
set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/wave2cnn

SUBMIT=0
[[ "$1" == "--submit" ]] && SUBMIT=1

PROJECT="xland-sfl-campaign"
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

LABELS=(  "cnn" "hybrid_linear" "hybrid_soft_handoff")
METHODS=( "cnn" "hybrid_linear" "hybrid_soft_handoff")
SEEDS=(1 2 3 4)

COUNT=0
for SEED in "${SEEDS[@]}"; do
    for i in "${!METHODS[@]}"; do
        LABEL="${LABELS[$i]}"
        METHOD="${METHODS[$i]}"
        RUN_NAME="xl_${LABEL}_seed${SEED}"
        JOB_NAME="w2c_${LABEL}_s${SEED}"
        SCRIPT_PATH="slurm_scripts/wave2cnn/${RUN_NAME}.sh"

        cat > "${SCRIPT_PATH}" << 'HEADER'
#!/bin/bash
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --qos=abaki
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --requeue
HEADER

        cat >> "${SCRIPT_PATH}" << BODY
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/%j_out.log
#SBATCH --error=slurm_logs/%j_err.log

# GPU-busy guard: this cluster does not track GPUs in SLURM (gres=null), so
# jobs can land on a node whose GPU is already occupied — by SLURM
# double-booking or by users running outside SLURM via ssh. That caused the
# 2026-08-31 OOM crashes. Wait up to 30 min for a free GPU; if still busy,
# requeue this job instead of crashing.
USED=99999
for i in \$(seq 1 60); do
    USED=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "\${USED:-99999}" -lt 1500 ]; then break; fi
    sleep 30
done
if [ "\${USED:-99999}" -ge 1500 ]; then
    echo "GPU busy on \$(hostname) (\${USED} MiB used) — requeueing"
    scontrol requeue \$SLURM_JOB_ID
    sleep 60
    exit 0
fi

cd ${WORKDIR}
source .venv/bin/activate
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/xland_sfl.py \\
    LEARN_METHOD=${METHOD} \\
    SEED=${SEED} \\
    RUN_NAME=${RUN_NAME} \\
    PROJECT=${PROJECT} \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=wave2-xland-${LABEL}
unset TF_GPU_ALLOCATOR
BODY

        COUNT=$((COUNT + 1))
        if [[ ${SUBMIT} -eq 1 ]]; then
            echo "Submitting: ${RUN_NAME}"
            sbatch "${SCRIPT_PATH}"
        else
            echo "Generated (NOT submitted): ${SCRIPT_PATH}"
        fi
    done
done
echo ""
echo "${COUNT} Wave-2 CNN-arm jobs $( [[ ${SUBMIT} -eq 1 ]] && echo submitted || echo generated )."
