#!/bin/bash
# =============================================================
# Wave 2 — xland difficult confirmatory (jaxnav campaign follow-up)
# UNBLOCKED ARMS ONLY (standard, dr): the cnn/hybrid arms are blocked
# until the xland scorers are retrained on the corrected dataset
# (THESIS_FINDINGS.md Finding 5).
#
# 2 arms x 4 paired seeds, seed-complete order, declared config
# (difficult ruleset, full 3e8 budget; ~10h/run measured on A5000).
# Free mixed queue: NvidiaAll,Abaki with abakus11/12 excluded (in use
# by another user).
#
# Default: GENERATE sbatch scripts only (slurm_scripts/wave2/).
# Submit with:  ./launch_wave2.sh --submit
# =============================================================

set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/wave2

SUBMIT=0
[[ "$1" == "--submit" ]] && SUBMIT=1

PROJECT="xland-sfl-campaign"   # xland lives in its own project, separate from jaxnav
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

LABELS=(  "dr"     "standard")
METHODS=( "random" "standard")
SEEDS=(1 2 3 4)

COUNT=0
for SEED in "${SEEDS[@]}"; do            # seed-complete blocks
    for i in "${!METHODS[@]}"; do
        LABEL="${LABELS[$i]}"
        METHOD="${METHODS[$i]}"
        RUN_NAME="xl_${LABEL}_seed${SEED}"
        JOB_NAME="w2_${LABEL}_s${SEED}"
        SCRIPT_PATH="slurm_scripts/wave2/${RUN_NAME}.sh"

        cat > "${SCRIPT_PATH}" << 'HEADER'
#!/bin/bash
# Free mixed queue; all Abaki nodes allowed (Malik, 2026-08-30).
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --qos=abaki
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --requeue
# NOTE: runs need 18-22h on NvidiaAll nodes (A5000/Abaki ~10-11h). Cluster
# reboots weekly Sunday ~04:45 (JobRequeue=1: a job running then restarts
# from scratch) — avoid submitting long runs that would straddle it.
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
if [[ ${SUBMIT} -eq 1 ]]; then
    echo "Submitted ${COUNT} Wave-2 jobs (unblocked arms)."
else
    echo "Generated ${COUNT} scripts. Review slurm_scripts/wave2/, then: ./launch_wave2.sh --submit"
fi
