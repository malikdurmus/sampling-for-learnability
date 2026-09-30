#!/bin/bash
# =============================================================
# Wave 2M — xland MEDIUM ruleset (added 2026-08-31 at Malik's
# request: difficult may be floor-locked at ~0 success, so medium
# provides a regime where methods can differentiate).
# Same 5 arms as the difficult wave (same pre-registered arm
# justification, Finding 6 addendum): dr, standard, cnn,
# hybrid_linear, hybrid_soft_handoff x seeds 1-4 = 20 runs.
# Config: xland-sfl-medium.yaml (medium ruleset + deployed
# xmg_medium AUG ensemble; input_domain resolved from the index).
# Runs xlm_<label>_seed<N>, groups wave2-xlandmed-<label>,
# project xland-sfl-campaign.
#
# Submit cap handling: submits until the 30-job cap, then feeds
# the remainder as slots free (log: slurm_logs/wave2m_feed.log).
# Default: GENERATE only; submit with --submit.
# =============================================================
set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/wave2med

SUBMIT=0
[[ "$1" == "--submit" ]] && SUBMIT=1

PROJECT="xland-sfl-campaign"
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

LABELS=(  "dr"     "standard" "cnn" "hybrid_linear" "hybrid_soft_handoff")
METHODS=( "random" "standard" "cnn" "hybrid_linear" "hybrid_soft_handoff")
SEEDS=(1 2 3 4)

SCRIPTS=()
for SEED in "${SEEDS[@]}"; do
    for i in "${!METHODS[@]}"; do
        LABEL="${LABELS[$i]}"
        METHOD="${METHODS[$i]}"
        RUN_NAME="xlm_${LABEL}_seed${SEED}"
        JOB_NAME="w2m_${LABEL}_s${SEED}"
        SCRIPT_PATH="slurm_scripts/wave2med/${RUN_NAME}.sh"

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
    --config-name=xland-sfl-medium \\
    LEARN_METHOD=${METHOD} \\
    SEED=${SEED} \\
    RUN_NAME=${RUN_NAME} \\
    PROJECT=${PROJECT} \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=wave2-xlandmed-${LABEL}
unset TF_GPU_ALLOCATOR
BODY
        SCRIPTS+=("${SCRIPT_PATH}")
    done
done

if [[ ${SUBMIT} -eq 0 ]]; then
    printf 'Generated (NOT submitted): %s\n' "${SCRIPTS[@]}"
    echo "Review slurm_scripts/wave2med/, then: ./launch_wave2_medium.sh --submit"
    exit 0
fi

REMAINING=()
for SCRIPT in "${SCRIPTS[@]}"; do
    if [[ ${#REMAINING[@]} -eq 0 ]]; then
        OUT=$(sbatch "$SCRIPT" 2>&1) && echo "Submitted $(basename $SCRIPT): ${OUT#Submitted batch job }" \
            || { echo "cap reached at $(basename $SCRIPT); feeding remainder in background"; REMAINING+=("$SCRIPT"); }
    else
        REMAINING+=("$SCRIPT")
    fi
done

if [[ ${#REMAINING[@]} -gt 0 ]]; then
    nohup bash -c '
        cd "'"$WORKDIR"'"
        for S in '"${REMAINING[*]}"'; do
            while true; do
                Q=$(squeue -u "$USER" -h | wc -l)
                if [ "$Q" -lt 30 ]; then
                    O=$(sbatch "$S" 2>&1)
                    if echo "$O" | grep -q Submitted; then
                        echo "$(date "+%F %T") submitted $S (${O#Submitted batch job })" >> slurm_logs/wave2m_feed.log
                        break
                    fi
                fi
                sleep 300
            done
        done
        echo "$(date "+%F %T") ALL_SUBMITTED" >> slurm_logs/wave2m_feed.log
    ' >/dev/null 2>&1 &
    echo "feeder started for ${#REMAINING[@]} remaining scripts (log: slurm_logs/wave2m_feed.log)"
fi
