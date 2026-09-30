#!/bin/bash
# =============================================================
# Wave 1 — headline comparison (jaxnav campaign)
# 7 methods x 10 paired seeds at the pre-declared defaults
# (scorer = jaxnav aug ensemble, CURRICULUM_STRATEGY =
# performance_adaptive, CNN_SCORE_METHOD = sigmoid — all from
# jaxnav-sfl.yaml).
#
# Submission order is SEED-COMPLETE: all 7 methods for seed 1,
# then seed 2, ... so a stalled queue still leaves complete
# paired sets for the Wilcoxon analysis.
#
# Default: GENERATE sbatch scripts only (into slurm_scripts/wave1/).
# Submit with:  ./launch_wave1.sh --submit
# =============================================================

set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/wave1

SUBMIT=0
[[ "$1" == "--submit" ]] && SUBMIT=1

# One W&B project for the whole campaign; waves are told apart by GROUP_NAME.
PROJECT="sfl-jaxnav-campaign"
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

# label -> LEARN_METHOD (label is used in run/group names; "dr" runs as
# LEARN_METHOD=random, i.e. domain randomization through the same loop)
LABELS=(  "dr"     "standard" "cnn" "hybrid_linear" "hybrid_soft_handoff" "hybrid_learnability_weighted" "hybrid_multiplicative")
METHODS=( "random" "standard" "cnn" "hybrid_linear" "hybrid_soft_handoff" "hybrid_learnability_weighted" "hybrid_multiplicative")

# 10 paired seeds (decided 2026-08-28): n=10 gives Wilcoxon min p = 0.00195,
# which survives Holm over the full 21-comparison pairwise family (0.05/21).
SEEDS=(1 2 3 4 5 6 7 8 9 10)

COUNT=0
for SEED in "${SEEDS[@]}"; do            # outer loop = seed (seed-complete blocks)
    for i in "${!METHODS[@]}"; do
        LABEL="${LABELS[$i]}"
        METHOD="${METHODS[$i]}"
        # One W&B run name per seed (never reuse a name across seeds)
        RUN_NAME="${LABEL}_seed${SEED}"
        JOB_NAME="w1_${LABEL}_s${SEED}"
        SCRIPT_PATH="slurm_scripts/wave1/${RUN_NAME}.sh"

        cat > "${SCRIPT_PATH}" << 'HEADER'
#!/bin/bash
# Both GPU partitions: SLURM starts the job wherever a node frees first.
# QOS=abaki is required by the Abaki partition (AllowQos=abaki) and is
# accepted by NvidiaAll (AllowQos=ALL) at identical priority.
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --qos=abaki
HEADER

        cat >> "${SCRIPT_PATH}" << BODY
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/%j_out.log
#SBATCH --error=slurm_logs/%j_err.log

cd ${WORKDIR}
source .venv/bin/activate
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py \\
    LEARN_METHOD=${METHOD} \\
    SEED=${SEED} \\
    RUN_NAME=${RUN_NAME} \\
    PROJECT=${PROJECT} \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=wave1-${LABEL}
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
    echo "Submitted ${COUNT} Wave-1 jobs (seed-complete order, ~3h15m each)."
else
    echo "Generated ${COUNT} scripts. Review slurm_scripts/wave1/, then run: ./launch_wave1.sh --submit"
fi
