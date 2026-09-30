#!/bin/bash
# =============================================================
# Wave 1-FIX — rerun of the jaxnav CNN arms with the corrected
# scorer input domain (THESIS_FINDINGS.md Finding 5 resolution).
#
# WHY: Wave 1's five CNN-using arms ran with the scorer fed TRUE
# images while it had been trained on color-inverted ones — it
# scored 0.7675 instead of 0.9075 and had degenerated into a wall
# counter. Those 50 runs are confounded and are rerun here.
#
# NOT rerun (still valid, reused as the paired baseline): the
# original wave1-dr and wave1-standard arms — they never call the
# CNN, so the fix cannot change their behaviour, and they share
# seeds 1-10 with these runs so the pairing is preserved.
#
# Old runs are RETAINED in W&B (nothing deleted); these use new
# names/groups so the comparison stays auditable.
#
# Default: GENERATE sbatch scripts only (slurm_scripts/wave1fix/).
# Submit with:  ./launch_wave1fix.sh --submit
# =============================================================

set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/wave1fix

SUBMIT=0
[[ "$1" == "--submit" ]] && SUBMIT=1

PROJECT="sfl-jaxnav-campaign-fixed"   # new project; Wave-1 dr/standard baselines copied in
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

# CNN-using arms only (dr/standard are unaffected and reused from Wave 1)
LABELS=(  "cnn" "hybrid_linear" "hybrid_soft_handoff" "hybrid_learnability_weighted" "hybrid_multiplicative")
METHODS=( "cnn" "hybrid_linear" "hybrid_soft_handoff" "hybrid_learnability_weighted" "hybrid_multiplicative")
SEEDS=(1 2 3 4 5 6 7 8 9 10)

COUNT=0
for SEED in "${SEEDS[@]}"; do            # seed-complete blocks
    for i in "${!METHODS[@]}"; do
        LABEL="${LABELS[$i]}"
        METHOD="${METHODS[$i]}"
        RUN_NAME="${LABEL}_fixed_seed${SEED}"
        JOB_NAME="w1f_${LABEL}_s${SEED}"
        SCRIPT_PATH="slurm_scripts/wave1fix/${RUN_NAME}.sh"

        cat > "${SCRIPT_PATH}" << 'HEADER'
#!/bin/bash
# Free mixed queue; abakus11/12 excluded (another user needs them).
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --qos=abaki
#SBATCH --exclude=abakus11,abakus12
# ~3.2h/run measured; limit lets backfill schedule and caps any hang.
# NOTE: the cluster reboots weekly, Sunday ~04:45 (nodes last booted
# 2026-08-23 04:43). Jobs running then are killed and, since JobRequeue=1,
# restart from scratch — submit with --begin after the reboot window.
#SBATCH --time=05:00:00
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
    GROUP_NAME=wave1fix-${LABEL}
unset TF_GPU_ALLOCATOR
BODY

        COUNT=$((COUNT + 1))
        if [[ ${SUBMIT} -eq 1 ]]; then
            OUT=$(sbatch ${SBATCH_EXTRA:-} "${SCRIPT_PATH}" 2>&1)
            if echo "$OUT" | grep -q Submitted; then
                echo "Submitted ${RUN_NAME}: ${OUT#Submitted batch job }"
            else
                echo "STOP at ${RUN_NAME}: ${OUT}"
                echo "(submit limit reached — run ./wave1fix_dripfeed.sh to feed the rest)"
                exit 0
            fi
        else
            echo "Generated (NOT submitted): ${SCRIPT_PATH}"
        fi
    done
done

echo ""
if [[ ${SUBMIT} -eq 1 ]]; then
    echo "Submitted ${COUNT} Wave-1-fix jobs."
else
    echo "Generated ${COUNT} scripts. Review slurm_scripts/wave1fix/, then: ./launch_wave1fix.sh --submit"
fi
