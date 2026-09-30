#!/bin/bash
# =============================================================
# Wave 0 — smoke runs (jaxnav campaign)
# 2 short runs (standard + hybrid_linear) verifying: ensemble
# checkpoint load, wrapper plumbing (probe panel + member std),
# run naming, diagnostics, W&B logging.
#
# Default: GENERATE sbatch scripts only (into slurm_scripts/wave0/).
# Submit with:  ./launch_wave0.sh --submit
# =============================================================

set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/wave0

SUBMIT=0
[[ "$1" == "--submit" ]] && SUBMIT=1

# One W&B project for the whole campaign; waves are told apart by GROUP_NAME.
PROJECT="sfl-jaxnav-campaign"
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
GROUP="wave0-smoke"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

# ~2.7e7 steps -> ~205 updates -> 4 eval cycles + checkpoint saves (~15-20 min)
# (plain integer so the hydra CLI override always parses as a number)
SMOKE_TIMESTEPS=27000000
SEED=0

METHODS=("standard" "hybrid_linear")

for METHOD in "${METHODS[@]}"; do
    RUN_NAME="smoke_${METHOD}_seed${SEED}"
    JOB_NAME="w0_${METHOD}"
    SCRIPT_PATH="slurm_scripts/wave0/${RUN_NAME}.sh"

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
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py \\
    LEARN_METHOD=${METHOD} \\
    SEED=${SEED} \\
    RUN_NAME=${RUN_NAME} \\
    PROJECT=${PROJECT} \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=${GROUP} \\
    learning.TOTAL_TIMESTEPS=${SMOKE_TIMESTEPS}
unset TF_GPU_ALLOCATOR
BODY

    if [[ ${SUBMIT} -eq 1 ]]; then
        echo "Submitting: ${RUN_NAME}"
        sbatch "${SCRIPT_PATH}"
    else
        echo "Generated (NOT submitted): ${SCRIPT_PATH}"
    fi
done

echo ""
if [[ ${SUBMIT} -eq 1 ]]; then
    echo "Submitted ${#METHODS[@]} Wave-0 smoke jobs."
else
    echo "Review the scripts in slurm_scripts/wave0/, then run: ./launch_wave0.sh --submit"
fi
