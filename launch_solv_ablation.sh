#!/bin/bash
# =============================================================
# Solvability-filtered ablation (thesis C6/C7, Malik 2026-09-03).
#
# QUESTION: does the late deficit of the prior-guided arms come from
# the unsolvable levels that the difficulty target drives them toward?
# Same pipeline as wave1fix, but the level generator is restricted to
# solvable maps via the env flag
#     env.env_params.map_params.valid_path_check=True
# (hydra override; the shared yaml is NOT edited). Because candidates,
# training resets and the frozen percentile reference all go through
# env.reset, the whole run sees the filtered generator consistently.
# The 100-level eval set and the scorer checkpoints are unchanged.
# Verified 2026-09-03: 300 sampled levels/seed are 100% passable under
# the flag (env.map_obj.dikstra_path), ~80% without it.
#
# ARMS: standard (baseline must be re-run on the same generator),
#       cnn (pure prior), hybrid_linear (best hybrid). 10 paired seeds.
# NAMES: project sfl-jaxnav-campaign-fixed, groups wave1fix-solv-<arm>,
#        runs solv_<arm>_seed<N>. Smoke: project sfl-jaxnav-campaign,
#        group wave0-smoke, run smoke_solv_cnn_seed0 (9% of the budget).
# PARTITION: NvidiaAll only (Abaki queue is occupied by the XLand arms).
#
# Usage:  ./launch_solv_ablation.sh            # generate scripts only
#         ./launch_solv_ablation.sh --smoke    # submit the smoke run
#         ./launch_solv_ablation.sh --submit   # submit seed sets until the
#                                              # 30-job cap refuses
#         ./launch_solv_ablation.sh --dripfeed # loop: submit the rest as
#                                              # slots free (nohup this)
# NOTE: cluster reboots Sunday ~04:40; jobs running then are requeued
# and restart from scratch (duplicate W&B names -> analysis keeps the
# newest full-length attempt, as for wave1fix).
# =============================================================
set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/solv

MODE="generate"
case "${1:-}" in
  --smoke)    MODE="smoke";;
  --submit)   MODE="submit";;
  --dripfeed) MODE="dripfeed";;
esac

ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
FLAG="env.env_params.map_params.valid_path_check=True"
ARMS=(standard cnn hybrid_linear)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

emit() {  # $1=script $2=jobname $3=runname $4=project $5=group $6=method $7=seed $8=extra
    cat > "$1" << 'HEADER'
#!/bin/bash
#SBATCH --partition=NvidiaAll
#SBATCH --qos=abaki
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --requeue
HEADER
    cat >> "$1" << BODY
#SBATCH --job-name=$2
#SBATCH --output=slurm_logs/%j_out.log
#SBATCH --error=slurm_logs/%j_err.log

# GPU-busy guard (no gres tracking on this cluster; 2026-08-31 incident)
USED=99999
for i in \$(seq 1 60); do
    USED=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "\${USED:-99999}" -lt 1500 ]; then break; fi
    sleep 30
done
if [ "\${USED:-99999}" -ge 1500 ]; then
    echo "GPU busy on \$(hostname) (\${USED} MiB used) - requeueing"
    scontrol requeue \$SLURM_JOB_ID
    sleep 60
    exit 0
fi

cd ${WORKDIR}
source .venv/bin/activate
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py \\
    LEARN_METHOD=$6 \\
    SEED=$7 \\
    RUN_NAME=$3 \\
    PROJECT=$4 \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=$5 \\
    ${FLAG} $8
unset TF_GPU_ALLOCATOR
BODY
}

# smoke: cnn arm (exercises ensemble + percentile reference + flag), 9% budget
emit slurm_scripts/solv/smoke_solv_cnn_seed0.sh solv_smoke smoke_solv_cnn_seed0 \
     sfl-jaxnav-campaign wave0-smoke cnn 0 "learning.TOTAL_TIMESTEPS=27000000"

# seed-complete order so a stalled queue still leaves complete paired sets
ORDERED=()
for SEED in "${SEEDS[@]}"; do
    for ARM in "${ARMS[@]}"; do
        F="slurm_scripts/solv/solv_${ARM}_seed${SEED}.sh"
        emit "$F" "sv_${ARM}_s${SEED}" "solv_${ARM}_seed${SEED}" \
             sfl-jaxnav-campaign-fixed "wave1fix-solv-${ARM}" "${ARM}" "${SEED}" ""
        ORDERED+=("$F")
    done
done

already_queued_or_done() {  # $1=script
    local JOB="sv_$(basename "$1" .sh | sed 's/^solv_//; s/_seed/_s/')"
    squeue -u "$USER" -h -o "%j" | grep -qx "$JOB" && return 0
    sacct -n -X --name="$JOB" --starttime=2026-09-03 -o State 2>/dev/null | grep -qE "COMPLETED|RUNNING|PENDING" && return 0
    return 1
}

case "$MODE" in
  generate)
    echo "Generated 1 smoke + ${#ORDERED[@]} seed scripts in slurm_scripts/solv/ (nothing submitted)";;
  smoke)
    sbatch slurm_scripts/solv/smoke_solv_cnn_seed0.sh;;
  submit)
    N=0
    for F in "${ORDERED[@]}"; do
        already_queued_or_done "$F" && continue
        OUT=$(sbatch "$F" 2>&1)
        if echo "$OUT" | grep -q Submitted; then echo "Submitted $(basename "$F"): ${OUT#Submitted batch job }"; N=$((N+1))
        else echo "STOP at $(basename "$F"): $OUT"; echo "(cap reached - run: nohup ./launch_solv_ablation.sh --dripfeed &)"; break; fi
    done
    echo "Submitted $N jobs.";;
  dripfeed)
    LOG=slurm_logs/solv_dripfeed.log
    for F in "${ORDERED[@]}"; do
        already_queued_or_done "$F" && continue
        while true; do
            if [ "$(squeue -u "$USER" -h | wc -l)" -lt 30 ]; then
                OUT=$(sbatch "$F" 2>&1)
                if echo "$OUT" | grep -q Submitted; then echo "$(date '+%F %T') submitted $(basename "$F") (${OUT#Submitted batch job })" >> "$LOG"; break; fi
                echo "$(date '+%F %T') refused $(basename "$F"): $OUT" >> "$LOG"
            fi
            sleep 300
        done
    done
    echo "$(date '+%F %T') drip-feed complete" >> "$LOG";;
esac
