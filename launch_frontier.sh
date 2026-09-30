#!/bin/bash
# =============================================================
# Frontier-controller wave (Malik, 2026-09-12): jaxnav, BOTH generators.
#
# QUESTION: does fixing the target-mu controller close the gap between the
# prior-guided hybrid and standard SFL, and remove the early-phase lag?
# Trainer: sfl/train/jaxnav_sfl_frontier.py (a COPY of jaxnav_sfl.py; the
# original is untouched) with config sfl/train/config/jaxnav-sfl-frontier.yaml.
# Generators (two full sets, 4 arms x 4 seeds each):
#   solv_*  valid_path_check=True  (solvable-only generator, as in the solv ablation:
#           the top of the difficulty axis is solvable)
#   norm_*  valid_path_check=False (the campaign's default generator, ~18% unreachable
#           levels; the frontier rule is expected to hold mu near 0.8 instead of 1.0)
#
# ARMS (4 paired seeds each, seeds 1-4, per generator):
#   standard    plain SFL baseline (re-run here so all arms share code/hardware;
#               logs best_maps / highest_in_curriculum / lowest_in_curriculum /
#               buffer_sample / unsolvable_maps images every cycle)
#   hybrid_pa   hybrid_linear, OLD performance_adaptive rule driven by the FIXED
#               measurement (buffer-only, per-env success)  -> isolates the
#               measurement fix
#   hybrid_cf   hybrid_linear, calibrated_frontier, FRONTIER_MODE=crossing
#               (mu = c where the isotonic p-hat(c) crosses 0.5; fallback to the
#               fixed performance_adaptive until >= 3 bins have p-hat > 0.05)
#   hybrid_cfp  hybrid_linear, calibrated_frontier, FRONTIER_MODE=priority
#               (pre-registered form: CNN term = 4 * p-hat(c)(1 - p-hat(c)))
# NAMES: project sfl-jaxnav-frontier, groups frontier-<gen>-<arm>, runs <gen>_<arm>_seed<N>.
# SMOKE: group smoke, runs smoke_<arm>_seed0 (9% of the budget = 4 cycles; solvable generator).
# PARTITION: NvidiaAll, qos normal (the abaki qos was withdrawn from the account
#            after 2026-09-03), 8 h (solv runs took 3.9-4.9 h).
#
# Usage:  ./launch_frontier.sh            # generate scripts only
#         ./launch_frontier.sh --smoke    # submit the two smoke runs (hybrid_cf, standard)
#         ./launch_frontier.sh --submit   # submit the 32 seed runs (stops at the 30-job cap)
#         ./launch_frontier.sh --dripfeed # loop: submit the rest as slots free (nohup this)
# =============================================================
set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/frontier

MODE="generate"
case "${1:-}" in
  --smoke)    MODE="smoke";;
  --submit)   MODE="submit";;
  --dripfeed) MODE="dripfeed";;
esac

ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
PROJECT="sfl-jaxnav-frontier"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
FLAG_SOLV="env.env_params.map_params.valid_path_check=True"
FLAG_NORM="env.env_params.map_params.valid_path_check=False"
GENS=(solv norm)
ARMS=(standard hybrid_pa hybrid_cf hybrid_cfp)
SEEDS=(1 2 3 4)
gen_flag() { case "$1" in solv) echo "$FLAG_SOLV";; norm) echo "$FLAG_NORM";; *) echo "unknown gen $1" >&2; exit 1;; esac; }

arm_overrides() {  # $1=arm -> hydra overrides for that arm
    case "$1" in
      standard)   echo "LEARN_METHOD=standard";;
      hybrid_pa)  echo "LEARN_METHOD=hybrid_linear CURRICULUM_STRATEGY=performance_adaptive MU_MEASUREMENT=buffer_env";;
      hybrid_cf)  echo "LEARN_METHOD=hybrid_linear CURRICULUM_STRATEGY=calibrated_frontier FRONTIER_MODE=crossing MU_MEASUREMENT=buffer_env";;
      hybrid_cfp) echo "LEARN_METHOD=hybrid_linear CURRICULUM_STRATEGY=calibrated_frontier FRONTIER_MODE=priority MU_MEASUREMENT=buffer_env";;
      *) echo "unknown arm $1" >&2; exit 1;;
    esac
}

emit() {  # $1=script $2=jobname $3=runname $4=group $5=arm $6=seed $7=extra $8=genflag
    cat > "$1" << 'HEADER'
#!/bin/bash
#SBATCH --partition=NvidiaAll
#SBATCH --qos=normal
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
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl_frontier.py \\
    $(arm_overrides "$5") \\
    SEED=$6 \\
    RUN_NAME=$3 \\
    PROJECT=${PROJECT} \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=$4 \\
    $8 $7
unset TF_GPU_ALLOCATOR
BODY
}

# smoke: the full new path (hybrid_cf) and the baseline (standard), 9% budget = 4 cycles
emit slurm_scripts/frontier/smoke_hybrid_cf_seed0.sh fr_smoke_cf smoke_hybrid_cf_seed0 smoke hybrid_cf 0 "learning.TOTAL_TIMESTEPS=27000000" "$FLAG_SOLV"
emit slurm_scripts/frontier/smoke_standard_seed0.sh fr_smoke_std smoke_standard_seed0 smoke standard 0 "learning.TOTAL_TIMESTEPS=27000000" "$FLAG_SOLV"

# seed-complete order (seed-major, then generator, then arm) so a stalled queue
# still leaves complete paired sets in both generators
rm -f slurm_scripts/frontier/[a-z]*_seed[1-9].sh
ORDERED=()
for SEED in "${SEEDS[@]}"; do
    for GEN in "${GENS[@]}"; do
        for ARM in "${ARMS[@]}"; do
            F="slurm_scripts/frontier/${GEN}_${ARM}_seed${SEED}.sh"
            emit "$F" "fr_${GEN}_${ARM}_s${SEED}" "${GEN}_${ARM}_seed${SEED}" "frontier-${GEN}-${ARM}" "${ARM}" "${SEED}" "" "$(gen_flag "$GEN")"
            ORDERED+=("$F")
        done
    done
done

already_queued_or_done() {  # $1=script
    local JOB="fr_$(basename "$1" .sh | sed 's/_seed/_s/')"
    squeue -u "$USER" -h -o "%j" | grep -qx "$JOB" && return 0
    sacct -n -X --name="$JOB" --starttime=2026-09-12 -o State 2>/dev/null | grep -qE "COMPLETED|RUNNING|PENDING" && return 0
    return 1
}

case "$MODE" in
  generate)
    echo "Generated 2 smoke + ${#ORDERED[@]} seed scripts (${#GENS[@]} generators x ${#ARMS[@]} arms x ${#SEEDS[@]} seeds) in slurm_scripts/frontier/ (nothing submitted)";;
  smoke)
    sbatch slurm_scripts/frontier/smoke_hybrid_cf_seed0.sh
    sbatch slurm_scripts/frontier/smoke_standard_seed0.sh;;
  submit)
    N=0
    for F in "${ORDERED[@]}"; do
        already_queued_or_done "$F" && continue
        OUT=$(sbatch "$F" 2>&1)
        if echo "$OUT" | grep -q Submitted; then echo "Submitted $(basename "$F"): ${OUT#Submitted batch job }"; N=$((N+1))
        else echo "STOP at $(basename "$F"): $OUT"; echo "(cap reached - run: nohup ./launch_frontier.sh --dripfeed &)"; break; fi
    done
    echo "Submitted $N jobs.";;
  dripfeed)
    LOG=slurm_logs/frontier_dripfeed.log
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
