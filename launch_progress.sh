#!/bin/bash
# =============================================================
# Goal-proximity progress-variance arm (Malik's dense learnability
# metric, ported 2026-09-01 from the testnewmetric fork with the
# pre-step distance fix — the fork measured d_end on the post-step
# state, which JaxNav auto-resets in place, so it recorded d_start).
#
# Same rollout protocol as standard SFL; only the per-level score
# changes: var over episodes of progress = clip(1 - d_end/d_start,0,1).
# Binary p(1-p) is logged alongside (sfl/*) plus correlation/overlap
# diagnostics (progress/*) so the density claim is testable in W&B.
#
# --smoke : one short run (seed 0, 2.7e7 steps) -> project
#           sfl-jaxnav-campaign, group wave0-smoke
# --submit: progress_seed1..10 -> project sfl-jaxnav-campaign-fixed,
#           group wave1fix-progress (beside the paired dr/standard/
#           cnn/dijkstra arms)
# Default: generate scripts only.
# =============================================================
set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/progress

MODE="generate"
[[ "${1:-}" == "--submit" ]] && MODE="submit"
[[ "${1:-}" == "--smoke"  ]] && MODE="smoke"

ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

emit() {  # $1=script $2=jobname $3=runname $4=project $5=group $6=seed $7=extra
    cat > "$1" << 'HEADER'
#!/bin/bash
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --qos=abaki
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --requeue
HEADER
    cat >> "$1" << BODY
#SBATCH --job-name=$2
#SBATCH --output=slurm_logs/%j_out.log
#SBATCH --error=slurm_logs/%j_err.log

# GPU-busy guard (no gres tracking on this cluster; see 2026-08-31 incident)
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
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py \\
    LEARN_METHOD=progress \\
    SEED=$6 \\
    RUN_NAME=$3 \\
    PROJECT=$4 \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=$5 $7
unset TF_GPU_ALLOCATOR
BODY
}

# smoke script
emit slurm_scripts/progress/smoke_progress_seed0.sh prg_smoke smoke_progress_seed0 \
     sfl-jaxnav-campaign wave0-smoke 0 "learning.TOTAL_TIMESTEPS=27000000"

# seed set
for SEED in 1 2 3 4 5 6 7 8 9 10; do
    emit "slurm_scripts/progress/progress_seed${SEED}.sh" "prg_s${SEED}" "progress_seed${SEED}" \
         sfl-jaxnav-campaign-fixed wave1fix-progress "${SEED}" ""
done

case "$MODE" in
  generate)
    echo "Generated 1 smoke + 10 seed scripts in slurm_scripts/progress/ (nothing submitted)";;
  smoke)
    sbatch slurm_scripts/progress/smoke_progress_seed0.sh;;
  submit)
    for SEED in 1 2 3 4 5 6 7 8 9 10; do
        sbatch "slurm_scripts/progress/progress_seed${SEED}.sh"
    done;;
esac
