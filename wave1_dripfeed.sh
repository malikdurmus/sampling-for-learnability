#!/bin/bash
# =============================================================
# Wave-1 drip-feed submitter.
# The cluster caps each user at 30 submitted / 15 running jobs
# (sacctmgr assoc: MaxSubmit=30, MaxJobs=15), so the 70-job wave
# cannot be submitted at once. This script submits the REMAINING
# Wave-1 sbatch scripts in the same seed-complete order whenever
# the queue drops below the submit limit.
#
# Started 2026-08-28 after the first 30 jobs (seeds 1-4 complete,
# dr_s5 + standard_s5) were submitted directly.
# Log: slurm_logs/wave1_dripfeed.log ; marker file on completion:
# slurm_logs/wave1_dripfeed_done.marker
# =============================================================

set -u
cd "$(dirname "$0")"

MAX_SUBMIT=30
LOG=slurm_logs/wave1_dripfeed.log

LABELS=("dr" "standard" "cnn" "hybrid_linear" "hybrid_soft_handoff" "hybrid_learnability_weighted" "hybrid_multiplicative")

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# Build the remaining list in seed-complete order, skipping anything
# already submitted (already in the queue or already finished with a
# W&B-run log marker is not checkable here, so we track by job name).
remaining=()
for SEED in 5 6 7 8 9 10; do
    for LABEL in "${LABELS[@]}"; do
        JOB_NAME="w1_${LABEL}_s${SEED}"
        SCRIPT="slurm_scripts/wave1/${LABEL}_seed${SEED}.sh"
        # skip if already queued/running under this name
        if squeue -u "$USER" -h -o "%j" | grep -qx "${JOB_NAME}"; then
            continue
        fi
        # skip the two seed-5 jobs submitted before the limit hit
        if sacct -n -X --name="${JOB_NAME}" --starttime=2026-08-28 2>/dev/null | grep -q .; then
            continue
        fi
        remaining+=("${SCRIPT}")
    done
done

log "drip-feed starting: ${#remaining[@]} scripts remaining"

for SCRIPT in "${remaining[@]}"; do
    while true; do
        QUEUED=$(squeue -u "$USER" -h | wc -l)
        if [ "$QUEUED" -lt "$MAX_SUBMIT" ]; then
            OUT=$(sbatch --partition=NvidiaAll,Abaki --qos=abaki "$SCRIPT" 2>&1)
            if echo "$OUT" | grep -q "Submitted"; then
                log "submitted ${SCRIPT} (${OUT#Submitted batch job }) queued=$((QUEUED+1))"
                break
            else
                log "sbatch failed for ${SCRIPT}: ${OUT} — retrying in 300s"
            fi
        fi
        sleep 300
    done
done

log "drip-feed complete: all Wave-1 jobs submitted"
echo "ALL_SUBMITTED $(date '+%F %T')" > slurm_logs/wave1_dripfeed_done.marker
