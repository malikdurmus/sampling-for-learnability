#!/bin/bash
# Feeds the remaining Wave-1-fix jobs as SLURM slots free (30-job submit cap).
# Started via `at` after the weekly Sunday reboot, since any process running
# at ~04:45 is killed with the node.
set -u
cd "$(dirname "$0")"
MAX_SUBMIT=30
LOG=slurm_logs/wave1fix_dripfeed.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# Seed-complete order: all arms for seed 1, then seed 2, ... so a stalled
# queue still leaves COMPLETE paired sets for the Wilcoxon analysis.
ORDERED=()
for SEED in 1 2 3 4 5 6 7 8 9 10; do
    for M in cnn hybrid_linear hybrid_soft_handoff hybrid_learnability_weighted hybrid_multiplicative; do
        F="slurm_scripts/wave1fix/${M}_fixed_seed${SEED}.sh"
        [ -f "$F" ] && ORDERED+=("$F")
    done
done

remaining=()
for SCRIPT in "${ORDERED[@]}"; do
    JOB="w1f_$(basename "$SCRIPT" .sh | sed 's/_fixed_seed/_s/')"
    squeue -u "$USER" -h -o "%j" | grep -qx "$JOB" && continue
    sacct -n -X --name="$JOB" --starttime=2026-08-30 -o State 2>/dev/null | grep -qE "COMPLETED|RUNNING|PENDING" && continue
    remaining+=("$SCRIPT")
done
log "drip-feed starting: ${#remaining[@]} scripts remaining"

for SCRIPT in "${remaining[@]}"; do
    while true; do
        if [ "$(squeue -u "$USER" -h | wc -l)" -lt "$MAX_SUBMIT" ]; then
            OUT=$(sbatch "$SCRIPT" 2>&1)
            if echo "$OUT" | grep -q Submitted; then
                log "submitted $(basename "$SCRIPT") (${OUT#Submitted batch job })"
                break
            fi
            log "sbatch refused $(basename "$SCRIPT"): $OUT — retry in 300s"
        fi
        sleep 300
    done
done
log "drip-feed complete: all Wave-1-fix jobs submitted"
echo "ALL_SUBMITTED $(date '+%F %T')" > slurm_logs/wave1fix_dripfeed_done.marker
