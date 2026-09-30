#!/bin/bash
# Robust submit-cap feeder for the remaining wave-2 scripts.
# Fully detached (setsid) so it survives terminal/session teardown — the
# 2026-08-31 nohup feeder died with its parent and left 6 runs unsubmitted.
# Idempotent: skips any script whose job name is already queued/running.
# Log: slurm_logs/wave2_feeder.log
set -u
cd "$(dirname "$0")"
LOG=slurm_logs/wave2_feeder.log
MAX=30

jobname_of() {   # slurm_scripts/wave2med/xlm_cnn_seed4.sh -> w2m_cnn_s4
    grep -oP '(?<=#SBATCH --job-name=).*' "$1"
}

for S in "$@"; do
    JN=$(jobname_of "$S")
    while true; do
        if squeue -u "$USER" -h -o "%j" | grep -qx "$JN"; then
            echo "$(date '+%F %T') skip $JN (already queued)" >> "$LOG"; break
        fi
        Q=$(squeue -u "$USER" -h | wc -l)
        if [ "$Q" -lt "$MAX" ]; then
            O=$(sbatch "$S" 2>&1)
            if echo "$O" | grep -q Submitted; then
                echo "$(date '+%F %T') submitted $JN (${O#Submitted batch job }) queue=$((Q+1))" >> "$LOG"
                break
            else
                echo "$(date '+%F %T') sbatch refused $JN: $O" >> "$LOG"
            fi
        fi
        sleep 300
    done
done
echo "$(date '+%F %T') ALL_SUBMITTED" >> "$LOG"
