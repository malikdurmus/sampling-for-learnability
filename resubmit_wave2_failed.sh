#!/bin/bash
# One-shot driver (2026-08-31): resubmit the failed xland runs with the
# corrected headers (--time=2-00:00:00, --exclusive, GPU-busy guard).
# Failed set: all 8 difficult baselines (14h TIMEOUT), 10 difficult CNN-arm
# runs (GPU-collision OOM), and all medium runs except the 2 healthy ones.
# Keeps running jobs untouched. Feeds past the 30-job cap via a background
# loop (log: slurm_logs/wave2_resubmit_feed.log).
set -u
cd "$(dirname "$0")"

LIST=(
  # difficult baselines (seed-complete)
  slurm_scripts/wave2/xl_dr_seed1.sh    slurm_scripts/wave2/xl_standard_seed1.sh
  slurm_scripts/wave2/xl_dr_seed2.sh    slurm_scripts/wave2/xl_standard_seed2.sh
  slurm_scripts/wave2/xl_dr_seed3.sh    slurm_scripts/wave2/xl_standard_seed3.sh
  slurm_scripts/wave2/xl_dr_seed4.sh    slurm_scripts/wave2/xl_standard_seed4.sh
  # difficult CNN arms that OOM-crashed (cnn_s1 / hybrid_linear_s1 still running)
  slurm_scripts/wave2cnn/xl_hybrid_soft_handoff_seed1.sh
  slurm_scripts/wave2cnn/xl_cnn_seed2.sh
  slurm_scripts/wave2cnn/xl_hybrid_linear_seed2.sh
  slurm_scripts/wave2cnn/xl_hybrid_soft_handoff_seed2.sh
  slurm_scripts/wave2cnn/xl_cnn_seed3.sh
  slurm_scripts/wave2cnn/xl_hybrid_linear_seed3.sh
  slurm_scripts/wave2cnn/xl_hybrid_soft_handoff_seed3.sh
  slurm_scripts/wave2cnn/xl_cnn_seed4.sh
  slurm_scripts/wave2cnn/xl_hybrid_linear_seed4.sh
  slurm_scripts/wave2cnn/xl_hybrid_soft_handoff_seed4.sh
  # medium wave (xlm_dr_seed1 / xlm_standard_seed1 still running)
  slurm_scripts/wave2med/xlm_cnn_seed1.sh
  slurm_scripts/wave2med/xlm_hybrid_linear_seed1.sh
  slurm_scripts/wave2med/xlm_hybrid_soft_handoff_seed1.sh
  slurm_scripts/wave2med/xlm_dr_seed2.sh
  slurm_scripts/wave2med/xlm_standard_seed2.sh
  slurm_scripts/wave2med/xlm_cnn_seed2.sh
  slurm_scripts/wave2med/xlm_hybrid_linear_seed2.sh
  slurm_scripts/wave2med/xlm_hybrid_soft_handoff_seed2.sh
  slurm_scripts/wave2med/xlm_dr_seed3.sh
  slurm_scripts/wave2med/xlm_standard_seed3.sh
  slurm_scripts/wave2med/xlm_cnn_seed3.sh
  slurm_scripts/wave2med/xlm_hybrid_linear_seed3.sh
  slurm_scripts/wave2med/xlm_hybrid_soft_handoff_seed3.sh
  slurm_scripts/wave2med/xlm_dr_seed4.sh
  slurm_scripts/wave2med/xlm_standard_seed4.sh
  slurm_scripts/wave2med/xlm_cnn_seed4.sh
  slurm_scripts/wave2med/xlm_hybrid_linear_seed4.sh
  slurm_scripts/wave2med/xlm_hybrid_soft_handoff_seed4.sh
)

REMAINING=()
for S in "${LIST[@]}"; do
    if [[ ${#REMAINING[@]} -eq 0 ]]; then
        OUT=$(sbatch "$S" 2>&1)
        if echo "$OUT" | grep -q Submitted; then
            echo "Submitted $(basename $S): ${OUT#Submitted batch job }"
        else
            echo "cap reached at $(basename $S); feeding remainder in background"
            REMAINING+=("$S")
        fi
    else
        REMAINING+=("$S")
    fi
done

if [[ ${#REMAINING[@]} -gt 0 ]]; then
    nohup bash -c '
        cd "'"$PWD"'"
        for S in '"${REMAINING[*]}"'; do
            while true; do
                Q=$(squeue -u "$USER" -h | wc -l)
                if [ "$Q" -lt 30 ]; then
                    O=$(sbatch "$S" 2>&1)
                    if echo "$O" | grep -q Submitted; then
                        echo "$(date "+%F %T") submitted $S (${O#Submitted batch job })" >> slurm_logs/wave2_resubmit_feed.log
                        break
                    fi
                fi
                sleep 300
            done
        done
        echo "$(date "+%F %T") ALL_SUBMITTED" >> slurm_logs/wave2_resubmit_feed.log
    ' >/dev/null 2>&1 &
    echo "feeder started for ${#REMAINING[@]} scripts"
fi
