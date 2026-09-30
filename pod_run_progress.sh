#!/bin/bash
# Sequential run driver for the three dense-learnability arms ON the pod.
#   bash pod_run_progress.sh smokes   -> 3 short smoke runs (wave0-smoke)
#   bash pod_run_progress.sh seeds    -> 3 arms x seeds 1..10 (wave1fix-*)
# Idempotent: a run whose DONE marker exists is skipped, so the driver can
# be restarted after a pod interruption without duplicating W&B runs.
# Run detached so it survives ssh disconnects:
#   setsid nohup bash pod_run_progress.sh seeds > logs/driver.log 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/sampling-for-learnability
source .venv/bin/activate
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
mkdir -p logs markers

run_one () {  # method seed runname project group extra
    local method=$1 seed=$2 runname=$3 project=$4 group=$5 extra=$6
    if [[ -f "markers/${runname}.DONE" ]]; then
        echo "SKIP ${runname} (marker exists)"; return 0
    fi
    echo "=== $(date -u +%FT%TZ) START ${runname} ==="
    TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py \
        LEARN_METHOD="${method}" SEED="${seed}" RUN_NAME="${runname}" \
        PROJECT="${project}" ENTITY="${ENTITY}" GROUP_NAME="${group}" ${extra} \
        > "logs/${runname}.log" 2>&1
    local rc=$?
    echo "=== $(date -u +%FT%TZ) END ${runname} exit=${rc} ==="
    [[ ${rc} -eq 0 ]] && touch "markers/${runname}.DONE"
    return ${rc}
}

# dijkstra moved here from SLURM 2026-09-01 (queue saturated; its pending
# smoke was cancelled there and the slot repaid to the medium wave)
ARMS=(progress progress_mindist progress_mean dijkstra)

case "${1:?usage: pod_run_progress.sh smokes|seeds}" in
  smokes)
    for m in "${ARMS[@]}"; do
        run_one "$m" 0 "smoke_${m}_seed0" sfl-jaxnav-campaign wave0-smoke \
                "learning.TOTAL_TIMESTEPS=27000000" || echo "SMOKE ${m} FAILED"
    done
    echo "ALL SMOKES ATTEMPTED $(date -u +%FT%TZ)"; touch markers/SMOKES_DONE;;
  seeds)
    # Seed-complete order (all 3 arms for seed 1, then seed 2, ...): an
    # interrupted campaign still leaves complete paired sets across arms.
    # 5 seeds (Malik 2026-09-01); full 3e8 budget — truncation at 1.5k
    # updates was declined because baselines still gain 2.5-5pt after
    # update 1500, which would confound final-checkpoint CVaR.
    for s in 1 2 3 4 5; do
        for m in "${ARMS[@]}"; do
            run_one "$m" "$s" "${m}_seed${s}" sfl-jaxnav-campaign-fixed \
                    "wave1fix-${m}" "" || echo "RUN ${m}_seed${s} FAILED (continuing)"
        done
    done
    echo "ALL SEED RUNS ATTEMPTED $(date -u +%FT%TZ)"; touch markers/SEEDS_DONE;;
esac
