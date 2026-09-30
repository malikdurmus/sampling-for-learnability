#!/bin/bash
# Run a list of arm:seed jobs sequentially on this pod (credit-outage
# rebuild, 2026-09-02). Tokens: <arm>:<seed>, e.g.
#   bash pod_run_arm.sh progress:10 progress_mindist:5 progress_mindist:7
# Also accepts the old form "<arm> <seed> <seed>..." for compatibility.
# Full 3e8 budget, project sfl-jaxnav-campaign-fixed, group wave1fix-<arm>,
# idempotent DONE markers.
set -uo pipefail
cd /workspace/sampling-for-learnability
source .venv/bin/activate
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
mkdir -p logs markers

JOBS=()
if [[ "${1:-}" == *:* ]]; then
    JOBS=("$@")
else
    ARM="${1:?arm}"; shift
    for S in "$@"; do JOBS+=("${ARM}:${S}"); done
fi

for J in "${JOBS[@]}"; do
    ARM="${J%%:*}"; SEED="${J##*:}"
    RUN="${ARM}_seed${SEED}"
    if [[ -f "markers/${RUN}.DONE" ]]; then echo "SKIP ${RUN}"; continue; fi
    echo "=== $(date -u +%FT%TZ) START ${RUN} ==="
    TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/jaxnav_sfl.py \
        LEARN_METHOD="${ARM}" SEED="${SEED}" RUN_NAME="${RUN}" \
        PROJECT=sfl-jaxnav-campaign-fixed ENTITY="${ENTITY}" \
        GROUP_NAME="wave1fix-${ARM}" > "logs/${RUN}.log" 2>&1
    rc=$?
    echo "=== $(date -u +%FT%TZ) END ${RUN} exit=${rc} ==="
    [[ ${rc} -eq 0 ]] && touch "markers/${RUN}.DONE" || echo "RUN ${RUN} FAILED (continuing)"
done
echo "ARM SEED SET ATTEMPTED $(date -u +%FT%TZ)"
