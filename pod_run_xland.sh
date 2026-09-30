#!/bin/bash
# Run ONE xland CNN/hybrid run on this pod (one run per pod = full parallel
# fan-out; 13-14h on A5000 compressed into the deadline window by using one
# fast GPU per run).
#   bash pod_run_xland.sh <variant> <arm> <seed>
#     variant: difficult | medium
#     arm    : cnn | hybrid_linear | hybrid_soft_handoff
# Idempotent via markers so a re-run after a disconnect does not duplicate
# the W&B run.
set -uo pipefail
cd /workspace/sampling-for-learnability
source .venv/bin/activate
ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
mkdir -p logs markers

VARIANT="${1:?variant}"; ARM="${2:?arm}"; SEED="${3:?seed}"
if [[ "$VARIANT" == "medium" ]]; then
    CFG="xland-sfl-medium"; PREFIX="xlm"; GROUP="wave2-xlandmed-${ARM}"
else
    CFG="xland-sfl";        PREFIX="xl";  GROUP="wave2-xland-${ARM}"
fi
RUN="${PREFIX}_${ARM}_seed${SEED}"

if [[ -f "markers/${RUN}.DONE" ]]; then echo "SKIP ${RUN} (done)"; exit 0; fi
echo "=== $(date -u +%FT%TZ) START ${RUN} on $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/xland_sfl.py \
    --config-name=${CFG} \
    LEARN_METHOD=${ARM} \
    SEED=${SEED} \
    RUN_NAME=${RUN} \
    PROJECT=xland-sfl-campaign \
    ENTITY=${ENTITY} \
    GROUP_NAME=${GROUP} > "logs/${RUN}.log" 2>&1
rc=$?
echo "=== $(date -u +%FT%TZ) END ${RUN} exit=${rc} ==="
[[ ${rc} -eq 0 ]] && touch "markers/${RUN}.DONE"
exit ${rc}
