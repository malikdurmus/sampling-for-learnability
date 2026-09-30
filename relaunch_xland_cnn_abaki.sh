#!/bin/bash
# =============================================================
# Abaki relaunch of the xland CNN/hybrid runs that died with CUDA OOM.
#
# 2026-09-01 diagnosis: the 5-member ensemble + 100x100 rendered level
# images do not fit on the NvidiaAll GPUs; every CNN-arm run that ever
# finished ran on an Abaki A5000/A4500 (w2c_cnn_s1 on abakus12, 13h48m).
# => partition restricted to Abaki ONLY, plus a >=20GB VRAM guard that
#    requeues rather than OOMs.
#
# SPLIT (Malik 2026-09-01, cost cap): 2 runs go to RunPod pods, the other
# 20 come here. The two sets are DISJOINT — a run name must never execute
# in both places (duplicate W&B display names break analysis pulls that
# assert one run per name).
#   POD 1 (RTX 5090): xl_cnn_seed2      [difficult]
#   POD 2 (RTX 5090): xlm_cnn_seed1     [medium]
#   ABAKI: everything else (9 difficult + 11 medium)
#
#   --submit : submit all 20
#   default  : generate scripts only
# =============================================================
set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs slurm_scripts/xland_abaki

MODE="generate"
[[ "${1:-}" == "--submit" ]] && MODE="submit"

ENTITY="malikdurmus-ludwig-maximilian-university-of-munich"
WORKDIR="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

# variant:arm:seed   (xl_cnn_seed2 and xlm_cnn_seed1 excluded -> on pods)
JOBS=(
  # difficult (9) — cnn seed1 + hybrid_linear seed1 already finished earlier
  "difficult:cnn:3" "difficult:cnn:4"
  "difficult:hybrid_linear:2" "difficult:hybrid_linear:3" "difficult:hybrid_linear:4"
  "difficult:hybrid_soft_handoff:1" "difficult:hybrid_soft_handoff:2"
  "difficult:hybrid_soft_handoff:3" "difficult:hybrid_soft_handoff:4"
  # medium (11) — nothing finished for these arms yet
  "medium:cnn:2" "medium:cnn:3" "medium:cnn:4"
  "medium:hybrid_linear:1" "medium:hybrid_linear:2" "medium:hybrid_linear:3" "medium:hybrid_linear:4"
  "medium:hybrid_soft_handoff:1" "medium:hybrid_soft_handoff:2"
  "medium:hybrid_soft_handoff:3" "medium:hybrid_soft_handoff:4"
)

for entry in "${JOBS[@]}"; do
    VARIANT="${entry%%:*}"; REST="${entry#*:}"; ARM="${REST%%:*}"; SEED="${REST##*:}"
    if [[ "$VARIANT" == "medium" ]]; then
        CFG="xland-sfl-medium"; PREFIX="xlm"; GROUP="wave2-xlandmed-${ARM}"; TAG="m"
    else
        CFG="xland-sfl";        PREFIX="xl";  GROUP="wave2-xland-${ARM}";    TAG="d"
    fi
    SHORT=$(echo "$ARM" | sed 's/hybrid_linear/hlin/;s/hybrid_soft_handoff/hsh/')
    SCRIPT="slurm_scripts/xland_abaki/${PREFIX}_${ARM}_seed${SEED}.sh"
    cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
#SBATCH --partition=Abaki
#SBATCH --qos=abaki
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --requeue
HEADER
    cat >> "$SCRIPT" << BODY
#SBATCH --job-name=xa${TAG}_${SHORT}_s${SEED}
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
    echo "GPU busy on \$(hostname) (\${USED} MiB used) — requeueing"
    scontrol requeue \$SLURM_JOB_ID
    sleep 60
    exit 0
fi

# VRAM guard: the CNN arms need >=20GB (this is what killed them on NvidiaAll)
TOTAL=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "\${TOTAL:-0}" -lt 20000 ]; then
    echo "GPU too small on \$(hostname) (\${TOTAL} MiB, need >=20GB) — requeueing"
    scontrol requeue \$SLURM_JOB_ID
    sleep 60
    exit 0
fi

cd ${WORKDIR}
source .venv/bin/activate
TF_GPU_ALLOCATOR=cuda_malloc_async python -u sfl/train/xland_sfl.py \\
    --config-name=${CFG} \\
    LEARN_METHOD=${ARM} \\
    SEED=${SEED} \\
    RUN_NAME=${PREFIX}_${ARM}_seed${SEED} \\
    PROJECT=xland-sfl-campaign \\
    ENTITY=${ENTITY} \\
    GROUP_NAME=${GROUP}
unset TF_GPU_ALLOCATOR
BODY
done

case "$MODE" in
  generate) echo "Generated ${#JOBS[@]} Abaki scripts in slurm_scripts/xland_abaki/ (nothing submitted)";;
  submit)
    n=0
    for entry in "${JOBS[@]}"; do
        VARIANT="${entry%%:*}"; REST="${entry#*:}"; ARM="${REST%%:*}"; SEED="${REST##*:}"
        [[ "$VARIANT" == "medium" ]] && PREFIX="xlm" || PREFIX="xl"
        sbatch --chdir="${WORKDIR}" "${WORKDIR}/slurm_scripts/xland_abaki/${PREFIX}_${ARM}_seed${SEED}.sh" && n=$((n+1))
    done
    echo "submitted $n jobs";;
esac
