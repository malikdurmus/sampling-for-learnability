#!/bin/bash
# Provision ONE jaxnav seed-extension pod and start its arm's seed range.
#   bash provision_jaxnav_pod.sh <host> <port> <arm> <seed> [<seed> ...]
# No scorer checkpoints needed (progress arms use no CNN; dijkstra uses the
# env's own Dijkstra). Venv goes on the LOCAL container disk (/root/venv,
# symlinked as .venv) — some hosts mount /workspace on a slow network FS
# (2026-09-01 pod2 lesson). Setup aborts if JAX cannot see the GPU.
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; ARM="${3:?arm}"; shift 3
SEEDS="$*"
KEY="$HOME/.ssh/id_ed25519"
SSHO="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
SSH="ssh $SSHO -i $KEY -p $PORT root@$HOST"
RS="ssh $SSHO -i $KEY -p $PORT"
SRC="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
VENVPKG="$SRC/.venv/lib/python3.12/site-packages"
TAG="[$HOST:$PORT $ARM seeds:$SEEDS]"

echo "$TAG waiting for sshd"
for i in $(seq 1 40); do $SSH "echo ok" >/dev/null 2>&1 && break; sleep 15; done
$SSH "echo sshd_ready" || { echo "$TAG SSH FAILED"; exit 1; }

echo "$TAG syncing repo"
rsync -rltz --no-o --no-g --timeout=120 -e "$RS" \
  --exclude .venv --exclude .git --exclude wandb --exclude slurm_logs \
  --exclude slurm_scripts --exclude checkpoints --exclude outputs --exclude results \
  --exclude '*.png' --exclude '*.pdf' --exclude '*.npz' --exclude __pycache__ --exclude .vscode \
  "$SRC/" "root@$HOST:/workspace/sampling-for-learnability/" || { echo "$TAG RSYNC REPO FAILED"; exit 1; }

echo "$TAG syncing patched jaxmarl (GitHub pin is NOT equivalent)"
$SSH "mkdir -p /workspace/sampling-for-learnability/.venv_pkgs"
rsync -rltz --no-o --no-g --timeout=120 -e "$RS" --exclude __pycache__ \
  "$VENVPKG/jaxmarl/" "root@$HOST:/workspace/sampling-for-learnability/.venv_pkgs/jaxmarl/" \
  || { echo "$TAG RSYNC JAXMARL FAILED"; exit 1; }

echo "$TAG installing wandb creds"
grep -A2 "api.wandb.ai" "$HOME/.netrc" | $SSH "cat >> ~/.netrc && chmod 600 ~/.netrc"

echo "$TAG venv setup on local disk (~8 min)"
$SSH "cd /workspace/sampling-for-learnability && { \
  python3 -m venv /root/venv && \
  /root/venv/bin/pip install --upgrade pip -q && \
  /root/venv/bin/pip install -r pod_env_freeze.txt -q && \
  /root/venv/bin/pip install -e . --no-deps -q && \
  cp -r .venv_pkgs/jaxmarl/. /root/venv/lib/python3.12/site-packages/jaxmarl/ && \
  ln -sfn /root/venv .venv && \
  /root/venv/bin/python -c 'import jax,sys; d=jax.devices(); print(\"jax devices:\",d); sys.exit(0 if d[0].platform==\"gpu\" else 1)' && \
  JAX_PLATFORMS=cpu /root/venv/bin/python verify_progress_arms.py > /dev/null 2>&1 && \
  echo SETUP_OK; } > setup.log 2>&1; tail -3 setup.log"
if ! $SSH "grep -q 'SETUP_OK' /workspace/sampling-for-learnability/setup.log"; then
    echo "$TAG SETUP FAILED"; $SSH "tail -8 /workspace/sampling-for-learnability/setup.log"; exit 1
fi

echo "$TAG launching seeds detached"
LOGN=$(echo "$ARM" | tr ':' '_')
$SSH "cd /workspace/sampling-for-learnability && mkdir -p logs markers && (setsid nohup bash pod_run_arm.sh $ARM $SEEDS > logs/driver_${LOGN}.log 2>&1 < /dev/null &); sleep 5; head -2 logs/driver_${LOGN}.log"
echo "$TAG LAUNCHED"
