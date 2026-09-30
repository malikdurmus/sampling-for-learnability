#!/bin/bash
# Provision ONE xland pod from the workstation and start its assigned run.
#   bash provision_xland_pod.sh <host> <port> <variant> <arm> <seed>
# Syncs repo + patched jaxmarl + the scorer checkpoints for that variant
# (at their original absolute path — configs and rlhf_utils hardcode it),
# installs the venv, verifies, then launches the run detached.
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; VARIANT="${3:?variant}"; ARM="${4:?arm}"; SEED="${5:?seed}"
KEY="$HOME/.ssh/id_ed25519"
SSHO="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
SSH="ssh $SSHO -i $KEY -p $PORT root@$HOST"
RS="ssh $SSHO -i $KEY -p $PORT"
SRC="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
UED="/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints"
VENVPKG="$SRC/.venv/lib/python3.12/site-packages"
TAG="[$HOST:$PORT $VARIANT/$ARM/s$SEED]"

echo "$TAG waiting for sshd"
for i in $(seq 1 40); do $SSH "echo ok" >/dev/null 2>&1 && break; sleep 15; done
$SSH "echo sshd_ready" || { echo "$TAG SSH FAILED"; exit 1; }

echo "$TAG syncing repo"
rsync -rltz --no-o --no-g --timeout=120 -e "$RS" \
  --exclude .venv --exclude .git --exclude wandb --exclude slurm_logs \
  --exclude slurm_scripts --exclude checkpoints --exclude outputs --exclude results \
  --exclude '*.png' --exclude '*.pdf' --exclude '*.npz' --exclude __pycache__ --exclude .vscode \
  "$SRC/" "root@$HOST:/workspace/sampling-for-learnability/" || { echo "$TAG RSYNC REPO FAILED"; exit 1; }

echo "$TAG syncing patched jaxmarl (GitHub pin is NOT equivalent - see 2026-09-01)"
$SSH "mkdir -p /workspace/sampling-for-learnability/.venv_pkgs"
rsync -rltz --no-o --no-g --timeout=120 -e "$RS" --exclude __pycache__ \
  "$VENVPKG/jaxmarl/" "root@$HOST:/workspace/sampling-for-learnability/.venv_pkgs/jaxmarl/" \
  || { echo "$TAG RSYNC JAXMARL FAILED"; exit 1; }

echo "$TAG syncing scorer checkpoints ($VARIANT) to original absolute path"
if [[ "$VARIANT" == "medium" ]]; then ENSDIR="xmg_medium"; else ENSDIR="xmg_difficult"; fi
$SSH "mkdir -p $UED/ensemble/$ENSDIR"
rsync -rltz --no-o --no-g --timeout=300 -e "$RS" "$UED/CHECKPOINT_INDEX.json" "root@$HOST:$UED/" \
  || { echo "$TAG RSYNC INDEX FAILED"; exit 1; }
rsync -rltz --no-o --no-g --timeout=600 -e "$RS" --exclude __pycache__ \
  "$UED/ensemble/$ENSDIR/" "root@$HOST:$UED/ensemble/$ENSDIR/" \
  || { echo "$TAG RSYNC CKPT FAILED"; exit 1; }

echo "$TAG installing wandb creds"
grep -A2 "api.wandb.ai" "$HOME/.netrc" | $SSH "cat >> ~/.netrc && chmod 600 ~/.netrc"

echo "$TAG venv setup (this takes ~8 min)"
$SSH "cd /workspace/sampling-for-learnability && XLAND_VARIANT=$VARIANT bash pod_setup_xland.sh > setup.log 2>&1; \
      cp -r .venv_pkgs/jaxmarl/. .venv/lib/python3.12/site-packages/jaxmarl/ 2>/dev/null; \
      tail -6 setup.log"
if ! $SSH "grep -q 'XLAND POD SETUP COMPLETE' /workspace/sampling-for-learnability/setup.log"; then
    echo "$TAG SETUP FAILED — see setup.log on pod"; exit 1
fi

echo "$TAG launching run detached"
$SSH "cd /workspace/sampling-for-learnability && mkdir -p logs markers && (setsid nohup bash pod_run_xland.sh $VARIANT $ARM $SEED > logs/driver_${ARM}_s${SEED}.log 2>&1 < /dev/null &) ; sleep 5; cat logs/driver_${ARM}_s${SEED}.log"
echo "$TAG LAUNCHED"
