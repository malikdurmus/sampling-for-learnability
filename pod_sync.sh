#!/bin/bash
# Push the repo to the pod over the RunPod SSH proxy. Run from the
# workstation:  bash pod_sync.sh <ssh-user>  (e.g. abc123xyz-64411dd2)
# Uses ~/.ssh/id_ed25519 (the "Key for CIP" key registered with the pod).
set -euo pipefail
POD_USER="${1:?usage: pod_sync.sh <runpod-ssh-user>}"
SSH="ssh -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519"
SRC="$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"

rsync -az --info=progress2 -e "$SSH" \
    --exclude .venv --exclude .git --exclude wandb --exclude slurm_logs \
    --exclude slurm_scripts --exclude checkpoints --exclude outputs \
    --exclude results --exclude '*.png' --exclude '*.pdf' --exclude '*.npz' \
    --exclude __pycache__ --exclude .vscode \
    "$SRC/" "${POD_USER}@ssh.runpod.io:/workspace/sampling-for-learnability/"

# wandb credentials (wandb reads ~/.netrc); only the api.wandb.ai stanza
grep -A2 "api.wandb.ai" "$HOME/.netrc" | $SSH "${POD_USER}@ssh.runpod.io" \
    "cat >> ~/.netrc && chmod 600 ~/.netrc"

echo "SYNC DONE. Next: ssh in and run: bash /workspace/sampling-for-learnability/pod_setup.sh"
