#!/bin/bash
# One-time setup ON the RunPod pod (run from /workspace after rsync).
# Expects the repo at /workspace/sampling-for-learnability (pushed by
# pod_sync.sh from the workstation) and WANDB_API_KEY in the environment
# (pod_sync.sh writes ~/.netrc instead, which wandb also honors).
set -euo pipefail
cd /workspace/sampling-for-learnability

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
# Exact clone of the workstation venv (jax 0.11.0 cuda12, jaxmarl pinned
# commit). The sfl package itself is installed editable from the synced repo
# (the freeze's git+ssh self-reference is stripped — no GitHub key on pod).
pip install -r pod_env_freeze.txt -q
pip install -e . --no-deps -q

echo "--- device check ---"
python -c "import jax; print(jax.__version__, jax.devices())"
echo "--- verification gates (production episode math) ---"
JAX_PLATFORMS=cpu python verify_progress_arms.py
echo "POD SETUP COMPLETE"
