#!/bin/bash
# Setup for an xland CNN/hybrid pod. Run ON the pod after the workstation
# has rsynced the repo, the patched jaxmarl, and the scorer checkpoints.
#
# The checkpoint tree is replicated at its ORIGINAL ABSOLUTE PATH because
# both the xland configs (CNN_CHECKPOINT_PATHS) and rlhf_utils
# (CHECKPOINT_INDEX_PATH, used by resolve_input_domain) hardcode
# /home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/...
# Replicating the path keeps the index-driven inversion branching intact
# with zero config edits (CLAUDE.md: never hardcode invert_input).
set -euo pipefail
cd /workspace/sampling-for-learnability

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r pod_env_freeze.txt -q
pip install -e . --no-deps -q

echo "--- device ---"
python -c "import jax;print(jax.__version__, jax.devices())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail the setup if JAX cannot see the GPU — a CPU fallback would silently
# run ~100x slower (some community hosts have broken CUDA: cuInit UNKNOWN)
python -c "import jax,sys; d=jax.devices(); print('jax devices:',d); sys.exit(0 if d[0].platform=='gpu' else 1)" \
  || { echo "JAX HAS NO GPU — aborting setup"; exit 1; }

echo "--- scorer index reachable + domain resolves (variant: ${XLAND_VARIANT:-medium}) ---"
python - <<'PY'
import sys, os, yaml
sys.path.insert(0, "sfl/train")
from rlhf_utils import resolve_input_domain, CHECKPOINT_INDEX_PATH
assert os.path.exists(CHECKPOINT_INDEX_PATH), f"index missing: {CHECKPOINT_INDEX_PATH}"
variant = os.environ.get("XLAND_VARIANT", "medium")
cfg = ("sfl/train/config/xland-sfl-medium.yaml" if variant == "medium"
       else "sfl/train/config/xland-sfl.yaml")
paths = yaml.safe_load(open(cfg))["CNN_CHECKPOINT_PATHS"]
missing = [p for p in paths if not os.path.isdir(p)]
assert not missing, f"{cfg}: missing checkpoint dirs {missing}"
print(f"  {os.path.basename(cfg)}: {len(paths)} members, domain={resolve_input_domain(paths)}")
PY
echo "XLAND POD SETUP COMPLETE"
