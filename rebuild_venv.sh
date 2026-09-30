#!/bin/bash
# =====================================================================
# Deterministic rebuild of the campaign venv (jaxnav + xland SFL runs).
#
# Reconstructed forensically 2026-09-01 from the venv's own metadata
# (pyvenv.cfg, dist-info direct_url.json/RECORD hashes, file mtimes) —
# see VENV_REPRODUCTION.md for the evidence. The two site-packages
# patches at the end are the part that is NOT captured by pyproject.toml,
# requirements.txt, install.sh, or `pip freeze`: without them a fresh
# install of the same pinned commit CRASHES. They are why rebuilding this
# environment was painful.
#
# Usage:  bash rebuild_venv.sh [target_dir]     (default: ./.venv)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
VENV="${1:-.venv}"

# Pinned commit of malikdurmus/JaxMARL that the campaign venv was built
# from. requirements.txt says "@experimental", a MOVING branch — pin here
# so a rebuild cannot silently pick up a different jaxmarl.
JAXMARL_COMMIT="8d57bf90ad2dc63604f7e5c38407fd3d9e3387e8"
JAXUED_COMMIT="0f8f1284677375b889e4f13a32c9617cd009f8c4"
PY="${PYTHON:-python3.12}"     # campaign venv used system Python 3.12.3

echo "=== [1/5] venv ($($PY --version)) ==="
$PY -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip -q

echo "=== [2/5] project + deps (jax cuda12 wheels from Google's index) ==="
# Mirrors install.sh, but with the git deps pinned to the campaign commits.
pip install -q \
    "jaxmarl @ git+https://github.com/malikdurmus/JaxMARL@${JAXMARL_COMMIT}" \
    "jaxued @ git+https://github.com/DramaCow/jaxued@${JAXUED_COMMIT}"
pip install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -q

echo "=== [3/5] xland extras (added 2026-07-31 for the xminigrid arm) ==="
pip install -q xminigrid==0.9.3 imageio-ffmpeg

echo "=== [4/5] site-packages patches to jaxmarl (REQUIRED) ==="
python - <<'PATCH'
import sys, pathlib, jaxmarl
root = pathlib.Path(jaxmarl.__file__).parent
applied, already = [], []

def patch(relpath, old, new, name):
    f = root / relpath
    src = f.read_text()
    if new in src and old not in src:
        already.append(name); return
    if old not in src:
        sys.exit(f"FATAL: patch target not found in {relpath} ({name}). "
                 "jaxmarl version differs from the pinned commit — do not "
                 "proceed; the campaign env cannot be reproduced blindly.")
    f.write_text(src.replace(old, new, 1))
    applied.append(name)

# PATCH 1 — dikstra_path returns a scalar distance, but the false branch of
# lax.select was a 2-vector => shape mismatch at trace time. Breaks every
# run that logs env-metrics shortest_path_length, and the whole dijkstra arm.
patch("environments/jaxnav/maps/grid_map.py",
      "jax.lax.select(valid, d, jnp.array([0.0,0.0]))",
      "jax.lax.select(valid, d, 0.0)",
      "grid_map.dikstra_path scalar-select")

# PATCH 2 — the pinned commit renamed Map's `fill` to `min_fill`/`max_fill`
# but left this singleton call site on the old kwarg => TypeError building
# the singleton eval maps (crashes at startup, before training).
patch("environments/jaxnav/jaxnav_singletons.py",
      '                "fill": 0.5,\n',
      '                "min_fill": 0.0,\n                "max_fill": 0.5,\n',
      "jaxnav_singletons min/max_fill kwargs")

for n in applied: print(f"  applied : {n}")
for n in already: print(f"  already : {n}")
PATCH

echo "=== [5/5] verification ==="
python - <<'VERIFY'
import jax, jax.numpy as jnp
print("  jax", jax.__version__, jax.devices())
# patch 2: singleton eval maps must construct
from jaxmarl.environments.jaxnav.jaxnav_singletons import make_jaxnav_singleton_collection
envs, ids = make_jaxnav_singleton_collection("multi")   # the campaign's test_set
print(f"  singleton collection OK ({len(envs)} envs)")
# patch 1: dikstra_path must trace and return a scalar length
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
e = JaxNav(num_agents=1, map_id="Grid-Rand-Poly",
           map_params={"map_size": (11, 11), "min_fill": 0.0, "max_fill": 0.6})
_, st = e.reset(jax.random.PRNGKey(0))
ok, ln = jax.jit(e.map_obj.dikstra_path)(st.map_data, st.pos[0], st.goal[0])
print(f"  dikstra_path OK (passable={bool(ok)}, len={float(ln):.2f})")
VERIFY

echo
echo "VENV REBUILD COMPLETE -> $VENV"
echo "Next: JAX_PLATFORMS=cpu python verify_progress_arms.py"
