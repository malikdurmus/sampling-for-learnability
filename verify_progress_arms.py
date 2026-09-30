"""Verification gates for the progress-based learnability arms
(LEARN_METHOD = progress / progress_mindist / progress_mean).

Tests the EXACT production episode-outcome code
(sfl.train.jaxnav_sfl.calc_progress_outcomes_by_agent) on synthetic
trajectories with hand-computed expected values.

Gate P1: episode segmentation + d_start/d_end/d_min extraction + progress
         formulas on a hand-built two-episode trajectory.
Gate P2: degenerate cases — zero episodes (no NaN, scores 0) and a single
         episode (variance exactly 0).
Gate P3: score semantics — mp*(1-mp) ranking is identical to |mp-0.5|
         ranking; closest-approach dominates end-based progress per episode
         (d_min <= d_end always => progress_min >= progress_end).

Run: JAX_PLATFORMS=cpu python verify_progress_arms.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfl", "train"))

import numpy as np
import jax.numpy as jnp

from jaxnav_sfl import calc_progress_outcomes_by_agent

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def run_calc(dones, dpre, goalr=None, T=None):
    """Wrap 1-actor arrays into the (T, num_actors) layout the vmap expects."""
    T = T or len(dones)
    dones = jnp.array(dones, dtype=bool).reshape(T, 1)
    info = {
        "DPre": jnp.array(dpre, dtype=jnp.float32).reshape(T, 1),
        "GoalR": jnp.array(goalr if goalr is not None else [0.0] * T,
                           dtype=jnp.float32).reshape(T, 1),
        "MapC": jnp.zeros((T, 1), dtype=jnp.float32),
        "AgentC": jnp.zeros((T, 1), dtype=jnp.float32),
        "TimeO": jnp.zeros((T, 1), dtype=jnp.float32),
    }
    returns = jnp.zeros((T, 1), dtype=jnp.float32)
    o = calc_progress_outcomes_by_agent(T, dones, returns, info)
    return {k: np.asarray(v)[0] for k, v in o.items()}  # actor 0


# ---------------- Gate P1: hand-built two-episode trajectory ----------------
# T=12. Episode 1 = steps 0..4 (done at 4), episode 2 = steps 5..9 (done at 9).
# Steps 10,11: unfinished tail (must be ignored).
# DPre = distance of the state each step was taken FROM.
#   ep1: 10 (start), 8, 1 (closest dip), 6, 2 (pre-terminal)
#        -> d_start=DPre[0+? ] ... start_idx=-1 -> d_start=DPre[0]=10
#           d_end=DPre[4]=2 -> progress_end = 1-2/10  = 0.8
#           d_min=min(10,8,1,6,2)=1 -> progress_min = 1-1/10 = 0.9
#   ep2 (after auto-reset, DPre[5] = reset distance): 4, 3, 5, 2, 3
#           d_start=DPre[5]=4, d_end=DPre[9]=3 -> progress_end = 1-3/4 = 0.25
#           d_min=min(4,3,5,2,3)=2 -> progress_min = 1-2/4 = 0.5
#   tail: DPre[10..11] = 9, 9 (must not affect anything)
dones = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
dpre = [10, 8, 1, 6, 2, 4, 3, 5, 2, 3, 9, 9]
goalr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # ep1 success, ep2 not
o = run_calc(dones, dpre, goalr)

exp_end_mean = (0.8 + 0.25) / 2                  # 0.525
exp_end_var = ((0.8 - exp_end_mean) ** 2 + (0.25 - exp_end_mean) ** 2) / 2  # 0.075625
exp_min_mean = (0.9 + 0.5) / 2                   # 0.7
exp_min_var = ((0.9 - 0.7) ** 2 + (0.5 - 0.7) ** 2) / 2                    # 0.04
exp_d_start = (10 + 4) / 2                       # 7.0

check("P1 num_episodes == 2", o["num_episodes"] == 2, f"got {o['num_episodes']}")
check("P1 progress_end_mean", np.isclose(o["progress_end_mean"], exp_end_mean, atol=1e-5),
      f"got {o['progress_end_mean']:.6f} want {exp_end_mean:.6f}")
check("P1 progress_end_var", np.isclose(o["progress_end_var"], exp_end_var, atol=1e-5),
      f"got {o['progress_end_var']:.6f} want {exp_end_var:.6f}")
check("P1 progress_min_mean", np.isclose(o["progress_min_mean"], exp_min_mean, atol=1e-5),
      f"got {o['progress_min_mean']:.6f} want {exp_min_mean:.6f}")
check("P1 progress_min_var", np.isclose(o["progress_min_var"], exp_min_var, atol=1e-5),
      f"got {o['progress_min_var']:.6f} want {exp_min_var:.6f}")
check("P1 d_start_mean", np.isclose(o["d_start_mean"], exp_d_start, atol=1e-5),
      f"got {o['d_start_mean']:.6f} want {exp_d_start}")
check("P1 success_rate == 0.5", np.isclose(o["success_rate"], 0.5),
      f"got {o['success_rate']}")

# Tail must be ignored: change tail DPre values, outcomes must not move
o2 = run_calc(dones, [10, 8, 1, 6, 2, 4, 3, 5, 2, 3, 0.001, 0.001], goalr)
check("P1 unfinished tail ignored",
      np.isclose(o2["progress_min_var"], o["progress_min_var"], atol=1e-7) and
      np.isclose(o2["progress_end_mean"], o["progress_end_mean"], atol=1e-7))

# Progress clipped at 0 when the agent ends FARTHER than it started
o3 = run_calc([0, 0, 1, 0], [5, 6, 9, 5], T=4)
check("P1 moving away clips to 0", np.isclose(o3["progress_end_mean"], 0.0),
      f"got {o3['progress_end_mean']}")
# ... but closest-approach still credits the best point (d_min=5 -> progress 0)
check("P1 min-progress uses closest point", np.isclose(o3["progress_min_mean"], 0.0),
      "d_min includes d_start so never negative")

# ---------------- Gate P2: degenerate cases ----------------
o4 = run_calc([0] * 8, [5] * 8)  # no episode terminates
check("P2 zero episodes: no NaN, scores 0",
      o4["num_episodes"] == 0 and o4["progress_end_var"] == 0.0
      and o4["progress_min_var"] == 0.0 and o4["progress_end_mean"] == 0.0
      and not np.isnan(o4["progress_end_var"]))

o5 = run_calc([0, 0, 0, 1, 0, 0], [8, 6, 4, 2, 7, 7])  # exactly one episode
check("P2 one episode: var == 0, mean == its progress",
      np.isclose(o5["progress_end_var"], 0.0) and
      np.isclose(o5["progress_end_mean"], 1 - 2 / 8) and
      np.isclose(o5["progress_min_mean"], 1 - 2 / 8),
      f"end_mean {o5['progress_end_mean']:.4f}")

# ---------------- Gate P3: score semantics ----------------
rng = np.random.default_rng(0)
mp = rng.uniform(0, 1, 5000)
# Mathematical property (float64): mp*(1-mp) ranking == |mp-0.5| ranking
rank_a = np.argsort(np.argsort(-(mp * (1 - mp))))
rank_b = np.argsort(np.argsort(np.abs(mp - 0.5)))
check("P3 mp*(1-mp) ranking == |mp-0.5| ranking (float64)",
      np.array_equal(rank_a, rank_b))
# float32 (production dtype): only near-tie swaps allowed (<1% of levels)
mp32 = mp.astype(np.float32)
rank_a32 = np.argsort(np.argsort(-(mp32 * (1 - mp32))))
rank_b32 = np.argsort(np.argsort(np.abs(mp32 - 0.5)))
frac_swapped = (rank_a32 != rank_b32).mean()
check("P3 float32 tie-swaps < 1%", frac_swapped < 0.01,
      f"swapped frac {frac_swapped:.4f}")

# progress_min >= progress_end for every episode (d_min <= d_end by definition)
for _ in range(200):
    T = 30
    d = rng.uniform(0.5, 10, T)
    dn = np.zeros(T); done_at = rng.choice(np.arange(3, T - 1), 2, replace=False)
    dn[sorted(done_at)] = 1
    oo = run_calc(dn.tolist(), d.tolist(), T=T)
    if oo["progress_min_mean"] < oo["progress_end_mean"] - 1e-6:
        check("P3 progress_min >= progress_end", False,
              f"min {oo['progress_min_mean']} < end {oo['progress_end_mean']}")
        break
else:
    check("P3 progress_min >= progress_end (200 random trajectories)", True)

print()
if FAILURES:
    print(f"GATES FAILED: {FAILURES}")
    sys.exit(1)
print("ALL PROGRESS-ARM GATES PASSED")
