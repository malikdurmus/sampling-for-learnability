"""Dijkstra-arm verification gates (all three BEFORE any RL run):
[1] 5 levels rendered with the BFS shortest path overlaid + length printed
    -> dijkstra_gate1_overlay.png
[2] reachability + length cross-check vs an INDEPENDENT numpy BFS on 2000
    DR levels (also adjudicates any x/y-convention doubt): zero
    disagreements required
[3] score distribution over 5000 DR levels: spread + tie structure
Writes verify_dijkstra_gates_output.txt.
"""
import sys, os, yaml
from collections import deque, Counter
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/home/d/durmusy/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
os.chdir(REPO); sys.path.insert(0, REPO + "/sfl/train")
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

lines = []
def P(*a):
    t = " ".join(str(x) for x in a); print(t); lines.append(t)

env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
env = JaxNav(num_agents=1, **env_cfg["env_params"])
H, W = env_cfg["env_params"]["map_params"]["map_size"]
SENTINEL = float(H * W)

def jit_scores(states):
    def _one(state):
        passable, plen = jax.vmap(env.map_obj.dikstra_path, in_axes=(None, 0, 0))(
            state.map_data, state.pos, state.goal)
        return jnp.where(passable, plen, SENTINEL).mean(), passable.all()
    return jax.vmap(_one)(states)

def np_bfs(map_data, pos, goal):
    """Independent reference: BFS on the occupancy grid, 4-connected.
    Returns (reachable, length_in_steps, path_cells). Cells = (row, col) =
    (floor(y), floor(x)) for world [x, y] positions."""
    grid = np.asarray(map_data) > 0
    sr, sc = int(np.floor(pos[1])), int(np.floor(pos[0]))
    gr, gc = int(np.floor(goal[1])), int(np.floor(goal[0]))
    if grid[sr, sc] or grid[gr, gc]:
        return False, None, []
    prev = {}
    dist = {(sr, sc): 0}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            break
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not grid[nr, nc] and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                prev[(nr, nc)] = (r, c)
                q.append((nr, nc))
    if (gr, gc) not in dist:
        return False, None, []
    path = [(gr, gc)]
    while path[-1] != (sr, sc):
        path.append(prev[path[-1]])
    return True, dist[(gr, gc)], path[::-1]

# ---------- data ----------
rr = jax.random.split(jax.random.PRNGKey(77), 5000)
_, states = jax.vmap(env.reset)(rr)
scores, allpass = map(np.asarray, jit_scores(states))

# ---------- GATE 2: cross-check vs numpy BFS on 2000 levels ----------
P("[GATE 2] jitted dikstra_path vs independent numpy BFS, 2000 levels")
mismatch_reach = mismatch_len = 0
deltas = []
for i in range(2000):
    st = jax.tree.map(lambda x: x[i], states)
    reach_np, len_np, _ = np_bfs(st.map_data, np.asarray(st.pos)[0], np.asarray(st.goal)[0])
    reach_jit = bool(allpass[i])
    if reach_np != reach_jit:
        mismatch_reach += 1
        if mismatch_reach <= 3: P(f"  REACH MISMATCH level {i}: np={reach_np} jit={reach_jit}")
    elif reach_np:
        deltas.append(float(scores[i]) - len_np)
P(f"  reachability disagreements: {mismatch_reach}/2000  (required: 0)")
d = np.array(deltas)
P(f"  length delta (jit - npBFS) on reachable: mean {d.mean():.3f}  max|.| {np.abs(d).max():.3f}"
  f"  exact matches {np.mean(d == 0):.1%}")
gate2 = mismatch_reach == 0 and np.abs(d).max() <= 1e-6

# ---------- GATE 1: overlay figure ----------
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
for k, ax in enumerate(axes):
    st = jax.tree.map(lambda x: x[k], states)
    env.init_render(ax, st, lidar=False, ticks_off=True)
    reach, ln, path = np_bfs(st.map_data, np.asarray(st.pos)[0], np.asarray(st.goal)[0])
    if reach:
        ys = [r + 0.5 for r, c in path]; xs = [c + 0.5 for r, c in path]
        ax.plot(xs, ys, "r.-", linewidth=2, markersize=4)
    ax.set_title(f"jit len: {scores[k]:.0f}  bfs: {ln if reach else 'unreach'}", fontsize=10)
    ax.set_aspect("equal", "box")
plt.tight_layout(); fig.savefig("dijkstra_gate1_overlay.png", dpi=150, bbox_inches="tight"); plt.close(fig)
P("[GATE 1] wrote dijkstra_gate1_overlay.png (eyeball: path hugs free cells, length matches)")

# ---------- GATE 3: distribution + ties ----------
P("\\n[GATE 3] score distribution, 5000 DR levels")
uniq, counts = np.unique(scores, return_counts=True)
P(f"  distinct values: {len(uniq)}   range [{scores.min():.0f}, {scores.max():.0f}]")
P(f"  unreachable (sentinel {SENTINEL:.0f}): {np.mean(scores >= SENTINEL):.1%}")
top = sorted(zip(counts, uniq), reverse=True)[:8]
P("  largest ties: " + ", ".join(f"len {v:.0f} x{c}" for c, v in top))
pct = np.argsort(np.argsort(scores)) / (len(scores) - 1)
cov = [(np.abs(pct - m) <= 0.05).sum() for m in (0.05, 0.3, 0.5, 0.7, 0.95)]
P(f"  percentile coverage at mu=.05/.3/.5/.7/.95 (+-0.05): {cov}  (uniform ref ~500)")
gate3 = len(uniq) >= 15

P(f"\\nGATES: 2({'PASS' if gate2 else 'FAIL'}) 3({'PASS' if gate3 else 'FAIL'}); gate 1 by eyeball")
open("verify_dijkstra_gates_output.txt", "w").write("\\n".join(lines) + "\\n")
