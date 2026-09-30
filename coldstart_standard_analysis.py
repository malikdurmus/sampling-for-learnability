"""Cold start of STANDARD SFL in the three environments (jaxnav default generator, jaxnav
solvable-only generator, xminigrid medium, xminigrid difficult).

What it measures, per cycle, from the scoring rollouts on the 5000 fresh candidates that
standard SFL performs every cycle (so these describe the environment + agent stage, not a
selection rule):
  - candidate success rate         solvability/batch_mean
  - candidate learnability         sfl/batch_mean, sfl/batch_max   (p(1-p), ceiling 0.25)
  - zero-score fraction            from hist/sfl_all: share of the 5000 with p(1-p) == 0
  - nonzero count / filler share   nonzero = 5000*(1-zero); filler = max(0, 1000-nonzero)/1000
  - buffer learnability            learnability_set_mean_score (mean p(1-p) of the 1000 selected)
  - worst-20 buffer learnability   worst_learnability_mean_score
  - eval                           jaxnav: sampled-test overall win rate; xland: eval_fixed success
plus, for jaxnav, the inside of cycle 0 from per-update keys (goal-reaches, collisions,
timeouts; new-wave runs also log the success on the fresh-DR half and the buffer half), and
the cycle-0 per-level picture from probe_coldstart.npz (untrained policy, 1000 levels).

Zero-score fraction from the histogram: wandb.Histogram uses 64 bins over [min, max]. The
smallest nonzero p(1-p) with <= 10 episodes per level is 0.09, and the bin width is <= 0.25/64,
so exact zeros are the only values below 0.05: zero_count = sum of bins whose right edge <= 0.05.
A cycle whose 5000 scores are all zero gives a degenerate histogram centred on 0, which the
same rule handles (its single occupied bin has right edge <= 0.05).

Runs (standard only):
  jaxnav norm : sfl-jaxnav-frontier  norm_standard_seed1-4  (primary; full key set)
                sfl-jaxnav-campaign-fixed standard_seed1-10 (cross-check: no histograms logged)
  jaxnav solv : sfl-jaxnav-frontier  solv_standard_seed1-4  (primary)
                sfl-jaxnav-campaign-fixed solv_standard_seed1-10 (cross-check, histograms present)
  xland medium: xland-sfl-campaign   xlm_standard_seed1-4
  xland diff. : xland-sfl-campaign   xl_standard_seed1-4
Usage: python coldstart_standard_analysis.py [--refresh]
  -> coldstart_standard_raw.json, coldstart_standard_analysis_output.txt,
     coldstart_standard_fig_envs.png, coldstart_standard_fig_jaxnav_cycle0.png
"""
import json, os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "coldstart_standard_raw.json")
OUT = os.path.join(HERE, "coldstart_standard_analysis_output.txt")
ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
NUM_CANDIDATES, NUM_TO_SAVE = 5000, 1000
MIN_NONZERO_SCORE = 0.09           # (1/10)*(9/10): smallest nonzero p(1-p) at <= 10 episodes
K_EVAL_JN = "sampled-test-metrics.eval-sampled/overall_win_rate"
K_EVAL_XL = "eval_fixed/success_rate_mean"
CYC_KEYS = ["solvability/batch_mean", "sfl/batch_mean", "sfl/batch_max", "hist/sfl_all", "learnability_set_mean_score",
            "worst_learnability_mean_score", "recent_success_rate", K_EVAL_JN, "singleton-test-metrics.eval/:overall_win_rate",
            K_EVAL_XL, "eval_bench/success_rate_mean", "train/success_rate_mean"]
UPD_KEYS = ["train-term.GoalR", "train-term.MapC", "train-term.TimeO", "train-term.AgentC", "train-term.NumC",
            "train-generated/.success_env_weighted", "train-buffer/.success_env_weighted",
            "train-generated/.episodes_per_env", "train-buffer/.episodes_per_env"]
SPECS = [  # env label, project, run-name template, seeds, role
    ("jaxnav_norm", "sfl-jaxnav-frontier", "norm_standard_seed{}", range(1, 5), "primary"),
    ("jaxnav_norm", "sfl-jaxnav-campaign-fixed", "standard_seed{}", range(1, 11), "crosscheck"),
    ("jaxnav_solv", "sfl-jaxnav-frontier", "solv_standard_seed{}", range(1, 5), "primary"),
    ("jaxnav_solv", "sfl-jaxnav-campaign-fixed", "solv_standard_seed{}", range(1, 11), "crosscheck"),
    ("xland_medium", "xland-sfl-campaign", "xlm_standard_seed{}", range(1, 5), "primary"),
    ("xland_difficult", "xland-sfl-campaign", "xl_standard_seed{}", range(1, 5), "primary"),
]
lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

# ------------------------------------------------------------------ pull
def fetch_all():
    import wandb
    api = wandb.Api(timeout=180)
    raw = {}
    for env, proj, tmpl, seeds, role in SPECS:
        runs = list(api.runs(f"{ENT}/{proj}", per_page=500))
        for s in seeds:
            name = tmpl.format(s)
            cand = [r for r in runs if r.name == name and r.state == "finished"]
            if not cand:
                print(f"MISSING {proj}/{name}", file=sys.stderr); continue
            r = max(cand, key=lambda r: ((r.summary.get("update_count") or 0), r.created_at))
            t0 = time.time()
            rows = list(r.history(samples=500000, pandas=False))
            cyc = {k: {} for k in CYC_KEYS}; upd = {k: {} for k in UPD_KEYS}
            for x in rows:
                u = x.get("update_count")
                if u is None: continue
                u = int(u)
                for k in CYC_KEYS:
                    v = x.get(k)
                    if v is None: continue
                    if k == "hist/sfl_all":
                        # the full-history API packs uniform bins as {min, size, count}; the keyed API expands them
                        if isinstance(v, dict) and "bins" in v:
                            cyc[k][u] = {"bins": list(v["bins"]), "values": list(v["values"])}
                        elif isinstance(v, dict) and "packedBins" in v:
                            pb = v["packedBins"]; n = int(pb["count"])
                            cyc[k][u] = {"bins": [float(pb["min"]) + float(pb["size"]) * i for i in range(n + 1)], "values": list(v["values"])}
                    elif not isinstance(v, dict):
                        cyc[k][u] = float(v)
                for k in UPD_KEYS:
                    v = x.get(k)
                    if v is not None and not isinstance(v, dict) and u <= 300:
                        upd[k][u] = float(v)
            raw[f"{env}|{role}|{s}"] = {"id": r.id, "name": name, "project": proj, "state": r.state,
                                        "update_count": r.summary.get("update_count"), "cycle": cyc, "upd": upd}
            print(f"{env:<16}{role:<11}{name:<24} rows={len(rows):>5} cycles={len(cyc['solvability/batch_mean']):>4} hist={len(cyc['hist/sfl_all']):>4} ({time.time()-t0:.1f}s)", file=sys.stderr)
    json.dump(raw, open(RAW, "w"))
    return raw

if "--refresh" in sys.argv or not os.path.exists(RAW):
    raw = fetch_all()
else:
    raw = json.load(open(RAW))

# ------------------------------------------------------------------ helpers
def zero_fraction(h):
    """Share of the 5000 candidate scores that are exactly 0.
    wandb.Histogram = numpy histogram, 64 equal bins over [min, max]. Whenever any score is 0
    the range starts at 0 and bin 0 = [0, max/64) with max <= 0.25, i.e. width <= 0.0039; the
    smallest nonzero p(1-p) with n episodes is ~1/n, so bin 0 holds nonzero scores only if a
    level completed > 256 episodes, impossible in a <= 2500-step rollout. If ALL scores are 0
    numpy centres the range on 0 and the bin with left edge 0 holds everything. Hence: the
    zero count is the bin whose LEFT edge is exactly 0 (NOT "every bin below 0.05": late in
    training an xminigrid agent completes 50+ episodes per level and a single failure gives
    a genuine nonzero score near 0.02)."""
    bins = np.asarray(h["bins"], float); vals = np.asarray(h["values"], float)
    tot = vals.sum()
    if tot <= 0: return np.nan, 0.0
    j = np.where(bins[:-1] == 0.0)[0]
    if len(j) == 0:                       # no zero-valued scores at all (range starts above 0)
        return 0.0, tot
    return float(vals[j[0]]) / tot, tot

def smallest_nonzero_score(h):
    bins = np.asarray(h["bins"], float); vals = np.asarray(h["values"], float)
    nz = np.where((vals > 0) & (bins[:-1] > 0))[0]
    return float(bins[nz[0]]) if len(nz) else np.nan

def implied_episode_cap(score):
    """n such that (1/n)(1-1/n) = score, i.e. the most episodes a level can have completed."""
    if not (score > 0): return np.nan
    return (1 + np.sqrt(max(0.0, 1 - 4 * score))) / (2 * score)

def cycle_series(run, key):
    """Values by cycle index (0-based, order of the logged rows carrying the key)."""
    d = run["cycle"].get(key, {})
    us = sorted(int(u) for u in d)
    return np.array([d[str(u)] if str(u) in d else d[u] for u in us], dtype=object if key == "hist/sfl_all" else float), us

def first_cycle_at_or_above(arr, thr):
    idx = np.where(np.asarray(arr, float) >= thr)[0]
    return int(idx[0]) if len(idx) else None

def fmt_cycles(lst):
    return "[" + " ".join("never" if v is None else str(v) for v in lst) + "]"

def stat(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    return (v.mean(), v.std(), len(v)) if len(v) else (np.nan, np.nan, 0)

# ------------------------------------------------------------------ per-run derived series
derived = {}
for key, run in raw.items():
    env, role, seed = key.split("|")
    succ, us = cycle_series(run, "solvability/batch_mean")
    sfl_m, _ = cycle_series(run, "sfl/batch_mean"); sfl_x, _ = cycle_series(run, "sfl/batch_max")
    lset, _ = cycle_series(run, "learnability_set_mean_score"); worst, _ = cycle_series(run, "worst_learnability_mean_score")
    hists, hus = cycle_series(run, "hist/sfl_all")
    zf = np.full(len(us), np.nan); tot = np.full(len(us), np.nan); mnz = np.full(len(us), np.nan)
    hmap = {u: h for u, h in zip(hus, hists)}
    for i, u in enumerate(us):
        if u in hmap:
            zf[i], tot[i] = zero_fraction(hmap[u]); mnz[i] = smallest_nonzero_score(hmap[u])
    nonzero = NUM_CANDIDATES * (1 - zf)
    filler = np.clip(NUM_TO_SAVE - nonzero, 0, NUM_TO_SAVE) / NUM_TO_SAVE
    evk = K_EVAL_JN if env.startswith("jaxnav") else K_EVAL_XL
    ev, _ = cycle_series(run, evk)
    derived[key] = dict(env=env, role=role, seed=int(seed), n_cycles=len(us), succ=succ, sfl_mean=sfl_m, sfl_max=sfl_x,
                        lset=lset, worst=worst, zero=zf, hist_total=tot, nonzero=nonzero, filler=filler, eval=ev, min_nonzero=mnz,
                        train_succ=cycle_series(run, "train/success_rate_mean")[0] if not env.startswith("jaxnav") else None)

# ------------------------------------------------------------------ 0. inventory + self-checks
p("COLD START OF STANDARD SFL — environment characterisation (generated %s)" % time.strftime("%Y-%m-%d %H:%MZ", time.gmtime()))
p("\n== 0. Runs and self-checks ==")
for key in sorted(derived):
    d = derived[key]; run = raw[key]
    hist_ok = np.all(np.isnan(d["hist_total"]) | (d["hist_total"] == NUM_CANDIDATES))
    # lower bound on the batch mean implied by the nonzero share (each nonzero score >= 0.09)
    lb = d["min_nonzero"] * (1 - d["zero"]); viol = np.nanmax(lb - d["sfl_mean"]) if np.any(~np.isnan(lb)) else np.nan
    p(f"  {key:<30} {run['name']:<24} id={run['id']} cycles={d['n_cycles']:>3} hist_cycles={int(np.sum(~np.isnan(d['zero']))):>3} "
      f"hist_total==5000:{'ok' if hist_ok else 'FAIL'} batchmean>=0.09*nonzero: {'ok' if (np.isnan(viol) or viol <= 1e-6) else f'FAIL({viol:.4f})'}")

# ------------------------------------------------------------------ 1. per-cycle tables
SHOW = {"jaxnav_norm": [0, 1, 2, 3, 5, 10, 25, 44], "jaxnav_solv": [0, 1, 2, 3, 5, 10, 25, 44],
        "xland_medium": [0, 1, 2, 5, 10, 25, 50, 75, 100, 150, 365], "xland_difficult": [0, 1, 2, 5, 10, 25, 50, 100, 200, 365]}
ENVS = ["jaxnav_norm", "jaxnav_solv", "xland_medium", "xland_difficult"]
def env_runs(env, role="primary"):
    return [d for d in derived.values() if d["env"] == env and d["role"] == role]
def agg(runs, field, c):
    return stat([d[field][c] if c < len(d[field]) else np.nan for d in runs])
p("\n== 1. Candidate batch (5000 fresh levels) and buffer per cycle — mean±std over seeds (population std) ==")
for env in ENVS:
    runs = env_runs(env)
    if not runs: continue
    p(f"\n-- {env} (n={len(runs)} standard runs, {runs[0]['n_cycles']} cycles) --")
    p(f"  {'cyc':>4}{'cand succ':>12}{'cand p(1-p)':>13}{'cand max':>10}{'zero%':>9}{'nonzero':>9}{'filler%':>9}{'buffer p(1-p)':>15}{'enrich':>8}{'worst20':>9}{'eval':>8}")
    for c in SHOW[env]:
        if c >= runs[0]["n_cycles"]: continue
        s = agg(runs, "succ", c); m = agg(runs, "sfl_mean", c); x = agg(runs, "sfl_max", c); z = agg(runs, "zero", c)
        nz = agg(runs, "nonzero", c); f = agg(runs, "filler", c); l = agg(runs, "lset", c); w = agg(runs, "worst", c); e = agg(runs, "eval", c)
        enr = l[0] / m[0] if m[0] and m[0] > 0 else np.nan
        p(f"  {c:>4}{s[0]:>8.4f}±{s[1]:.3f}{m[0]:>8.4f}±{m[1]:.3f}{x[0]:>10.3f}{100*z[0]:>8.1f}%{nz[0]:>9.0f}{100*f[0]:>8.1f}%{l[0]:>10.4f}±{l[1]:.3f}{enr:>8.2f}{w[0]:>9.4f}{e[0]:>8.3f}")

# ------------------------------------------------------------------ 2. cold-start durations
p("\n== 1b. Episode budget behind the per-level estimate (from the smallest nonzero score in the cycle-0 and cycle-25 histograms) ==")
for env in ENVS:
    runs = env_runs(env)
    if not runs: continue
    for c in (0, 25):
        s0 = stat([d["min_nonzero"][c] for d in runs])[0]
        p(f"  {env:<16} cycle {c:>2}: smallest nonzero p(1-p) = {s0:.4f} -> most episodes a level completed ~ {implied_episode_cap(s0):.0f}; "
          f"a level with true p = 0.1 then scores 0 with probability {0.9 ** implied_episode_cap(s0):.2f}")
p("\n== 2. Cold-start durations: first cycle (0-based) at which the quantity reaches the threshold, per seed ==")
for env in ENVS:
    runs = env_runs(env)
    if not runs: continue
    p(f"-- {env} --")
    for thr in (0.05, 0.2, 0.5):
        p(f"  candidate success >= {thr:<4}: {fmt_cycles([first_cycle_at_or_above(d['succ'], thr) for d in runs])}")
    for thr in (0.25, 0.5):
        p(f"  nonzero-score share >= {thr:<4}: {fmt_cycles([first_cycle_at_or_above(1 - d['zero'], thr) for d in runs])}   (buffer no longer padded when nonzero >= 1000, i.e. share >= 0.2)")
    p(f"  buffer padded (nonzero < 1000) in cycles: {[int(np.nansum(d['filler'] > 0)) for d in runs]} of {runs[0]['n_cycles']}; padded in ALL of the first 10: {[bool(np.all(d['filler'][:10] > 0)) for d in runs]}")
    for thr in (0.1, 0.5, 0.9):
        p(f"  eval >= {thr:<4}: {fmt_cycles([first_cycle_at_or_above(d['eval'], thr) for d in runs])}")
    cross = env_runs(env, "crosscheck")
    if cross:
        p(f"  cross-check, old campaign (n={len(cross)}): candidate success >= 0.5 at {fmt_cycles([first_cycle_at_or_above(d['succ'], 0.5) for d in cross])}; eval >= 0.5 at {fmt_cycles([first_cycle_at_or_above(d['eval'], 0.5) for d in cross])}"
          + (f"; zero% at cycles 0/1/2: {[round(100*float(np.nanmean([d['zero'][c] for d in cross])),1) for c in (0,1,2)]}" if not np.all(np.isnan([d['zero'][0] for d in cross])) else "; (no histograms logged in these runs)"))

# ------------------------------------------------------------------ 3. inside cycle 0, jaxnav
p("\n== 3. Inside cycle 0 (jaxnav, per PPO update; 256 training envs = 128 fresh DR + 128 from the buffer) ==")
def upd_arr(run, key, n=100):
    d = run["upd"].get(key, {}); return np.array([d.get(str(u), d.get(u, np.nan)) for u in range(1, n + 1)], float)
for env in ["jaxnav_norm", "jaxnav_solv"]:
    runs = [(k, raw[k]) for k in raw if derived[k]["env"] == env and derived[k]["role"] == "primary"]
    if not runs: continue
    p(f"-- {env} (n={len(runs)}) --")
    G = np.array([upd_arr(r, "train-term.GoalR") for _, r in runs]); M = np.array([upd_arr(r, "train-term.MapC") for _, r in runs])
    T = np.array([upd_arr(r, "train-term.TimeO") for _, r in runs]); A = np.array([upd_arr(r, "train-term.AgentC") for _, r in runs])
    N = np.array([upd_arr(r, "train-term.NumC") for _, r in runs])
    SG = np.array([upd_arr(r, "train-generated/.success_env_weighted") for _, r in runs]); SB = np.array([upd_arr(r, "train-buffer/.success_env_weighted") for _, r in runs])
    p(f"  {'updates':>9}{'goal-reaches':>13}{'collisions':>12}{'timeouts':>10}{'episodes':>10}{'goal share':>12}{'succ fresh-DR half':>20}{'succ buffer half':>18}")
    for lo, hi in [(1, 1), (2, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100)]:
        sl = slice(lo - 1, hi)
        g, m, t, a, n = (np.nanmean(X[:, sl]) for X in (G, M, T, A, N))
        p(f"  {f'{lo}-{hi}':>9}{g:>13.1f}{m:>12.1f}{t:>10.1f}{n:>10.1f}{g / max(n, 1e-9):>12.3f}{np.nanmean(SG[:, sl]):>20.3f}{np.nanmean(SB[:, sl]):>18.3f}")
    p(f"  first update with fresh-DR success >= 0.2 / 0.5, per seed: {[first_cycle_at_or_above(SG[i], 0.2) for i in range(len(runs))]} / {[first_cycle_at_or_above(SG[i], 0.5) for i in range(len(runs))]}  (update index = position+1)")

# ------------------------------------------------------------------ 4. per-level cycle-0 picture (probe)
p("\n== 4. Per-level picture at cycle 0: untrained policy on 1000 random jaxnav levels (probe_coldstart.npz, default generator) ==")
probe = os.path.join(HERE, "probe_coldstart.npz")
if os.path.exists(probe):
    d = np.load(probe); pp, n, s = d["p"], d["n_eps"], d["s"]
    p(f"  levels {len(pp)}; share with p == 0: {np.mean(pp == 0):.3f}; share with p == 1: {np.mean(pp == 1):.3f}; pooled success (episode-weighted): {np.sum(pp * n) / np.sum(n):.4f}; mean p (level-weighted): {pp.mean():.4f}")
    p(f"  episodes per level: median {np.median(n):.0f}, min {n.min()}, max {n.max()}, share with <= 2 episodes {np.mean(n <= 2):.3f}, share with >= 10 episodes {np.mean(n >= 10):.3f}")
    p(f"  score p(1-p): mean {s.mean():.4f} (ceiling 0.25), share nonzero {np.mean(s > 0):.3f}, top-1000-equivalent: with 1000 of 5000 slots the buffer would hold {min(1000, int(round(5000 * np.mean(s > 0))))} nonzero levels -> filler {max(0, 1000 - int(round(5000 * np.mean(s > 0)))) / 10:.0f}%")
    nz = pp[pp > 0]
    p(f"  among levels with p > 0: median p {np.median(nz):.3f}, share with p <= 0.2: {np.mean(nz <= 0.2):.3f}; corr(n_episodes, p) = {np.corrcoef(n, pp)[0, 1]:+.3f}")
else:
    p("  probe_coldstart.npz not found")

# ------------------------------------------------------------------ 5. figures
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C_SUCC, C_SFL, C_ZERO, C_FILL, C_EVAL, C_MAX = "#2a78d6", "#eb6834", "#4a3aa7", "#e34948", "#1baf7a", "#9a9a95"
XLIM = {"jaxnav_norm": (0, 44), "jaxnav_solv": (0, 44), "xland_medium": (0, 120), "xland_difficult": (0, 365)}
TITLE = {"jaxnav_norm": "JaxNav, default generator", "jaxnav_solv": "JaxNav, solvable-only generator", "xland_medium": "XLand-MiniGrid, medium ruleset", "xland_difficult": "XLand-MiniGrid, difficult ruleset"}
def mean_over(runs, field):
    L = max(len(d[field]) for d in runs); M = np.full((len(runs), L), np.nan)
    for i, d in enumerate(runs): M[i, :len(d[field])] = d[field]
    return np.nanmean(M, axis=0), np.nanmin(M, axis=0), np.nanmax(M, axis=0)
fig, axes = plt.subplots(4, 2, figsize=(12, 15))
for ri, env in enumerate(ENVS):
    runs = env_runs(env); x0, x1 = XLIM[env]
    if not runs: continue
    ax = axes[ri, 0]
    for field, col, lab in [("succ", C_SUCC, "candidate success rate p"), ("sfl_mean", C_SFL, "candidate learnability p(1-p)"), ("sfl_max", C_MAX, "max p(1-p) in batch")]:
        m, lo, hi = mean_over(runs, field); x = np.arange(len(m))
        ax.plot(x, m, color=col, lw=2 if field != "sfl_max" else 1.2, ls="-" if field != "sfl_max" else ":", label=lab)
        if field != "sfl_max": ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axhline(0.25, color=C_MAX, lw=0.8); ax.text(x1 * 0.99, 0.26, "p(1-p) ceiling 0.25", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.set_ylim(0, 1.02); ax.set_xlim(x0, x1); ax.set_title(f"{TITLE[env]}: what the agent can do on fresh random levels", fontsize=9.5, loc="left")
    ax = axes[ri, 1]
    for field, col, lab in [("zero", C_ZERO, "share of the 5000 with score exactly 0"), ("filler", C_FILL, "share of the 1000-slot buffer that is zero-score padding")]:
        m, lo, hi = mean_over(runs, field); x = np.arange(len(m))
        ax.plot(x, m, color=col, lw=2, label=lab); ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.set_ylim(0, 1.02); ax.set_xlim(x0, x1); ax.set_title("how sparse the learnability signal is", fontsize=9.5, loc="left")
    for ax in axes[ri]:
        ax.set_xlabel("cycle"); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="center right" if ri == 3 else "best")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "coldstart_standard_fig_envs.png"), dpi=130); plt.close(fig)

# figure 2: inside cycle 0 (jaxnav) + probe histograms
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
ax = axes[0]
for env, col in [("jaxnav_norm", C_SUCC), ("jaxnav_solv", C_SFL)]:
    runs = [raw[k] for k in raw if derived[k]["env"] == env and derived[k]["role"] == "primary"]
    if not runs: continue
    SG = np.array([upd_arr(r, "train-generated/.success_env_weighted", 150) for r in runs]); SB = np.array([upd_arr(r, "train-buffer/.success_env_weighted", 150) for r in runs])
    x = np.arange(1, 151)
    ax.plot(x, np.nanmean(SG, 0), color=col, lw=2, label=f"{TITLE[env].split(', ')[1]}: fresh-DR half")
    ax.plot(x, np.nanmean(SB, 0), color=col, lw=1.4, ls="--", label=f"{TITLE[env].split(', ')[1]}: buffer half")
ax.axvline(50, color=INK2, lw=0.8); ax.axvline(100, color=INK2, lw=0.8); ax.text(51, 0.02, "cycle 1", fontsize=7.5, color=INK2); ax.text(101, 0.02, "cycle 2", fontsize=7.5, color=INK2)
ax.set_xlabel("PPO update"); ax.set_ylabel("success rate of the training envs (per env)"); ax.set_ylim(0, 1.02); ax.set_title("JaxNav standard SFL: the first 150 updates", fontsize=9.5, loc="left")
ax = axes[1]
runs = [raw[k] for k in raw if derived[k]["env"] == "jaxnav_norm" and derived[k]["role"] == "primary"]
if runs:
    G = np.nanmean([upd_arr(r, "train-term.GoalR", 150) for r in runs], 0); M = np.nanmean([upd_arr(r, "train-term.MapC", 150) for r in runs], 0); T = np.nanmean([upd_arr(r, "train-term.TimeO", 150) for r in runs], 0)
    x = np.arange(1, 151)
    ax.plot(x, G, color=C_EVAL, lw=2, label="goal reached"); ax.plot(x, M, color=C_FILL, lw=2, label="collision"); ax.plot(x, T, color=C_ZERO, lw=2, label="timeout")
    ax.set_xlabel("PPO update"); ax.set_ylabel("episode ends per update, 256 envs"); ax.set_title("JaxNav default generator: how episodes end", fontsize=9.5, loc="left")
ax = axes[2]
if os.path.exists(probe):
    d = np.load(probe); pp = d["p"]
    ax.hist(pp, bins=np.linspace(0, 1, 21), color=C_SUCC, edgecolor="white", lw=0.5)
    ax.set_yscale("log"); ax.set_xlabel("success rate p of a level, untrained policy, <= 10 episodes"); ax.set_ylabel("levels (of 1000, log scale)")
    ax.set_title(f"JaxNav cycle 0 per level: {100*np.mean(pp==0):.0f}% never solved", fontsize=9.5, loc="left")
for ax in axes:
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "coldstart_standard_fig_jaxnav_cycle0.png"), dpi=130); plt.close(fig)
p("\nfigures: coldstart_standard_fig_envs.png (4 environments x [candidate success & learnability | zero-score & padding share]),")
p("         coldstart_standard_fig_jaxnav_cycle0.png (per-update success of both training halves, episode-end composition, cycle-0 per-level histogram)")
open(OUT, "w").write("\n".join(lines) + "\n"); print("saved", OUT)
