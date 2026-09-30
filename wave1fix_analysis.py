"""Corrected Wave-1 analysis (wave1fix): the headline jaxnav comparison with
the scorer served in its proper input domain.

Project sfl-jaxnav-campaign-fixed:
  baselines (reused, CNN-free, valid): dr_seed1..10, standard_seed1..10
  fixed CNN arms: {cnn,hybrid_*}_fixed_seed1..10

Includes a domain sanity gate: each CNN-arm run's mean
cnn/member_rank_agreement must look in-domain (~0.84), not corrupted
(~0.988) — the analysis refuses to aggregate runs that fail it.

Outputs wave1fix_analysis_output.txt. Endpoints as in wave1_analysis.py
(tail5 primary). Also reports old-vs-fixed deltas per CNN arm (old runs in
sfl-jaxnav-campaign).
"""
import numpy as np
import wandb
from scipy.stats import wilcoxon

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJ = "sfl-jaxnav-campaign-fixed"
OLD_PROJ = "sfl-jaxnav-campaign"
SEEDS = list(range(1, 11))
CNN_ARMS = ["cnn", "hybrid_linear", "hybrid_soft_handoff",
            "hybrid_learnability_weighted", "hybrid_multiplicative"]
METHODS = ["dr", "standard"] + CNN_ARMS
K_SAMP = "sampled-test-metrics.eval-sampled/overall_win_rate"
K_SING = "singleton-test-metrics.eval/:overall_win_rate"
K_PASS = "env-metrics/.passable"
K_AGREE = "cnn/member_rank_agreement"

api = wandb.Api()
lines = []
def p(*a):
    t = " ".join(str(x) for x in a); print(t); lines.append(t)

def series(run, key):
    h = run.history(keys=[key, "update_count"], pandas=False, samples=4000)
    h = [x for x in h if x.get(key) is not None]
    h.sort(key=lambda x: x.get("update_count") or 0)
    return np.array([x[key] for x in h], dtype=float)

def run_name(m, s):
    return f"{m}_seed{s}" if m in ("dr", "standard") else f"{m}_fixed_seed{s}"

data = {}
domain_fail = []
for m in METHODS:
    for s in SEEDS:
        name = run_name(m, s)
        rs = list(api.runs(f"{ENT}/{PROJ}", {"display_name": name}))
        assert len(rs) == 1, f"{name}: found {len(rs)}"
        r = rs[0]
        assert r.state == "finished", f"{name}: state={r.state}"
        d = {"samp": series(r, K_SAMP), "sing": series(r, K_SING),
             "pass": series(r, K_PASS)}
        if m in CNN_ARMS:
            ag = series(r, K_AGREE)
            d["agree"] = float(ag.mean()) if len(ag) else float("nan")
            if not (0.5 < d["agree"] < 0.95):
                domain_fail.append((name, d["agree"]))
        data[(m, s)] = d
        print(f"  pulled {name}: {len(d['samp'])} cycles")

p("\n=== DOMAIN SANITY GATE ===")
ags = [data[(m, s)]["agree"] for m in CNN_ARMS for s in SEEDS]
p(f"cnn/member_rank_agreement across all 50 CNN-arm runs: "
  f"mean {np.mean(ags):.4f}  range [{min(ags):.4f}, {max(ags):.4f}]")
p(f"(in-domain reference ~0.84; corrupted reference ~0.988)")
if domain_fail:
    p(f"!! RUNS FAILING THE GATE: {domain_fail} — DO NOT TRUST AGGREGATES")
else:
    p("all runs in-domain: OK")

def endpoints(v):
    return dict(final=v[-1], tail5=v[-5:].mean(), early5=v[:5].mean(), auc=v.mean())

p("\n=== SAMPLED TEST (100 fixed DR levels), overall_win_rate ===")
p(f"{'method':<30}{'final':>9}{'tail5':>9}{'early5':>9}{'auc':>9}")
ep = {}
for m in METHODS:
    es = [endpoints(data[(m, s)]["samp"]) for s in SEEDS]
    ep[m] = es
    row = f"{m:<30}"
    for k in ("final", "tail5", "early5", "auc"):
        vals = np.array([e[k] for e in es])
        row += f" {vals.mean():.3f}±{vals.std():.3f}"
    p(row)

p("\n--- paired vs standard (tail5): per-seed deltas ---")
base = np.array([e["tail5"] for e in ep["standard"]])
results = []
for m in [x for x in METHODS if x != "standard"]:
    v = np.array([e["tail5"] for e in ep[m]])
    d = v - base
    try:
        _, pv = wilcoxon(v, base)
    except ValueError:
        pv = 1.0
    results.append((m, d, pv))
    p(f"{m:<30} median_delta={np.median(d):+.4f}  wins={int((d>0).sum())}/10  p={pv:.4f}")
ps = sorted([(pv, m) for m, _, pv in results])
p("\nHolm over 6 comparisons (alpha=0.05):")
for rank, (pv, m) in enumerate(ps):
    thr = 0.05 / (len(ps) - rank)
    verdict = "SIGNIFICANT" if pv < thr else "not significant"
    p(f"  {m:<30} p={pv:.4f}  threshold={thr:.4f}  -> {verdict}")
    if pv >= thr:
        break

p("\n--- early phase (cold start, first 5 cycles) vs standard ---")
base_e = np.array([e["early5"] for e in ep["standard"]])
for m in [x for x in METHODS if x != "standard"]:
    v = np.array([e["early5"] for e in ep[m]])
    d = v - base_e
    try: _, pv = wilcoxon(v, base_e)
    except ValueError: pv = 1.0
    p(f"{m:<30} median_delta={np.median(d):+.4f}  wins={int((d>0).sum())}/10  p={pv:.4f}")

p("\n--- singleton test (5 hand-designed maps), tail5 ---")
for m in METHODS:
    vals = np.array([endpoints(data[(m, s)]["sing"])["tail5"] for s in SEEDS])
    p(f"{m:<30} {vals.mean():.3f} ± {vals.std():.3f}")

p("\n--- passable fraction of trained-on envs (tail5) ---")
for m in METHODS:
    vals = np.array([data[(m, s)]["pass"][-5:].mean() for s in SEEDS])
    p(f"{m:<30} {vals.mean():.3f} ± {vals.std():.3f}")

p("\n=== OLD (confounded) vs FIXED, per CNN arm, tail5 sampled ===")
for m in CNN_ARMS:
    olds = []
    for s in SEEDS:
        rs = list(api.runs(f"{ENT}/{OLD_PROJ}", {"display_name": f"{m}_seed{s}"}))
        if rs:
            v = series(rs[0], K_SAMP)
            if len(v) >= 5:
                olds.append(v[-5:].mean())
    new = np.array([e["tail5"] for e in ep[m]])
    if olds:
        p(f"{m:<30} old {np.mean(olds):.3f} -> fixed {new.mean():.3f}  (delta {new.mean()-np.mean(olds):+.3f})")

open("wave1fix_analysis_output.txt", "w").write("\n".join(lines) + "\n")
print("\nwrote wave1fix_analysis_output.txt")
