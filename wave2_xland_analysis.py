"""Wave-2 xland difficult analysis (confirmatory/directional; n=4 paired
seeds, so Wilcoxon minimum two-sided p = 0.125 — report effect sizes, win
counts and curves; no Holm family can reach significance at this n and none
is claimed).

Project xland-sfl-campaign, runs xl_<method>_seed1..4, methods:
dr, standard, cnn, hybrid_linear, hybrid_soft_handoff.

Endpoints (per run):
  PRIMARY  eval_fixed/success_rate_mean  — fixed difficult-ruleset eval
  SECONDARY eval_bench/success_rate_mean — benchmark-sampled rulesets
  tail5 / early10 / auc; plus full learning curves (the cold-start
  question lives in the early phase).
Mechanism panels: target_mu trajectory, hybrid/alpha_soft_handoff (does the
gate engage?), cnn/member_rank_agreement (domain gate ~0.96 in-domain for
the xland ensemble), solvability/batch_mean.
Writes wave2_xland_analysis_output.txt.
"""
import argparse
import numpy as np
import wandb
from scipy.stats import wilcoxon

_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", choices=["difficult", "medium"], default="difficult",
                 help="difficult = runs xl_* (groups wave2-xland-*); "
                      "medium = runs xlm_* (groups wave2-xlandmed-*)")
_args = _ap.parse_args()
PREFIX = "xl_" if _args.variant == "difficult" else "xlm_"
OUT = f"wave2_xland_{_args.variant}_analysis_output.txt"

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJ = "xland-sfl-campaign"
METHODS = ["dr", "standard", "cnn", "hybrid_linear", "hybrid_soft_handoff"]
CNN_ARMS = ["cnn", "hybrid_linear", "hybrid_soft_handoff"]
SEEDS = [1, 2, 3, 4]
K_FIX = "eval_fixed/success_rate_mean"
K_BEN = "eval_bench/success_rate_mean"
KEYS2 = ["target_mu", "hybrid/alpha_soft_handoff", "cnn/member_rank_agreement",
         "solvability/batch_mean", "recent_success_rate"]

api = wandb.Api()
lines = []
def p(*a):
    t = " ".join(str(x) for x in a); print(t); lines.append(t)

def series(run, key):
    h = [x for x in run.history(keys=[key, "update_count"], pandas=False, samples=4000)
         if x.get(key) is not None]
    h.sort(key=lambda x: x.get("update_count") or 0)
    return np.array([x[key] for x in h], dtype=float)

data = {}
for m in METHODS:
    for s in SEEDS:
        name = f"{PREFIX}{m}_seed{s}"
        rs = list(api.runs(f"{ENT}/{PROJ}", {"display_name": name}))
        # crashed/failed attempts (2026-08-31 TIMEOUT/OOM incident) are
        # RETAINED in W&B; resolve to the finished attempt, newest first
        fin = sorted([x for x in rs if x.state == "finished"],
                     key=lambda x: x.created_at, reverse=True)
        assert fin, f"{name}: no finished attempt (states: {[x.state for x in rs]})"
        if len(rs) > 1:
            print(f"    ({name}: {len(rs)} attempts, using finished {fin[0].id})")
        r = fin[0]
        d = {"fix": series(r, K_FIX), "ben": series(r, K_BEN)}
        for k in KEYS2:
            d[k] = series(r, k)
        data[(m, s)] = d
        print(f"  pulled {name}: {len(d['fix'])} eval cycles")

p("\n=== DOMAIN GATE (CNN arms; xland in-domain member agreement ~0.96) ===")
for m in CNN_ARMS:
    ags = [data[(m, s)]["cnn/member_rank_agreement"] for s in SEEDS]
    ags = [float(a.mean()) for a in ags if len(a)]
    if ags:
        p(f"  {m:<22} mean agreement {np.mean(ags):.4f}  ({'OK' if 0.85 < np.mean(ags) < 1.0 else 'CHECK'})")

def endpoints(v):
    return dict(final=v[-1], tail5=v[-5:].mean(), early10=v[:10].mean(),
                early50=v[:50].mean(), auc=v.mean())

for key, label in (("fix", "PRIMARY: eval_fixed success rate"),
                   ("ben", "SECONDARY: eval_bench success rate")):
    p(f"\n=== {label} ===")
    p(f"{'method':<22}{'final':>9}{'tail5':>9}{'early10':>9}{'early50':>9}{'auc':>9}")
    ep = {}
    for m in METHODS:
        es = [endpoints(data[(m, s)][key]) for s in SEEDS]
        ep[m] = es
        row = f"{m:<22}"
        for k in ("final", "tail5", "early10", "early50", "auc"):
            vals = np.array([e[k] for e in es])
            row += f" {vals.mean():.4f}±{vals.std():.4f}"[:18].ljust(18) if False else f" {vals.mean():.3f}±{vals.std():.3f}"
        p(row)
    if key == "fix":
        p("\n--- paired vs standard (tail5 and early50; n=4 => descriptive only) ---")
        for stat in ("tail5", "early50"):
            base = np.array([e[stat] for e in ep["standard"]])
            p(f"  [{stat}]")
            for m in [x for x in METHODS if x != "standard"]:
                v = np.array([e[stat] for e in ep[m]])
                d = v - base
                try: _, pv = wilcoxon(v, base)
                except ValueError: pv = 1.0
                p(f"    {m:<22} median_delta={np.median(d):+.4f}  wins={int((d>0).sum())}/4  p={pv:.3f}")

p("\n=== MECHANISM PANELS (mean over seeds) ===")
p("--- target_mu: first/mid/last ---")
for m in CNN_ARMS:
    mus = np.stack([data[(m, s)]["target_mu"] for s in SEEDS])
    n = mus.shape[1]
    p(f"  {m:<22} start {mus[:,0].mean():.2f}  mid {mus[:,n//2].mean():.2f}  end {mus[:,-1].mean():.2f}")
p("--- soft_handoff gate: fraction of cycles with alpha < 1 ---")
al = [data[("hybrid_soft_handoff", s)]["hybrid/alpha_soft_handoff"] for s in SEEDS]
al = [a for a in al if len(a)]
if al:
    fr = np.mean([np.mean(a < 0.999) for a in al])
    p(f"  alpha<1 in {fr:.1%} of cycles (jaxnav: 2.2% — gate engaged means the arm is a real method here)")
p("--- batch solvability (fresh DR levels), first10/last10 mean ---")
for m in METHODS:
    sv = [data[(m, s)]["solvability/batch_mean"] for s in SEEDS]
    sv = [x for x in sv if len(x) >= 10]
    if sv:
        p(f"  {m:<22} first10 {np.mean([x[:10].mean() for x in sv]):.4f}   last10 {np.mean([x[-10:].mean() for x in sv]):.4f}")

p("\n--- learning curves (fixed eval, mean over seeds, every 20th cycle) ---")
for m in METHODS:
    cur = np.stack([data[(m, s)]["fix"] for s in SEEDS]).mean(0)
    pts = " ".join(f"{cur[i]:.3f}" for i in range(0, len(cur), 20))
    p(f"  {m:<22} {pts}")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
