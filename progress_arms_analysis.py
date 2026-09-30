"""Paired analysis of the 4 new arms (progress, progress_mindist,
progress_mean, dijkstra; seeds 1-5, pod campaign 2026-09-01/02) against the
wave1fix baselines (dr, standard, cnn, hybrid_linear; seeds 1-5 of their
10-seed sets — pairing restricted to common seeds).

Endpoints (per uedrlhf conventions / wave1fix_analysis.py):
  final : last sampled-test overall win rate
  early5: mean of first 5 evals (cold-start signal)
  auc   : mean over the whole eval curve
Stats: paired Wilcoxon signed-rank vs standard, Holm over the 4 new arms.
NOTE n=5 -> exact two-sided Wilcoxon minimum p = 0.0625: significance after
Holm is unreachable; report effect directions + per-seed wins, treat as
descriptive with the pre-registered caveat.

Also aggregates the score-level density diagnostics (first-cycle
frac_nonzero of each candidate score) that back the density claim.

Output: progress_arms_analysis_output.txt
"""
import numpy as np
import wandb
from scipy.stats import wilcoxon

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJ = "sfl-jaxnav-campaign-fixed"
K_SAMP = "sampled-test-metrics.eval-sampled/overall_win_rate"
SEEDS = [1, 2, 3, 4, 5]
NEW_ARMS = ["progress", "progress_mindist", "progress_mean", "dijkstra"]
BASELINES = ["standard", "dr", "cnn", "hybrid_linear"]

api = wandb.Api()
lines = []
def p(*a):
    t = " ".join(str(x) for x in a); print(t); lines.append(t)

def run_name(m, s):
    if m in ("dr", "standard"):
        return f"{m}_seed{s}"
    if m in ("cnn", "hybrid_linear", "hybrid_soft_handoff",
             "hybrid_learnability_weighted", "hybrid_multiplicative"):
        return f"{m}_fixed_seed{s}"
    return f"{m}_seed{s}"  # new arms

def newest_finished(name):
    rs = [r for r in api.runs(f"{ENT}/{PROJ}", {"display_name": name})
          if r.state == "finished"]
    rs.sort(key=lambda r: r.created_at)
    assert rs, f"{name}: no finished run"
    return rs[-1]

def series(run, key):
    h = [x for x in run.history(keys=[key, "update_count"], pandas=False,
                                samples=4000) if x.get(key) is not None]
    h.sort(key=lambda x: x.get("update_count") or 0)
    return np.array([x[key] for x in h], dtype=float)

def scalar_first(run, key):
    h = [x for x in run.history(keys=[key, "update_count"], pandas=False,
                                samples=200) if x.get(key) is not None]
    h.sort(key=lambda x: x.get("update_count") or 0)
    return float(h[0][key]) if h else float("nan")

data = {}
for m in NEW_ARMS + BASELINES:
    for s in SEEDS:
        r = newest_finished(run_name(m, s))
        curve = series(r, K_SAMP)
        assert len(curve) >= 40, f"{run_name(m,s)}: only {len(curve)} evals"
        data[(m, s)] = {"final": curve[-1], "early5": curve[:5].mean(),
                        "auc": curve.mean(), "run": r}

p("=" * 78)
p("PROGRESS-ARMS + DIJKSTRA PAIRED ANALYSIS (seeds 1-5)  [2026-09-02]")
p("endpoint source: " + K_SAMP)
p("=" * 78)

for ep in ["final", "early5", "auc"]:
    p(f"\n--- endpoint: {ep} ---")
    p(f"{'method':18s} " + " ".join(f"s{s}" for s in SEEDS) + "   mean+-sd")
    for m in NEW_ARMS + BASELINES:
        v = np.array([data[(m, s)][ep] for s in SEEDS])
        p(f"{m:18s} " + " ".join(f"{x:.3f}" for x in v) +
          f"   {v.mean():.4f}+-{v.std(ddof=1):.4f}")

p("\n--- paired Wilcoxon vs standard (n=5; exact min p=0.0625 — descriptive) ---")
for ep in ["final", "early5", "auc"]:
    pvals = {}
    for m in NEW_ARMS:
        a = np.array([data[(m, s)][ep] for s in SEEDS])
        b = np.array([data[("standard", s)][ep] for s in SEEDS])
        d = a - b
        try:
            stat, pv = wilcoxon(a, b, method="exact")
        except ValueError:
            pv = float("nan")
        pvals[m] = (d.mean(), (d > 0).sum(), pv)
    order = sorted(pvals, key=lambda m: pvals[m][2])
    holm = {}
    for i, m in enumerate(order):
        holm[m] = min(1.0, pvals[m][2] * (len(order) - i))
    p(f"  [{ep}]")
    for m in NEW_ARMS:
        dm, wins, pv = pvals[m]
        p(f"    {m:18s} mean-delta {dm:+.4f}  wins {wins}/5  "
          f"p={pv:.4f}  holm={holm[m]:.4f}")

p("\n--- density claim: first-cycle frac_nonzero (mean over the arm's 5 runs) ---")
dens_keys = {"binary p(1-p)": "sfl/frac_nonzero",
             "end-var": "progress/end_var_frac_nonzero",
             "min-var": "progress/min_var_frac_nonzero",
             "mp-score": "progress/mp_score_frac_nonzero"}
for m in ["progress", "progress_mindist", "progress_mean"]:
    vals = {lbl: np.nanmean([scalar_first(data[(m, s)]["run"], k) for s in SEEDS])
            for lbl, k in dens_keys.items()}
    p(f"  {m:18s} " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in vals.items()))

p("\n--- cross-metric agreement (run-mean Spearman, progress arms pooled) ---")
agree_keys = ["agree/binary_vs_end_var_spearman", "agree/end_var_vs_min_var_spearman",
              "agree/binary_vs_mp_score_spearman", "agree/end_var_vs_mp_score_spearman"]
for k in agree_keys:
    v = np.nanmean([series(data[(m, s)]["run"], k).mean()
                    for m in ["progress", "progress_mindist", "progress_mean"]
                    for s in SEEDS])
    p(f"  {k:45s} {v:.3f}")

with open("progress_arms_analysis_output.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwritten: progress_arms_analysis_output.txt")
