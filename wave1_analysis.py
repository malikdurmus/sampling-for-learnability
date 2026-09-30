"""Wave-1 first-pass outcome analysis (descriptive + paired tests).

Pulls, for all 70 wave-1 runs (7 methods x seeds 1-10, W&B project
sfl-jaxnav-campaign), the two held-out eval series:
  - sampled-test-metrics.eval-sampled/overall_win_rate   (PRIMARY: 100 fixed
    DR levels x 10 episodes, identical set for every run/method)
  - singleton-test-metrics.eval/:overall_win_rate        (5 hand-designed maps)
plus curriculum-characterization series (env-metrics/.passable,
recent_success_rate).

Endpoints reported per run:
  final    = last eval cycle
  tail5    = mean of last 5 cycles   (primary endpoint: robust to eval noise)
  early5   = mean of first 5 cycles  (cold-start probe)
  auc      = mean over all 45 cycles (learning-speed summary)

Stats: per method vs 'standard', paired by seed: Wilcoxon signed-rank
(two-sided) on tail5, Holm over the 6 comparisons; per-seed win counts.
Writes wave1_analysis_output.txt.
"""
import numpy as np
import wandb
from scipy.stats import wilcoxon

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJ = "sfl-jaxnav-campaign"
METHODS = ["dr","standard","cnn","hybrid_linear","hybrid_soft_handoff",
           "hybrid_learnability_weighted","hybrid_multiplicative"]
SEEDS = list(range(1, 11))
K_SAMP = "sampled-test-metrics.eval-sampled/overall_win_rate"
K_SING = "singleton-test-metrics.eval/:overall_win_rate"
K_PASS = "env-metrics/.passable"
K_RSR  = "recent_success_rate"

api = wandb.Api()
lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

def series(run, key):
    h = run.history(keys=[key, "update_count"], pandas=False, samples=4000)
    h = [x for x in h if x.get(key) is not None]
    h.sort(key=lambda x: x.get("update_count") or 0)
    return np.array([x[key] for x in h], dtype=float)

data = {}   # (method, seed) -> dict of series
for m in METHODS:
    for s in SEEDS:
        name = f"{m}_seed{s}"
        rs = list(api.runs(f"{ENT}/{PROJ}", {"display_name": name}))
        assert len(rs) == 1 and rs[0].state == "finished", name
        r = rs[0]
        data[(m, s)] = {"samp": series(r, K_SAMP), "sing": series(r, K_SING),
                        "pass": series(r, K_PASS), "rsr": series(r, K_RSR)}
        print(f"  pulled {name}: {len(data[(m,s)]['samp'])} eval cycles")

def endpoints(v):
    return dict(final=v[-1], tail5=v[-5:].mean(), early5=v[:5].mean(), auc=v.mean())

p("\n================ SAMPLED TEST (100 fixed DR levels) — overall_win_rate ================")
p(f"{'method':<30}{'final':>9}{'tail5':>9}{'early5':>9}{'auc':>9}   (mean +/- std over 10 seeds)")
ep = {}
for m in METHODS:
    es = [endpoints(data[(m, s)]["samp"]) for s in SEEDS]
    ep[m] = es
    row = f"{m:<30}"
    for k in ("final","tail5","early5","auc"):
        vals = np.array([e[k] for e in es])
        row += f" {vals.mean():.3f}±{vals.std():.3f}"
    p(row)

p("\n---- paired vs standard (tail5, sampled test): per-seed deltas ----")
base = np.array([e["tail5"] for e in ep["standard"]])
results = []
for m in [x for x in METHODS if x != "standard"]:
    v = np.array([e["tail5"] for e in ep[m]])
    d = v - base
    try:
        stat, pv = wilcoxon(v, base)
    except ValueError:
        pv = 1.0
    results.append((m, d, pv))
    p(f"{m:<30} median_delta={np.median(d):+.4f}  wins={int((d>0).sum())}/10  p={pv:.4f}"
      f"  deltas={np.round(d,3)}")
ps = sorted([(pv, m) for m, _, pv in results])
p("\nHolm over 6 comparisons (alpha=0.05):")
mtests = len(ps)
for rank, (pv, m) in enumerate(ps):
    thr = 0.05 / (mtests - rank)
    p(f"  {m:<30} p={pv:.4f}  threshold={thr:.4f}  -> "
      + ("SIGNIFICANT" if pv < thr else "not significant"))
    if pv >= thr: break

p("\n---- early phase (cold start): early5 paired vs standard ----")
base_e = np.array([e["early5"] for e in ep["standard"]])
for m in [x for x in METHODS if x != "standard"]:
    v = np.array([e["early5"] for e in ep[m]])
    d = v - base_e
    try: _, pv = wilcoxon(v, base_e)
    except ValueError: pv = 1.0
    p(f"{m:<30} median_delta={np.median(d):+.4f}  wins={int((d>0).sum())}/10  p={pv:.4f}")

p("\n---- singleton test (5 hand-designed maps), tail5 ----")
for m in METHODS:
    vals = np.array([endpoints(data[(m, s)]["sing"])["tail5"] for s in SEEDS])
    p(f"{m:<30} {vals.mean():.3f} ± {vals.std():.3f}")

p("\n---- curriculum characterization ----")
p(f"{'method':<30}{'passable(train envs) tail5':>28}{'recent_success tail5':>22}")
for m in METHODS:
    pa = np.array([data[(m, s)]["pass"][-5:].mean() for s in SEEDS])
    rs_ = np.array([data[(m, s)]["rsr"][-5:].mean() for s in SEEDS])
    p(f"{m:<30}{pa.mean():>15.3f} ± {pa.std():.3f}{rs_.mean():>15.3f} ± {rs_.std():.3f}")

p("\n---- learning curves: sampled win_rate mean over seeds at cycles 1..10 ----")
for m in METHODS:
    cur = np.stack([data[(m, s)]["samp"][:10] for s in SEEDS]).mean(0)
    p(f"{m:<30} " + " ".join(f"{x:.2f}" for x in cur))

open("wave1_analysis_output.txt", "w").write("\n".join(lines) + "\n")
print("\nwrote wave1_analysis_output.txt")
