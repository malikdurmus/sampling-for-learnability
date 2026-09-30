
import numpy as np, wandb
from scipy.stats import wilcoxon

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJ = "sfl-jaxnav-campaign-fixed"
K_SAMP = "sampled-test-metrics.eval-sampled/overall_win_rate"
NEW = ["progress", "progress_mindist", "progress_mean", "dijkstra"]
SEEDS = list(range(1, 11))

api = wandb.Api()
lines = []
def p(*a):
    t = " ".join(str(x) for x in a); print(t); lines.append(t)

def resolve(name):
    rs = sorted(api.runs(f"{ENT}/{PROJ}", {"display_name": name}),
                key=lambda r: r.created_at)
    fin = [r for r in rs if r.state == "finished"
           and (r.summary.get("update_count") or 0) >= 2200]
    assert fin, f"{name}: no finished full-length run ({len(rs)} attempts)"
    return fin[-1], len(rs)

def series(run, key):
    h = [x for x in run.history(keys=[key, "update_count"], pandas=False,
                                samples=4000) if x.get(key) is not None]
    h.sort(key=lambda x: x.get("update_count") or 0)
    return np.array([x[key] for x in h], dtype=float)

data, attempts = {}, {}
for m in NEW + ["standard", "dr"]:
    for s in SEEDS:
        r, att = resolve(f"{m}_seed{s}")
        v = series(r, K_SAMP)
        assert len(v) >= 40, f"{m}_seed{s}: only {len(v)} eval points"
        data[(m, s)] = v
        attempts[(m, s)] = att

p("NEW-ARMS ANALYSIS (generated", __import__("datetime").datetime.utcnow()
  .strftime("%Y-%m-%d %H:%MZ") + ")")
p(f"project {PROJ}; endpoint {K_SAMP}; n=10 paired seeds; duplicate names")
p("resolved to newest finished full-length attempt "
  f"(multi-attempt names: {sum(1 for a in attempts.values() if a > 1)})")
p("")

def tail5(v): return v[-5:].mean()
def early5(v): return v[:5].mean()
def auc(v): return v.mean()

p("== summary (mean ± std over 10 seeds) ==")
p(f"{'method':18s} {'tail5':>14s} {'early5':>14s} {'AUC':>8s}")
stats = {}
for m in ["standard"] + NEW + ["dr"]:
    t = np.array([tail5(data[(m, s)]) for s in SEEDS])
    e = np.array([early5(data[(m, s)]) for s in SEEDS])
    a = np.array([auc(data[(m, s)]) for s in SEEDS])
    stats[m] = (t, e, a)
    p(f"{m:18s} {t.mean():.4f}±{t.std():.4f} {e.mean():.4f}±{e.std():.4f} {a.mean():.4f}")
p("")

p("== paired vs standard (two-sided Wilcoxon, Holm over the 4 new arms) ==")
raw = {}
for m in NEW:
    d = stats[m][0] - stats["standard"][0]
    stat, pv = wilcoxon(stats[m][0], stats["standard"][0])
    raw[m] = (d, pv)
order = sorted(NEW, key=lambda m: raw[m][1])
holm = {}
for i, m in enumerate(order):
    holm[m] = min(1.0, raw[m][1] * (len(order) - i))
for j in range(1, len(order)):  # enforce monotonicity
    holm[order[j]] = max(holm[order[j]], holm[order[j - 1]])
p(f"{'method':18s} {'med delta':>10s} {'wins':>6s} {'p':>8s} {'p(Holm)':>8s}")
for m in NEW:
    d, pv = raw[m]
    p(f"{m:18s} {np.median(d):+10.4f} {int((d > 0).sum()):>4d}/10 "
      f"{pv:8.4f} {holm[m]:8.4f}")
p("")

p("== paired vs dr (context: do the new arms beat random?) ==")
for m in NEW:
    d = stats[m][0] - stats["dr"][0]
    stat, pv = wilcoxon(stats[m][0], stats["dr"][0])
    p(f"{m:18s} med delta {np.median(d):+.4f}  wins {int((d>0).sum())}/10  p={pv:.4f}")
p("")

p("== cold start, paired vs standard (early-5 mean) ==")
for m in NEW:
    d = stats[m][1] - stats["standard"][1]
    stat, pv = wilcoxon(stats[m][1], stats["standard"][1])
    p(f"{m:18s} med delta {np.median(d):+.4f}  wins {int((d>0).sum())}/10  p={pv:.4f}")
p("")

p("== density endpoint (dense arms only; within-run measurement) ==")
fb, lb, fd, ld = [], [], [], []
for m in NEW[:3]:
    for s in SEEDS:
        r, _ = resolve(f"{m}_seed{s}")
        b = series(r, "sfl/frac_nonzero"); dsc = series(r, "selection/frac_nonzero")
        if len(b) >= 40:
            fb.append(b[0]); lb.append(b[-1]); fd.append(dsc[0]); ld.append(dsc[-1])
p(f"n={len(fb)} runs")
p(f"binary p(1-p) frac_nonzero: cycle1 {np.mean(fb):.3f} "
  f"[{min(fb):.3f},{max(fb):.3f}]  final {np.mean(lb):.3f} [{min(lb):.3f},{max(lb):.3f}]")
p(f"dense score  frac_nonzero: cycle1 {np.mean(fd):.3f}  final {np.mean(ld):.3f}")

with open("newarms_analysis_output.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwritten: newarms_analysis_output.txt")
