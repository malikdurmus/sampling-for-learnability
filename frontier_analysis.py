"""Frontier-controller wave analysis (project sfl-jaxnav-frontier, launched 2026-09-12).

Question: does fixing the target-mu controller (measurement fix; calibrated frontier) change
the hybrid_linear curriculum's behaviour and outcome, within each generator and relative to
the old controller runs of sfl-jaxnav-campaign-fixed?

Data: frontier_analysis_raw.json (from frontier_analysis_pull.py). Cycle c (0-based) is the
row logged at update_count = 50*(c+1); per-update keys are averaged over (50c, 50(c+1)].
Windows are computed per seed only when EVERY cycle of the window is present, so running
runs contribute to early windows but not to late ones. Paired tests: exact two-sided
Wilcoxon over seeds (n=4 -> smallest attainable p = 0.125; medians and wins carry the
evidence). Cross-project contrasts (new vs old campaign) pair by seed number but are
DESCRIPTIVE: code copy, qos/hardware and date differ.
Usage: python frontier_analysis.py  -> frontier_analysis_output.txt + frontier_analysis_*.png
"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "frontier_analysis_raw.json")
OUT = os.path.join(HERE, "frontier_analysis_output.txt")
K_MAIN = "sampled-test-metrics.eval-sampled/overall_win_rate"
K_SING = "singleton-test-metrics.eval/:overall_win_rate"
NCYC = 45
ARMS_NEW = ["standard", "hybrid_pa", "hybrid_cf", "hybrid_cfp"]
GENS = ["solv", "norm"]
WINDOWS = [("first5", range(0, 5)), ("first10", range(0, 10)), ("c11-21", range(11, 22)), ("c22-32", range(22, 33)),
           ("c33-44", range(33, 45)), ("tail5", range(40, 45))]
lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

raw = json.load(open(RAW))

def cyc_array(run, key):
    d = run["cycle"].get(key, {})
    arr = np.full(NCYC, np.nan)
    for u, v in d.items():
        u = int(u)
        if u % 50 == 0 and 1 <= u // 50 <= NCYC:
            arr[u // 50 - 1] = v
    return arr

def upd_cycle_mean(run, key):
    d = run["upd"].get(key, {})
    acc = [[] for _ in range(NCYC)]
    for u, v in d.items():
        u = int(u); c = (u - 1) // 50
        if 0 <= c < NCYC:
            acc[c].append(v)
    return np.array([np.mean(a) if a else np.nan for a in acc])

# ---------------- organise ----------------
# new project: name = <gen>_<arm>_seed<N>; old: standard_seedN / hybrid_linear_fixed_seedN (norm), solv_standard_seedN / solv_hybrid_linear_seedN (solv)
runs = {}   # (proj, gen, arm) -> {seed: run}
for key, r in raw.items():
    label, name = key.split(":", 1)
    seed = int(name.rsplit("seed", 1)[1])
    if label == "new":
        gen, rest = name.split("_", 1); arm = rest.rsplit("_seed", 1)[0]
    else:
        if name.startswith("solv_"):
            gen = "solv"; arm = "old_" + name[5:].rsplit("_seed", 1)[0]
        else:
            gen = "norm"; arm = "old_" + name.rsplit("_seed", 1)[0].replace("_fixed", "")
    runs.setdefault((gen, arm), {})[seed] = r

p("FRONTIER WAVE ANALYSIS (generated %s)" % __import__("time").strftime("%Y-%m-%d %H:%MZ", __import__("time").gmtime()))
p("raw file:", os.path.basename(RAW), "| runs:", len(raw))
p("\n== 1. Inventory (cycles of 45 logged; state) ==")
for gen in GENS:
    for arm in ARMS_NEW + ["old_standard", "old_hybrid_linear"]:
        d = runs.get((gen, arm), {})
        if not d: continue
        cells = []
        for s in sorted(d):
            r = d[s]; n = int(np.sum(~np.isnan(cyc_array(r, K_MAIN))))
            cells.append(f"s{s}:{n}{'' if r['state']=='finished' else '(' + r['state'][:3] + ')'}")
        p(f"  {gen:<5}{arm:<20}" + "  ".join(cells) + f"   ids: " + ",".join(f"{d[s]['id']}" for s in sorted(d)))

# ---------------- sanity gates ----------------
p("\n== 2. Sanity gates ==")
for gen in GENS:
    for arm in ARMS_NEW:
        d = runs.get((gen, arm), {})
        if not d: continue
        vals = [np.nanmean(upd_cycle_mean(r, "env-metrics/.passable")) for r in d.values()]
        agree = [np.nanmean(cyc_array(r, "cnn/member_rank_agreement")) for r in d.values()]
        cfg = next(iter(d.values()))["config"]
        p(f"  {gen:<5}{arm:<11} passable(train envs) mean {np.nanmean(vals):.3f} [{np.nanmin(vals):.3f},{np.nanmax(vals):.3f}]"
          + (f"  member_rank_agreement {np.nanmean(agree):.4f}" if arm != "standard" else "")
          + f"  cfg: CS={cfg['CURRICULUM_STRATEGY']} MM={cfg['MU_MEASUREMENT']} FM={cfg['FRONTIER_MODE']} vpc={cfg['valid_path_check']} variant={cfg['TRAINER_VARIANT']}")

# ---------------- helpers ----------------
def window_means(d, key, cycles):
    """seed -> mean over the window (NaN unless every cycle is present)."""
    out = {}
    for s, r in d.items():
        a = cyc_array(r, key)[list(cycles)]
        out[s] = float(a.mean()) if not np.isnan(a).any() else np.nan
    return out

def paired(a, b):
    """a, b: seed -> value. Returns (n, median delta, wins, p or nan, deltas)."""
    seeds = [s for s in sorted(set(a) & set(b)) if not (np.isnan(a[s]) or np.isnan(b[s]))]
    if not seeds: return 0, np.nan, 0, np.nan, []
    dl = np.array([a[s] - b[s] for s in seeds])
    pv = np.nan
    if len(dl) >= 4 and np.any(dl != 0):
        try: pv = wilcoxon(dl, zero_method="wilcox", alternative="two-sided", method="exact").pvalue
        except Exception: pv = np.nan
    return len(dl), float(np.median(dl)), int((dl > 0).sum()), pv, dl

def fmt_delta(res):
    n, md, w, pv, _ = res
    if n == 0: return "n=0"
    return f"med {md:+.4f} wins {w}/{n}" + (f" p={pv:.3f}" if not np.isnan(pv) else "")

# ---------------- 3. outcome tables ----------------
for K, kname in [(K_MAIN, "sampled test (100 fixed levels) overall win rate"), (K_SING, "singleton suite overall win rate")]:
    p(f"\n== 3. Outcome: {kname} ==")
    for gen in GENS:
        p(f"\n-- generator {gen} --")
        hdr = f"  {'arm':<20}" + "".join(f"{w:>16}" for w, _ in WINDOWS)
        p(hdr)
        for arm in ARMS_NEW + ["old_standard", "old_hybrid_linear"]:
            d = runs.get((gen, arm), {})
            if not d: continue
            cells = []
            for w, cyc in WINDOWS:
                m = window_means(d, K, cyc); v = np.array([x for x in m.values() if not np.isnan(x)])
                cells.append(f"{v.mean():.3f}±{v.std():.3f} n{len(v)}" if len(v) else "   (pending)   ")
            p(f"  {arm:<20}" + "".join(f"{c:>16}" for c in cells))
        p("  paired within the new project (vs new standard; and the frontier contrasts):")
        for w, cyc in WINDOWS:
            std = window_means(runs.get((gen, "standard"), {}), K, cyc)
            row = [f"{w:<8}"]
            for arm in ["hybrid_pa", "hybrid_cf", "hybrid_cfp"]:
                row.append(f"{arm}-std: {fmt_delta(paired(window_means(runs.get((gen, arm), {}), K, cyc), std))}")
            row.append(f"cf-pa: {fmt_delta(paired(window_means(runs.get((gen, 'hybrid_cf'), {}), K, cyc), window_means(runs.get((gen, 'hybrid_pa'), {}), K, cyc)))}")
            row.append(f"cfp-cf: {fmt_delta(paired(window_means(runs.get((gen, 'hybrid_cfp'), {}), K, cyc), window_means(runs.get((gen, 'hybrid_cf'), {}), K, cyc)))}")
            p("   " + " | ".join(row))
        p("  cross-project, paired by seed number (DESCRIPTIVE):")
        for w, cyc in WINDOWS:
            ostd = window_means(runs.get((gen, "old_standard"), {}), K, cyc); ohl = window_means(runs.get((gen, "old_hybrid_linear"), {}), K, cyc)
            row = [f"{w:<8}", f"newstd-oldstd: {fmt_delta(paired(window_means(runs.get((gen, 'standard'), {}), K, cyc), ostd))}",
                   f"pa-oldhyb: {fmt_delta(paired(window_means(runs.get((gen, 'hybrid_pa'), {}), K, cyc), ohl))}",
                   f"cf-oldhyb: {fmt_delta(paired(window_means(runs.get((gen, 'hybrid_cf'), {}), K, cyc), ohl))}",
                   f"cfp-oldhyb: {fmt_delta(paired(window_means(runs.get((gen, 'hybrid_cfp'), {}), K, cyc), ohl))}",
                   f"oldhyb-oldstd: {fmt_delta(paired(ohl, ostd))}"]
            p("   " + " | ".join(row))

# per-cycle paired delta vs new standard (K_MAIN) — where does any difference live?
p("\n== 4. Per-cycle mean win rate (sampled test) and paired delta vs new standard (median over seeds with that cycle) ==")
for gen in GENS:
    p(f"\n-- generator {gen} --")
    p(f"  {'cyc':>3} " + "".join(f"{a[:10]:>10}" for a in ARMS_NEW) + f"{'old_std':>9}{'old_hyb':>9} | " + "".join(f"{('d_'+a[7:] if a!='standard' else ''):>12}" for a in ARMS_NEW[1:]))
    for c in range(NCYC):
        means = []; deltas = []
        std_vals = {s: cyc_array(r, K_MAIN)[c] for s, r in runs.get((gen, "standard"), {}).items()}
        for arm in ARMS_NEW + ["old_standard", "old_hybrid_linear"]:
            vals = [cyc_array(r, K_MAIN)[c] for r in runs.get((gen, arm), {}).values()]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(f"{np.mean(vals):>10.3f}" if vals else f"{'-':>10}")
        for arm in ARMS_NEW[1:]:
            dl = [cyc_array(r, K_MAIN)[c] - std_vals[s] for s, r in runs.get((gen, arm), {}).items() if s in std_vals and not np.isnan(cyc_array(r, K_MAIN)[c]) and not np.isnan(std_vals[s])]
            deltas.append(f"{np.median(dl):+.3f}({sum(x>0 for x in dl)}/{len(dl)})" if dl else "-")
        p(f"  {c:>3} " + "".join(means[:4]) + f"{means[4][1:]:>9}{means[5][1:]:>9} | " + "".join(f"{x:>12}" for x in deltas))

# ---------------- 5. controller behaviour ----------------
p("\n== 5. Controller behaviour (mean over seeds at selected cycles) ==")
CK = [("target_mu", "mu"), ("frontier/mu_star", "mu*"), ("frontier/n_informative_bins", "infB"), ("buffer_success_rate", "sBuf"),
      ("recent_success_rate", "sLeg"), ("generated_success_rate", "sGen"), ("solvability/selected_mean", "solvSel"),
      ("frontier/selected_frac_intermediate", "interm"), ("cnn/selected_mean", "cnnSel"), ("cnn/tracking_error", "trkE"),
      ("sfl/selected_mean", "sflSel"), ("sfl/batch_mean", "sflBat")]
SHOW = [0, 1, 2, 3, 4, 5, 8, 11, 16, 22, 28, 33, 39, 44]
for gen in GENS:
    for arm in ["hybrid_pa", "hybrid_cf", "hybrid_cfp", "old_hybrid_linear", "standard"]:
        d = runs.get((gen, arm), {})
        if not d: continue
        p(f"\n-- {gen} / {arm} (n={len(d)}) --")
        p("  cyc " + "".join(f"{lab:>8}" for _, lab in CK) + f"{'passTr':>8}{'eval':>8}")
        for c in SHOW:
            cells = []
            for k, _ in CK:
                v = [cyc_array(r, k)[c] for r in d.values()]; v = [x for x in v if not np.isnan(x)]
                cells.append(f"{np.mean(v):>8.3f}" if v else f"{'-':>8}")
            pv = [upd_cycle_mean(r, "env-metrics/.passable")[c] for r in d.values()]; pv = [x for x in pv if not np.isnan(x)]
            ev = [cyc_array(r, K_MAIN)[c] for r in d.values()]; ev = [x for x in ev if not np.isnan(x)]
            p(f"  {c:>3} " + "".join(cells) + (f"{np.mean(pv):>8.3f}" if pv else f"{'-':>8}") + (f"{np.mean(ev):>8.3f}" if ev else f"{'-':>8}"))

p("\n== 6. Controller summary statistics per arm ==")
for gen in GENS:
    for arm in ["hybrid_pa", "hybrid_cf", "hybrid_cfp", "old_hybrid_linear"]:
        d = runs.get((gen, arm), {})
        if not d: continue
        mu_first_ge = []; mu_final = []; mu_dec = []; solv_late = []; interm_late = []; sbuf_late = []; sleg_late = []
        for r in d.values():
            mu = cyc_array(r, "target_mu"); ok = ~np.isnan(mu)
            if ok.sum() == 0: continue
            mu_final.append(mu[ok][-1]); mu_dec.append(int(np.sum(np.diff(mu[ok]) < -1e-6)))
            ge = np.where(mu >= 0.6)[0]; mu_first_ge.append(int(ge[0]) if len(ge) else -1)
            sl = cyc_array(r, "solvability/selected_mean"); sl = sl[22:][~np.isnan(sl[22:])]; solv_late.append(sl.mean() if len(sl) else np.nan)
            it = cyc_array(r, "frontier/selected_frac_intermediate"); it = it[22:][~np.isnan(it[22:])]; interm_late.append(it.mean() if len(it) else np.nan)
            sb = cyc_array(r, "buffer_success_rate"); sb = sb[22:][~np.isnan(sb[22:])]; sbuf_late.append(sb.mean() if len(sb) else np.nan)
            sg = cyc_array(r, "recent_success_rate"); sg = sg[22:][~np.isnan(sg[22:])]; sleg_late.append(sg.mean() if len(sg) else np.nan)
        p(f"  {gen:<5}{arm:<18} first cycle with mu>=0.6: {mu_first_ge}  final mu: {[round(x,2) for x in mu_final]}  #decreases: {mu_dec}"
          f"  | cycles>=22: solvability(selected) {np.nanmean(solv_late):.3f}, intermediate frac {np.nanmean(interm_late):.3f}, buffer success {np.nanmean(sbuf_late):.3f} vs legacy {np.nanmean(sleg_late):.3f}")

# ---------------- 7. figures ----------------
COL = {"standard": "#52514e", "hybrid_pa": "#eda100", "hybrid_cf": "#2a78d6", "hybrid_cfp": "#1baf7a", "old_standard": "#9a9a95", "old_hybrid_linear": "#e34948"}
LAB = {"standard": "standard (new)", "hybrid_pa": "hybrid_pa (old rule, fixed measurement)", "hybrid_cf": "hybrid_cf (calibrated frontier, crossing)",
       "hybrid_cfp": "hybrid_cfp (calibrated frontier, priority)", "old_standard": "old standard (campaign-fixed)", "old_hybrid_linear": "old hybrid_linear (old controller)"}
def mean_curve(d, key, upd=False):
    arrs = np.array([upd_cycle_mean(r, key) if upd else cyc_array(r, key) for r in d.values()])
    return np.nanmean(arrs, axis=0), np.sum(~np.isnan(arrs), axis=0)
fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))
for gi, gen in enumerate(GENS):
    ax = axes[gi, 0]
    for arm in ["old_standard", "old_hybrid_linear", "standard", "hybrid_pa", "hybrid_cf", "hybrid_cfp"]:
        d = runs.get((gen, arm), {})
        if not d: continue
        m, n = mean_curve(d, K_MAIN); x = np.arange(NCYC)[n >= 2]
        ax.plot(x, m[n >= 2], color=COL[arm], lw=2 if not arm.startswith("old") else 1.5, ls="-" if not arm.startswith("old") else "--", label=LAB[arm])
    ax.set_title(f"{gen}: sampled-test win rate (mean over seeds)", fontsize=10, loc="left"); ax.set_xlabel("cycle"); ax.set_ylim(0.85 if gen == "solv" else 0.8, 1.0); ax.set_xlim(2, 44)
    ax.grid(True, color="#e6e5e1"); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    if gi == 0: ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[gi, 1]
    for arm in ["old_hybrid_linear", "hybrid_pa", "hybrid_cf", "hybrid_cfp"]:
        d = runs.get((gen, arm), {})
        if not d: continue
        m, n = mean_curve(d, "target_mu"); x = np.arange(NCYC)[n >= 1]
        ax.plot(x, m[n >= 1], color=COL[arm], lw=2, ls="--" if arm.startswith("old") else "-", label=LAB[arm])
    ax.set_title(f"{gen}: target mu actually used", fontsize=10, loc="left"); ax.set_xlabel("cycle"); ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e6e5e1"); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax = axes[gi, 2]
    for arm in ["old_hybrid_linear", "hybrid_pa", "hybrid_cf", "hybrid_cfp"]:
        d = runs.get((gen, arm), {})
        if not d: continue
        m, n = mean_curve(d, "solvability/selected_mean"); x = np.arange(NCYC)[n >= 1]
        ax.plot(x, m[n >= 1], color=COL[arm], lw=2, ls="--" if arm.startswith("old") else "-", label=LAB[arm])
    ax.set_title(f"{gen}: solvability of the selected buffer (per level)", fontsize=10, loc="left"); ax.set_xlabel("cycle"); ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e6e5e1"); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    if gi == 0: ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "frontier_analysis_curves.png"), dpi=140); plt.close(fig)
p("\nfigure: frontier_analysis_curves.png (eval curves, mu used, selected-buffer solvability; both generators; old-project references dashed)")
open(OUT, "w").write("\n".join(lines) + "\n"); print("saved", OUT)
