"""XLand campaign analysis (thesis §6.7) — reproducible from W&B alone.

Usage:
    python xland_analysis.py --ruleset difficult   # writes xland_difficult_analysis_output.txt
    python xland_analysis.py --ruleset medium      # once the medium arms are complete

Data source: W&B project xland-sfl-campaign, groups wave2-xland-<arm>
(difficult) / wave2-xlandmed-<arm> (medium), runs xl_/xlm_<arm>_seed{1..4}.
Dedup rule (established for this campaign): among runs sharing a name, use
the newest FINISHED run with the highest _step; crashed/failed attempts are
ignored. Every run's config is asserted (arm, seed, ruleset, budget, eval
protocol) before any number is computed.

Endpoints (n=4 seeds -> DESCRIPTIVE ONLY; the exact two-sided Wilcoxon
cannot go below p=0.125 at n=4, so no significance claims are possible):
  - eval_fixed/success_rate_mean : success rate on 512 freshly sampled envs
    of the fixed training ruleset x 10 episodes, every eval cycle. NOTE:
    the eval envs are RE-SAMPLED each cycle from the run's rng chain (the
    config key eval_seed is unused in xland_sfl.py) — this is NOT a frozen
    level set like JaxNav's 100-level suite.
  - eval_bench/success_rate_mean : same but rulesets sampled from the
    high-3m benchmark (generalization).
  Windows: tail-5 (primary), final, early-10, early-50 (cold-start).

Raw pulled series are cached to xland_<ruleset>_raw.json next to this file;
delete the cache to force a re-pull.
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

ENTITY  = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJECT = "xland-sfl-campaign"
SEEDS   = [1, 2, 3, 4]
ARMS    = ["dr", "standard", "cnn", "hybrid_linear", "hybrid_soft_handoff"]
EXPECTED_METHOD = {"dr": "random", "standard": "standard", "cnn": "cnn",
                   "hybrid_linear": "hybrid_linear",
                   "hybrid_soft_handoff": "hybrid_soft_handoff"}
KEYS = ["eval_fixed/success_rate_mean", "eval_bench/success_rate_mean",
        "eval_fixed/returns_mean", "eval_bench/returns_mean", "update_count"]

ap = argparse.ArgumentParser()
ap.add_argument("--ruleset", choices=["difficult", "medium"], default="difficult")
args = ap.parse_args()
PFX   = {"difficult": ("wave2-xland-", "xl_"), "medium": ("wave2-xlandmed-", "xlm_")}[args.ruleset]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"xland_{args.ruleset}_raw.json")
OUT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"xland_{args.ruleset}_analysis_output.txt")

def fetch():
    import wandb
    api = wandb.Api(timeout=120)
    data = {}
    for arm in ARMS:
        group = PFX[0] + arm
        runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})]
        for seed in SEEDS:
            name = f"{PFX[1]}{arm}_seed{seed}"
            cand = [r for r in runs if r.name == name and r.state == "finished"]
            assert cand, f"no finished run named {name} in group {group}"
            r = max(cand, key=lambda r: (r.summary.get("_step") or 0, str(r.created_at)))
            c = r.config; env = c["env"]; lrn = c["learning"]
            assert c["LEARN_METHOD"] == EXPECTED_METHOD[arm], (name, c["LEARN_METHOD"])
            assert int(c["SEED"]) == seed, (name, c["SEED"])
            assert env["ruleset"] == args.ruleset, (name, env["ruleset"])
            assert int(float(lrn["TOTAL_TIMESTEPS"])) == 300000000, (name, lrn["TOTAL_TIMESTEPS"])
            assert env["eval_num_envs"] == 512 and env["eval_num_episodes"] == 10, name
            assert env["benchmark_id"] == "high-3m", name
            rows = list(r.history(keys=KEYS, pandas=False, samples=100000))
            assert rows, f"{name}: no eval rows"
            series = {k: [row[k] for row in rows] for k in KEYS}
            uc = series["update_count"]
            assert all(b > a for a, b in zip(uc, uc[1:])), f"{name}: update_count not increasing"
            assert not any(v is None for k in KEYS for v in series[k]), f"{name}: null values"
            data[f"{arm}/{seed}"] = {"run_id": r.id, "series": series}
            print(f"fetched {name}: {len(rows)} eval rows (run {r.id})", file=sys.stderr)
    return data

if os.path.exists(CACHE):
    data = json.load(open(CACHE))
    print(f"using cache {CACHE}", file=sys.stderr)
else:
    data = fetch()
    json.dump(data, open(CACHE, "w"))

lengths = {k: len(v["series"]["update_count"]) for k, v in data.items()}
assert len(set(lengths.values())) == 1, f"unequal eval-row counts: {lengths}"
N_EVAL = next(iter(lengths.values()))

def window(arm, key, sl):
    return np.array([np.mean(np.asarray(data[f"{arm}/{s}"]["series"][key], dtype=float)[sl])
                     for s in SEEDS])

def fmt_pair(a, b):
    d = a - b
    nz = d[d != 0]
    p = wilcoxon(nz, method="exact", alternative="two-sided").pvalue if len(nz) else 1.0
    return f"delta_median={np.median(d):+.4f} wins={int((d > 0).sum())}/4 p={p:.3f}"

lines = [f"XLand {args.ruleset.upper()} ruleset campaign analysis",
         f"project={PROJECT}  arms={ARMS}  seeds={SEEDS}  eval rows/run={N_EVAL}",
         f"run ids: " + ", ".join(f"{k}:{v['run_id']}" for k, v in sorted(data.items())),
         "n=4 -> exact two-sided Wilcoxon minimum p=0.125: ALL comparisons are",
         "descriptive; no significance claims are possible at this seed count.", ""]
for key, label in [("eval_fixed/success_rate_mean", "SUCCESS RATE, fixed training ruleset (512 envs x 10 eps, resampled per cycle)"),
                   ("eval_bench/success_rate_mean", "SUCCESS RATE, high-3m benchmark rulesets (generalization)")]:
    lines.append(f"== {label} ==")
    for wname, sl in [("tail-5", slice(-5, None)), ("final", slice(-1, None)),
                      ("early-10", slice(0, 10)), ("early-50", slice(0, 50))]:
        lines.append(f"  [{wname}]")
        vals = {arm: window(arm, key, sl) for arm in ARMS}
        for arm in ARMS:
            v = vals[arm]
            lines.append(f"    {arm:22s} {v.mean():.4f} +- {v.std(ddof=1):.4f}   per-seed: "
                         + " ".join(f"{x:.4f}" for x in v))
        for arm in ["cnn", "hybrid_linear", "hybrid_soft_handoff"]:
            lines.append(f"    {arm:22s} vs standard: {fmt_pair(vals[arm], vals['standard'])}"
                         f"   vs dr: {fmt_pair(vals[arm], vals['dr'])}")
        lines.append(f"    {'standard':22s} vs dr: {fmt_pair(vals['standard'], vals['dr'])}")
    lines.append("")
# ---- EXPLORATORY (not pre-registered): time to threshold on the training ruleset ----
# First eval cycle (1-based, of N_EVAL) at which eval_fixed success >= thr; "never" if not reached.
# Answers "does the prior speed up learning?" where floor/ceiling endpoints are uninformative.
KF = "eval_fixed/success_rate_mean"
lines.append("== EXPLORATORY, not pre-registered: first eval cycle reaching a success threshold (fixed ruleset) ==")
for thr in (0.1, 0.5, 0.9):
    lines.append(f"  [threshold {thr}]")
    T = {}
    for arm in ARMS:
        t_arm = []
        for s in SEEDS:
            v = np.asarray(data[f"{arm}/{s}"]["series"][KF], float)
            hit = np.nonzero(v >= thr)[0]
            t_arm.append(int(hit[0]) + 1 if len(hit) else None)
        T[arm] = t_arm
        shown = " ".join("never" if x is None else str(x) for x in t_arm)
        reached = [x for x in t_arm if x is not None]
        summ = (f"median={np.median(reached):.0f} mean={np.mean(reached):.1f} +- {np.std(reached, ddof=1) if len(reached) > 1 else 0:.1f}"
                if reached else "never reached")
        lines.append(f"    {arm:22s} cycles per seed: {shown:24s} ({len(reached)}/4 reached; {summ})")
    if all(x is not None for arm in ARMS for x in T[arm]):
        for arm in ["cnn", "hybrid_linear", "hybrid_soft_handoff"]:
            a, b, c = np.array(T[arm], float), np.array(T["standard"], float), np.array(T["dr"], float)
            lines.append(f"    {arm:22s} vs standard: {fmt_pair(a, b)} (positive = LATER)   vs dr: {fmt_pair(a, c)}")
        lines.append(f"    {'standard':22s} vs dr: {fmt_pair(np.array(T['standard'], float), np.array(T['dr'], float))}")
lines.append("")
report = "\n".join(lines)
open(OUT, "w").write(report + "\n")
print(report)
print(f"\nwritten: {OUT}", file=sys.stderr)
