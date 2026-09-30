"""Solvability-filtered ablation analysis (thesis; launched 2026-09-03).

Question: does restricting the level generator to solvable maps
(valid_path_check=True) close the gap between the prior-guided arms and
standard SFL on JaxNav?

Usage:  python solv_ablation_analysis.py        # writes solv_ablation_analysis_output.txt

Data: W&B project sfl-jaxnav-campaign-fixed.
  Filtered arms   : groups wave1fix-solv-{standard,cnn,hybrid_linear},
                    runs solv_<arm>_seed{1..10}  (valid_path_check=True asserted)
  Unfiltered ref  : wave1-standard/standard_seedN, wave1fix-cnn/cnn_fixed_seedN,
                    wave1fix-hybrid_linear/hybrid_linear_fixed_seedN
                    (valid_path_check=False asserted)
Dedup: newest FINISHED run with the highest _step per run name (the crashed
partial solv_standard_seed10 from the suspended Slurm job is thereby skipped).

Primary endpoint: tail-5 mean of the frozen 100-level eval set win rate
('sampled-test-metrics.eval-sampled/overall_win_rate'; the eval set is a
fixed pkl, identical for filtered and unfiltered runs). Secondary: singleton
suite; early windows (first 5 / first 10 evals). Paired exact Wilcoxon over
10 seeds WITHIN the filtered generator (cnn and hybrid_linear vs standard),
Holm over these two primary comparisons. Filtered-vs-unfiltered contrasts
are reported descriptively with per-seed deltas; note that seeds 1-2 of the
filtered arms ran on RTX 4090 pods and seeds 3-10 on cluster nodes, while
hardware is matched WITHIN each seed across the three filtered arms.
"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon

ENTITY  = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJECT = "sfl-jaxnav-campaign-fixed"
SEEDS   = list(range(1, 11))
SPECS = {  # label -> (group, name_template, expected LEARN_METHOD, expect_filtered)
  "solv_standard":      ("wave1fix-solv-standard",      "solv_standard_seed{}",      "standard",      True),
  "solv_cnn":           ("wave1fix-solv-cnn",           "solv_cnn_seed{}",           "cnn",           True),
  "solv_hybrid_linear": ("wave1fix-solv-hybrid_linear", "solv_hybrid_linear_seed{}", "hybrid_linear", True),
  "standard":           ("wave1-standard",              "standard_seed{}",           "standard",      False),
  "cnn":                ("wave1fix-cnn",                "cnn_fixed_seed{}",          "cnn",           False),
  "hybrid_linear":      ("wave1fix-hybrid_linear",      "hybrid_linear_fixed_seed{}","hybrid_linear", False),
}
K_MAIN = "sampled-test-metrics.eval-sampled/overall_win_rate"
K_SING = "singleton-test-metrics.eval/:overall_win_rate"
HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "solv_ablation_raw.json")
OUT   = os.path.join(HERE, "solv_ablation_analysis_output.txt")

def fetch():
    import wandb
    api = wandb.Api(timeout=120)
    data = {}
    for label, (group, tmpl, method, filtered) in SPECS.items():
        runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group}))
        for seed in SEEDS:
            name = tmpl.format(seed)
            cand = [r for r in runs if r.name == name and r.state == "finished"]
            assert cand, f"no finished run {name} in {group}"
            r = max(cand, key=lambda r: (r.summary.get("_step") or 0, str(r.created_at)))
            c = r.config; mp = c["env"]["env_params"]["map_params"]
            assert c["LEARN_METHOD"] == method, (name, c["LEARN_METHOD"])
            assert int(c["SEED"]) == seed, (name, c["SEED"])
            assert bool(mp["valid_path_check"]) == filtered, (name, mp["valid_path_check"])
            assert mp["max_fill"] == 0.6, (name, mp["max_fill"])
            assert int(float(c["learning"]["TOTAL_TIMESTEPS"])) == 300000000, name
            ser = {}
            for k in (K_MAIN, K_SING):  # separate pulls: keys log at different cadences
                rows = list(r.history(keys=[k], pandas=False, samples=100000))
                ser[k] = [row[k] for row in rows]
                assert ser[k] and not any(v is None for v in ser[k]), (name, k)
            data[f"{label}/{seed}"] = {"run_id": r.id, "series": ser}
            print(f"fetched {name}: main={len(ser[K_MAIN])} sing={len(ser[K_SING])} ({r.id})", file=sys.stderr)
    return data

if os.path.exists(CACHE):
    data = json.load(open(CACHE)); print(f"using cache {CACHE}", file=sys.stderr)
else:
    data = fetch(); json.dump(data, open(CACHE, "w"))

for k in (K_MAIN, K_SING):
    L = {key: len(v["series"][k]) for key, v in data.items()}
    assert len(set(L.values())) == 1, f"unequal series lengths for {k}: {L}"
N_EVAL = len(next(iter(data.values()))["series"][K_MAIN])

def win(label, key, sl):
    return np.array([np.mean(np.asarray(data[f"{label}/{s}"]["series"][key], float)[sl]) for s in SEEDS])

def pair(a, b):
    d = a - b; nz = d[d != 0]
    p = wilcoxon(nz, method="exact", alternative="two-sided").pvalue if len(nz) else 1.0
    return np.median(d), int((d > 0).sum()), p

lines = [f"Solvability-filtered ablation (JaxNav, valid_path_check=True), {N_EVAL} evals/run",
         "run ids: " + ", ".join(f"{k}:{v['run_id']}" for k, v in sorted(data.items())), ""]
for key, kname, windows in [
    (K_MAIN, "frozen 100-level eval set win rate",
     [("tail-5", slice(-5, None)), ("first-5", slice(0, 5)), ("first-10", slice(0, 10))]),
    (K_SING, "singleton suite overall win rate", [("tail-5", slice(-5, None))])]:
    lines.append(f"== {kname} ==")
    for wname, sl in windows:
        lines.append(f"  [{wname}]")
        V = {lab: win(lab, key, sl) for lab in SPECS}
        for lab in SPECS:
            v = V[lab]
            lines.append(f"    {lab:20s} {v.mean():.3f} +- {v.std(ddof=1):.3f}   per-seed: "
                         + " ".join(f"{x:.3f}" for x in v))
        lines.append("    -- paired WITHIN the filtered generator (10 seeds, exact Wilcoxon):")
        raw = {}
        for lab in ("solv_cnn", "solv_hybrid_linear"):
            m, w, p = pair(V[lab], V["solv_standard"]); raw[lab] = p
            lines.append(f"    {lab:20s} vs solv_standard: delta_median={m:+.4f} wins={w}/10 p_raw={p:.4f}")
        if wname == "tail-5" and key == K_MAIN:
            order = sorted(raw, key=raw.get)
            holm, prev = {}, 0.0
            for i, lab in enumerate(order):
                prev = max(prev, min(1.0, (2 - i) * raw[lab])); holm[lab] = prev
            for lab in ("solv_cnn", "solv_hybrid_linear"):
                lines.append(f"    {lab:20s} Holm-adjusted (m=2): p={holm[lab]:.4f}")
        lines.append("    -- filtered vs unfiltered, same arm (DESCRIPTIVE; treatment=generator; seeds matched, hardware differs for seeds 1-2):")
        for a, b in (("solv_standard", "standard"), ("solv_cnn", "cnn"), ("solv_hybrid_linear", "hybrid_linear")):
            m, w, p = pair(V[a], V[b])
            lines.append(f"    {a:20s} vs {b}: delta_median={m:+.4f} wins={w}/10 (p_raw={p:.4f}, descriptive)")
    lines.append("")
report = "\n".join(lines)
open(OUT, "w").write(report + "\n")
print(report)
print(f"\nwritten: {OUT}", file=sys.stderr)
