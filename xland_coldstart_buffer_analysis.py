"""XLand MEDIUM cold start: does the difficulty prior's easy band concentrate solvable levels
while the rollouts are 'uninformative'? (Malik, 2026-09-12; follow-up to the frontier wave.)

Data: W&B project xland-sfl-campaign, medium-ruleset runs standard / hybrid_linear /
hybrid_soft_handoff, seeds 1-4 (run ids from xland_medium_analysis_output.txt).
Per cycle the loop logs the mean p(1-p) and mean solvability of the 5000 candidates and of the
1000 selected levels (standard logs only the selected p(1-p), as learnability_set_mean_score).
Enrichment = selected mean / candidate mean; 5.0 means every level carrying the quantity was
admitted (1000 of 5000 slots). For the hybrid, solvability enrichment > p(1-p) enrichment would
mean the CNN-chosen filler holds solvable levels beyond those the rollouts identified.
Usage: python xland_coldstart_buffer_analysis.py -> xland_coldstart_buffer_analysis_output.txt
"""
import os, numpy as np, wandb
ENT = "malikdurmus-ludwig-maximilian-university-of-munich"; PROJ = "xland-sfl-campaign"
IDS = {"standard": ["gxc6234y", "4d4gd5v9", "8liiejhd", "hym2l37s"],
       "hybrid_linear": ["x4ct2ony", "jq6ueo0l", "lwq1hj4b", "ti4scxd7"],
       "hybrid_soft_handoff": ["yjtkl0xh", "yxk3im34", "v47r9q1j", "cyfiapc2"]}
KEYS = ["solvability/selected_mean", "solvability/batch_mean", "sfl/selected_mean", "sfl/batch_mean", "cnn/selected_mean",
        "recent_success_rate", "target_mu", "hybrid/alpha_soft_handoff", "learnability_set_mean_score", "eval_fixed/success_rate_mean"]
PHASES = [("c0-10", 0, 11), ("c11-25", 11, 26), ("c26-50", 26, 51), ("c51-75", 51, 76), ("c76-100", 76, 101), ("c101-366", 101, 366)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xland_coldstart_buffer_analysis_output.txt")
lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
api = wandb.Api(timeout=120)
def series(run, key):
    h = run.history(keys=[key, "update_count"], pandas=False, samples=4000)
    return {int(x["update_count"]): float(x[key]) for x in h if x.get(key) is not None}
data = {}
for arm, ids in IDS.items():
    for i, rid in enumerate(ids, 1):
        r = api.run(f"{ENT}/{PROJ}/{rid}"); ser = {k: series(r, k) for k in KEYS}
        ucs = sorted(ser["eval_fixed/success_rate_mean"])
        data[f"{arm}/{i}"] = {k: np.array([ser[k].get(u, np.nan) for u in ucs], float) for k in KEYS}
p("XLand MEDIUM cold-start buffer analysis (xland-sfl-campaign, seeds 1-4; run ids:", ", ".join(f"{a}/{i}:{r}" for a, ids in IDS.items() for i, r in enumerate(ids, 1)) + ")")
p("\n== Table 1: selected buffer vs candidates, means over 4 seeds x cycles of the phase ==")
p(f"{'arm':<22}{'phase':<10}{'solv_sel':>9}{'solv_bat':>9}{'solv_enr':>9}{'sfl_sel':>8}{'sfl_bat':>8}{'sfl_enr':>8}{'cnn_sel':>8}{'mu':>6}{'alpha':>7}{'rsr':>6}{'eval':>7}")
for arm in IDS:
    for ph, lo, hi in PHASES:
        rows = []
        for i in range(1, 5):
            d = data[f"{arm}/{i}"]; g = lambda k: np.nanmean(d[k][lo:hi])
            fs = g("learnability_set_mean_score") if arm == "standard" else g("sfl/selected_mean")
            ss, sb, fb = g("solvability/selected_mean"), g("solvability/batch_mean"), g("sfl/batch_mean")
            rows.append([ss, sb, ss / sb if sb > 0 else np.nan, fs, fb, fs / fb if fb > 0 else np.nan, g("cnn/selected_mean"), g("target_mu"), g("hybrid/alpha_soft_handoff"), g("recent_success_rate"), g("eval_fixed/success_rate_mean")])
        m = np.nanmean(np.array(rows, float), axis=0)
        p(f"{arm:<22}{ph:<10}{m[0]:>9.4f}{m[1]:>9.4f}{m[2]:>9.2f}{m[3]:>8.4f}{m[4]:>8.4f}{m[5]:>8.2f}{m[6]:>8.3f}{m[7]:>6.2f}{m[8]:>7.2f}{m[9]:>6.3f}{m[10]:>7.3f}")
p("\n== Table 2: share of the learnable (p(1-p) > 0) mass admitted to the buffer, max 5.0 ==")
p(f"{'phase':<10}{'standard':>10}{'hybrid_linear':>15}{'hybrid solv_enr':>17}{'soft_handoff':>14}{'soft_handoff solv_enr':>23}")
for ph, lo, hi in PHASES:
    def enr(arm, num, den):
        v = []
        for i in range(1, 5):
            d = data[f"{arm}/{i}"]; n, m = np.nanmean(d[num][lo:hi]), np.nanmean(d[den][lo:hi]); v.append(n / m if m > 0 else np.nan)
        return np.nanmean(v)
    p(f"{ph:<10}{enr('standard', 'learnability_set_mean_score', 'sfl/batch_mean'):>10.2f}{enr('hybrid_linear', 'sfl/selected_mean', 'sfl/batch_mean'):>15.2f}{enr('hybrid_linear', 'solvability/selected_mean', 'solvability/batch_mean'):>17.2f}{enr('hybrid_soft_handoff', 'sfl/selected_mean', 'sfl/batch_mean'):>14.2f}{enr('hybrid_soft_handoff', 'solvability/selected_mean', 'solvability/batch_mean'):>23.2f}")
p("\nReading: in cycles 0-25 the hybrid buffer's solvability enrichment equals its p(1-p) enrichment, i.e. the")
p("CNN-chosen filler (mu = 0, easiest band) adds no solvable levels beyond those the rollouts identified, while")
p("the CNN term displaces rollout-identified learnable levels: standard admits 4.8 of 5.0 of the learnable mass,")
p("hybrid_linear 3.9, soft_handoff 2.6 (a zero-sfl level at CNN percentile c scores 1 - c, which outranks")
p("4 p(1-p) for every level with p outside ~[0.43, 0.57]).")
open(OUT, "w").write("\n".join(lines) + "\n"); print("saved", OUT)
