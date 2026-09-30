"""Pull the full W&B histories for the frontier wave (project sfl-jaxnav-frontier) and the
seed-matched reference runs of the old campaign (sfl-jaxnav-campaign-fixed) into
frontier_analysis_raw.json. Re-run to refresh (running runs are re-fetched).
Usage: python frontier_analysis_pull.py [--out FILE]"""
import json, sys, time
import wandb

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "frontier_analysis_raw.json"
K_MAIN = "sampled-test-metrics.eval-sampled/overall_win_rate"
K_SING = "singleton-test-metrics.eval/:overall_win_rate"
CYC_KEYS = [K_MAIN, K_SING, "target_mu", "recent_success_rate", "buffer_success_rate", "generated_success_rate",
            "buffer_success_term_weighted", "frontier/mu_star", "frontier/mu_used", "frontier/use_frontier",
            "frontier/n_informative_bins", "frontier/selected_frac_intermediate", "solvability/selected_mean",
            "solvability/batch_mean", "sfl/selected_mean", "sfl/batch_mean", "cnn/selected_mean", "cnn/selected_std",
            "cnn/tracking_error", "cnn/member_rank_agreement", "learnability_set_mean_score", "worst_learnability_mean_score"]
UPD_KEYS = ["env-metrics/.passable", "train-buffer/.success_env_weighted", "train-generated/.success_env_weighted",
            "train-buffer/.success_term_weighted", "train-term.GoalR", "train-term.TimeO", "train-term.MapC", "train-term.NumC"]
# (project, run-name regex prefix list, label) — old project references, seeds 1-4 only
SPECS = [("sfl-jaxnav-frontier", None, "new"),
         ("sfl-jaxnav-campaign-fixed", ["standard_seed", "hybrid_linear_fixed_seed", "solv_standard_seed", "solv_hybrid_linear_seed"], "old")]

def fetch(run):
    rows = list(run.history(samples=500000, pandas=False))
    out = {"cycle": {k: {} for k in CYC_KEYS}, "upd": {k: {} for k in UPD_KEYS}}
    for x in rows:
        u = x.get("update_count")
        if u is None:
            continue
        u = int(u)
        for k in CYC_KEYS:
            v = x.get(k)
            if v is not None and not isinstance(v, dict):
                out["cycle"][k][u] = float(v)
        for k in UPD_KEYS:
            v = x.get(k)
            if v is not None and not isinstance(v, dict):
                out["upd"][k][u] = float(v)
    return out, len(rows)

api = wandb.Api(timeout=180)
raw = {}
for proj, prefixes, label in SPECS:
    runs = list(api.runs(f"{ENT}/{proj}", per_page=500))
    sel = []
    for r in runs:
        if r.name.startswith("smoke"):
            continue
        if prefixes is not None:
            if not any(r.name.startswith(p) and r.name[len(p):].isdigit() and 1 <= int(r.name[len(p):]) <= 4 for p in prefixes):
                continue
        sel.append(r)
    # dedup by name: prefer the highest update_count, then newest
    by = {}
    for r in sel:
        key = f"{label}:{r.name}"
        uc = r.summary.get("update_count") or 0
        prev = by.get(key)
        if prev is None or (uc, r.created_at) > (prev[1], prev[0].created_at):
            by[key] = (r, uc)
    for key, (r, uc) in sorted(by.items()):
        t0 = time.time()
        hist, n = fetch(r)
        cfg = r.config
        mp = ((cfg.get("env") or {}).get("env_params") or {}).get("map_params") or {}
        raw[key] = {"id": r.id, "name": r.name, "project": proj, "state": r.state, "created": r.created_at, "group": r.group,
                    "config": {"LEARN_METHOD": cfg.get("LEARN_METHOD"), "CURRICULUM_STRATEGY": cfg.get("CURRICULUM_STRATEGY"),
                               "MU_MEASUREMENT": cfg.get("MU_MEASUREMENT"), "FRONTIER_MODE": cfg.get("FRONTIER_MODE"),
                               "SEED": cfg.get("SEED"), "valid_path_check": mp.get("valid_path_check"),
                               "TRAINER_VARIANT": cfg.get("TRAINER_VARIANT")},
                    "update_count": uc, **hist}
        print(f"{key:<40} {r.state:<9} upd={uc:>5} rows={n:>5} cycles={len(hist['cycle']['target_mu']):>3} ({time.time()-t0:.1f}s)", flush=True)
json.dump(raw, open(OUT, "w"))
print("saved", OUT, len(raw), "runs")
