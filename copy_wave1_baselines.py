"""Copy the Wave-1 dr/standard runs into the fixed-scorer project.

Those 20 runs never call the CNN, so the scorer-domain fix cannot change
their behaviour — they are reused verbatim as the paired baseline for the
rerun CNN arms (same seeds 1-10). Copying them into the new project lets
every arm be viewed and analysed together.

Scalar history is replayed row-by-row keyed by the original _step; media
(map panels) and histograms are not copied. Each copy is tagged
`copied-from-wave1` and its config records the source run id, so provenance
is auditable and copies can never be mistaken for fresh runs.
"""
import sys
import numpy as np
import wandb

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
SRC = "sfl-jaxnav-campaign"
DST = "sfl-jaxnav-campaign-fixed"
SKIP_SUBSTR = ("maps", "buffer_sample", "curriculum", "hist/", "cnn_input_check")

api = wandb.Api()
todo = [f"{m}_seed{s}" for m in ("dr", "standard") for s in range(1, 11)]

# skip anything already copied (idempotent — safe to re-run)
try:
    done = {r.name for r in api.runs(f"{ENT}/{DST}")}
except Exception:
    done = set()
todo = [n for n in todo if n not in done]
print(f"copying {len(todo)} runs into {DST} (skipping {len(done)} already present)")

for name in todo:
    src = list(api.runs(f"{ENT}/{SRC}", {"display_name": name}))
    assert len(src) == 1, f"{name}: expected 1 source run, found {len(src)}"
    src = src[0]
    hist = src.history(pandas=False, samples=100000)
    hist.sort(key=lambda x: x.get("_step", 0))

    cfg = {k: v for k, v in src.config.items() if not k.startswith("_")}
    cfg["COPIED_FROM_RUN_ID"] = src.id
    cfg["COPIED_FROM_PROJECT"] = SRC
    cfg["COPY_NOTE"] = ("verbatim copy of the Wave-1 run; this arm never calls the "
                        "CNN so the scorer input-domain fix does not affect it")

    run = wandb.init(entity=ENT, project=DST, name=name, group=src.group,
                     config=cfg, tags=list(src.tags) + ["copied-from-wave1"],
                     reinit=True, settings=wandb.Settings(silent=True))
    run.define_metric("update_count")
    run.define_metric("*", step_metric="update_count")

    n = 0
    for row in hist:
        payload = {k: v for k, v in row.items()
                   if not k.startswith("_")
                   and not any(s in k for s in SKIP_SUBSTR)
                   and isinstance(v, (int, float, np.integer, np.floating))
                   and v is not None and np.isfinite(v)}
        if payload:
            run.log(payload, step=int(row["_step"]))
            n += 1
    run.summary["COPY_ROWS_REPLAYED"] = n
    run.finish()
    print(f"  {name}: replayed {n}/{len(hist)} rows (from {src.id})")

print("done")
