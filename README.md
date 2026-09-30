# Learned Difficulty Scorers for Curriculum Discovery

Bachelor's thesis, Malik Durmus, Ludwig-Maximilians-Universität München.

This repository holds the code for my thesis on curriculum learning for reinforcement
learning agents. It extends the Sampling For Learnability (SFL) codebase of Rutherford
et al. (NeurIPS 2024) with level selection driven by a learned difficulty model, a
second training domain, and the tooling used to run the experiments. What came from the
original work is listed under [Built on](#built-on).

<p align="center">
  <img src="./docs/images/jaxnav-sa.gif" alt="JaxNav" width="30%">
  <img src="./docs/images/xland.gif" alt="XLand-MiniGrid" width="30%">
</p>
<p align="center"><sub>The two domains used here: JaxNav 2D navigation (left) and
XLand-MiniGrid (right).</sub></p>

## Background

SFL picks training levels by learnability, `p(1-p)`, where `p` is the agent's success
rate on a level. Getting `p` means rolling out the current policy over thousands of
candidate levels at every refresh cycle, which is expensive. Early in training it is
also uninformative, because the agent fails almost everything and the estimates are
mostly binomial noise.

My thesis asks whether a difficulty model trained offline can supply that signal
instead. The models are ResNets trained on pairwise difficulty comparisons in a
separate repository, and are used here as frozen ensembles that score rendered levels
before any rollout happens. Alongside them I added selection methods that keep SFL's
agent-relative idea but measure something denser than a binary win or loss.

## What this adds

### Level selection methods

All of these sit behind one config key, `LEARN_METHOD`:

| Method | What drives selection | Uses a difficulty model |
|---|---|---|
| `standard` | measured `p(1-p)`, the SFL baseline | no |
| `dr` / `random` | uniform sampling | no |
| `cnn` | predicted difficulty against a target difficulty `mu` | yes |
| `hybrid_linear` | `4*sfl + cnn_proximity` | yes |
| `hybrid_soft_handoff` | blend from SFL to the model as training proceeds | yes |
| `hybrid_learnability_weighted` | model score weighted by inverse SFL | yes |
| `hybrid_multiplicative` | Gaussian-filtered product of both | yes |
| `progress` | variance over episodes of `clip(1 - d/d_start, 0, 1)`, using the distance to the goal at the end of the episode | no |
| `progress_mindist` | same, but using the closest the agent ever got | no |
| `progress_mean` | `mp*(1-mp)`, which targets levels the agent half finishes | no |
| `dijkstra` | shortest path length over the occupancy grid | no |

### Serving the difficulty models

`sfl/train/rlhf_utils.py` loads five independently trained networks per domain and
combines them by averaging their scalar logits, never their weights. The spread between
members is kept as an uncertainty signal and logged, together with the rank agreement
between them. Loading a single network is the same code path with K=1.

### Score normalisation

The models are trained on pairwise comparisons, so they learn a ranking and their logit
offset is arbitrary. Squashing those logits with a sigmoid pushes every level into one
corner of the range, which makes the target difficulty meaningless. Selection therefore
converts each score into its rank inside a fixed 10,000 level sample drawn from the same
generator. That reference is built once and shared across runs and seeds so scores stay
comparable. Sigmoid and min-max normalisation are still available as config options for
comparison. `score_normalization_comparison.py` and `percentile_score_distribution.py`
produce the figures behind this.

### Target difficulty controllers

`sfl/train/jaxnav_sfl_frontier.py` replaces the preset `mu` ramp with three
alternatives. Two are step rules, one driven by the success rate over all training
environments and one by the success rate on the selected levels only. The third bins the
candidate levels by predicted difficulty, fits an isotonic success curve over those
bins, and puts `mu` where the curve crosses a target success rate. While too few bins
carry any signal, which is the situation at the start of training, it falls back to the
step rule. The curve is computed and logged on every run, so the diagnostic is available
whichever controller is active.

### Second domain

`sfl/train/xland_sfl.py` ports the methods to XLand-MiniGrid. The ruleset is a config
key, with `difficult` as the main setting and `medium` as a comparison, each using the
difficulty model trained on that ruleset.

### Robustness evaluation

`cvar_eval/` implements the CVaR protocol from the paper over a frozen set of 10 by 1000
solvable levels. Levels are ranked using one rollout seed and scored using a second,
which is the bias correction the original authors use. It runs in three stages: generate
the level set once, roll out twice, then analyse. Rollouts skip levels that already have
a CSV, so a killed job can be resubmitted.

### Experiment infrastructure

SLURM launchers for each wave of runs, with drip feeding so a shared partition is not
monopolised, requeue guards for node reboots, and VRAM pinning for the methods that need
a large GPU. `pod_*.sh` and `provision_*_pod.sh` set up rented cloud instances for the
runs the cluster could not host. Roughly 200 training runs went through these.

## Running it

Build the environment with `rebuild_venv.sh`. The lockfiles on their own produce a
broken environment, because two files inside `site-packages/jaxmarl` need patching after
install; the script does that.

```bash
# one JaxNav run
python -m sfl.train.jaxnav_sfl --config-name jaxnav-sfl LEARN_METHOD=hybrid_linear SEED=1

# one XLand run on the difficult ruleset
python -m sfl.train.xland_sfl --config-name xland-sfl LEARN_METHOD=cnn SEED=1

# a full wave on SLURM
./launch_wave1fix.sh

# CVaR evaluation
python cvar_eval/cvar_0_generate_levels.py
sbatch cvar_eval/launch_cvar.sbatch
python cvar_eval/cvar_2_analyse.py
```

The difficulty model checkpoints are not in this repository. They are several GB of
Orbax checkpoints from a separate training repo, and `CNN_CHECKPOINT_PATHS` in the
configs points at them with absolute local paths. Any method that needs a difficulty
model therefore will not run elsewhere without editing those paths. `standard`, `dr`,
the `progress` variants and `dijkstra` need no model and run as they are.

## Repository layout

Scripts sit at the top level, grouped here by what they do.

| | |
|---|---|
| Training | `sfl/train/jaxnav_sfl.py` (main trainer), `jaxnav_sfl_frontier.py` (target difficulty controllers), `xland_sfl.py` (XLand), `rlhf_utils.py` (difficulty models), configs under `sfl/train/config/` |
| Evaluation | `cvar_eval/`, plus the 10k level result CSVs in `sfl/data/eval/results/` |
| Analysis | `wave1fix_analysis.py`, `newarms_analysis.py`, `frontier_analysis.py`, `xland_analysis.py`, `solv_ablation_analysis.py`, `coldstart_*`, `*_score_distribution.py` |
| Checks | `verify_progress_arms.py`, `verify_dijkstra_gates.py`, `verify_inversion_integrated.py`, `test_inversion_*.py`, `xrender_*` (renderer comparison), `member_structure_diagnostics.py` |
| Infrastructure | `launch_*.sh`, `*_dripfeed.sh`, `pod_*.sh`, `provision_*_pod.sh`, `rebuild_venv.sh` |

Each analysis script commits the `*_output.txt` it printed, and where it pulled numbers
from Weights and Biases it commits the `*_raw.json` it pulled. Anything reported can
be recomputed from the repository without a network connection or a live W&B project.

## Built on

This codebase started from the reference implementation of *No Regrets: Investigating
and Improving Regret Approximations for Curriculum Discovery* by Rutherford, Beukman,
Willi, Lacerda, Hawes and Foerster (NeurIPS 2024):
[paper](https://arxiv.org/abs/2408.15099),
[original repository](https://github.com/amacrutherford/sampling-for-learnability),
Apache-2.0, kept in [LICENSE](LICENSE).

Still used here from that work: SFL itself, which is the baseline everything is measured
against, the PLR, Robust PLR, ACCEL and DR baselines, and the environments. JaxNav comes
from [JaxMARL](https://github.com/FLAIROx/JaxMARL), MiniGrid from
[JaxUED](https://github.com/DramaCow/jaxued), and
[XLand-MiniGrid](https://github.com/corl-team/xland-minigrid) from its own repository.
XLand needs a different JAX version, so its code stays separate under `xland/` with its
own Dockerfile. `sfl_guide.md` is the original walkthrough of the JaxNav trainer.

If you use SFL or JaxNav, cite the original authors:

```bibtex
@inproceedings{rutherford2024noregrets,
    title={No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery},
    author={Alexander Rutherford and Michael Beukman and Timon Willi and Bruno Lacerda and Nick Hawes and Jakob Nicolaus Foerster},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://arxiv.org/abs/2408.15099}
}
```
