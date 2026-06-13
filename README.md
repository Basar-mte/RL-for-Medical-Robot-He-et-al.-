# Reproducing He et al. (2025) — Adaptive Reward Optimization for Robot Crowd Navigation

*Computers, Materials & Continua 84(2):2733–2749 — DOI 10.32604/cmc.2025.065205*

This is a step-by-step reproduction of the paper you uploaded (your "main competitor").
The key thing to understand before writing any code:

> **The paper does not introduce a new simulator, a new network, or a new RL algorithm.
> Its entire contribution is two reward equations layered on an existing codebase.**

Everything else — the CrowdSim environment, the GST trajectory predictor, the
attention-graph policy, and PPO — is the **`Shuijing725/CrowdNav_Prediction_AttnGraph`**
repo (ICRA 2023, the "GST + HH Attn" baseline). TGRF (the other baseline in their tables)
is `JinnnK/TGRF`, which forks the same base. So "remaking" this paper means:

1. Stand up the base repo.
2. Replace its reward function with the paper's Eqs. (2)–(7).
3. Train with their hyperparameters.
4. Evaluate with their five metrics across the density gradient.

The files in this folder implement step 2 (the actual novelty) and step 4, validated
against the paper's own figures.

---

## What's in this folder

| File | Paper section | What it is |
|---|---|---|
| `risk_field.py` | §3.1, Eq. 2 | Gaussian-kernel scene-complexity `C(σ)` + helpers to redraw Figs 2 & 3 |
| `adaptive_reward.py` | §3.2, Eqs. 3–7 | The reward terms `r_col`, `r_disc`, `r_pred`, `r_pot` and the full `r(s,a)` |
| `adaptive_reward_patch.py` | §3.2 | Drop-in `calc_reward_adaptive(self, action)` for the base repo's env |
| `metrics.py` | §4.2 | SR, NT, PL, ITR, SD — fills Tables 1–3 |
| `config.py` | §4.1 | Every hyperparameter the paper reports |
| `validate_figs.py` | §3.1 | Regenerates Figs 2 & 3 to prove the math is faithful |

---

## The five equations, mapped to code

```
Eq.2  C(σ)   = Σ_i v_i · exp(−d_i² / 2σ²)              risk_field.scene_complexity()
Eq.3  r_col  = −10 · C(σ)                              adaptive_reward.r_col()
Eq.4  r_disc =  C(σ)/2 · exp(1 − d_min·λ)              adaptive_reward.r_disc()
Eq.5  r_pred = min_i min_k ( 1{collide} · r_col/2^k )  adaptive_reward.r_pred()
Eq.6  r_pot  = d_goal^{t+1} − d_goal^{t}               adaptive_reward.r_pot()
Eq.7  piecewise by region {goal, collision, confined, free}   adaptive_reward.adaptive_reward()
```

**One ambiguity to flag in your write-up:** Eq. (4) is typeset as a *positive*
quantity, but the text (p.5, p.7) says the danger-zone term must assign
"exponentially increasing **negative** rewards" / "be **punished**". A positive
`r_disc` would reward approaching humans, which contradicts the design. This
reproduction applies it as a **penalty** by default (`DISC_AS_PENALTY=True`).
Note this discrepancy — it is the kind of thing reviewers and competitors look for.

---

## Step 0 — Validate the math (no GPU needed)

```bash
python3 validate_figs.py        # writes fig2_repro.png, fig3_repro.png
python3 risk_field.py           # C(σ) grows with σ
python3 adaptive_reward.py      # dense scene → big penalty; sparse → mild
python3 metrics.py              # SR/NT/PL/ITR/SD aggregation
```

Confirmed against the paper:
- **Fig 2**: effective risk radius ≈ 1.2 m (σ=0.5), ≈ 2.5 m (σ=1.0), >6 m (σ=3.0)
  — matches their "within 1 m / 1–3 m / >3 m" description.
- **Fig 3**: low-density peak ≈ 0.50 (discrete peaks); high-density peak ≈ 1.69
  — the two adjacent pedestrians superpose into one hotspot (their "crowd effect").

## Step 1 — Stand up the base repo

```bash
git clone https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph
cd CrowdNav_Prediction_AttnGraph
# conda env, Python 3.6–3.8 per their README; install requirements;
# download/point to the pretrained GST predictor as documented there.
```
Run their default training once unmodified so you know the pipeline works
(this reproduces the plain `GST + HH Attn` baseline rows in Table 1).

## Step 2 — Splice in the adaptive reward

Copy `risk_field.py` and `adaptive_reward_patch.py` next to
`crowd_sim/envs/crowd_sim.py`. In the env's `__init__`/`configure`, add:

```python
self.risk_sigma   = 8.0     # Eq.2  (config.RISK_SIGMA)
self.decay_lambda = 0.1     # Eq.4  (config.DECAY_LAMBDA)
self.fov_radius   = 5.0
self.prev_d_goal  = None    # set to initial goal distance on reset()
self.use_adaptive_reward = True
```

Then, in `step()`, replace the existing reward block (the part that computes
`dmin`, checks collision / reaching_goal / discomfort, and adds `r_pred`) with:

```python
from adaptive_reward_patch import calc_reward_adaptive
reward, done, info_tag = calc_reward_adaptive(self, action,
                                              danger_collide_flags=pred_flags)
```

`pred_flags` is the `(n_humans, K)` predicted-collision indicator the pred-env
already has the data to build (feed it for the full Eq. 7; pass `None` to drop
`r_pred` to 0). On `reset()`, initialise `self.prev_d_goal` to the start-to-goal
distance so Eq. 6 is well-defined on the first step.

## Step 3 — Train (paper Sec 4.1)

Set in the base repo's PPO/trainer config (see `config.py`):

```
γ=0.99   lr=4e-5   clip=0.2   16 parallel envs   ~20,820 iterations
12 m × 12 m arena, 360° FoV, 5 m lidar, robot v_max=1 m/s, r=0.3 m
humans: ORCA, r∈[0.3,0.5] m, v∈[0.5,1.5] m/s, invisible-to-robot
train density ρ=0.15  (20 pedestrians)
```

## Step 4 — Evaluate (Sec 4.2–4.3)

Run 500 fixed-seed test episodes. Wrap each episode in `metrics.EpisodeRecorder`
(call `.step(robot_pos, min_surface_gap)` each timestep, `.finish(outcome)` at the
end), then `metrics.summarize(records)` to get the table row.

Reproduce **Table 2** by re-running evaluation across the density gradient in
`config.DENSITY_GRADIENT` (10→30 pedestrians, ρ = 0.07→0.21). The headline claims
to reproduce: **+9.0% SR and −10.7% ITR vs TGRF at ρ=0.21**.

## Step 5 — Hyperparameter study (Table 3 / Figs 5–6)

Sweep `σ ∈ {2,6,8,10} × λ ∈ {0.005,0.05,0.1,0.2}` (16 runs). Expected from the
paper: `σ=6, λ=0.1` is the sweet spot (SR≈94%); extremes like `σ=10, λ=0.2`
collapse to SR≈7%. Recommended per-density settings are in `config.TUNING_GUIDE`.

---

## If you're "remaking" it to *beat* it (since it's your competitor)

The reproduction also exposes the paper's soft spots, which are natural angles
for a stronger follow-up:

- **2D-only, ORCA-only crowds.** They admit (§5) the sim oversimplifies occlusion
  and sensor noise. Validating in Isaac Sim with real sensor models / 3D occlusion
  would be a clean differentiator — and is squarely in your wheelhouse.
- **`C(σ)` ignores heading.** It uses speed magnitude only; a pedestrian walking
  *away* is weighted the same as one walking *at* the robot. Adding a relative-
  velocity / time-to-collision term is a low-risk, high-payoff extension.
- **Quadratic cost they complain about, they don't actually fix.** Eq. 2 is still
  an O(n) sum per step with a costly `exp`; they criticise pairwise cost but their
  own field is a vectorizable kernel sum (see `risk_field.py`) — easy to argue and
  benchmark.
- **The Eq. 4 sign issue** above is a genuine reproducibility gap worth citing.

I can help build any of these out next, or wire the patch into a clone you push.
