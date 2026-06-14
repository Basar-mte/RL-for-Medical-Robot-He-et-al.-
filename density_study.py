"""
density_study.py
================
Standalone analogue of Table 2 / Fig 4 from He et al. (2025).

For each crowd density in config.DENSITY_GRADIENT we:
  1. Place n_humans pedestrians uniformly in a 12x12 m arena.
  2. Compute C(sigma) and the reward terms along a straight robot path
     from start to goal (10 m, 40 steps).
  3. Record per-step metrics, then aggregate SR-proxy, mean r, mean C,
     ITR-proxy (fraction of steps in confined zone).

This is a simulation of the REWARD FIELD, not a trained policy.
The trends (collision risk, ITR) should agree with the paper's Table 2 direction.

Writes:
  results/density_study.txt
  results/density_curves.png
"""

import numpy as np
import matplotlib.pyplot as plt
from adaptive_reward import adaptive_reward
from risk_field import scene_complexity
from config import DENSITY_GRADIENT, RISK_SIGMA, DECAY_LAMBDA, DISCOMFORT_DIST

ROBOT_RADIUS = 0.3
ARENA = 12.0
N_TRIALS = 20        # random trials per density
STEPS = 40           # steps per trial (straight-line path, 10 m at 0.25 m/step)
SIGMA = RISK_SIGMA
LAMBDA = DECAY_LAMBDA
RNG = np.random.default_rng(0)


def run_trial(n_humans, trial_seed):
    rng = np.random.default_rng(trial_seed)
    # place humans randomly in the arena
    human_pos = rng.uniform(-ARENA/2, ARENA/2, (n_humans, 2))
    human_vel = rng.uniform(0.5, 1.5, (n_humans, 2)) * rng.choice([-1, 1], (n_humans, 2))

    # straight-line robot path: start=(-5,0), goal=(5,0)
    start = np.array([-5.0, 0.0])
    goal  = np.array([ 5.0, 0.0])
    path  = [start + t * (goal - start) / STEPS for t in range(STEPS + 1)]

    rewards, C_vals, confined_steps = [], [], 0
    d_goal_prev = np.linalg.norm(start - goal)
    collision = False

    for i, pos in enumerate(path[:-1]):
        r, info = adaptive_reward(
            robot_pos=pos, robot_vel=[0.25, 0],
            robot_radius=ROBOT_RADIUS, goal_pos=goal,
            human_pos=human_pos, human_vel=human_vel,
            d_goal_prev=d_goal_prev, sigma=SIGMA, lam=LAMBDA)
        rewards.append(r)
        C_vals.append(info["C"])
        d_goal_prev = info["d_goal"]
        if info["region"] == "confined":
            confined_steps += 1
        if info["region"] == "collision":
            collision = True
            break

    success = not collision and d_goal_prev <= ROBOT_RADIUS
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_C": float(np.mean(C_vals)),
        "itr_proxy": 100.0 * confined_steps / max(len(rewards), 1),
        "collision": collision,
        "success": success,
    }


# ----- run ------------------------------------------------------------------
density_rows = []
header = f"{'density':>9} {'n_humans':>9} {'SR_proxy%':>10} {'mean_r':>8} {'mean_C':>8} {'ITR_proxy%':>11}"
rows = [header, "-" * len(header)]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax_sr, ax_r, ax_C, ax_itr = axes.flat
density_labels = []
sr_vals, r_vals, C_vals_all, itr_vals = [], [], [], []

for rho, n_h in sorted(DENSITY_GRADIENT.items()):
    trials = [run_trial(n_h, seed) for seed in range(N_TRIALS)]
    sr  = 100.0 * sum(t["success"] for t in trials) / N_TRIALS
    mr  = float(np.mean([t["mean_reward"] for t in trials]))
    mC  = float(np.mean([t["mean_C"]      for t in trials]))
    itr = float(np.mean([t["itr_proxy"]   for t in trials]))

    rows.append(f"{rho:>9.2f} {n_h:>9d} {sr:>10.1f} {mr:>8.3f} {mC:>8.3f} {itr:>11.2f}")
    density_labels.append(f"rho={rho}")
    sr_vals.append(sr); r_vals.append(mr); C_vals_all.append(mC); itr_vals.append(itr)

rows += [
    "",
    "Note: SR_proxy = fraction of straight-line trials reaching goal without collision.",
    "      ITR_proxy = fraction of steps inside personal-space zone (discomfort_dist=0.25m).",
    f"      sigma={SIGMA}, lambda={LAMBDA} (paper training values)",
    "      This is reward-field simulation, not trained-policy evaluation.",
]

output = "\n".join(rows)
print(output)

with open("results/density_study.txt", "w", encoding="utf-8") as f:
    f.write(output + "\n")
print("\nSaved results/density_study.txt")

# plot
for ax, vals, ylabel, title in [
    (ax_sr,  sr_vals,      "SR proxy (%)",       "Success rate vs density"),
    (ax_r,   r_vals,       "Mean reward",         "Mean reward vs density"),
    (ax_C,   C_vals_all,   "Mean C(sigma)",       "Scene complexity vs density"),
    (ax_itr, itr_vals,     "ITR proxy (%)",       "Intrusion-time ratio vs density"),
]:
    ax.plot(range(len(density_labels)), vals, "o-", lw=2)
    ax.set_xticks(range(len(density_labels)))
    ax.set_xticklabels(density_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Density study (Table 2 analogue) — reward-field simulation\n"
             f"sigma={SIGMA}, lambda={LAMBDA}, {N_TRIALS} trials/density, straight-line path",
             fontsize=10)
fig.tight_layout()
fig.savefig("results/density_curves.png", dpi=150)
print("Saved results/density_curves.png")
plt.close(fig)
