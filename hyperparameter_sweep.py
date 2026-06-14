"""
hyperparameter_sweep.py
=======================
Standalone sweep over the paper's sigma x lambda grid (Sec 4.3, Table 3 / Figs 5-6).

Because we don't have the full RL training loop here, we measure THREE
proxy signals that the paper's own analysis correlates with final SR:

  1. mean_r_confined  – mean reward when the robot is in the confined zone
                        (d_min < discomfort_dist).  Higher = safer shaping.
  2. mean_r_free      – mean reward in the free zone (goal-approach + r_pred=0).
                        Should be positive regardless of sigma/lambda.
  3. r_col_at_crowd   – collision penalty for a dense 8-human crowd (C proxy).
                        More negative = stronger deterrent.

The paper reports:
  Sweet spot : sigma=6,  lambda=0.1  -> SR ~94%
  Too tight  : sigma=2,  lambda=0.2  -> reward collapses, poor guidance
  Too loose  : sigma=10, lambda=0.2  -> SR collapses to ~7%
  Training   : sigma=8,  lambda=0.1

These are proxies, not SR — the trends match the paper's qualitative conclusions
without needing 20k PPO iterations.

Writes:
  results/hyperparameter_sweep.txt  (table)
  results/sweep_heatmaps.png        (3 heatmaps)
"""

import numpy as np
import matplotlib.pyplot as plt
from adaptive_reward import adaptive_reward, r_col
from risk_field import scene_complexity
from config import DISCOMFORT_DIST, GOAL_REWARD

# ----- sweep grid (paper Table 3) ------------------------------------------
SIGMAS  = [2, 6, 8, 10]
LAMBDAS = [0.005, 0.05, 0.1, 0.2]

# ----- fixed test scenarios -------------------------------------------------
# Confined scenario: center-to-center = 0.7 m -> surface gap = 0.7 - 0.3 - 0.3 = 0.1 m
# (inside discomfort_dist=0.25 m, but NOT a collision). lambda matters here.
CONFINED_ROBOT  = np.array([0.0, 0.0])
CONFINED_HUMAN  = np.array([[0.7, 0.0]])
CONFINED_VEL    = np.array([[0.8, 0.0]])
CONFINED_GOAL   = np.array([5.0, 0.0])

# Dense crowd scenario: 8 humans within 2m radius (collision scenario)
rng = np.random.default_rng(42)
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
DENSE_POS = 1.5 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
DENSE_VEL = rng.uniform(0.5, 1.5, (8, 2))
DENSE_GOAL = np.array([6.0, 0.0])

# Free scenario: no humans nearby
FREE_ROBOT = np.array([0.0, 0.0])
FREE_GOAL  = np.array([5.0, 0.0])

ROBOT_RADIUS = 0.3


def eval_combo(sigma, lam):
    """Return proxy metrics for one (sigma, lambda) combo."""
    # 1. confined reward
    r_conf, _ = adaptive_reward(
        robot_pos=CONFINED_ROBOT, robot_vel=[0, 0],
        robot_radius=ROBOT_RADIUS, goal_pos=CONFINED_GOAL,
        human_pos=CONFINED_HUMAN, human_vel=CONFINED_VEL,
        d_goal_prev=5.15, sigma=sigma, lam=lam)

    # 2. free reward (d_goal improving by 0.1 m)
    r_free, _ = adaptive_reward(
        robot_pos=FREE_ROBOT, robot_vel=[1, 0],
        robot_radius=ROBOT_RADIUS, goal_pos=FREE_GOAL,
        human_pos=np.zeros((0, 2)), human_vel=np.zeros((0, 2)),
        d_goal_prev=5.1, sigma=sigma, lam=lam)

    # 3. collision penalty in dense crowd
    C_dense = scene_complexity(CONFINED_ROBOT, DENSE_POS, DENSE_VEL, sigma, fov_radius=5.0)
    rc_dense = r_col(C_dense)

    return float(r_conf), float(r_free), float(rc_dense), float(C_dense)


# ----- run sweep ------------------------------------------------------------
results = {}
header = f"{'sigma':>6} {'lambda':>7} {'r_confined':>12} {'r_free':>10} {'r_col_dense':>12} {'C_dense':>10}"
rows = [header, "-" * len(header)]

for sig in SIGMAS:
    for lam in LAMBDAS:
        rc, rf, rcd, C = eval_combo(sig, lam)
        results[(sig, lam)] = (rc, rf, rcd, C)
        rows.append(f"{sig:>6} {lam:>7.3f} {rc:>12.4f} {rf:>10.4f} {rcd:>12.4f} {C:>10.4f}")
    rows.append("")

# summary
rows.append("Paper headline results (from Sec 4.3):")
rows.append("  Sweet spot : sigma=6,  lambda=0.1  -> SR ~94%")
rows.append("  Too loose  : sigma=10, lambda=0.2  -> SR ~7%  (reward guidance fails)")
rows.append("  Training   : sigma=8,  lambda=0.1")
rows.append("")
rows.append("Proxy interpretation:")
rows.append("  r_confined: more negative = stronger penalty in danger zone (good deterrent)")
rows.append("  r_free:     should be ~+0.1 (potential reward when closing 0.1 m to goal)")
rows.append("  r_col_dense: more negative = stronger collision avoidance signal")

output = "\n".join(rows)
print(output)

with open("results/hyperparameter_sweep.txt", "w", encoding="utf-8") as f:
    f.write(output + "\n")
print("\nSaved results/hyperparameter_sweep.txt")

# ----- heatmaps -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics_info = [
    ("r_confined",  0, "RdBu",   "Confined-zone reward\n(more negative = stronger deterrent)"),
    ("r_free",      1, "RdYlGn", "Free-zone reward\n(should be ~+0.1 everywhere)"),
    ("r_col_dense", 2, "Reds_r", "Collision penalty – dense crowd\n(more negative = stronger signal)"),
]

for label, idx, cmap, title in metrics_info:
    mat = np.array([[results[(s, l)][idx] for l in LAMBDAS] for s in SIGMAS])
    ax = axes[idx]
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(LAMBDAS))); ax.set_xticklabels(LAMBDAS)
    ax.set_yticks(range(len(SIGMAS)));  ax.set_yticklabels(SIGMAS)
    ax.set_xlabel("lambda"); ax.set_ylabel("sigma")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax)
    # annotate cells
    for i, s in enumerate(SIGMAS):
        for j, l in enumerate(LAMBDAS):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

fig.suptitle("Hyperparameter sweep: sigma x lambda (He et al. 2025, Table 3 proxies)",
             fontsize=11)
fig.tight_layout()
fig.savefig("results/sweep_heatmaps.png", dpi=150)
print("Saved results/sweep_heatmaps.png")
plt.close(fig)
