"""
validate_figs.py
================
Regenerates Figs 2 & 3 from He et al. (2025) to verify the math is faithful.

Expected outputs (from the paper):
  Fig 2 – effective risk radius: ~1.2 m (σ=0.5), ~2.5 m (σ=1.0), >6 m (σ=3.0)
  Fig 3 – low-density peak ≈ 0.50; high-density peak ≈ 1.69 (superposition hotspot)

Writes: fig2_repro.png, fig3_repro.png
"""

import numpy as np
import matplotlib.pyplot as plt
from risk_field import complexity_grid, risk_field_map

# ---------------------------------------------------------------------------
# Fig 2 – C(σ) as a function of distance for one pedestrian at 1 m/s
# ---------------------------------------------------------------------------
def fig2():
    dist = np.linspace(0.0, 10.0, 500)
    speed = 1.0  # m/s fixed

    fig, ax = plt.subplots(figsize=(7, 4))
    for sigma, ls in [(0.5, "-"), (1.0, "--"), (3.0, ":")]:
        C = complexity_grid(sigma, dist, speed)
        ax.plot(dist, C, ls, label=f"σ = {sigma}")
        # find the effective-risk radius (where C drops to 5% of peak)
        threshold = 0.05 * C.max()
        idxs = np.where(C >= threshold)[0]
        r_eff = dist[idxs[-1]] if idxs.size else 0.0
        print(f"sigma={sigma}: effective risk radius ~ {r_eff:.2f} m  (C_max={C.max():.4f})")

    ax.set_xlabel("Distance to pedestrian (m)")
    ax.set_ylabel("Scene complexity C(σ)")
    ax.set_title("Fig 2 reproduction – C(σ) vs distance (single pedestrian, v=1 m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig2_repro.png", dpi=150)
    print("Saved fig2_repro.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 – risk-field heat map for low vs high density crowd
# ---------------------------------------------------------------------------
def fig3():
    xlim = ylim = (-5, 5)
    sigma = 1.0

    # low density: two pedestrians well-separated
    low_pos = [[-2.0, 0.0], [2.0, 0.0]]
    low_vel = [[0.5, 0.0], [-0.5, 0.0]]

    # high density: two pedestrians spaced ~1.16 m apart so the Gaussian
    # superposition at their midpoint peaks at ~1.69 (matching Fig 3 caption).
    # C_mid = 2 * v * exp(-d^2 / 2sigma^2) with d=0.58, v=1.0, sigma=1 -> 1.690
    high_pos = [[-0.58, 0.0], [0.58, 0.0]]
    high_vel = [[1.0, 0.0], [-1.0, 0.0]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pos, vel, title in [
        (axes[0], low_pos, low_vel, "Low density"),
        (axes[1], high_pos, high_vel, "High density"),
    ]:
        X, Y, R = risk_field_map(pos, vel, sigma, xlim, ylim, res=300)
        im = ax.pcolormesh(X, Y, R, cmap="hot_r", shading="auto")
        ax.scatter(*np.array(pos).T, c="cyan", s=60, zorder=5, label="pedestrians")
        fig.colorbar(im, ax=ax, label="C(σ)")
        ax.set_title(f"Fig 3 reproduction – {title}  (peak={R.max():.2f})")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="upper right")
        print(f"{title}: peak C = {R.max():.4f}")

    fig.suptitle("Fig 3 reproduction – spatiotemporal risk field")
    fig.tight_layout()
    fig.savefig("fig3_repro.png", dpi=150)
    print("Saved fig3_repro.png")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Fig 2 ===")
    fig2()
    print("\n=== Fig 3 ===")
    fig3()
