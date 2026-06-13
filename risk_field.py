"""
risk_field.py
=============
Re-implementation of the spatiotemporal risk field from
He et al. (2025), "Research on Adaptive Reward Optimization Method for Robot
Navigation in Complex Dynamic Environment", CMC 84(2):2733-2749.

This file implements ONLY the paper's Section 3.1 contribution:
the Gaussian-kernel scene-complexity score C(sigma)  --  Eq. (2).

    C(sigma) = sum_i  v_i / exp( d_i^2 / (2 * sigma^2) )

where, for every pedestrian i inside the robot's field of view:
    d_i    = Euclidean distance from robot to pedestrian i   [m]
    v_i    = speed (magnitude of velocity) of pedestrian i   [m/s]
    sigma  = risk-field range factor (decay rate). Paper uses sigma=8 for
             training (Sec 4.1) and sigma in {0.5,1.0,3.0} for the
             illustrative surfaces in Fig. 2.

Everything is plain numpy so it can be dropped straight into the CrowdSim
environment loop (which is also numpy-based).
"""

from __future__ import annotations
import numpy as np


def scene_complexity(robot_pos, human_pos, human_vel, sigma, fov_radius=None):
    """
    Eq. (2): scene-complexity score C(sigma) for the current frame.

    Parameters
    ----------
    robot_pos  : (2,) array_like           robot (x, y)
    human_pos  : (n, 2) array_like         pedestrian positions
    human_vel  : (n, 2) array_like         pedestrian velocities (vx, vy)
    sigma      : float                     risk-field range factor
    fov_radius : float or None             only count pedestrians within this
                                           radius (paper: 5 m lidar range).
                                           None  ->  use all pedestrians.

    Returns
    -------
    C : float    non-negative scene-complexity score (0 if FoV is empty)
    """
    robot_pos = np.asarray(robot_pos, dtype=float)
    human_pos = np.asarray(human_pos, dtype=float).reshape(-1, 2)
    human_vel = np.asarray(human_vel, dtype=float).reshape(-1, 2)

    if human_pos.shape[0] == 0:
        return 0.0

    d = np.linalg.norm(human_pos - robot_pos[None, :], axis=1)   # (n,) distances
    v = np.linalg.norm(human_vel, axis=1)                        # (n,) speeds

    if fov_radius is not None:
        mask = d <= fov_radius
        d, v = d[mask], v[mask]
        if d.size == 0:
            return 0.0

    # Gaussian kernel weighting:  v_i / exp(d_i^2 / 2 sigma^2)
    weight = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return float(np.sum(v * weight))


def complexity_grid(sigma, dist, speed):
    """
    Single-pedestrian complexity surface used to reproduce Fig. 2:
    C(sigma) for ONE pedestrian as a function of (distance, speed).

    Parameters
    ----------
    sigma : float
    dist  : (...,) array      distance grid [m]
    speed : (...,) array      speed grid   [m/s]   (broadcastable with dist)

    Returns
    -------
    C : ndarray (broadcast shape of dist & speed)
    """
    dist = np.asarray(dist, dtype=float)
    speed = np.asarray(speed, dtype=float)
    return speed * np.exp(-(dist ** 2) / (2.0 * sigma ** 2))


def risk_field_map(human_pos, human_vel, sigma, xlim, ylim, res=200):
    """
    Continuous risk-intensity field over a 2D plane, used to reproduce Fig. 3.
    At each grid point p the risk is the superposition over all pedestrians:
        R(p) = sum_i v_i * exp(-||p - x_i||^2 / 2 sigma^2)

    Returns (X, Y, R) meshgrids suitable for pcolormesh/contourf.
    """
    human_pos = np.asarray(human_pos, dtype=float).reshape(-1, 2)
    human_vel = np.asarray(human_vel, dtype=float).reshape(-1, 2)
    v = np.linalg.norm(human_vel, axis=1)

    xs = np.linspace(xlim[0], xlim[1], res)
    ys = np.linspace(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(xs, ys)
    R = np.zeros_like(X)
    for (hx, hy), vi in zip(human_pos, v):
        d2 = (X - hx) ** 2 + (Y - hy) ** 2
        R += vi * np.exp(-d2 / (2.0 * sigma ** 2))
    return X, Y, R


if __name__ == "__main__":
    # quick smoke test
    robot = [0.0, 0.0]
    humans = [[1.0, 0.0], [3.0, 0.0]]
    vels = [[1.0, 0.0], [0.5, 0.0]]
    for s in (0.5, 1.0, 3.0, 8.0):
        print(f"sigma={s:>4}:  C = {scene_complexity(robot, humans, vels, s):.4f}")
