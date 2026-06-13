"""
adaptive_reward_patch.py
========================
Drop-in replacement for the reward computation inside the base repo
Shuijing725/CrowdNav_Prediction_AttnGraph.

WHERE THIS GOES
---------------
In that repo the reward is computed inside crowd_sim/envs/crowd_sim.py
(and the prediction variant crowd_sim_pred.py) in the step() method, where a
block computes `dmin` (closest robot-human separation), checks collision /
reaching_goal / discomfort, and (in the pred variant) adds the prediction
reward r_pred.

Replace that block with a single call to `calc_reward_adaptive(self, action)`
defined below, and add the listed __init__ attributes. This file deliberately
mirrors the variable names used in that codebase (self.robot, self.humans,
self.discomfort_dist, self.time_step, etc.).

NOTE: copy risk_field.py + adaptive_reward.py next to crowd_sim.py, or import
the equation helpers from them. Below we inline the two equations so the patch
is self-contained.
"""

import numpy as np


# ----- equation helpers (inlined; identical to risk_field/adaptive_reward) ---
def _scene_complexity(robot_xy, humans_xy, humans_speed, sigma, fov_radius):
    d = np.linalg.norm(humans_xy - robot_xy[None, :], axis=1)
    mask = d <= fov_radius
    if not mask.any():
        return 0.0
    d, v = d[mask], humans_speed[mask]
    return float(np.sum(v * np.exp(-(d ** 2) / (2.0 * sigma ** 2))))


# ----- attributes to add in CrowdSim.__init__ (or configure() ) --------------
#   self.risk_sigma     = 8.0    # Eq.2 range factor   (paper Sec 4.1)
#   self.decay_lambda   = 0.1    # Eq.4 decay rate
#   self.fov_radius     = 5.0    # lidar range (m)
#   self.use_adaptive_reward = True
#   (self.discomfort_dist, self.collision_penalty=-20, self.success_reward=10,
#    self.time_step already exist in the base repo.)


def calc_reward_adaptive(self, action, danger_collide_flags=None):
    """
    Implements Eqs. (2)-(7). Returns (reward, done, episode_info).

    `self` is the CrowdSim env instance. We read robot/human state through the
    same accessors the base repo uses.

    `danger_collide_flags` (optional): (n_humans, K) array of predicted-collision
    indicators for r_pred (Eq.5). The base pred-env already computes the data
    needed for this; pass it through if you want the full Eq.7 (otherwise r_pred=0).
    """
    # --- gather state (same accessors as the base repo) ---
    robot_xy = np.array([self.robot.px, self.robot.py], dtype=float)
    goal_xy = np.array([self.robot.gx, self.robot.gy], dtype=float)

    humans_xy, humans_speed, surface_gaps = [], [], []
    for h in self.humans:
        humans_xy.append([h.px, h.py])
        humans_speed.append(float(np.hypot(h.vx, h.vy)))
        center = np.hypot(self.robot.px - h.px, self.robot.py - h.py)
        surface_gaps.append(center - h.radius - self.robot.radius)
    humans_xy = np.array(humans_xy, dtype=float).reshape(-1, 2)
    humans_speed = np.array(humans_speed, dtype=float)
    surface_gaps = np.array(surface_gaps, dtype=float)

    dmin = float(surface_gaps.min()) if surface_gaps.size else np.inf
    d_goal = float(np.linalg.norm(robot_xy - goal_xy))

    # --- Eq.2 scene complexity, Eq.3 density-scaled collision penalty ---
    C = _scene_complexity(robot_xy, humans_xy, humans_speed,
                          self.risk_sigma, self.fov_radius)
    r_collision = -10.0 * C                                  # Eq.3

    collision = dmin < 0.0
    reaching_goal = d_goal < self.robot.radius

    done, info_tag = False, "nothing"

    if self.global_time >= self.time_limit - 1:
        reward, done, info_tag = 0.0, True, "timeout"

    elif collision:                                          # Eq.7 line 2
        reward, done, info_tag = r_collision, True, "collision"

    elif reaching_goal:                                      # Eq.7 line 1
        reward, done, info_tag = self.success_reward, True, "goal"

    elif dmin < self.discomfort_dist:                        # Eq.7 line 3 (confined)
        # Eq.4 danger-zone exponential decay, applied as a penalty (see docstring)
        r_disc = -(C / 2.0) * np.exp(1.0 - max(dmin, 0.0) * self.decay_lambda)
        r_pred = _r_pred(danger_collide_flags, r_collision)
        reward, info_tag = r_pred + r_disc, "danger"

    else:                                                    # Eq.7 line 4 (free)
        r_pot = self.prev_d_goal - d_goal                    # Eq.6
        r_pred = _r_pred(danger_collide_flags, r_collision)
        reward, info_tag = r_pred + r_pot, "free"

    self.prev_d_goal = d_goal                                # remember for next step
    return reward, done, info_tag


def _r_pred(flags, col_penalty):
    """Eq.5 trajectory-conflict penalty; 0 if no flags supplied."""
    if flags is None:
        return 0.0
    flags = np.asarray(flags, float)
    if flags.size == 0:
        return 0.0
    K = flags.shape[1]
    disc = col_penalty / (2.0 ** np.arange(K))
    per_human = np.min(np.where(flags > 0, disc[None, :], 0.0), axis=1)
    return float(min(per_human.min(), 0.0))
