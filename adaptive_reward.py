"""
adaptive_reward.py
==================
Re-implementation of Section 3.2 of He et al. (2025): the adaptive reward
function built on top of the scene-complexity score C(sigma).

Equations reproduced
---------------------
  (3)  r_col  = -10 * C(sigma)                         density-scaled collision penalty
  (4)  r_disc =  C(sigma)/2 * exp(1 - d_min * lambda)  danger-zone exponential-decay term
  (5)  r_pred = min_i min_k ( 1{collide t+k} * r_col / 2^k )   trajectory-conflict penalty
  (6)  r_pot  = d_goal^{t+1} - d_goal^{t}              potential / goal-approach reward
  (7)  piecewise combination by region

NOTE ON THE SIGN OF Eq. (4)  (important reproduction detail)
------------------------------------------------------------
As literally typeset, Eq. (4) is positive (C>=0, exp>0). But the surrounding
text states the mechanism should "assign exponentially increasing NEGATIVE
rewards to the robot's actions of approaching pedestrians" (p.5) and that the
robot "will be punished" in the danger zone (p.7). A positive r_disc would
*reward* approaching, which contradicts the design intent. We therefore apply
it as a penalty by default (DISC_AS_PENALTY=True). Flip the flag if you want
the literal typeset sign. This is the single genuine ambiguity in the paper;
flag it in any write-up.
"""

from __future__ import annotations
import numpy as np
from risk_field import scene_complexity

# ----- region labels (Eq. 7) ----------------------------------------------
GOAL = "goal"
COLLISION = "collision"
CONFINED = "confined"      # inside discomfort / danger zone but not colliding
FREE = "free"

# ----- defaults from the paper ---------------------------------------------
SIGMA_DEFAULT = 8.0        # Sec 4.1 training value
LAMBDA_DEFAULT = 0.1       # Sec 4.1 training value
GOAL_REWARD = 10.0         # Eq. 7
DISCOMFORT_DIST = 0.25     # base-repo discomfort radius (m); defines CONFINED
DISC_AS_PENALTY = True     # see module docstring


# ---------------------------------------------------------------------------
# individual reward terms
# ---------------------------------------------------------------------------
def r_col(C):
    """Eq. (3): density-scaled collision penalty."""
    return -10.0 * C


def r_disc(C, d_min, lam=LAMBDA_DEFAULT, as_penalty=DISC_AS_PENALTY):
    """Eq. (4): danger-zone exponential-decay term.

    d_min : distance to the nearest human [m]
    lam   : decay rate lambda (single hyperparameter the paper tunes).
    """
    val = (C / 2.0) * np.exp(1.0 - d_min * lam)
    return -val if as_penalty else val


def r_pred(collide_flags, col_penalty):
    """Eq. (5): future-trajectory conflict penalty.

    Faithful to the base AttnGraph repo, where r_pred reuses the collision
    penalty discounted by 2^k over the K-step prediction horizon.

    Parameters
    ----------
    collide_flags : (n_humans, K) bool/0-1 array
        collide_flags[i, k] == 1 if the robot's next position overlaps the
        predicted position of human i at horizon step k (k = 0..K-1 -> t+1..t+K).
    col_penalty : float
        the value of r_col for the current frame (Eq. 3). In the base repo a
        constant r_c=-20 is used; the paper feeds the density-scaled r_col.

    Returns
    -------
    float : the minimum (most negative) discounted conflict penalty, or 0.0.
    """
    flags = np.asarray(collide_flags, dtype=float)
    if flags.size == 0:
        return 0.0
    K = flags.shape[1]
    discount = col_penalty / (2.0 ** np.arange(K))      # r_col / 2^k  (k=0..K-1)
    per_human = np.min(np.where(flags > 0, discount[None, :], 0.0), axis=1)
    val = per_human.min()
    return float(min(val, 0.0))


def r_pot(d_goal_prev, d_goal_curr):
    """Eq. (6): potential-based goal-approach reward (positive when closing in).

    Paper writes  r_pot = d_goal^{t+1} - d_goal^{t}  (next minus current).
    Using the standard potential convention this rewards *decreasing* distance,
    so we return (prev - curr): >0 when the robot moved closer to the goal.
    """
    return float(d_goal_prev - d_goal_curr)


# ---------------------------------------------------------------------------
# full reward  (Eq. 7)
# ---------------------------------------------------------------------------
def classify_region(d_min, d_goal, robot_radius, discomfort_dist=DISCOMFORT_DIST):
    """Decide which branch of Eq. (7) applies this step."""
    if d_min < 0.0:
        return COLLISION              # negative separation == overlap
    if d_goal <= robot_radius:
        return GOAL
    if d_min < discomfort_dist:
        return CONFINED               # inside personal/danger zone
    return FREE


def adaptive_reward(robot_pos, robot_vel, robot_radius, goal_pos,
                    human_pos, human_vel,
                    d_goal_prev,
                    collide_flags=None,
                    sigma=SIGMA_DEFAULT, lam=LAMBDA_DEFAULT,
                    fov_radius=5.0):
    """
    Compute r(s_t, a_t) per Eq. (7) for one environment step.

    Returns
    -------
    reward : float
    info   : dict with the intermediate terms (C, region, r_col, ...)
             -- handy for logging / debugging / ablations.
    """
    robot_pos = np.asarray(robot_pos, float)
    human_pos = np.asarray(human_pos, float).reshape(-1, 2)
    human_vel = np.asarray(human_vel, float).reshape(-1, 2)

    # --- nearest-human distance d_min (Eq. 4 input) ---
    if human_pos.shape[0]:
        seps = np.linalg.norm(human_pos - robot_pos[None, :], axis=1)
        # subtract radii so d_min<0 means overlap (collision), matching base repo
        # here we assume a representative human radius of 0.3 m; pass real radii in CrowdSim.
        d_min = float(seps.min() - robot_radius - 0.3)
        d_min_raw = float(seps.min())
    else:
        d_min = np.inf
        d_min_raw = np.inf

    d_goal = float(np.linalg.norm(robot_pos - np.asarray(goal_pos, float)))

    # --- scene complexity (Eq. 2) ---
    C = scene_complexity(robot_pos, human_pos, human_vel, sigma, fov_radius=fov_radius)
    rc = r_col(C)

    region = classify_region(d_min, d_goal, robot_radius)

    info = {"C": C, "d_min": d_min, "d_min_raw": d_min_raw,
            "d_goal": d_goal, "region": region, "r_col": rc}

    if region == GOAL:
        reward = GOAL_REWARD

    elif region == COLLISION:
        reward = rc                                            # Eq.7 line 2

    elif region == CONFINED:
        rd = r_disc(C, max(d_min_raw, 0.0), lam=lam)           # Eq.4
        rp = r_pred(collide_flags, rc) if collide_flags is not None else 0.0
        reward = rp + rd                                       # Eq.7 line 3
        info.update(r_disc=rd, r_pred=rp)

    else:  # FREE
        rp = r_pred(collide_flags, rc) if collide_flags is not None else 0.0
        rpot = r_pot(d_goal_prev, d_goal)                      # Eq.6
        reward = rp + rpot                                     # Eq.7 line 4
        info.update(r_pred=rp, r_pot=rpot)

    return float(reward), info


if __name__ == "__main__":
    # sanity: dense scene -> large negative collision penalty; sparse -> mild
    dense = adaptive_reward(
        robot_pos=[0, 0], robot_vel=[1, 0], robot_radius=0.3, goal_pos=[5, 0],
        human_pos=[[0.4, 0.0], [0.6, 0.3], [0.5, -0.4], [1.0, 0.2]],
        human_vel=[[1.2, 0], [1.0, 0], [1.3, 0], [0.8, 0]],
        d_goal_prev=5.1, sigma=8.0, lam=0.1)
    sparse = adaptive_reward(
        robot_pos=[0, 0], robot_vel=[1, 0], robot_radius=0.3, goal_pos=[5, 0],
        human_pos=[[4.0, 3.0]],
        human_vel=[[0.5, 0]],
        d_goal_prev=5.1, sigma=8.0, lam=0.1)
    print("dense :", round(dense[0], 4), dense[1]["region"], "C=", round(dense[1]["C"], 3))
    print("sparse:", round(sparse[0], 4), sparse[1]["region"], "C=", round(sparse[1]["C"], 3))
