"""
metrics.py
==========
The five evaluation metrics from Section 4.2 of He et al. (2025), used to fill
Tables 1-3. Accumulate per-episode records, then call summarize().

  SR  : success rate (%)            higher better
  NT  : navigation time (s)         lower better  (successful episodes only)
  PL  : path length (m)             lower better  (successful episodes only)
  ITR : intrusion-to-time ratio (%) lower better  (fraction of steps the robot
                                                    is inside a pedestrian's
                                                    personal space)
  SD  : social distance (m)         higher better (mean min robot-human distance
                                                    *during intrusion events*)
"""

from __future__ import annotations
import numpy as np


class EpisodeRecorder:
    """Accumulate one episode's trajectory, then emit a per-episode summary."""

    def __init__(self, time_step=0.25, intrusion_dist=0.25):
        self.time_step = time_step           # CrowdSim default dt
        self.intrusion_dist = intrusion_dist # personal-space threshold (m)
        self.reset()

    def reset(self):
        self.n_steps = 0
        self.path_length = 0.0
        self.prev_pos = None
        self.intrusion_steps = 0
        self.intrusion_min_dists = []
        self.outcome = None                  # 'success' | 'collision' | 'timeout'

    def step(self, robot_pos, min_human_dist):
        """Call once per timestep. min_human_dist = surface-to-surface gap (m)."""
        robot_pos = np.asarray(robot_pos, float)
        if self.prev_pos is not None:
            self.path_length += float(np.linalg.norm(robot_pos - self.prev_pos))
        self.prev_pos = robot_pos
        self.n_steps += 1
        if min_human_dist < self.intrusion_dist:
            self.intrusion_steps += 1
            self.intrusion_min_dists.append(min_human_dist)

    def finish(self, outcome):
        self.outcome = outcome
        return {
            "outcome": outcome,
            "nav_time": self.n_steps * self.time_step,
            "path_length": self.path_length,
            "n_steps": self.n_steps,
            "intrusion_steps": self.intrusion_steps,
            "intrusion_min_dists": list(self.intrusion_min_dists),
        }


def summarize(records):
    """Aggregate a list of per-episode dicts (from EpisodeRecorder.finish)."""
    n = len(records)
    succ = [r for r in records if r["outcome"] == "success"]

    sr = 100.0 * len(succ) / n if n else 0.0
    nt = float(np.mean([r["nav_time"] for r in succ])) if succ else float("nan")
    pl = float(np.mean([r["path_length"] for r in succ])) if succ else float("nan")

    tot_intr = sum(r["intrusion_steps"] for r in records)
    tot_steps = sum(r["n_steps"] for r in records)
    itr = 100.0 * tot_intr / tot_steps if tot_steps else 0.0

    all_dists = [d for r in records for d in r["intrusion_min_dists"]]
    sd = float(np.mean(all_dists)) if all_dists else float("nan")

    return {"SR(%)": round(sr, 1), "NT(s)": round(nt, 2), "PL(m)": round(pl, 2),
            "ITR(%)": round(itr, 2), "SD": round(sd, 2), "N": n}


if __name__ == "__main__":
    # toy demo over 3 fake episodes
    recs = []
    rng = np.random.default_rng(0)
    for ep in range(3):
        rec = EpisodeRecorder()
        pos = np.array([0.0, 0.0])
        for _ in range(40):
            pos = pos + rng.normal(0, 0.1, 2)
            rec.step(pos, min_human_dist=abs(rng.normal(0.4, 0.2)))
        recs.append(rec.finish("success" if ep < 2 else "collision"))
    print(summarize(recs))
