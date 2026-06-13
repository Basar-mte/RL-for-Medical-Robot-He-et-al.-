"""
config.py
=========
All training/environment hyperparameters reported in He et al. (2025), Sec 4.1.
Mirror these into the base repo's arguments/config files.
"""

# --- risk field + reward (the paper's contribution) ---
RISK_SIGMA = 8.0        # Eq.2 range factor (best training value; sigma=6 also strong)
DECAY_LAMBDA = 0.1      # Eq.4 exponential decay rate
GOAL_REWARD = 10.0      # Eq.7
DISCOMFORT_DIST = 0.25  # personal-space radius defining the "confined" zone (m)

# --- PPO (base repo trainer) ---
PPO = dict(
    gamma=0.99,
    lr=4e-5,
    clip_param=0.2,
    num_processes=16,      # parallel environments
    total_iterations=20820,
)

# --- CrowdSim environment (base repo) ---
ENV = dict(
    workspace_size=12.0,   # 12 m x 12 m square
    robot_fov_deg=360,
    lidar_range=5.0,       # m
    robot_max_speed=1.0,   # m/s
    robot_radius=0.3,      # m
    human_radius_range=(0.3, 0.5),   # m
    human_speed_range=(0.5, 1.5),    # m/s
    time_step=0.25,        # s  (CrowdSim default)
    human_policy="orca",
    robot_visible_to_humans=False,   # unidirectional interaction
    train_n_humans=20,     # rho = 0.15 persons/m^2 for training
)

# --- evaluation (Sec 4.2 / 4.3) ---
EVAL = dict(
    n_test_cases=500,
    intrusion_dist=0.25,   # threshold for ITR / SD
    fixed_seed=True,
)

# --- density gradient for the generalization experiment (Table 2) ---
DENSITY_GRADIENT = {
    0.07: 10,   # persons/m^2 : n_humans
    0.10: 15,
    0.14: 20,
    0.17: 25,
    0.21: 30,
}

# --- recommended (sigma, lambda) by density (Sec 4.3.3) ---
TUNING_GUIDE = {
    "low_density (rho<0.10)":   dict(sigma=2, lam=0.05),   # favour efficiency
    "med_high (rho>=0.15)":     dict(sigma=6, lam=0.1),    # or sigma=8; balanced
    "avoid":                    "extreme combos e.g. sigma=10, lam=0.2 (SR collapses to 7%)",
}
