"""
Physics-inspired potential field reward shaping for naval RL.

Each function accepts position arrays and config dicts, returning per-ship
reward vectors.  All functions are pure (no side effects) so they can be
unit-tested in isolation.

References
----------
- Lennard-Jones potential (atomic physics, adapted): LJ(D) = -ε [(σ/D)^n - (σ/D)^m]
- Predictive intercept: dead-reckoning future position of opponent ships
- Modified gravity: 1/D attraction + linear term for gradient stability at range
"""

from __future__ import annotations

import numpy as np


EPS = 1e-6   # guard against division by zero


# ---------------------------------------------------------------------------
# Lennard-Jones family
# ---------------------------------------------------------------------------

def lennard_jones(D: np.ndarray, sigma: float, epsilon: float,
                  n: int = 4, m: int = 2) -> np.ndarray:
    """
    Generalised Lennard-Jones potential.

    Returns negative (repulsive) values at D < D_min, positive (attractive)
    at D > D_min, where D_min = sigma * (n/m)^(1/(n-m)).

    Parameters
    ----------
    D       : distance matrix (any shape), strictly positive
    sigma   : characteristic distance (equilibrium ∝ sigma)
    epsilon : energy scale
    n, m    : repulsive / attractive exponents (n > m)
    """
    r = sigma / (D + EPS)
    return -epsilon * (r**n - r**m)


def lj_supremacy(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    weapon_range_A: float,
    weapon_range_B: float,
    epsilon: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lennard-Jones engagement-range potential.

    Encourages ships with range superiority to hold at optimal engagement
    distance; discourages closing inside minimum safe range.

    Parameters
    ----------
    pos_A, pos_B      : (nA, 2) and (nB, 2) position arrays
    weapon_range_A/B  : effective weapon range for each fleet (metres)
    epsilon           : energy scale

    Returns
    -------
    r_A : (nA,) reward per Alice ship
    r_B : (nB,) reward per Bob ship
    """
    D = np.linalg.norm(
        pos_A[:, None, :] - pos_B[None, :, :], axis=-1
    ) + EPS                                           # (nA, nB)

    sigma_A = weapon_range_A * np.sqrt(2)
    sigma_B = weapon_range_B * np.sqrt(2)

    r_A = lennard_jones(D, sigma_A, epsilon).sum(axis=1)  # (nA,)
    r_B = lennard_jones(D, sigma_B, epsilon).sum(axis=0)  # (nB,)
    return r_A, r_B


def lj_formation(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    d_form_A: float,
    d_form_B: float,
    epsilon: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lennard-Jones formation-keeping potential between friendly ships.

    Parameters
    ----------
    pos_A, pos_B  : (nA, 2) and (nB, 2) position arrays
    d_form_A/B    : optimal inter-ship spacing (metres)
    epsilon       : energy scale

    Returns
    -------
    r_A : (nA,) reward per Alice ship
    r_B : (nB,) reward per Bob ship
    """
    def _fleet_reward(pos: np.ndarray, d_form: float) -> np.ndarray:
        n = pos.shape[0]
        if n < 2:
            return np.zeros(n)
        D = np.linalg.norm(
            pos[:, None, :] - pos[None, :, :], axis=-1
        ) + EPS
        np.fill_diagonal(D, np.inf)
        sigma = d_form * np.sqrt(2)
        return lennard_jones(D, sigma, epsilon).sum(axis=1)

    return _fleet_reward(pos_A, d_form_A), _fleet_reward(pos_B, d_form_B)


# ---------------------------------------------------------------------------
# Modified gravity
# ---------------------------------------------------------------------------

def modified_gravity(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    G: float = 1.0,
    k_linear: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1/D gravitational attraction with an added linear term for gradient
    stability at large separations (avoids vanishing gradient at range).

    reward_i = G/D - k_linear * D   (summed over opponents)

    Returns
    -------
    r_A : (nA,) reward per Alice ship
    r_B : (nB,) reward per Bob ship
    """
    D = np.linalg.norm(
        pos_A[:, None, :] - pos_B[None, :, :], axis=-1
    ) + EPS                                           # (nA, nB)

    field = G / D - k_linear * D
    return field.sum(axis=1), field.sum(axis=0)


# ---------------------------------------------------------------------------
# Confining boundary potential
# ---------------------------------------------------------------------------

def boundary_penalty(
    pos: np.ndarray,
    grid_half: float,
    R_conf: float = 1.0,
    margin: float = 0.9,
) -> np.ndarray:
    """
    Inverse-distance penalty for approaching grid boundaries.

    Parameters
    ----------
    pos        : (n, 2) position array (values in [-grid_half, grid_half])
    grid_half  : half-size of the operational area (metres)
    R_conf     : penalty scale
    margin     : fraction of grid_half at which penalty activates

    Returns
    -------
    penalties : (n,) penalty per ship (≤ 0)
    """
    threshold = margin * grid_half
    dists_to_wall = np.stack([
        np.abs(pos[:, 0] - grid_half),
        np.abs(pos[:, 0] + grid_half),
        np.abs(pos[:, 1] - grid_half),
        np.abs(pos[:, 1] + grid_half),
    ], axis=1).min(axis=1)                           # (n,)

    delta = np.clip(threshold - dists_to_wall, 0.0, None)
    return -R_conf / (delta + EPS) * (delta > 0)


# ---------------------------------------------------------------------------
# Predictive intercept
# ---------------------------------------------------------------------------

def predictive_intercept(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    vel_A: np.ndarray,
    vel_B: np.ndarray,
    v_max_A: float,
    v_max_B: float,
    R_pred: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dead-reckoning reward: place an attractive field at the *predicted future
    position* of opponent ships, encouraging interception rather than pursuit.

    intercept_time[i,j] = ||pos_B[j] - pos_A[i]|| / v_max_A
    future_B[i,j]       = pos_B[j] + intercept_time[i,j] * vel_B[j]
    r_A[i]              = -R_pred * Σ_j ||future_B[i,j] - pos_A[i]||

    Returns
    -------
    r_A : (nA,) reward per Alice ship
    r_B : (nB,) reward per Bob ship
    """
    # Alice chasing Bob
    D_AB = np.linalg.norm(
        pos_B[None, :, :] - pos_A[:, None, :], axis=-1
    ) + EPS                                           # (nA, nB)
    t_AB = D_AB / (v_max_A + EPS)                    # (nA, nB)
    future_B = pos_B[None, :, :] + t_AB[:, :, None] * vel_B[None, :, :]  # (nA,nB,2)
    pred_dist_A = np.linalg.norm(future_B - pos_A[:, None, :], axis=-1) + EPS
    r_A = -R_pred * pred_dist_A.sum(axis=1)

    # Bob chasing Alice
    D_BA = np.linalg.norm(
        pos_A[None, :, :] - pos_B[:, None, :], axis=-1
    ) + EPS                                           # (nB, nA)
    t_BA = D_BA / (v_max_B + EPS)
    future_A = pos_A[None, :, :] + t_BA[:, :, None] * vel_A[None, :, :]  # (nB,nA,2)
    pred_dist_B = np.linalg.norm(future_A - pos_B[:, None, :], axis=-1) + EPS
    r_B = -R_pred * pred_dist_B.sum(axis=1)

    return r_A, r_B


# ---------------------------------------------------------------------------
# Composite reward calculator
# ---------------------------------------------------------------------------

def compute_rewards(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    vel_A: np.ndarray,
    vel_B: np.ndarray,
    cfg_A: dict,
    cfg_B: dict,
    grid_half: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate all potential-field rewards for one timestep.

    Parameters
    ----------
    pos_A, pos_B : (nA/nB, 2) fleet positions
    vel_A, vel_B : (nA/nB, 2) fleet velocity vectors
    cfg_A, cfg_B : reward config dicts (see configs/ YAML files)
    grid_half    : half-size of operational grid (metres)

    Returns
    -------
    r_A : (nA,) total reward per Alice ship
    r_B : (nB,) total reward per Bob ship
    """
    r_A = np.zeros(pos_A.shape[0])
    r_B = np.zeros(pos_B.shape[0])

    # 1. Gravity / simple attraction
    if cfg_A.get("w_gravity", 0.0) or cfg_B.get("w_gravity", 0.0):
        g_A, g_B = modified_gravity(
            pos_A, pos_B,
            G=cfg_A.get("gravity_G", 1.0),
            k_linear=cfg_A.get("gravity_k", 0.3),
        )
        r_A += cfg_A.get("w_gravity", 0.0) * g_A
        r_B += cfg_B.get("w_gravity", 0.0) * g_B

    # 2. LJ supremacy (engagement range)
    if cfg_A.get("w_lj_sup", 0.0) or cfg_B.get("w_lj_sup", 0.0):
        s_A, s_B = lj_supremacy(
            pos_A, pos_B,
            weapon_range_A=cfg_A.get("weapon_range", 10_000.0),
            weapon_range_B=cfg_B.get("weapon_range", 10_000.0),
        )
        r_A += cfg_A.get("w_lj_sup", 0.0) * s_A
        r_B += cfg_B.get("w_lj_sup", 0.0) * s_B

    # 3. LJ formation (friendly cohesion)
    if cfg_A.get("w_lj_form", 0.0) or cfg_B.get("w_lj_form", 0.0):
        f_A, f_B = lj_formation(
            pos_A, pos_B,
            d_form_A=cfg_A.get("d_form", 5_000.0),
            d_form_B=cfg_B.get("d_form", 5_000.0),
        )
        r_A += cfg_A.get("w_lj_form", 0.0) * f_A
        r_B += cfg_B.get("w_lj_form", 0.0) * f_B

    # 4. Predictive intercept
    if cfg_A.get("w_pred", 0.0) or cfg_B.get("w_pred", 0.0):
        p_A, p_B = predictive_intercept(
            pos_A, pos_B, vel_A, vel_B,
            v_max_A=cfg_A.get("v_max", 617.0),
            v_max_B=cfg_B.get("v_max", 617.0),
        )
        r_A += cfg_A.get("w_pred", 0.0) * p_A
        r_B += cfg_B.get("w_pred", 0.0) * p_B

    # 5. Boundary penalty
    if cfg_A.get("w_boundary", 0.0):
        r_A += cfg_A.get("w_boundary", 0.0) * boundary_penalty(pos_A, grid_half)
    if cfg_B.get("w_boundary", 0.0):
        r_B += cfg_B.get("w_boundary", 0.0) * boundary_penalty(pos_B, grid_half)

    # 6. Per-step time penalty
    r_A += cfg_A.get("time_penalty", 0.0)
    r_B += cfg_B.get("time_penalty", 0.0)

    return r_A, r_B
