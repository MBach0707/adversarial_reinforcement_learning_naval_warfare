"""
NavalEnv — a Gymnasium-compliant adversarial naval warfare environment.

Two independent commanders (Alice, Bob) each control a fleet of surface
combatants on a continuous 2-D plane.  Ships move, fire, and intercept
missiles each timestep.  Rewards are shaped by physics-inspired potential
fields defined in naval_rl.rewards.potential_fields.

Action space (per fleet, flattened)
------------------------------------
For each ship:  [speed_fraction ∈ [0,1],  course ∈ [-π, π],
                 fire_fraction ∈ [0,1],   target_fraction ∈ [0,1]]

  speed_fraction  : fraction of max speed
  course          : desired heading (radians, mathematical convention)
  fire_fraction   : > fire_threshold → fire; magnitude encodes salvo size
  target_fraction : mapped to opponent index via floor(f * n_opponents)

Observation space
-----------------
Flat vector:  [alice_ships × (x, y, course, speed, health),
               bob_ships   × (x, y, course, speed, health)]
All values in raw metres / radians / knots — normalisation is handled
externally by the agent's RunningMeanStd.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from naval_rl.envs.entities import Ship, Weapon, ADMeasure
from naval_rl.rewards.potential_fields import compute_rewards


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURES_PER_SHIP = 5   # x, y, course, speed, health
_ACTIONS_PER_SHIP  = 4   # speed_frac, course, fire_frac, target_frac
_FIRE_THRESHOLD    = 0.5


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fleet_velocity(fleet: List[Ship]) -> np.ndarray:
    """Return (n, 2) velocity vectors (m/min) for a fleet."""
    vels = []
    for s in fleet:
        vels.append([
            s.speed * math.cos(s.course),
            s.speed * math.sin(s.course),
        ])
    return np.array(vels, dtype=np.float32)


def _fleet_pos(fleet: List[Ship]) -> np.ndarray:
    return np.array([[s.x, s.y] for s in fleet], dtype=np.float32)


def _scalar_to_salvo(f: float, max_salvo: int) -> int:
    return int(np.clip(round(f * max_salvo), 0, max_salvo))


def _scalar_to_index(f: float, n: int) -> int:
    return int(np.clip(round(0.5 * (f + 1) * (n - 1)), 0, n - 1))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NavalEnv(gym.Env):
    """
    Adversarial multi-agent naval warfare environment.

    Parameters
    ----------
    fleet_alice, fleet_bob  : lists of Ship objects
    cfg_alice, cfg_bob      : reward / scenario config dicts (see configs/)
    max_steps               : episode length (minutes of simulated time)
    grid_half               : half-size of operational area (metres)
    fire_threshold          : fire_fraction above which a shot is attempted
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        fleet_alice: List[Ship],
        fleet_bob: List[Ship],
        cfg_alice: Dict[str, Any],
        cfg_bob: Dict[str, Any],
        max_steps: int = 500,
        grid_half: float = 100_000.0,
        fire_threshold: float = _FIRE_THRESHOLD,
    ) -> None:
        super().__init__()

        self.fleet_alice = fleet_alice
        self.fleet_bob   = fleet_bob
        self.cfg_alice   = cfg_alice
        self.cfg_bob     = cfg_bob
        self.max_steps   = max_steps
        self.grid_half   = grid_half
        self.fire_threshold = fire_threshold

        self.n_alice = len(fleet_alice)
        self.n_bob   = len(fleet_bob)

        # Step / episode counters
        self.step_count   = 0
        self.episode_count = 0

        # Per-episode rare-event flags (used by replay buffer)
        self.rare_event         = False
        self.rare_event_attempt = False

        # --- Spaces ---
        n_total  = self.n_alice + self.n_bob
        obs_dim  = n_total * _FEATURES_PER_SHIP
        act_dim  = n_total * _ACTIONS_PER_SHIP

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low =np.array([-np.inf] * act_dim, dtype=np.float32),
            high=np.array([ np.inf] * act_dim, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.episode_count += 1
        self.step_count = 0
        self.rare_event = False
        self.rare_event_attempt = False

        for ship in self.fleet_alice + self.fleet_bob:
            ship.reset()

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """
        Execute one timestep.

        Parameters
        ----------
        action : flat action vector for ALL ships (alice first, then bob)

        Returns
        -------
        obs, rewards (shape (2,)), terminated, truncated, info
        rewards[0] = Alice's mean per-ship reward
        rewards[1] = Bob's mean per-ship reward
        """
        self.step_count += 1
        self.rare_event = False
        self.rare_event_attempt = False

        # Clear shot logs
        for s in self.fleet_alice + self.fleet_bob:
            s.last_shots = []
            for w in s.weapons:
                w.tick()
            for ad in s.ad_measures:
                ad.tick()

        # Unpack actions
        action = np.asarray(action, dtype=np.float32)
        act_mat = action.reshape(-1, _ACTIONS_PER_SHIP)
        act_A = act_mat[: self.n_alice]
        act_B = act_mat[self.n_alice :]

        # --- Movement ---
        for ship, a in zip(self.fleet_alice, act_A):
            if ship.alive:
                ship.move(course=float(a[1]), speed_fraction=float(a[0]))
        for ship, a in zip(self.fleet_bob, act_B):
            if ship.alive:
                ship.move(course=float(a[1]), speed_fraction=float(a[0]))

        # --- Positions & velocities ---
        pos_A = _fleet_pos(self.fleet_alice)
        pos_B = _fleet_pos(self.fleet_bob)
        vel_A = _fleet_velocity(self.fleet_alice)
        vel_B = _fleet_velocity(self.fleet_bob)

        # --- Potential-field rewards ---
        r_A, r_B = compute_rewards(
            pos_A, pos_B, vel_A, vel_B,
            self.cfg_alice, self.cfg_bob,
            self.grid_half,
        )

        # --- Firing ---
        dmg_to_bob   = np.zeros(self.n_bob,   dtype=np.float32)
        dmg_to_alice = np.zeros(self.n_alice, dtype=np.float32)

        D = np.linalg.norm(
            pos_A[:, None, :] - pos_B[None, :, :], axis=-1
        )  # (nA, nB)

        # Alice fires
        for i, (ship, a) in enumerate(zip(self.fleet_alice, act_A)):
            if not ship.alive:
                continue
            fire_frac = float(a[2])
            if fire_frac > self.fire_threshold:
                self.rare_event_attempt = True
                for w in ship.weapons:
                    if not w.can_fire():
                        continue
                    j = _scalar_to_index(float(a[3]), self.n_bob)
                    if D[i, j] <= w.range:
                        salvo = _scalar_to_salvo(fire_frac, w.max_salvo_size)
                        fired = w.fire(salvo)
                        dmg_to_bob[j] += fired
                        r_A[i] += self.cfg_alice.get("reward_fire", 0.0)
                        self.rare_event = True
                        ship.last_shots.append({
                            "source": (ship.x, ship.y),
                            "target": (self.fleet_bob[j].x, self.fleet_bob[j].y),
                            "missiles_fired": fired,
                            "intercepted": 0,
                            "missiles_hit": fired,
                        })

        # Bob fires
        for i, (ship, a) in enumerate(zip(self.fleet_bob, act_B)):
            if not ship.alive:
                continue
            fire_frac = float(a[2])
            if fire_frac > self.fire_threshold:
                self.rare_event_attempt = True
                for w in ship.weapons:
                    if not w.can_fire():
                        continue
                    j = _scalar_to_index(float(a[3]), self.n_alice)
                    if D[j, i] <= w.range:
                        salvo = _scalar_to_salvo(fire_frac, w.max_salvo_size)
                        fired = w.fire(salvo)
                        dmg_to_alice[j] += fired
                        r_B[i] += self.cfg_bob.get("reward_fire", 0.0)
                        self.rare_event = True

        # AD intercept + damage application
        for j, dmg in enumerate(dmg_to_bob):
            if dmg <= 0:
                continue
            target = self.fleet_bob[j]
            remaining = int(dmg)
            for ad in target.ad_measures:
                if ad.can_intercept():
                    intercepted = ad.intercept(remaining)
                    remaining -= intercepted
                    if remaining <= 0:
                        break
            if remaining > 0:
                # stochastic hit resolution
                hits = sum(
                    1 for _ in range(remaining)
                    if np.random.rand() < self.cfg_alice.get("p_hit", 0.8)
                )
                if hits > 0:
                    target.take_damage(hits)
                    r_A += self.cfg_alice.get("kill_reward", 100.0) * (not target.alive)
                    r_B[j] -= self.cfg_bob.get("death_penalty", 100.0) * (not target.alive)

        for j, dmg in enumerate(dmg_to_alice):
            if dmg <= 0:
                continue
            target = self.fleet_alice[j]
            remaining = int(dmg)
            for ad in target.ad_measures:
                if ad.can_intercept():
                    intercepted = ad.intercept(remaining)
                    remaining -= intercepted
                    if remaining <= 0:
                        break
            if remaining > 0:
                hits = sum(
                    1 for _ in range(remaining)
                    if np.random.rand() < self.cfg_bob.get("p_hit", 0.8)
                )
                if hits > 0:
                    target.take_damage(hits)
                    r_B += self.cfg_bob.get("kill_reward", 100.0) * (not target.alive)
                    r_A[j] -= self.cfg_alice.get("death_penalty", 100.0) * (not target.alive)

        # --- Termination ---
        alice_dead = all(not s.alive for s in self.fleet_alice)
        bob_dead   = all(not s.alive for s in self.fleet_bob)
        terminated = alice_dead or bob_dead
        truncated  = self.step_count >= self.max_steps

        if terminated:
            if bob_dead:
                r_A += self.cfg_alice.get("victory_bonus", 200.0)
                r_B -= self.cfg_bob.get("defeat_penalty", 200.0)
            if alice_dead:
                r_B += self.cfg_bob.get("victory_bonus", 200.0)
                r_A -= self.cfg_alice.get("defeat_penalty", 200.0)

        rewards = np.array([r_A.mean(), r_B.mean()], dtype=np.float32)

        info = {
            "alice_alive": sum(s.alive for s in self.fleet_alice),
            "bob_alive":   sum(s.alive for s in self.fleet_bob),
            "rare_event":  self.rare_event,
        }
        return self._get_obs(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        parts = []
        for s in self.fleet_alice + self.fleet_bob:
            parts.append([s.x, s.y, s.course, s.speed, float(s.health)])
        return np.concatenate(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium render stub
    # ------------------------------------------------------------------

    def render(self) -> None:
        pos_A = [(s.x, s.y) for s in self.fleet_alice]
        pos_B = [(s.x, s.y) for s in self.fleet_bob]
        print(
            f"[Step {self.step_count:4d}] "
            f"Alice: {pos_A}  |  Bob: {pos_B}"
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def all_ships(self) -> List[Ship]:
        return self.fleet_alice + self.fleet_bob
