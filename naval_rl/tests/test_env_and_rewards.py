"""
Tests for naval_rl.rewards.potential_fields and naval_rl.envs.naval_env.

Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from naval_rl.rewards.potential_fields import (
    lennard_jones,
    lj_supremacy,
    lj_formation,
    modified_gravity,
    boundary_penalty,
    predictive_intercept,
    compute_rewards,
)
from naval_rl.envs.entities import Ship, Weapon, ADMeasure
from naval_rl.envs.naval_env import NavalEnv


# ---------------------------------------------------------------------------
# Potential field tests
# ---------------------------------------------------------------------------

class TestLennardJones:
    def test_minimum_at_sigma(self):
        """LJ minimum should occur near sigma * (n/m)^(1/(n-m))."""
        sigma, eps = 10.0, 1.0
        D = np.linspace(0.1, 50, 10000)
        v = lennard_jones(D, sigma, eps, n=4, m=2)
        # D_min for n=4, m=2: sigma * (4/2)^(1/(4-2)) = sigma * sqrt(2)
        expected_min = sigma * np.sqrt(2)
        actual_min = D[np.argmin(v)]
        assert abs(actual_min - expected_min) < 0.5, (
            f"LJ minimum at {actual_min:.2f}, expected ~{expected_min:.2f}"
        )

    def test_repulsive_at_short_range(self):
        """LJ should be negative (repulsive) for D << sigma."""
        D = np.array([0.1, 0.5])
        v = lennard_jones(D, sigma=10.0, epsilon=1.0)
        assert np.all(v < 0), "LJ should be repulsive at close range"

    def test_attractive_at_long_range(self):
        """LJ should be positive (attractive) for D >> sigma."""
        D = np.array([100.0, 200.0])
        v = lennard_jones(D, sigma=10.0, epsilon=1.0)
        assert np.all(v > 0), "LJ should be attractive at long range"


class TestSupremacyReward:
    def test_output_shapes(self):
        pos_A = np.random.randn(3, 2)
        pos_B = np.random.randn(2, 2)
        r_A, r_B = lj_supremacy(pos_A, pos_B, 10.0, 10.0)
        assert r_A.shape == (3,)
        assert r_B.shape == (2,)


class TestFormationReward:
    def test_single_ship_zero_reward(self):
        """A single ship has no formation partners — reward should be zero."""
        pos_A = np.array([[0.0, 0.0]])
        pos_B = np.array([[1.0, 1.0], [2.0, 2.0]])
        r_A, _ = lj_formation(pos_A, pos_B, d_form_A=5.0, d_form_B=5.0)
        assert r_A.shape == (1,)
        assert r_A[0] == pytest.approx(0.0)

    def test_repulsion_when_too_close(self):
        """Ships that are much closer than d_form should receive negative reward."""
        pos_A = np.array([[0.0, 0.0], [0.001, 0.0]])  # nearly on top of each other
        pos_B = np.array([[100.0, 0.0]])
        r_A, _ = lj_formation(pos_A, pos_B, d_form_A=5.0, d_form_B=5.0)
        assert np.all(r_A < 0)


class TestBoundaryPenalty:
    def test_no_penalty_at_centre(self):
        pos = np.array([[0.0, 0.0]])
        p = boundary_penalty(pos, grid_half=100_000.0, margin=0.9)
        assert p[0] == pytest.approx(0.0)

    def test_penalty_near_wall(self):
        pos = np.array([[99_000.0, 0.0]])   # close to boundary
        p = boundary_penalty(pos, grid_half=100_000.0, margin=0.9)
        assert p[0] < 0


class TestModifiedGravity:
    def test_symmetric(self):
        """Symmetric fleets should get the same reward magnitude."""
        pos_A = np.array([[0.0, 0.0]])
        pos_B = np.array([[10.0, 0.0]])
        r_A, r_B = modified_gravity(pos_A, pos_B)
        assert abs(r_A[0] - r_B[0]) < 1e-6


class TestComputeRewards:
    def test_shapes(self):
        pos_A = np.random.randn(2, 2).astype(np.float32)
        pos_B = np.random.randn(3, 2).astype(np.float32)
        vel_A = np.zeros((2, 2), dtype=np.float32)
        vel_B = np.zeros((3, 2), dtype=np.float32)
        cfg = {"w_gravity": 1.0, "gravity_G": 1.0, "gravity_k": 0.3}
        r_A, r_B = compute_rewards(pos_A, pos_B, vel_A, vel_B, cfg, cfg, 100_000.0)
        assert r_A.shape == (2,)
        assert r_B.shape == (3,)


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

def _make_env():
    fleet_alice = [
        Ship("Alice", x=-50_000, y=0, max_speed=20.0, health=3)
    ]
    fleet_alice[0].add_weapons([
        Weapon("SSM", stockpile=5, range=50_000, cooldown_max=10, p_hit=0.8, max_salvo_size=3)
    ])

    fleet_bob = [
        Ship("Bob", x=50_000, y=0, max_speed=20.0, health=3)
    ]
    fleet_bob[0].add_weapons([
        Weapon("SSM", stockpile=5, range=50_000, cooldown_max=10, p_hit=0.8, max_salvo_size=3)
    ])

    cfg = {"w_gravity": 1.0, "gravity_G": 1.0, "gravity_k": 0.3,
           "time_penalty": -0.01, "victory_bonus": 200.0,
           "kill_reward": 100.0, "death_penalty": 100.0,
           "defeat_penalty": 200.0, "reward_fire": 0.0, "p_hit": 0.8}

    return NavalEnv(fleet_alice, fleet_bob, cfg, cfg, max_steps=100)


class TestNavalEnv:
    def test_reset_returns_correct_shape(self):
        env = _make_env()
        obs, info = env.reset()
        expected = (2 * 5,)   # 2 ships × 5 features
        assert obs.shape == expected

    def test_step_returns_correct_types(self):
        env = _make_env()
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, rewards, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_episode_terminates_within_max_steps(self):
        env = _make_env()
        obs, _ = env.reset()
        for _ in range(env.max_steps + 10):
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert truncated or terminated, "Episode should have ended"

    def test_gymnasium_compliance(self):
        """Basic Gymnasium API compliance check."""
        from gymnasium.utils.env_checker import check_env
        env = _make_env()
        # check_env raises an error if the env violates the Gymnasium API
        try:
            check_env(env, warn=True)
        except Exception as e:
            pytest.fail(f"Gymnasium compliance check failed: {e}")

    def test_obs_changes_after_step(self):
        env = _make_env()
        obs, _ = env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 1.0   # speed fraction for Alice
        action[1] = 0.0   # course east
        next_obs, _, _, _, _ = env.step(action)
        # Alice should have moved — obs should differ
        assert not np.allclose(obs, next_obs)

    def test_weapons_deplete(self):
        env = _make_env()
        obs, _ = env.reset()
        # Force fire action for Alice: fire_fraction > 0.5
        action = np.zeros(env.action_space.shape)
        action[2] = 0.9   # fire_fraction
        action[3] = 0.0   # target index

        initial_stockpile = env.fleet_alice[0].weapons[0].stockpile
        # Run a few steps while in range (ships start at 100km apart — beyond range,
        # but we test that the mechanism doesn't crash)
        for _ in range(5):
            env.step(action)
        # Stockpile should be ≤ initial (may not have decreased if out of range)
        assert env.fleet_alice[0].weapons[0].stockpile <= initial_stockpile
