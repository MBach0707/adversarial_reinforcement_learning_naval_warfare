#!/usr/bin/env python3
"""
train.py — Main training entry point for adversarial naval RL.

Usage
-----
  python scripts/train.py --config configs/simple_attraction.yaml
  python scripts/train.py --config configs/cat_and_mouse.yaml --wandb

The script:
  1. Loads scenario config from YAML
  2. Builds fleets, environment, and two TD3 commanders
  3. Runs the training loop with W&B logging (optional)
  4. Saves checkpoints and trajectory snapshots
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from naval_rl.agents.td3 import TD3Agent
from naval_rl.envs.entities import ADMeasure, Ship, Weapon
from naval_rl.envs.naval_env import NavalEnv


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fleet factory
# ---------------------------------------------------------------------------

def build_fleet(fleet_cfg: list) -> list[Ship]:
    ships = []
    for s_cfg in fleet_cfg:
        ship = Ship(
            name      = s_cfg["name"],
            x         = s_cfg["x"],
            y         = s_cfg["y"],
            max_speed = s_cfg.get("max_speed", 20.0),
            health    = s_cfg.get("health", 5),
            grid_size = tuple(s_cfg.get("grid_size", [100_000, 100_000])),
        )
        for w_cfg in s_cfg.get("weapons", []):
            ship.add_weapons([Weapon(
                name           = w_cfg["name"],
                stockpile      = w_cfg["stockpile"],
                range          = w_cfg["range"],
                cooldown_max   = w_cfg["cooldown"],
                p_hit          = w_cfg.get("p_hit", 0.8),
                max_salvo_size = w_cfg.get("max_salvo_size", 3),
            )])
        for ad_cfg in s_cfg.get("ad_measures", []):
            ship.add_ad_measures([ADMeasure(
                name         = ad_cfg["name"],
                stockpile    = ad_cfg["stockpile"],
                range        = ad_cfg["range"],
                cooldown_max = ad_cfg["cooldown"],
            )])
        ships.append(ship)
    return ships


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any], use_wandb: bool = False) -> None:
    run_name = cfg.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir  = Path(cfg.get("output_dir", "outputs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    if use_wandb:
        import wandb
        wandb.init(
            project = cfg.get("wandb_project", "naval-adversarial-rl"),
            name    = run_name,
            config  = cfg,
        )

    # --- Build environment ---
    fleet_alice = build_fleet(cfg["fleet_alice"])
    fleet_bob   = build_fleet(cfg["fleet_bob"])
    env = NavalEnv(
        fleet_alice   = fleet_alice,
        fleet_bob     = fleet_bob,
        cfg_alice     = cfg["reward_alice"],
        cfg_bob       = cfg["reward_bob"],
        max_steps     = cfg["training"]["max_steps_per_episode"],
        grid_half     = cfg.get("grid_half", 100_000.0),
    )

    state_dim  = env.observation_space.shape[0]
    n_alice    = len(fleet_alice)
    n_bob      = len(fleet_bob)
    act_dim_A  = 4 * n_alice
    act_dim_B  = 4 * n_bob

    agent_cfg  = cfg["agent"]
    device     = cfg.get("device", "cpu")

    # --- Build agents ---
    alice = TD3Agent(
        state_dim  = state_dim,
        action_dim = act_dim_A,
        noise_cfg  = agent_cfg["noise_alice"],
        lr_actor   = agent_cfg["lr_actor"],
        lr_critic  = agent_cfg["lr_critic"],
        gamma      = agent_cfg["gamma"],
        tau        = agent_cfg["tau"],
        hidden     = agent_cfg.get("hidden", 256),
        device     = device,
    )
    bob = TD3Agent(
        state_dim  = state_dim,
        action_dim = act_dim_B,
        noise_cfg  = agent_cfg["noise_bob"],
        lr_actor   = agent_cfg["lr_actor"],
        lr_critic  = agent_cfg["lr_critic"],
        gamma      = agent_cfg["gamma"],
        tau        = agent_cfg["tau"],
        hidden     = agent_cfg.get("hidden", 256),
        device     = device,
    )

    train_cfg     = cfg["training"]
    num_episodes  = train_cfg["num_episodes"]
    warmup_steps  = train_cfg["warmup_steps"]
    batch_size    = train_cfg["batch_size"]
    save_every    = train_cfg.get("save_every", 500)
    rollout_every = train_cfg.get("rollout_every", 200)

    global_step = 0
    all_trajectories = []

    # --- Main loop ---
    for ep in range(num_episodes):
        obs, _ = env.reset()
        alice.noise.reset()
        bob.noise.reset()

        ep_reward  = np.zeros(2)
        done = False

        traj_A, traj_B = [], []

        while not done:
            a_alice = alice.select_action(obs)
            a_bob   = bob.select_action(obs)
            action  = np.concatenate([a_alice, a_bob])

            next_obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rare = info["rare_event"]
            alice.store(obs, a_alice, rewards[0], next_obs, done, rare=rare)
            bob.store(obs, a_bob,     rewards[1], next_obs, done, rare=rare)

            if global_step >= warmup_steps:
                alice.train(batch_size)
                bob.train(batch_size)

            ep_reward += rewards
            obs = next_obs
            global_step += 1

        # --- Episode logging ---
        log = {
            "episode": ep,
            "reward_alice": float(ep_reward[0]),
            "reward_bob":   float(ep_reward[1]),
            "alice/actor_loss":  alice.actor_loss,
            "alice/critic_loss": alice.critic_loss,
            "alice/mean_q":      alice.mean_q,
            "alice/noise_rms":   alice.noise_rms,
            "bob/actor_loss":    bob.actor_loss,
            "bob/critic_loss":   bob.critic_loss,
            "bob/mean_q":        bob.mean_q,
            "bob/noise_rms":     bob.noise_rms,
        }

        if ep % 50 == 0:
            print(
                f"Ep {ep:5d} | "
                f"Alice: {ep_reward[0]:8.2f}  Bob: {ep_reward[1]:8.2f} | "
                f"Step: {global_step:7d}"
            )

        if use_wandb:
            import wandb
            wandb.log(log, step=global_step)

        # --- Greedy rollout for trajectory capture ---
        if ep % rollout_every == 0 and ep > 0:
            traj_obs, _ = env.reset()
            traj_done = False
            traj_A, traj_B = [], []
            while not traj_done:
                traj_A.append(np.array([[s.x, s.y] for s in fleet_alice]))
                traj_B.append(np.array([[s.x, s.y] for s in fleet_bob]))
                a_A = alice.select_action_deterministic(traj_obs)
                a_B = bob.select_action_deterministic(traj_obs)
                traj_obs, _, t1, t2, _ = env.step(np.concatenate([a_A, a_B]))
                traj_done = t1 or t2
            all_trajectories.append({
                "episode": ep,
                "alice": np.array(traj_A, dtype=np.float32),
                "bob":   np.array(traj_B, dtype=np.float32),
            })

        # --- Checkpointing ---
        if ep % save_every == 0 and ep > 0:
            alice.save(str(out_dir / f"alice_ep{ep}.pt"))
            bob.save(str(out_dir / f"bob_ep{ep}.pt"))

    # --- Final save ---
    alice.save(str(out_dir / "alice_final.pt"))
    bob.save(str(out_dir / "bob_final.pt"))

    if all_trajectories:
        traj_path = out_dir / "trajectories.npy"
        np.save(traj_path, np.array(all_trajectories, dtype=object), allow_pickle=True)
        print(f"Saved {len(all_trajectories)} trajectory snapshots → {traj_path}")

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"Training complete. Outputs in: {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adversarial naval RL agents")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--wandb",  action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, use_wandb=args.wandb)
