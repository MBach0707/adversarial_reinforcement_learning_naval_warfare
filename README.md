# Adversarial Reinforcement Learning in Naval Warfare

*MSc Thesis · University of Copenhagen, Faculty of Science · 2025*  
*In collaboration with TERMA A/S*

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
[![CI](https://github.com/MBach0707/adversarial-reinforcement-learning-naval-warfare/actions/workflows/ci.yml/badge.svg)](https://github.com/MBach0707/adversarial-reinforcement-learning-naval-warfare/actions)

---

## Overview

This project investigates whether adversarial reinforcement learning can surface **emergent naval tactics** — manoeuvres that human commanders might not discover within the constraints of real-world exercise time and risk.

The central analogy: chess engines transformed professional chess by revealing tactics beyond human calculation horizons. Could a similar approach give commanding officers an edge in real anti-surface warfare?

Two independent TD3 agents — **Alice** and **Bob** — command opposing fleets on a continuous 200×200 km battlespace. They learn simultaneously, each adapting to the other's evolving strategy, producing an adversarial arms race of tactical discovery.

*The author served as a Lieutenant Commander and Senior Navigation Officer aboard HDMS ABSALON — the same class of frigate modelled in the simulation. All physics parameters and scenario designs are grounded in operational experience.*

---

## Demo

All animations were generated from trained agent checkpoints. **Engagement** views show missile fire arcs and hit markers. **Ghost** views show trajectory trails across the full episode.

### 1v1 — Simple Attraction

| Engagement | Ghost trails |
|:---:|:---:|
| ![1v1 engagement](outputs/1v1_engagement.gif) | ![1v1 ghost](outputs/1v1_ghost.gif) |

Both ships rewarded purely for closing distance. Agents learn a direct approach within ~500 episodes. The baseline scenario used to validate TD3 convergence.

---

### 2v2 — Symmetric Engagement

| Engagement | Ghost trails |
|:---:|:---:|
| ![2v2 engagement](outputs/2v2_engagement.gif) | ![2v2 ghost](outputs/2v2_ghost.gif) |

Two-ship fleets trained adversarially. Formation cohesion (Lennard-Jones formation reward) causes Alice's ships to manoeuvre as a pair rather than independently.

---

### 3v3 — Fleet Engagement

![3v3 ghost](outputs/3v3_ghost.gif)

Three ships per side. Ghost trails illustrate how fleet-level structure emerges: ships spread to cover arc, then converge on engagement.

---

### Emergent Tactics

Distinct tactical patterns that emerged without being explicitly programmed — discovered through reward shaping and adversarial co-training.

| Wolfpack | Hit and Run |
|:---:|:---:|
| ![wolfpack](outputs/wolfpack_ghost.gif) | ![hit and run](outputs/hit_run_ghost.gif) |

| Ambush | Escort |
|:---:|:---:|
| ![ambush](outputs/ambush_ghost.gif) | ![escort](outputs/escort_ghost.gif) |

**Wolfpack** — multiple attackers converge from different bearings to saturate a single defender's AD systems.
**Hit and Run** — attacker fires at range and immediately retreats before counter-fire can land.
**Ambush** — attacker holds position along a predicted transit corridor.
**Escort** — a protecting ship interposes between the threat and a high-value unit.

---

## Key Technical Contributions

### 1. Custom Gymnasium Environment

A full `gymnasium.Env`-compliant naval warfare simulator supporting:

- Continuous 2-D battlespace with physics-constrained movement (knots → m/min)
- Hybrid action space: `[speed_fraction, course, fire_fraction, target_fraction]` per ship
- Multi-ship fleets with configurable weapons (SSM, artillery) and air-defence systems (SAM)
- Stochastic missile engagement resolution with AD intercept mechanics
- Configurable via YAML — swap scenarios without touching code

### 2. Physics-Inspired Reward Shaping

Rather than hand-coding tactical rules, tactics *emerge* from energy minimisation over five composable potential fields:

| Field | Physics Analogy | Naval Behaviour Encouraged |
|---|---|---|
| **Modified Gravity** | 1/D attraction + linear term | Approach opponent at range |
| **Lennard-Jones Supremacy** | Atomic equilibrium distance | Hold at optimal weapon range |
| **Lennard-Jones Formation** | Inter-molecular cohesion | Concentrate friendly forces |
| **Predictive Intercept** | Dead-reckoning | Cut off retreating opponent |
| **Boundary Confinement** | Inverse-distance wall | Stay in operational area |

All fields are implemented as pure functions — independently testable and composable via config weights.

### 3. TD3 with Rare-Event Replay

Standard replay buffers severely undersample missile engagements (~5 per 100 episodes). The `MixedReplayBuffer` maintains a separate rare-event buffer and oversamples firing transitions at a configurable ratio, improving gradient signal for the Q-function near weapon-use events.

### 4. Composable Noise Architecture

Exploration is handled by a hierarchy of composable noise sources: `GaussianNoise`, `OUNoise`, `EpsilonGreedyNoise`, `SparseWeaponNoise` (sparse firing impulses), and `ExpDecayNoise` (annealing wrapper) — all configurable via YAML.

---

## Repository Structure

```
adversarial-reinforcement-learning-naval-warfare/
├── src/
│   └── naval_rl/
│       ├── envs/
│       │   ├── entities.py        # Ship, Weapon, ADMeasure
│       │   └── naval_env.py       # Gymnasium-compliant environment
│       ├── agents/
│       │   ├── td3.py             # TD3 with dual critics + normalisation
│       │   ├── replay_buffer.py   # Replay + rare-event oversampling
│       │   └── noise.py           # Composable exploration noise
│       └── rewards/
│           └── potential_fields.py # Physics-based reward shaping
├── configs/
│   ├── simple_attraction.yaml     # 1v1 baseline (validates convergence)
│   └── cat_and_mouse.yaml         # 2v1 asymmetric pursuit scenario
├── scripts/
│   ├── train.py                   # Training entry point (CLI + W&B)
│   └── evaluate.py                # Load checkpoint + render trajectory
├── tests/
│   └── test_env_and_rewards.py    # Unit tests (pytest)
├── notebooks/legacy/              # Original research notebooks
├── thesis.pdf
├── pyproject.toml
├── requirements.txt
└── .github/workflows/ci.yml      # Lint + test on push
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/MBach0707/adversarial-reinforcement-learning-naval-warfare.git
cd adversarial-reinforcement-learning-naval-warfare
pip install -e ".[dev,wandb,viz]"

# Run unit tests
pytest tests/ -v

# Train — simple 1v1 baseline
python scripts/train.py --config configs/simple_attraction.yaml

# Train — cat and mouse with W&B logging
python scripts/train.py --config configs/cat_and_mouse.yaml --wandb

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/cat_and_mouse/alice_final.pt \
                            --config configs/cat_and_mouse.yaml
```

---

## Scenarios

### Simple Attraction (`configs/simple_attraction.yaml`)

Symmetric 1v1. Both agents rewarded for reducing inter-fleet distance. Used to validate that the TD3 implementation converges — agents should learn a direct approach trajectory within ~500 episodes.

### Cat and Mouse (`configs/cat_and_mouse.yaml`)

Asymmetric 2v1. Alice commands two frigates (20 knots, Harpoon SSMs, ESSM air defence). Bob commands a single slower vessel (7 knots) and is rewarded for surviving each timestep. The speed asymmetry creates a structurally interesting pursuit-evasion problem: Alice must use formation and predictive intercept rewards to cut off Bob rather than simply chasing.

---

## Agent: Twin Delayed DDPG (TD3)

```
Actor:  Linear(obs) → ReLU → Linear → ReLU → Linear → ReLU → Linear → Tanh
Critic: Linear(obs ∥ act) → ReLU → Linear → ReLU → Linear → ReLU → Linear(1)

Two critics per agent (clipped double Q-learning)
Delayed actor updates (every policy_delay critic steps)
Target policy smoothing with clipped Gaussian noise
Gradient clipping: max_norm = 1.0 on all networks
Optional online RunningMeanStd normalisation for obs and rewards
```

---

## Honest Assessment of Results

Training stability in the adversarial multi-agent setting remains the central unsolved challenge. Several failure modes were observed and documented:

**Circular reward hacking:** In symmetric scenarios, agents converge to orbital dynamics at mutual weapon range — a locally optimal Nash equilibrium that avoids engagement entirely.

**Policy collapse in multi-ship scenarios:** The second ship in a fleet often fails to develop meaningful state-action coupling while the opponent's policy stagnates.

**Hybrid action space instability:** The discrete firing decision embedded in a continuous action space creates near-discontinuities in the reward landscape that the critic approximates poorly.

These failure modes are documented in detail in `thesis.pdf` along with proposed remedies (alternating training, centralised critic, entropy-regularised noise).

---

## Tools & Libraries

`Python 3.10+` · `PyTorch 2.0+` · `Gymnasium 0.29+` · `NumPy` · `SciPy` · `Matplotlib` · `W&B` · `PyYAML` · `pytest`

---

## Background

The author served as a **Lieutenant Commander and Senior Navigation Officer** aboard HDMS ABSALON — the same class of vessel modelled in this simulation — including live naval exercises and anti-surface warfare operations in the Gulf of Guinea. This operational experience informed the environment physics, scenario design, and qualitative evaluation of whether emergent agent behaviours are tactically plausible.

---

## Contact

Michael Bach · [michaelbach0707@gmail.com](mailto:michaelbach0707@gmail.com) · [linkedin.com/in/michaelbach07](https://linkedin.com/in/michaelbach07)
