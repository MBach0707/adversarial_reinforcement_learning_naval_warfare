# Boosted Decision-Making with Adversarial Reinforcement Learning Agents in Naval Warfare

*MSc Thesis · University of Copenhagen, Faculty of Science · 2025*  
*In collaboration with TERMA A/S*  
*University Advisor: Professor Troels Christian Petersen*  
*Company Advisor: Jens Egeløkke Frederiksen, TERMA*

---

## Overview

This thesis investigates whether adversarial reinforcement learning can be used to discover emergent tactics in naval warfare simulations — and whether those tactics could boost the decision-making of real warfare officers.

The central question, motivated by how chess engines have transformed professional chess:

> *"If AI can boost decision-making for players in games, could a similar approach give advantages in real-world military contexts?"*

The approach draws a direct analogy between chess engines augmenting grandmasters and a naval simulator augmenting commanding officers. Just as chess players now train with engines to find tactics beyond human calculation horizons, the hypothesis is that adversarial RL agents could explore the vast tactical state-space of naval engagements and surface novel maneuver strategies.

---

## What Was Built

### Mathematical Framework for Tactics

Using game theory, a formal definition of tactics is developed from first principles — first in the context of chess, then extended to a continuous naval environment. A tactic is defined as a set of trajectories navigating from a set of starting states to end states with higher estimated evaluation value, while anticipating opponent counterplay. Three categories of tactics are introduced: human, superhuman, and inconceivable — mirroring the evolution seen in chess since Deep Blue.

### Naval Simulation Environment

A custom naval warfare simulator was built using the standard `gymnasium` Python framework. The environment models anti-surface warfare on a continuous 2D plane (200×200 km), with:

- Ships as point agents with continuous position, heading, and speed
- A hybrid action space: continuous (course, speed) × discrete (missile salvo, targeting)
- Physics-constrained movement with realistic naval parameters (max speed ~20 knots, weapon range 10 km)
- Stochastic missile engagements modeled with hit probabilities
- Discrete status flags (alive/dead, firing threshold, targeting) embedded in a continuous state vector

### Adversarial Commander Framework

Two independent DDPG agents — named Alice and Bob — are trained simultaneously in the environment, each commanding a fleet. Both observe the same state, act simultaneously each timestep, and receive separate reward signals. The framework implements:

- Twin Delayed Deep Deterministic Policy Gradient (TD3) with dual critics per agent
- Separate replay buffers per agent, persisting across episodes
- Soft target network updates (Polyak averaging)
- Delayed actor updates to allow critic stabilization
- Target policy smoothing to prevent overfit on sharp Q-value peaks

### Reward Shaping Through Physical Potential Fields

Rather than hand-coding tactical rules, rewards are shaped using potential field mechanics borrowed from physics — allowing tactics to *emerge* from learned behavior rather than being prescribed. Fields implemented include:

- **Modified Gravity**: Encourages fleets to seek out opponents at long range, with a linear attraction term added for gradient stability at distance
- **Lennard-Jones Potential**: Guides ships with range superiority to maintain optimal engagement distances (attractive at long range, repulsive at close range), adapted from atomic physics with exponents n=4, m=2
- **Formation Cohesion Potential**: LJ-style potential between friendly ships, rewarding concentration of forces while preventing collision
- **Confining Boundary Potential**: Inverse-distance penalty for approaching grid edges, keeping engagements in the open sea
- **Predictive Intercept Field**: Dead-reckoning based potential field placed at the anticipated future position of opponent ships, encouraging interception rather than reactive pursuit

This approach treats reward maximization as analogous to energy minimization in physical systems.

---

## Results

### What Worked

The simple single-agent setup (one ship navigating to a goal) converges cleanly, with the policy flow field evolving from random to optimal direct-route trajectories across training steps. This validates the DDPG implementation.

In the **Simple Attraction** scenario (1v1, annihilation objective), agents initially converge toward expected direct approach trajectories toward each other — qualitatively correct behavior.

In the **Cat and Mouse** scenario (asymmetric: Alice hunts, Bob flees), the asymmetric reward structure produced more stable early training than symmetric setups, and the correct directional behavior (Alice approaching, Bob retreating) was observed in early episodes.

### Stability Issues — Honest Assessment

The core challenge of the thesis is training stability in the adversarial multi-agent setting. Several failure modes were consistently encountered:

**Circular reward hacking (hysteresis):** As noise decays and the scenario becomes increasingly symmetric, agents enter circular orbital dynamics rather than engaging. This is a known pathology of symmetric adversarial DDPG — the only asymmetry driving behavior is noise, which anneals away. The agents discover that orbiting outside weapon range while accumulating step rewards is locally optimal.

**Policy collapse in multi-ship scenarios:** Adding a second ship to Alice's fleet caused training to degrade — the second ship failed to develop meaningful coupling between its state and actions, while Bob's policy stagnated entirely.

**Hybrid action space instability:** Extending the action space to include discrete firing mechanics (via continuous-to-discrete rounding) introduced near-discontinuities in the reward landscape that the critic could not reliably approximate. The rare occurrence of shooting events (~5 per 100 episodes) caused severe sampling imbalance, and biased sampling from a separate rare-event replay buffer did not resolve the issue.

**Mutual overtraining:** The adversarial feedback loop caused agents to overtrain against each other's current policy rather than developing robust general strategies. The policies oscillated between local maxima rather than converging.

### Identified Root Causes

1. DDPG's sensitivity to near-discontinuities from the discrete action transformation
2. Hyperparameter search space likely insufficient given the environment's complexity
3. Energy potential reward gradients may be too steep at close range for stable critic approximation
4. Fully decoupled simultaneous training amplifies instability — each agent's environment is non-stationary from the other's perspective

---

## Key Technical Takeaways

The hybrid state-action space (continuous movement × discrete engagements) is the central unsolved problem. Standard DDPG and TD3 are designed for fully continuous spaces. The approximation of discrete actions via rounding introduces gradient discontinuities that destabilize the critic's Q-value estimates, particularly for rare events like missile engagements.

Future directions suggested in the thesis include:

- **Alternating training**: Lock one agent while the other trains, reducing mutual non-stationarity
- **MADDPG with centralized critic**: Potentially more stable but introduces bias toward cooperative behavior
- **Reward redesign**: Physics-based potential fields may impose overly deterministic dynamics — simpler behavioral reward signals (e.g., rewarding closing rate, penalizing idle behavior) may provide better gradient landscapes
- **Entropy-based or cyclical noise schemes**: To maintain exploration at different phases without the asymmetry collapse seen with decaying epsilon-greedy noise

---

## Repository Structure

```
naval-combat-rl/
├── notebooks/          # Experiments and training runs
├── figures/            # Training curves, trajectory plots, loss plots
├── thesis.pdf          # Full MSc thesis
└── requirements.txt    # Python dependencies
```

---

## Tools and Libraries

`Python` · `PyTorch` · `TorchRL` · `Stable-Baselines3` · `Gymnasium` · `Pandas` · `SciPy` · `NumPy` · `Matplotlib`

---

## Background

The author served as a Lieutenant Commander and Senior Navigation Officer aboard HDMS ABSALON — the same class of frigate modeled in the simulation. The environment design, scenario selection, and tactical framing are grounded in direct operational experience, including live naval exercises and anti-surface warfare operations in the Gulf of Guinea.

This domain expertise informed both the physics of the simulation and the qualitative evaluation of whether emergent agent behaviors were tactically plausible.

---

## Contact

Michael Bach · michaelbach0707@gmail.com · [linkedin.com/in/michaelbach07](https://linkedin.com/in/michaelbach07)
