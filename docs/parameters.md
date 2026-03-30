# Parameter Reference

Complete reference for all configurable parameters across the naval adversarial RL system.
Covers fleet entities, reward shaping, agent hyperparameters, noise schedules, and training.

---

## Table of Contents

1. [Environment](#environment)
2. [Fleet Entities](#fleet-entities)
   - [Ship](#ship)
   - [Weapon](#weapon)
   - [ADMeasure (Air Defence)](#admeasure-air-defence)
3. [Reward Shaping](#reward-shaping)
   - [Potential Field Weights](#potential-field-weights)
   - [Field Parameters](#field-parameters)
   - [Terminal & Event Rewards](#terminal--event-rewards)
4. [Agent (TD3)](#agent-td3)
5. [Noise Schedules](#noise-schedules)
   - [Noise Types](#noise-types)
   - [Composing Noise](#composing-noise)
6. [Training Schedule](#training-schedule)
7. [Scenario Config Layout (YAML)](#scenario-config-layout-yaml)
8. [Worked Examples](#worked-examples)

---

## Environment

Defined in `NavalEnv.__init__` and the top-level YAML keys.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid_half` | float (m) | `100_000` | Half-side of the square operational area. Ships exist in `[-grid_half, grid_half]²`. |
| `max_steps` | int | `500` | Episode length in timesteps. Each timestep = 1 simulated minute. |
| `fire_threshold` | float | `0.5` | `fire_fraction` must exceed this value for a firing attempt to be made. |

**Units throughout the codebase**

| Quantity | Unit |
|---|---|
| Position | metres |
| Speed (config) | knots |
| Speed (internal) | m/min (converted via `knots × 1852 / 60`) |
| Course | radians, mathematical convention (0 = East, CCW positive) |
| Range | metres |
| Time | timesteps (minutes) |

---

## Fleet Entities

### Ship

Defined in `entities.Ship`. Each entry under `fleet_alice` / `fleet_bob` in the YAML.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | str | yes | Display name, used in logs and visualisation. |
| `x` | float (m) | yes | Starting x-position. |
| `y` | float (m) | yes | Starting y-position. |
| `course` | float (rad) | no, default `0.0` | Starting heading (East = 0, CCW positive). |
| `max_speed` | float (kn) | no, default `20.0` | Maximum speed in knots. |
| `health` | int | no, default `5` | Hit points. Ship is destroyed when `health ≤ 0`. |
| `grid_size` | [float, float] | no, default `[100000, 100000]` | Operational area half-extents. Should match top-level `grid_half`. |
| `weapons` | list[Weapon] | no | List of weapon systems fitted to the ship. |
| `ad_measures` | list[ADMeasure] | no | List of air-defence systems fitted to the ship. |

**Action space per ship** (4 values, all in `[-1, 1]` before environment remapping):

| Index | Name | Raw range | Remapped to |
|---|---|---|---|
| 0 | `speed_fraction` | `[-1, 1]` clipped to `[0, 1]` | Fraction of `max_speed` |
| 1 | `course` | `[-1, 1]` × π | Heading in `[-π, π]` radians |
| 2 | `fire_fraction` | `[-1, 1]` | `> fire_threshold` triggers a shot; magnitude encodes salvo size |
| 3 | `target_fraction` | `[-1, 1]` | Mapped to opponent ship index via `floor((f+1)/2 × (n_opp−1))` |

**Observation per ship** (5 values, raw — normalisation handled by agent's `RunningMeanStd`):

| Index | Name | Unit |
|---|---|---|
| 0 | x | m |
| 1 | y | m |
| 2 | course | rad |
| 3 | speed | m/min |
| 4 | health | hp |

Total observation vector length = `(n_alice + n_bob) × 5`.

---

### Weapon

Defined in `entities.Weapon`. Listed under a ship's `weapons` key.

| Parameter | Type | Description |
|---|---|---|
| `name` | str | Display name (e.g. `SSM-HARPOON`, `Artillery`). |
| `stockpile` | int | Number of missiles / shells at episode start. Restored on `reset()`. |
| `range` | float (m) | Maximum engagement range. A shot is only attempted if the target is within this range. |
| `cooldown` | int (steps) | Timesteps the weapon must wait between firings. `cooldown=1` = fires every step; `cooldown=50` = fires once per 50 steps. |
| `p_hit` | float `[0,1]` | Per-missile probability of hitting the target (stochastic resolution). |
| `max_salvo_size` | int | Maximum missiles per single firing command. Actual salvo = `round(fire_fraction × max_salvo_size)`, clamped to remaining stockpile. |

**Firing resolution sequence (per weapon, per step):**

1. `fire_fraction > fire_threshold` → attempt
2. `weapon.can_fire()` → stockpile > 0 and cooldown == 0
3. Target must be within `weapon.range`
4. Salvo size computed: `round(fire_fraction × max_salvo_size)`
5. Intercepted by target's AD systems (see below)
6. Remaining missiles each hit independently with probability `p_hit`

---

### ADMeasure (Air Defence)

Defined in `entities.ADMeasure`. Listed under a ship's `ad_measures` key.

| Parameter | Type | Description |
|---|---|---|
| `name` | str | Display name (e.g. `SAM-ESSM`). |
| `stockpile` | int | Number of interceptor missiles at episode start. |
| `range` | float (m) | Currently stored but not used for range-gating intercepts (all inbound missiles are eligible). Reserved for future extension. |
| `cooldown` | int (steps) | Timesteps between intercept attempts. |

AD systems intercept inbound missiles **before** hit resolution. Each active AD system intercepts up to `stockpile` missiles from the inbound salvo, first-come-first-served across all AD systems on the target ship. Remainder proceeds to stochastic hit resolution.

---

## Reward Shaping

Each fleet has an independent reward config (`reward_alice`, `reward_bob` in YAML). This enables asymmetric objectives.

### Potential Field Weights

These weights scale the contribution of each field to the per-ship reward at every timestep.

| Key | Default | Description |
|---|---|---|
| `w_gravity` | `0.0` | Weight for **modified gravity** field (attraction/repulsion toward opponents). Set positive to attract, negative to repel. |
| `w_lj_sup` | `0.0` | Weight for **LJ supremacy** field (engagement-range optimisation). Encourages holding at optimal weapon range. |
| `w_lj_form` | `0.0` | Weight for **LJ formation** field (friendly-fleet cohesion). Only meaningful for fleets with ≥ 2 ships. |
| `w_pred` | `0.0` | Weight for **predictive intercept** field. Rewards heading toward the dead-reckoned future position of opponents. |
| `w_boundary` | `0.0` | Weight for **boundary confinement** penalty. Discourages ships from approaching the grid edge. |

### Field Parameters

#### Modified Gravity (`w_gravity`)

`reward_i = G / D_ij − k_linear × D_ij`   (summed over all opponent ships `j`)

The `1/D` term provides close-range attraction; the `−k·D` term prevents gradient vanishing at long range.

| Key | Default | Description |
|---|---|---|
| `gravity_G` | `1.0` | Gravitational constant — scales the `1/D` attraction term. |
| `gravity_k` | `0.3` | Linear coefficient — scales the `−k·D` long-range gradient term. Higher values pull ships together more aggressively from range. |

> Setting `w_gravity` negative (e.g. Bob's flee config) inverts the field: `1/D` becomes repulsive and the ship is pushed away.

#### LJ Supremacy (`w_lj_sup`)

Lennard-Jones potential centred on the weapon engagement range. Minimum energy (maximum reward) is at `sigma × (n/m)^(1/(n−m))`.

| Key | Default | Description |
|---|---|---|
| `weapon_range` | `10_000.0` m | Effective weapon range used to set the LJ equilibrium distance (`σ_A = weapon_range × √2`). Should match the ship's primary weapon range. |

> Uses fixed exponents `n=4, m=2`. Equilibrium distance ≈ `weapon_range × √2 × √(4/2)^(1/2)` ≈ `weapon_range × 2`.

#### LJ Formation (`w_lj_form`)

Lennard-Jones potential between **friendly** ships. Only active when a fleet has ≥ 2 ships.

| Key | Default | Description |
|---|---|---|
| `d_form` | `5_000.0` m | Desired inter-ship spacing. Equilibrium at `d_form × √2 × √2` ≈ `d_form × 2`. |

#### Predictive Intercept (`w_pred`)

Dead-reckoning reward: uses current opponent velocity to predict their position `t` seconds ahead and rewards closing toward that predicted position.

| Key | Default | Description |
|---|---|---|
| `v_max` | `617.0` m/min | Own fleet's maximum speed in m/min (used to compute intercept time `t = D / v_max`). Convert from knots: `knots × 1852 / 60`. 20 kn ≈ 617 m/min. |

> Higher `v_max` → shorter predicted intercept time → more conservative lead angle. Lower values cause the agent to lead more aggressively.

#### Boundary Confinement (`w_boundary`)

Inverse-distance penalty that activates when a ship enters the outer `(1 − margin)` fraction of the grid.

| Key | Default | Description |
|---|---|---|
| `margin` | `0.9` (hardcoded) | Penalty activates when ship is within `(1 − 0.9) × grid_half = 0.1 × grid_half` of any wall. |

> `margin` is currently hardcoded in `potential_fields.boundary_penalty`. Adjust `w_boundary` scale to tune aggressiveness.

### Terminal & Event Rewards

Applied as one-time additions at the moment of the event, not per-step.

| Key | Default | Description |
|---|---|---|
| `kill_reward` | `100.0` | Added to the shooter's reward when a hit reduces a target ship's HP to 0. |
| `death_penalty` | `100.0` | Subtracted from a ship's reward when it is destroyed. |
| `victory_bonus` | `200.0` | Added to the winning fleet's total reward when the entire enemy fleet is annihilated. |
| `defeat_penalty` | `200.0` | Subtracted from the losing fleet's total reward upon total defeat. |
| `reward_fire` | `0.0` | Added each time a ship fires (regardless of hit). Can be used to encourage aggression. Usually left at 0 to avoid rewarding spamming. |
| `time_penalty` | `0.0` | Added to every ship's reward every timestep. Negative values pressure for efficiency (Alice). Positive values reward survival (Bob). |
| `p_hit` | `0.8` | Per-missile hit probability used in damage resolution. Set here for reward config; should match the weapon's own `p_hit` for consistency. |

---

## Agent (TD3)

Defined in `TD3Agent.__init__`. Set under the `agent` key in YAML (both agents share these unless overridden).

### Network Architecture

| Parameter | Default | Description |
|---|---|---|
| `hidden` | `256` | Width of all hidden layers (actor and both critics). All networks use 3 hidden layers with ReLU activations. Actor output uses Tanh (clips to `[-1, 1]`). |

Input dimensions are inferred automatically:
- Actor input: `obs_dim` → output: `act_dim`
- Critic input: `obs_dim + act_dim` → output: `1`

### Optimisation

| Parameter | Default | Description |
|---|---|---|
| `lr_actor` | `1e-4` | Adam learning rate for the actor network. |
| `lr_critic` | `1e-3` | Adam learning rate for both critic networks. |
| `gamma` | `0.99` | Discount factor for future rewards. |
| `tau` | `0.005` | Polyak averaging coefficient for target network soft updates: `θ_target ← τ·θ + (1−τ)·θ_target`. |
| `grad_clip` | `1.0` | Max gradient norm applied to all networks via `clip_grad_norm_`. Prevents instability from discontinuous reward spikes (e.g. firing events). |

### TD3-Specific

| Parameter | Default | Description |
|---|---|---|
| `policy_delay` | `2` | Actor and target networks only update every `policy_delay` critic steps. Reduces feedback oscillation in adversarial training. |
| `policy_noise` | `0.1` | Std of Gaussian noise added to target policy actions during critic updates (target policy smoothing). Smooths the Q-function and prevents exploitation of sharp maxima. |
| `noise_clip` | `0.3` | Clamp range for target policy smoothing noise: `noise.clamp(-noise_clip, noise_clip)`. |

### Normalisation

| Parameter | Default | Description |
|---|---|---|
| `normalize_obs` | `True` | Online Welford running-mean-std normalisation of observations. Updated on every `store()` call. Applied before inference and training. |
| `normalize_rew` | `True` | Online normalisation of rewards. Rewards are clipped to `±5σ` after normalisation during training. Prevents large terminal rewards from dominating dense step rewards. |

### Replay Buffer

| Parameter | Default | Description |
|---|---|---|
| `buffer_capacity` | `200_000` | Main replay buffer size (circular deque). |
| `rare_ratio` | `0.1` | Fraction of each training batch drawn from the rare-event buffer. `rare_ratio=0.1` with `batch_size=64` → 6 rare transitions + 58 main transitions per update. |

**Rare-event buffer**: Missile engagement transitions (any step where a shot is fired) are stored in both the main and rare buffers. This corrects for the ~5 firing events per 100 episodes base rate and ensures the critic develops a meaningful firing Q-function.

---

## Noise Schedules

Configured per-agent under `agent.noise_alice` and `agent.noise_bob`. Built by `make_noise(cfg)`.

### Noise Types

#### `gaussian` / `gauss` / `normal`

Independent Gaussian noise on each action dimension.

| Key | Description |
|---|---|
| `dim` | Action vector length. |
| `sigma` | Standard deviation. |

#### `ou` / `ornstein-uhlenbeck`

Temporally correlated noise — smoother action trajectories, better for continuous control.

| Key | Default | Description |
|---|---|---|
| `dim` | — | Action vector length. |
| `theta` | `0.15` | Mean-reversion speed. Higher = reverts to zero faster. |
| `sigma` | `0.2` | Diffusion coefficient (noise magnitude). |
| `dt` | `1e-2` | Euler integration timestep. |

#### `epsilon` / `epsilon-greedy`

With probability `epsilon`, returns a uniform random action in `[-1, 1]`. Otherwise returns zero.

| Key | Default | Description |
|---|---|---|
| `dim` | — | Action vector length. |
| `epsilon` | `0.3` | Exploration probability. |

#### `sparse-weapon` / `weapon`

Injects random firing impulses specifically into the `fire_fraction` and `target_fraction` action dimensions. Encourages the agent to discover weapon use during early exploration without corrupting movement actions.

| Key | Default | Description |
|---|---|---|
| `n_ships` | — | Number of ships in the fleet. Noise is applied per-ship independently. |
| `p_fire` | `0.05` | Per-ship probability of injecting a firing impulse on any given step. |
| `fire_low` | `0.6` | Lower bound of the injected `fire_fraction` value (must be > `fire_threshold=0.5`). |
| `fire_high` | `1.0` | Upper bound of the injected `fire_fraction` value. |

> `dim` is inferred automatically as `4 × n_ships`.

#### `decay` / `annealed` / `exp-decay`

Wraps any base noise and multiplies its output by a decaying scale factor.

| Key | Default | Description |
|---|---|---|
| `base` | — | Nested noise config dict for the inner noise source. |
| `scale_init` | `1.0` | Starting scale multiplier. |
| `gamma` | `0.999` | Multiplicative decay factor applied per `per` event. |
| `scale_min` | `0.05` | Floor — scale never drops below this value. |
| `per` | `"step"` | Decay trigger: `"step"` decays every environment step; `"episode"` decays on `noise.reset()` (episode start). |

Schedule: `scale(t) = max(scale_min, scale_init × gamma^t)`

#### `composite` / `combo`

Sums multiple noise sources element-wise. All sources must have the same `dim`.

| Key | Description |
|---|---|
| `parts` | List of noise config dicts to sum. |

### Composing Noise

The typical production pattern for a fleet with firing:

```yaml
noise_alice:
  kind: composite
  parts:
    - kind: decay           # decaying movement noise
      per: episode
      gamma: 0.995
      scale_min: 0.02
      base:
        kind: gaussian
        dim: 8              # 4 actions × 2 ships
        sigma: 0.2
    - kind: sparse-weapon   # independent firing exploration per ship
      n_ships: 2
      p_fire: 0.3
```

The `sparse-weapon` component fires at high `p_fire` early on (when movement noise is large) and continues at the same rate throughout — it does not decay. This is intentional: firing is rare enough that sustained encouragement is needed.

---

## Training Schedule

Set under the `training` key in YAML.

| Key | Description |
|---|---|
| `num_episodes` | Total number of training episodes. |
| `max_steps_per_episode` | Maximum steps before an episode is force-terminated (`truncated=True`). Also sets `NavalEnv.max_steps`. |
| `warmup_steps` | Global steps before any gradient updates begin. Both agents collect experience into their buffers but do not train. Ensures the buffer has enough diversity before learning starts. |
| `batch_size` | Number of transitions sampled per gradient update step. With `rare_ratio=0.1`, `batch_size=64` → 58 main + 6 rare samples. |
| `save_every` | Save agent checkpoints (`alice_epN.pt`, `bob_epN.pt`) every N episodes. |
| `rollout_every` | Capture a deterministic (no-noise) trajectory snapshot every N episodes for visualisation. Stored in `outputs/<run_name>/trajectories.npy`. |

**Checkpoint files saved to `outputs/<run_name>/`:**

| File | Contents |
|---|---|
| `alice_epN.pt` / `bob_epN.pt` | Periodic checkpoints: actor, actor_target, critic1, critic2 weights + `obs_rms` state. |
| `alice_final.pt` / `bob_final.pt` | Final weights after all episodes. |
| `trajectories.npy` | Array of `{"episode": int, "alice": (T, nA, 2), "bob": (T, nB, 2)}` dicts. |
| `tb/` | TensorBoard event files. Run `tensorboard --logdir outputs/<run_name>/tb`. |

---

## Scenario Config Layout (YAML)

Full annotated skeleton:

```yaml
run_name: my_scenario          # used as output subdirectory name
output_dir: outputs            # root output directory
device: cpu                    # "cpu" or "cuda"
grid_half: 100000              # operational area half-size (metres)

fleet_alice:
  - name: Alice-1
    x: -60000                  # starting x (metres)
    y: 0                       # starting y (metres)
    max_speed: 20.0            # knots
    health: 5                  # hit points
    grid_size: [100000, 100000]
    weapons:
      - name: SSM-HARPOON
        stockpile: 6
        range: 50000           # metres
        cooldown: 50           # timesteps
        p_hit: 0.8
        max_salvo_size: 3
    ad_measures:
      - name: SAM-ESSM
        stockpile: 4
        range: 60000
        cooldown: 5

fleet_bob:
  - name: Bob-1
    # ... same structure as fleet_alice

reward_alice:
  # --- Potential field weights ---
  w_gravity:    1.0
  gravity_G:    1.0
  gravity_k:    0.3
  w_lj_sup:     0.0
  weapon_range: 50000
  w_lj_form:    0.0
  d_form:       5000
  w_pred:       0.0
  v_max:        617.0          # 20 knots in m/min
  w_boundary:   0.5
  # --- Terminal / event rewards ---
  kill_reward:   100.0
  death_penalty: 100.0
  victory_bonus: 200.0
  defeat_penalty: 200.0
  reward_fire:   0.0
  time_penalty:  -0.005
  p_hit:         0.8

reward_bob:
  # ... same keys; set independently for asymmetric objectives

agent:
  lr_actor:  0.0001
  lr_critic: 0.001
  gamma:     0.99
  tau:       0.005
  hidden:    256

  noise_alice:
    kind: decay
    dim: 4                     # 4 actions × 1 ship; 8 for 2 ships
    per: episode
    gamma: 0.995
    scale_min: 0.02
    base:
      kind: gaussian
      dim: 4
      sigma: 0.2

  noise_bob:
    # ... same structure

training:
  num_episodes:           1000
  max_steps_per_episode:  300
  warmup_steps:           1000
  batch_size:             64
  save_every:             250
  rollout_every:          100

wandb_project: naval-adversarial-rl   # only used with --wandb flag
```

---

## Worked Examples

### Simple Attraction — Symmetric 1v1

Both ships attract each other. Baseline to validate TD3 converges before adding complexity.

```yaml
reward_alice:
  w_gravity:   1.0    # pure attraction
  gravity_G:   1.0
  gravity_k:   0.3
  w_boundary:  0.5    # keep inside grid
  time_penalty: -0.005

reward_bob:            # identical — symmetric
  w_gravity:   1.0
  gravity_k:   0.3
  w_boundary:  0.5
  time_penalty: -0.005
```

> Hysteresis risk: symmetric + decaying noise → circular orbiting. Expected at high episode counts. Use asymmetry or `w_pred` to break it.

---

### Cat and Mouse — Asymmetric 2v1

Alice (2 ships, faster) hunts Bob (1 ship, slower). Asymmetric rewards produce more stable early training.

```yaml
reward_alice:
  w_lj_sup:    1.0    # hold at engagement range
  weapon_range: 50000
  w_lj_form:   0.5    # keep Alice's 2 ships in formation
  d_form:      8000
  w_pred:      0.5    # lead pursuit
  v_max:       617.0
  w_boundary:  1.0
  time_penalty: -0.01  # pressure to close

reward_bob:
  w_gravity:   -0.5   # negative = repulsive = flee
  w_boundary:  1.0
  time_penalty: +0.05  # rewarded for surviving each step

noise_alice:
  kind: composite
  parts:
    - kind: decay
      per: episode
      gamma: 0.995
      scale_min: 0.02
      base: {kind: gaussian, dim: 8, sigma: 0.2}
    - kind: sparse-weapon
      n_ships: 2
      p_fire: 0.3       # high — Alice needs to discover firing
```

---

### Tuning Guide

| Symptom | Likely cause | Suggested fix |
|---|---|---|
| Agents orbit instead of engaging | Symmetric rewards + decaying noise | Add `w_pred`, use asymmetric `time_penalty`, or increase `p_fire` |
| No firing ever learned | Rare-event undersampling | Increase `rare_ratio`, increase `p_fire` on `sparse-weapon` |
| Training diverges / Q explodes | Learning rate too high or `grad_clip` too loose | Reduce `lr_critic`, tighten `grad_clip` to `0.5` |
| Second ship in fleet does nothing | No formation incentive, no independent noise | Add `w_lj_form > 0`, use `composite` noise with `sparse-weapon` |
| Agents flee to grid edges | `w_boundary` too low | Increase `w_boundary` to `≥ 1.0` |
| Alice over-pursues (never fires) | `w_pred` too high relative to `w_lj_sup` | Reduce `w_pred` or increase `w_lj_sup` to hold at range |
| Reward signal noisy / unstable early | Dense + sparse rewards on different scales | Enable `normalize_rew: true`, increase `warmup_steps` |
