# Self-Driving Car Simulation with Deep RL — Reward vs. Observation Design

> **A Comparative Study of Reward and Observation Design**
> Department of Computer Science, Tunghai University — Undergraduate Capstone Project (專題報告)
> Advisor: Prof. 陳仕偉 · Authors: 黃子修, 鄭聿宏, 侯姿佑 · June 2026

This project takes the open-source [NeuralNine/ai-car-simulation](https://github.com/NeuralNine/ai-car-simulation) — which trains a 2D self-driving car with **NEAT**, a gradient-free neuroevolution method — and rebuilds it into a standard **Gymnasium environment** trained with **PPO (Stable-Baselines3)**, turning it into a platform for systematic ablation.

With the algorithm, hyperparameters, and seed all held fixed, we ask:

> **Which matters more for policy quality — reward shaping or observation design — and which aspect of quality does each one actually improve?**

This matters practically: when engineering effort is limited, should it go into *reward engineering* or into *representation design*? We answer it with a **2×2 factorial ablation** in which the only variables are the reward composition and the observation composition.

**Documents:** [Full written report and presentation slides](https://drive.google.com/drive/folders/1uO0yORbg89gwcLztfRHHgSC6nyjCx31R?usp=sharing) (Google Drive)

---

## TL;DR — Key Findings

1. **Reward shaping and observation augmentation are complementary, not interchangeable.** Each alone lifts mean survival from 276 to ~437 steps — a statistical dead heat. The difference is *which* aspect of quality each one fixes:
   - **Observation augmentation collapses variance** (std 88 → **5** steps) but triples control jerk (0.85 → 2.25).
   - **Reward shaping preserves smoothness** (jerk 0.82, the best of all four) but leaves variance high (± 94).
   - Only the **`full`** condition gets high mean *and* low variance together (470 ± 12).
2. **Their gains overlap on any single metric.** Combined, survival reaches only +70% over baseline, far short of the ~150% that independent effects would predict — the two interventions are relieving much of the same bottleneck.
3. **Naive PPO fails completely across track geometries.** A policy trained on the easy oval reaches **0% crash / full 1500 steps** in-distribution, but scores **100% crash / 22 ± 0.7 steps** on the winding track. We give a geometric explanation for *why* in [§7](#7-geometric-analysis-why-transfer-fails-and-why-policies-slow-down).
4. **Two honest caveats up front:** all runs use a **single seed**, and at 500k steps **none of the training curves had plateaued**. These results are therefore a **sample-efficiency comparison, not asymptotic performance.**

> This repository is a controlled ablation study, not a showcase of a finished racing agent. Every number below is taken verbatim from the `eval_results.json` files and TensorBoard logs produced by the runs described in §4 — **all of which are committed to this repository** (see [§11](#11-verifying-the-results)), so any claim here can be independently re-run.

---

## Contributions

| Role | Person |
|---|---|
| Environment refactor (Gymnasium), PPO training pipeline, modular reward/observation design, ablation design, evaluation metrics, geometric analysis — i.e. all code and experiments | **黃子修** |
| Presentation delivery | 鄭聿宏, 侯姿佑 |

Submitted as a team capstone; the report PDF lists all three authors per course submission requirements.

---

## 1. Background: Why Replace NEAT with PPO?

The original project uses **NEAT** (Stanley & Miikkulainen, 2002), which evolves network weights *and* topology via a genetic algorithm. It never computes a gradient — so strictly speaking it falls outside deep learning, which was the first problem for a deep-learning course project.

The deeper motivation was methodological. **NEAT's fitness is a single scalar computed only at the end of an episode**, which makes it nearly impossible to attribute an improvement to any individual design decision. Recasting the task as PPO with *pluggable* reward and observation components is precisely what makes component-wise ablation possible.

| Aspect | NEAT (original) | PPO (this project) |
|---|---|---|
| Learning method | Genetic algorithm | Policy gradient (Actor–Critic) |
| Parameter update | Mutation + crossover | Backpropagation |
| Uses gradients | No | Yes (clipped surrogate objective) |
| Sample efficiency | Low | High |
| Deep learning | ✗ | ✓ |

For comparability, the `progress` reward is deliberately defined as `speed / 30` — its cumulative return is proportional to total distance travelled, mirroring the distance-based fitness of the original NEAT setup, so the baseline is a fair analogue of the original.

---

## 2. Environment Engineering (`car_env.py`)

The original simulation was tightly coupled to NEAT and the pygame draw loop. Three refactors made it trainable:

1. **Gymnasium API** — implemented `reset` / `step` / `observation_space` / `action_space` so SB3's PPO can drive it directly, and so it can be vectorized with `SubprocVecEnv`.
2. **Headless + array-based sensing** — `SDL_VIDEODRIVER=dummy` removes the window during training, and collision/radar lookups read a NumPy array (`pygame.surfarray.array3d`) instead of per-pixel `Surface.get_at()` calls.
3. **Modular components** — reward and observation are both decomposed into independently toggleable pieces driven by a config dict, so **any ablation combination requires only a config change, never a code change.**

### Task Definition (MDP)
- **State** — 5 radar distances at −90°, −45°, 0°, +45°, +90° (baseline); optionally extended (§3.2)
- **Action** — `Discrete(4)`: turn left 10°, turn right 10°, decelerate −2, accelerate +2; speed clamped to `[12, 40]`
- **Termination** — collision with the white boundary, **or** the 1500-step cap

> **Note: the action space has no no-op.** Every step must change heading or speed. This becomes important in §7.

### Lap Detection (`_check_lap`)
Implemented as **segment intersection plus a directionality test**: a lap counts only when the car's centre path crosses the finish line *and* the displacement vector has positive dot product with `forward_dir`. This rejects false laps from reversing or jittering on the line. A 5-step grace period follows each reset.

---

## 3. Design Axes

### 3.1 Reward Components
Total reward is the sum of enabled components. **Formulas and weights are never modified during ablation** — only toggled — so the comparison stays clean.

| Component | Formula | Intent | Magnitude @ speed ≈ 13 |
|---|---|---|---|
| `progress` | `speed / 30` | Encourage forward travel (NEAT distance proxy) | **≈ 0.44** ← dominant |
| `center` | `0.1 × (1 − \|L−R\| / (L+R))` | Reward staying mid-track | ≤ 0.10 |
| `smooth` | `−0.05` on action change | Reduce jittery control | ≈ −0.03 |
| `crash` | `−1.0` on collision (terminal) | Penalize crashing | −1.0 (one-off) |
| `speed` | `0.01 × (speed−12) / 28` | Encourage higher speed | **≈ 0.0005** |

### 3.2 Observation Components
All normalized to `[-1, 1]`.

| Component | Dims | Intent |
|---|---|---|
| `radar` (baseline) | 5 | Distance to boundary (L / FL / F / FR / R) |
| `speed` | 1 | Own current speed |
| `angle` (sin/cos) | 2 | Heading, encoded to avoid the 0°/360° discontinuity |
| `action_history` | 16 | One-hot of the last 4 actions — a hand-built short-term memory |
| **Total** | **5 → 24** | baseline → full |

---

## 4. Experimental Design

| Condition | Observation | Reward |
|---|---|---|
| `baseline` | radar (5) | `progress` only |
| `reward_only` | radar (5) | all 5 components |
| `obs_only` | full (24) | `progress` only |
| `full` | full (24) | all 5 components |

**Held constant across all four:** PPO hyperparameters (SB3 defaults: `lr=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `γ=0.99`, `gae_lambda=0.95`, `clip=0.2`, `net_arch=[64,64]`, `tanh`), 500,000 timesteps, 16 parallel envs, `seed=42`, and the training map `map3.png`.

**Evaluation protocol:** 20 episodes, `seed=999`, deterministic policy, with a **±15° random perturbation of the starting heading** to probe robustness to initial pose. Metrics: survival steps, crash rate, **jerk** (mean norm of the third difference of position), and **action switch rate**.

**Generalization test:** a model trained on the easy oval `map.png` is evaluated directly on `map3.png`, unchanged.

**Training on CPU.** With 5–24 observation dims and a `[64, 64]` MLP, GPU kernel-launch and transfer overhead outweighs the compute; CPU measured faster.

---

## 5. Results

### 5.1 Training Curves (map3, 500k steps)

Final `rollout/ep_rew_mean`, read from the TensorBoard logs:

| Rank | Condition | ep_rew_mean @ 500k | ep_len_mean @ 500k | Slope over final quarter |
|---|---|---|---|---|
| 1 | `reward_only` | **177** | 359 | +64 / 100k steps |
| 2 | `full` | 167 | 331 | +60 / 100k steps |
| 3 | `obs_only` | 131 | 261 | **+72 / 100k steps** |
| 4 | `baseline` | 102 | 217 | +33 / 100k steps |

**All four curves were still climbing at 500k with no plateau.** This ranking is therefore a snapshot of **sample efficiency**, not converged performance — and note that `obs_only`, ranked 3rd, had the *steepest* final slope and was still accelerating, so the ordering could well change with a longer budget.

### 5.2 Ablation Results (map3, eval, n = 20)

| Condition | Obs | Reward | Survival (steps) | Distance | Mean speed | Jerk ↓ | Switch rate ↓ | Crash |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 5 | progress | 276 ± 88 | 3379 ± 1069 | 12.26 | **0.85** | **32.1%** | 100% |
| `reward_only` | 5 | all | 437 ± 94 | 5309 ± 1140 | 12.15 | **0.82** | **32.6%** | 100% |
| `obs_only` | 24 | progress | 436 ± **5** | **6370 ± 30** | 14.60 | 2.25 | 51.0% | 100% |
| `full` | 24 | all | **470 ± 12** | 6241 ± 48 | 13.28 | 2.35 | 58.6% | 100% |

*(Evaluation numbers exceed the training curves because evaluation uses a deterministic policy while training samples stochastically.)*

**Improvement over baseline:**

| | Survival | Distance |
|---|---|---|
| `reward_only` | +58.3% | +57.1% |
| `obs_only` | +58.1% | +88.5% |
| `full` | +70.3% | +84.7% |
| *if the two effects were independent (predicted)* | *+150%* | *+196%* |

### 5.3 Generalization Failure

| | Crash rate | Survival |
|---|---|---|
| **In-distribution** (train `map.png` → eval `map.png`) | **0%** | 1500 steps (full episode) |
| **Out-of-distribution** (train `map.png` → eval `map3.png`) | **100%** | **22 ± 0.7 steps** |

A policy that looks *perfect* on its training track fails at the very first corner of a different one. The std of 0.7 steps shows the failure is not stochastic — it is a deterministic, structural mismatch. §7 explains the mechanism.

---

## 6. Analysis: Reward and Observation Play Different Roles

The headline finding is that these two axes are **complementary rather than interchangeable**:

| Aspect | Reward shaping | Observation augmentation |
|---|---|---|
| Mean performance | ✓ 276 → 437 | ✓ 276 → 436 |
| Variance (std) | ✗ high (± 94) | ✓ **very low (± 5)** |
| Action smoothness | ✓ **low jerk (0.82)** | ✗ high jerk (2.25) |

Three points follow.

**(a) On mean survival they are a dead heat, and their gains overlap.** +58.3% vs +58.1% individually, but only +70.3% together — far below the +150% independent effects would give. On *distance*, `full` (6241) is actually *below* `obs_only` (6370). Both interventions appear to be relieving much of the same bottleneck (getting the car to slow down and stay centred), so once one has done so, the other has little headroom and may even interfere through competition among reward terms.

**(b) Variance reduction is the observation axis's job alone.** Enriching observations cuts the survival std from 88 to **5** and the distance std from 1069 to 30, making the policy almost insensitive to the ±15° start perturbation. This is evidence that the 5-dim radar-only setup makes the task **partially observable (a POMDP)** — the policy cannot infer its own speed or heading from range readings alone, so behaviour is not reproducible until proprioceptive state is supplied.

**(c) Smoothness is the reward axis's job — but the `smooth` term still failed.** `full` adds a `−0.05` action-change penalty on top of `obs_only`, yet the switch rate *rose* (51.0% → 58.6%) and jerk rose slightly (2.25 → 2.35). Magnitude analysis explains why: at speed ≈ 13, `progress` contributes ≈ 0.44 per step while the expected `smooth` penalty is only ≈ −0.03, roughly 7% of it — not enough to shift the policy's preferences.

### A methodological caveat on the reward ablation
Two of the five reward terms are **numerically inert**:

- **`speed`** caps at 0.01, which is **0.75%** of `progress`'s maximum of 1.33. At the observed speeds it contributes ≈ 0.0005 — indistinguishable from being switched off.
- **`crash`** applies a one-off `−1.0`, roughly **0.5%** of the true opportunity cost of crashing early (~400 lost steps × 0.44 ≈ 176 of forgone return). `progress` already encodes a far stronger *implicit* crash penalty, making this term largely redundant.

So `reward_only`'s gain most plausibly comes from **`center` alone**. The lesson is that toggling reward components on and off is not sufficient — **their magnitudes must be aligned first**, or a nominally "enabled" component may do nothing and the ablation conclusion becomes misleading.

---

## 7. Geometric Analysis: Why Transfer Fails, and Why Policies Slow Down

The generalization failure in §5.3 is usually reported as an empirical fact. Measuring the environment's geometry gives a concrete mechanism.

### 7.1 Minimum turning radius is set by speed

The car advances a fixed `speed` pixels per step while turning a fixed 10°, so its path curvature radius is:

```
R = speed / radians(10) = speed / 0.1745
```

| Speed | Min turning radius | Turning diameter |
|---|---|---|
| 12 (floor) | 69 px | 138 px |
| 20 (initial) | 115 px | 229 px |
| **27.2** (learned on `map.png`) | **156 px** | **312 px** |
| 40 (ceiling) | 229 px | 458 px |

Measuring drivable clearance with a distance transform: **the maximum clearance radius anywhere on `map3.png` is 83 px** — i.e. the widest corridor on the entire map is about 166 px across, and the car's own collision radius is 30 px.

### 7.2 This explains the transfer failure

The `map.png`-trained policy cruises at **27.2 px/step**. That is optimal on a wide, gently-curving oval, where the required turn radius is large. But it *couples* the policy to a **minimum turning circle of 312 px** — nearly **twice the width of the widest corridor on `map3`**.

The policy is therefore not merely under-trained on `map3`; at its learned cruising speed it is **geometrically incapable of negotiating the first tight corner**. That is exactly what a 22 ± 0.7 step deterministic crash looks like.

**Speed and minimum curvature are coupled through the fixed 10° steering increment.** A policy that learns a fast cruising speed on an easy track acquires a large minimum turning radius as a side effect, and that radius is what fails to transfer.

### 7.3 The same mechanism explains the speed floor

All four `map3` conditions converge to a mean speed of **12.15–14.60**, hugging the floor of 12 despite a ceiling of 40. Under the coupling above, **slowing down is the only lever the agent has for shrinking its turning radius** into the feasible range for a winding track. The convergence is not incidental — it is forced by the geometry.

### 7.4 A related artifact: the missing no-op

Because the action space offers only {left, right, decelerate, accelerate}, "drive straight" must be approximated by alternating ±10°, producing a zig-zag by construction. But note that in `step()`, decelerating does nothing when `speed − 2 < 12` — so **once speed is pinned to the floor, `decelerate` degenerates into an effective no-op**, and the policy gains a way to travel straight.

The data is consistent with policies exploiting this:

| Condition | Mean speed | Jerk | Switch rate |
|---|---|---|---|
| `reward_only` | 12.15 | 0.82 | 32.6% |
| `baseline` | 12.26 | 0.85 | 32.1% |
| `full` | 13.28 | 2.35 | 58.6% |
| `obs_only` | 14.60 | 2.25 | 51.0% |

The two conditions pinned to the floor show jerk ≈ 0.8; the two that drift above it — and must therefore alternate turns to go straight — rise past 2.2.

> ⚠️ §7.3 and §7.4 are **mechanistic inferences consistent with the measurements**, not yet confirmed by a controlled re-run with an explicit no-op action and a finer steering increment. Because training had also not converged (§5.1), we **cannot claim the action space is the sole binding constraint** — only that it is a measurable, previously unexamined one. Distinguishing the two explanations is the top-priority follow-up.

---

## 8. Limitations

Stated plainly — these mark future work, not a retraction of the results.

1. **Single seed.** All runs use `seed=42`. The standard for RL reproducibility is 3–5 seeds (Henderson et al., 2018). The `±` values reported here reflect variance **across evaluation episodes** (from start-angle perturbation), **not across training runs**, so between-condition differences carry no statistical significance test.
2. **500k steps is not converged.** All four curves were still rising. Results compare **sample efficiency**, not asymptotic performance.
3. **No lap completions on `map3`.** Every condition crashes before finishing a lap, so **survival steps serve as a proxy** for the real objective of lap time. Note that episodes end at 276–470 steps against a 1500-step cap — they are terminated by *crashing*, not by exhausting the step budget.
4. **Hand-tuned reward weights.** Coefficients (0.05, 0.1, …) were set by intuition, with no weight-sensitivity ablation — and §6 shows two of them are numerically inert as a result.
5. **Limited map calibration.** `assets/maps/` holds 6 maps, but only `map.png` and `map3.png` have finish lines and start poses configured in `MAP_INFO`; `map2/4/5/6` are not yet calibrated and cannot be used for lap measurement.

---

## 9. Future Work

1. **Domain randomization** — train across mixed maps to attack the generalization failure directly.
2. **Curriculum learning** — `map` → `map2` → `map3` → `map4`, easiest first.
3. **Recurrent policy (LSTM/GRU)** to replace the hand-built `action_history` with learned temporal memory.
4. **Multi-seed replication** (3–5 seeds) with statistical significance testing.
5. **Longer training** past the point where curves plateau, so asymptotic performance can be compared.
6. **Finer action space** — add an explicit no-op, reduce the steering increment (10° → 2–5°), or move to continuous control with SAC/TD3. This is the controlled test of the §7 hypothesis.
7. **Reward-weight sensitivity ablation**, after normalizing component magnitudes.

---

## 10. Installation and Reproduction

### Requirements
- Python 3.8+; runs on native Windows and WSL2
- Training is headless by default — no display required
- **CPU-bound**, not GPU-bound: a multi-core CPU to raise `--n-envs` matters far more than a graphics card

```bash
git clone https://github.com/h-s-i-u/rl-car-reward-vs-observation
cd rl-car-reward-vs-observation

pip install -r requirements.txt
# trajectory_viz.py additionally needs:
pip install matplotlib pillow
```

### Train
```bash
python train.py --exp baseline --map assets/maps/map3.png \
                --timesteps 500000 --seed 42 --n-envs 16 --vec subproc
# --exp accepts: baseline / reward_only / obs_only / full
```
Output lands in `logs/ppo_{exp}_{map}_{timestamp}_s{seed}/` with `final_model.zip`, periodic checkpoints, and an `exp_config.json` recording every setting used — which is what makes runs reproducible and evaluation self-configuring.

Full ablation sweep (all four conditions): `./run_all.ps1`

### Evaluate
```bash
python eval.py logs/ppo_full_map3_<timestamp>_s42/final_model.zip --episodes 20
```
`eval.py` reads the obs/reward configuration back from the `exp_config.json` beside the model, so evaluation can never be silently mismatched to training. Add `--render` to watch live, or `--map` to run the cross-track generalization test.

### Visualize and monitor
```bash
python trajectory_viz.py logs/ppo_full_map3_<ts>_s42/final_model.zip --episodes 10 --out trajectory.png
tensorboard --logdir ./logs/tb/
```

---

## 11. Verifying the Results

Every trained policy, evaluation output, and training curve quoted in this README is **committed to the repository** — none of the numbers above require taking our word for it:

| Artifact | Path | What it lets you check |
|---|---|---|
| Trained policies | `logs/*/final_model.zip` | Re-run any condition's evaluation yourself |
| Evaluation metrics | `logs/*/eval_results.json` | The exact per-episode and aggregate numbers in §5.2 |
| Run configuration | `logs/*/exp_config.json` | Which reward/obs components and hyperparameters each run used |
| Training curves | `logs/tb/` | The §5.1 curves, including the non-convergence claim |

To reproduce the §5.2 table without retraining:

```bash
python eval.py logs/ppo_full_map3_20260527_042616_s42/final_model.zip --episodes 20
python eval.py logs/ppo_obs_only_map3_20260527_040739_s42/final_model.zip --episodes 20
python eval.py logs/ppo_reward_only_map3_20260527_034916_s42/final_model.zip --episodes 20
python eval.py logs/ppo_baseline_map3_20260527_033037_s42/final_model.zip --episodes 20
```

To reproduce the §5.3 generalization failure — the same policy on its own track, then on the winding one:

```bash
python eval.py logs/ppo_baseline_map_20260527_010212_s42/final_model.zip --map assets/maps/map.png  --episodes 20
python eval.py logs/ppo_baseline_map_20260527_010212_s42/final_model.zip --map assets/maps/map3.png --episodes 20
```

> ⚠️ **Known issue:** `eval.py` writes `eval_results.json` next to the model without recording *which* map it evaluated on, so running the two commands above in sequence overwrites the first result. The committed `eval_results.json` for that run is therefore the **`map3` (out-of-distribution)** evaluation. Recording the eval map in the output is a pending fix.

Mid-training checkpoints are excluded from version control (~8.6 MB of intermediate state that adds nothing the final models and curves don't already provide).

---

## 12. Project Structure

```
├── car_env.py              # Gymnasium env: kinematics, radar ray-casting, pixel
│                           #   collision, pluggable obs/reward, lap detection
├── configs.py              # Single source of truth: map metadata, locked PPO
│                           #   hyperparameters, reward/obs sets, 4 conditions
├── train.py                # PPO training entry point (vectorized, checkpointed)
├── eval.py                 # Crash rate, lap time, jerk, switch rate, distance
├── trajectory_viz.py       # Trajectory overlay analysis
├── lap.py                  # Standalone LapDetector (earlier iteration; the live
│                           #   logic is inlined in car_env._check_lap)
├── tools/pick_coords.py    # Click a map to print pixel coords — used to
│                           #   calibrate finish lines and start poses
├── run_all.ps1             # Batch training over the four conditions
├── eval_all.ps1            # Batch cross-map evaluation
├── assets/maps/            # map.png ~ map6.png (map and map3 calibrated)
└── logs/                   # Committed results: trained models, eval metrics,
                            #   run configs, TensorBoard curves (see §11)
```

---

## 13. References

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
2. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," *ICLR*, 2016.
3. K. O. Stanley, R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," *Evolutionary Computation*, 10(2), pp. 99–127, 2002.
4. P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger, "Deep Reinforcement Learning that Matters," *AAAI*, 32(1), pp. 3207–3214, 2018.
5. A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR*, 22(268), pp. 1–8, 2021.
6. NeuralNine, "ai-car-simulation," GitHub. https://github.com/NeuralNine/ai-car-simulation
