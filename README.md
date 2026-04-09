# PPO Water Allocation Example

This project is a minimal PPO example for water allocation over 5 decision periods.

## Problem setup

- Each episode has 5 time steps.
- Input state:
  - current water demand of `N` channels
  - previous gate openings of `N` channels
- Output action:
  - PPO internally outputs normalized actions of `N` channels in `[0, 1]`
  - the environment maps them to real gate openings in `[gate_open_min, gate_open_max]`
- Hydraulic interface:
  - input: gate openings of `N` channels
  - output: actual water supply of `N` channels within one time period
- Demand transition:
  - next-period demand is determined by previous demand and previous-period actual supply

## Why keep previous gate openings

There is no cross-period total water budget in this version, so the policy does not need:

- remaining total water ratio
- current time-step ratio

The previous gate opening is kept only as a weak auxiliary input so the agent can learn smoother control. If you want a fully simplified version, it can also be removed and the state can be reduced to only the current demands.

## Constraints and reward

- The environment reserves a hydraulic simulator interface for you.
- PPO outputs gate openings, and the environment calls your simulator to get actual supply.
- A default placeholder simulator is included only to keep the example runnable.
- Initial demand is sampled once at the start of an episode.
- The next demand is updated by:

```text
next_demand = max(previous_demand - actual_supply, 0)
```

- Reward focuses on meeting demand as much as possible:
  - less unmet demand is better
  - oversupply has a small penalty
  - gate changes between adjacent periods have a small smoothness penalty
  - channel importance can be controlled by `channel_weights`
  - safety violations on `Z_res`, `Q_res`, and `Qf` can trigger a large penalty
  - early completion can receive an additional bonus based on remaining steps

## Files

- `water_allocation_env.py`: environment definition
- `ppo_agent.py`: PPO policy and value network
- `train.py`: training and evaluation entry point

## Hydraulic simulator interface

The environment provides this method:

```python
def simulate_supply(self, gate_openings: np.ndarray) -> np.ndarray:
    ...
```

You can connect your own model in either of two ways.

Method 1: pass a callback when creating the environment

```python
def my_hydraulic_simulator(gate_openings: np.ndarray) -> np.ndarray:
    # call your hydraulic model here
    return supply

env = WaterAllocationEnv(config, hydraulic_simulator=my_hydraulic_simulator)
```

Method 2: directly replace the placeholder function in `train.py`

```python
def hydraulic_simulator(gate_openings: np.ndarray) -> np.ndarray:
    return your_supply
```

## Run

Install dependencies first:

```bash
pip install numpy torch
```

Then run:

```bash
python train.py --num-channels 4 --train-iterations 200
```

To enable parallel rollout sampling:

```bash
python train.py --num-channels 3 --train-iterations 20 --rollout-episodes 8 --num-workers 4
```

To periodically save named checkpoints and run evaluation:

```bash
python train.py --num-channels 3 --train-iterations 50 --checkpoint-interval 5 --eval-interval 5 --eval-episodes 5
```

Each run creates a timestamped folder under `--output-dir` and stores:

- `args.json`
- `ppo_config.json`
- `env_config.json`
- `training_details.jsonl`
- `evaluation_details.jsonl`
- `checkpoint_latest.pt`
- periodic `checkpoint_iter_XXXX.pt`
- final model file

To resume training from a saved checkpoint:

```bash
python train.py --resume runs/20260408_120000/checkpoint_latest.pt
```

## How to adapt to real data

1. Replace `_sample_initial_demand()` if your initial demand comes from real data.
2. Connect your real hydraulic simulator through `simulate_supply()` or the callback interface.
3. Replace `_transition_demand()` if your demand evolution is more complex than unmet-demand carryover.
4. If some channels are more important, set `channel_weights` in `WaterAllocationConfig`.
5. If gate switching cost is truly unimportant, reduce `smoothness_penalty` further or set it to `0.0`.

Example:

```python
config = WaterAllocationConfig(
    num_channels=3,
    channel_weights=np.array([2.0, 1.0, 0.5], dtype=np.float32),
)
```

Gate opening range example:

```python
config = WaterAllocationConfig(
    num_channels=3,
    gate_open_min=0.03,
    gate_open_max=1.5,
)
```

Safety example:

```python
config = WaterAllocationConfig(
    num_channels=3,
    safe_z_max=1.2,
    safe_q_max=3.0,
    safe_qf_max=np.array([2000.0, 1800.0, 1500.0], dtype=np.float32),
    safety_penalty=5.0,
)
```

Early completion example:

```python
config = WaterAllocationConfig(
    num_channels=3,
    early_completion_bonus=1.0,
)
```
