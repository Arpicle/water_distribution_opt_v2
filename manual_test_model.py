from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppo_agent import ActorCritic
from simulation import hydraulic_simulator
from water_allocation_env import WaterAllocationConfig, WaterAllocationEnv


@dataclass
class ManualTestConfig:
    run_dir: Path = Path("runs") / "your_run_dir"
    model_name: str | None = None
    manual_demands: list[list[float]] = field(
        default_factory=lambda: [
            [8000.0, 12000.0, 15000.0],
        ]
    )
    seed: int = 1000
    deterministic: bool = True
    device: str = "cuda"
    output_dir: Path | None = None


CONFIG = ManualTestConfig(
    # Edit these values before running: python manual_test_model.py
    run_dir=Path("runs") / "your_run_dir",
    model_name=None,  # None means auto-select checkpoint_latest.pt or latest checkpoint_iter_*.pt.
    manual_demands=[
        [8000.0, 12000.0, 15000.0],
        [10000.0, 18000.0, 22000.0],
    ],
    seed=1000,
    deterministic=True,
    device="cuda",
    output_dir=None,
)


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env_config(run_dir: Path) -> WaterAllocationConfig:
    config = _load_json(run_dir / "env_config.json")
    if config.get("channel_weights") is not None:
        config["channel_weights"] = np.asarray(config["channel_weights"], dtype=np.float32)
    if config.get("safe_qf_max") is not None:
        config["safe_qf_max"] = np.asarray(config["safe_qf_max"], dtype=np.float32)
    return WaterAllocationConfig(**config)


def find_model_file(run_dir: Path, model_name: str | None) -> Path:
    if model_name:
        model_path = run_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    for name in ("checkpoint_latest.pt", "ppo_water_model.pt", "best_model.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    checkpoints = sorted(run_dir.glob("checkpoint_iter_*.pt"))
    if checkpoints:
        return checkpoints[-1]

    pt_files = sorted(run_dir.glob("*.pt"))
    if pt_files:
        return pt_files[-1]

    raise FileNotFoundError(f"No .pt model/checkpoint file found in {run_dir}.")


def load_model(run_dir: Path, model_path: Path, obs_dim: int, action_dim: int, device: str):
    ppo_config = _load_json(run_dir / "ppo_config.json")
    hidden_dim = int(ppo_config.get("hidden_dim", 128))

    torch_device = torch.device(device)
    model = ActorCritic(obs_dim, action_dim, hidden_dim=hidden_dim).to(torch_device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")
    model.eval()
    return model, torch_device, ppo_config


def _validate_manual_demands(manual_demands: list[list[float]], num_channels: int) -> list[np.ndarray]:
    if not manual_demands:
        raise ValueError("manual_demands cannot be empty.")

    validated = []
    for idx, demand in enumerate(manual_demands, start=1):
        demand_array = np.asarray(demand, dtype=np.float32)
        if demand_array.shape != (num_channels,):
            raise ValueError(
                f"manual_demands[{idx}] must have shape {(num_channels,)}, "
                f"got {demand_array.shape}."
            )
        if np.any(demand_array < 0):
            raise ValueError(f"manual_demands[{idx}] contains negative values.")
        validated.append(demand_array)
    return validated


def reset_with_manual_demand(
    env: WaterAllocationEnv,
    manual_demand: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    env.reset(seed=seed)
    env.current_demands = manual_demand.astype(np.float32).copy()
    return env._get_obs()


@torch.no_grad()
def select_action(model, obs: np.ndarray, device: torch.device, deterministic: bool) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    alpha, beta, _ = model(obs_t)
    if deterministic:
        action = alpha / (alpha + beta)
    else:
        dist = torch.distributions.Beta(alpha, beta)
        action = dist.sample()
    return np.clip(action.squeeze(0).cpu().numpy(), 0.0, 1.0)


def run_one_case(
    env: WaterAllocationEnv,
    model,
    device: torch.device,
    seed: int,
    deterministic: bool,
    manual_demand: np.ndarray,
    case_id: int,
) -> dict:
    obs = reset_with_manual_demand(env, manual_demand, seed=seed)
    initial_demand = env.current_demands.copy()
    done = False
    total_reward = 0.0
    steps = []

    while not done:
        step_id = env.current_step
        demand_before = env.current_demands.copy()
        obs_before = obs.copy()
        action = select_action(model, obs, device, deterministic)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        steps.append(
            {
                "step": step_id,
                "obs": obs_before,
                "demand_before": demand_before,
                "normalized_action": info["normalized_action"],
                "gate_action": info["gate_action"],
                "actual_supply": info["actual_supply"],
                "unmet_demand": info["unmet_demand"],
                "oversupply": info["oversupply"],
                "unmet_ratio": info["unmet_ratio"],
                "oversupply_ratio": info["oversupply_ratio"],
                "smoothness_cost": info["smoothness_cost"],
                "unmet_penalty": info["unmet_penalty"],
                "oversupply_penalty_value": info["oversupply_penalty_value"],
                "smoothness_penalty_value": info["smoothness_penalty_value"],
                "safety_violation": info["safety_violation"],
                "safety_penalty": info["safety_penalty"],
                "completion_bonus": info["completion_bonus"],
                "reward": reward,
                "early_finished": info["early_finished"],
                "all_demands_satisfied": info["all_demands_satisfied"],
            }
        )

    final_step = steps[-1] if steps else {}
    return {
        "case_id": case_id,
        "seed": seed,
        "initial_demand": initial_demand,
        "total_reward": total_reward,
        "episode_length": len(steps),
        "final_unmet_ratio": final_step.get("unmet_ratio", 0.0),
        "final_oversupply_ratio": final_step.get("oversupply_ratio", 0.0),
        "all_demands_satisfied": any(step["all_demands_satisfied"] for step in steps),
        "early_finished": any(step["early_finished"] for step in steps),
        "total_safety_penalty": sum(float(step["safety_penalty"]) for step in steps),
        "steps": steps,
    }


def summarize(cases: list[dict]) -> dict[str, float]:
    if not cases:
        return {}
    return {
        "num_cases": len(cases),
        "avg_total_reward": float(np.mean([case["total_reward"] for case in cases])),
        "avg_episode_length": float(np.mean([case["episode_length"] for case in cases])),
        "avg_final_unmet_ratio": float(np.mean([case["final_unmet_ratio"] for case in cases])),
        "avg_final_oversupply_ratio": float(np.mean([case["final_oversupply_ratio"] for case in cases])),
        "all_satisfied_rate": float(np.mean([case["all_demands_satisfied"] for case in cases])),
        "early_finished_rate": float(np.mean([case["early_finished"] for case in cases])),
        "avg_total_safety_penalty": float(np.mean([case["total_safety_penalty"] for case in cases])),
    }


def main(config: ManualTestConfig = CONFIG) -> None:
    run_dir = config.run_dir.resolve()
    env_config = load_env_config(run_dir)
    env = WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)
    manual_demands = _validate_manual_demands(config.manual_demands, env_config.num_channels)
    model_path = find_model_file(run_dir, config.model_name)
    model, device, ppo_config = load_model(run_dir, model_path, env.obs_dim, env.action_dim, config.device)

    cases = []
    for case_idx, manual_demand in enumerate(manual_demands):
        seed = config.seed + case_idx
        print(
            f"test_case={case_idx + 1:04d}/{len(manual_demands)} "
            f"seed={seed} initial_demand={np.round(manual_demand, 2)}"
        )
        cases.append(
            run_one_case(
                env=env,
                model=model,
                device=device,
                seed=seed,
                deterministic=config.deterministic,
                manual_demand=manual_demand,
                case_id=case_idx + 1,
            )
        )

    summary = summarize(cases)
    output_dir = config.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_log_name = _safe_name(model_path.stem)
    output_path = output_dir / f"manual_test_log_{model_log_name}.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_config": config,
        "run_dir": run_dir,
        "model_path": model_path,
        "env_config": asdict(env_config),
        "ppo_config": ppo_config,
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim,
        "summary": summary,
        "cases": cases,
    }
    output_path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Manual Test Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\nSaved details to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
