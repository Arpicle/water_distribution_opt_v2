from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppo_agent import ActorCritic
from simulation import hydraulic_simulator
from water_allocation_env import WaterAllocationConfig, WaterAllocationEnv


@dataclass
class ModelSpec:
    run_dir: Path = Path("runs") / "your_run_dir"
    model_name: str | None = None
    name: str | None = None


@dataclass
class TestConfig:
    models: list[ModelSpec]
    num_demands: int = 100
    seed: int = 1000
    deterministic: bool = True
    device: str = "cuda"
    output_dir: Path | None = None


CONFIG = TestConfig(
    # Edit these values before running: python test_model.py
    # The first model's env_config.json is used only to generate common initial demands.
    models=[
        ModelSpec(
            run_dir=Path("runs") / "your_first_run_dir",
            model_name=None,
            name="model_1",
        ),
        ModelSpec(
            run_dir=Path("runs") / "your_second_run_dir",
            model_name=None,
            name="model_2",
        ),
        ModelSpec(
            run_dir=Path("runs") / "your_third_run_dir",
            model_name=None,
            name="model_3",
        ),
    ],
    num_demands=100,
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


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU for model testing.")
        return torch.device("cpu")
    return torch.device(device)


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


def infer_checkpoint_dims(model_path: Path) -> tuple[int | None, int | None]:
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        return None, None
    first_layer = state_dict.get("shared.0.weight")
    actor_head = state_dict.get("actor_head.weight")
    obs_dim = int(first_layer.shape[1]) if first_layer is not None else None
    action_dim = int(actor_head.shape[0] // 2) if actor_head is not None else None
    return obs_dim, action_dim


def load_model(run_dir: Path, model_path: Path, obs_dim: int, action_dim: int, device: str):
    ppo_config = _load_json(run_dir / "ppo_config.json")
    hidden_dim = int(ppo_config.get("hidden_dim", 128))

    torch_device = resolve_device(device)
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


def build_test_data(env_config: WaterAllocationConfig, num_demands: int, seed_start: int) -> list[dict]:
    env = WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)
    test_data = []
    for case_idx in range(num_demands):
        seed = seed_start + case_idx
        env.reset(seed=seed)
        test_data.append(
            {
                "case": case_idx + 1,
                "seed": seed,
                "initial_demand": env.current_demands.copy(),
            }
        )
    return test_data


def reset_with_initial_demand(
    env: WaterAllocationEnv,
    seed: int,
    initial_demand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    demand = np.asarray(initial_demand, dtype=np.float32)
    if demand.shape != env.current_demands.shape:
        raise ValueError(
            f"Initial demand shape {demand.shape} does not match this environment's "
            f"demand shape {env.current_demands.shape}. Models with different numbers "
            "of channels cannot share the same initial demand set."
        )
    env.current_demands = demand.copy()
    return env._get_obs(), demand.copy()


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
    initial_demand: np.ndarray,
) -> dict:
    if not deterministic:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    obs, initial_demand = reset_with_initial_demand(env, seed, initial_demand)
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


def main(config: TestConfig = CONFIG) -> None:
    if not config.models:
        raise ValueError("CONFIG.models cannot be empty.")

    model_names = [spec.name or (spec.model_name or spec.run_dir.name) for spec in config.models]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Each model in CONFIG.models must have a unique name/model_name/run_dir name.")

    demand_source_run_dir = config.models[0].run_dir.resolve()
    demand_source_env_config = load_env_config(demand_source_run_dir)
    test_data = build_test_data(demand_source_env_config, config.num_demands, config.seed)

    model_results = {}
    model_metadata = {}
    for model_idx, model_spec in enumerate(config.models):
        run_dir = model_spec.run_dir.resolve()
        env_config = load_env_config(run_dir)
        env = WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)
        model_path = find_model_file(run_dir, model_spec.model_name)
        model_name = model_spec.name or model_path.stem
        expected_obs_dim, expected_action_dim = infer_checkpoint_dims(model_path)
        if expected_obs_dim is not None and expected_obs_dim != env.obs_dim:
            raise ValueError(
                f"Model '{model_name}' expects obs_dim={expected_obs_dim}, "
                f"but its own env_config builds obs_dim={env.obs_dim}. "
                "Check that the model folder contains the matching env_config.json."
            )
        if expected_action_dim is not None and expected_action_dim != env.action_dim:
            raise ValueError(
                f"Model '{model_name}' expects action_dim={expected_action_dim}, "
                f"but its own env_config builds action_dim={env.action_dim}. "
                "Check that the model folder contains the matching env_config.json."
            )

        print(f"\n=== Testing {model_name} ({model_idx + 1}/{len(config.models)}) ===")
        model, device, ppo_config = load_model(
            run_dir,
            model_path,
            env.obs_dim,
            env.action_dim,
            config.device,
        )

        cases = []
        for case in test_data:
            case_idx = int(case["case"])
            seed = int(case["seed"])
            print(f"model={model_name} test_case={case_idx:04d}/{config.num_demands} seed={seed}")
            cases.append(
                run_one_case(
                    env,
                    model,
                    device,
                    seed,
                    config.deterministic,
                    np.asarray(case["initial_demand"], dtype=np.float32),
                )
            )

        model_results[model_name] = {
            "summary": summarize(cases),
            "cases": cases,
        }
        model_metadata[model_name] = {
            "run_dir": run_dir,
            "model_path": model_path,
            "env_config": env_config,
            "ppo_config": ppo_config,
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
        }

    output_dir = config.output_dir or demand_source_run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_log_name = "_".join(_safe_name(spec.name or (spec.model_name or spec.run_dir.name)) for spec in config.models)
    output_path = output_dir / f"test_log_{model_log_name}.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_config": config,
        "demand_source_run_dir": demand_source_run_dir,
        "demand_source_env_config": asdict(demand_source_env_config),
        "test_data": test_data,
        "model_metadata": model_metadata,
        "model_results": model_results,
    }
    output_path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Test Summary ===")
    for model_name, result in model_results.items():
        print(f"\n[{model_name}]")
        for key, value in result["summary"].items():
            print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\nSaved details to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
