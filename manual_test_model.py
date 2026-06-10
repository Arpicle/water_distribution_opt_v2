from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ManualTestConfig:
    run_dir: Path = Path("runs") / "your_run_dir"
    model_name: str | None = None
    code_dir: Path | None = None
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


def load_env_config(run_dir: Path, config_cls=None):
    if config_cls is None:
        from water_allocation_env import WaterAllocationConfig

        config_cls = WaterAllocationConfig

    config = _load_json(run_dir / "env_config.json")
    if config.get("channel_weights") is not None:
        config["channel_weights"] = np.asarray(config["channel_weights"], dtype=np.float32)
    if config.get("safe_qf_max") is not None:
        config["safe_qf_max"] = np.asarray(config["safe_qf_max"], dtype=np.float32)
    return config_cls(**config)


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


def load_module_from_file(module_name: str, path: Path, extra_sys_path: Path | None = None):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_path = str(extra_sys_path) if extra_sys_path is not None else None
    if added_path is not None:
        sys.path.insert(0, added_path)
    try:
        spec.loader.exec_module(module)
    finally:
        if added_path is not None and sys.path and sys.path[0] == added_path:
            sys.path.pop(0)
    return module


def find_first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_required_project_module(
    module_name: str,
    candidate_paths: list[Path],
    extra_sys_path: Path,
):
    module_path = find_first_existing_path(candidate_paths)
    if module_path is None:
        checked = "\n".join(f"  - {path}" for path in candidate_paths)
        raise FileNotFoundError(
            f"Cannot find required module '{module_name}.py'. Checked:\n{checked}"
        )
    return load_module_from_file(module_name, module_path, extra_sys_path=extra_sys_path), module_path


def load_training_modules(run_dir: Path, code_dir: Path | None = None):
    run_dir = run_dir.resolve()
    code_dir = (code_dir or run_dir).resolve()
    script_dir = Path(__file__).resolve().parent

    simulation_module, simulation_path = load_required_project_module(
        "simulation",
        [code_dir / "simulation.py", run_dir / "simulation.py", script_dir / "simulation.py"],
        code_dir,
    )
    # Make model-local imports like "from simulation import ..." resolve to the selected file.
    sys.modules["simulation"] = simulation_module

    ppo_module, ppo_agent_path = load_required_project_module(
        "manual_test_model_ppo_agent",
        [code_dir / "ppo_agent.py", run_dir / "ppo_agent.py", script_dir / "ppo_agent.py"],
        code_dir,
    )

    env_module, env_path = load_required_project_module(
        "manual_test_model_water_allocation_env",
        [
            code_dir / "water_allocation_env.py",
            run_dir / "water_allocation_env.py",
            script_dir / "water_allocation_env.py",
        ],
        code_dir,
    )

    return {
        "run_dir": run_dir,
        "code_dir": code_dir,
        "ppo_module": ppo_module,
        "env_module": env_module,
        "simulation_module": simulation_module,
        "ppo_agent_path": ppo_agent_path,
        "env_path": env_path,
        "simulation_path": simulation_path,
    }


def load_model(
    ppo_module,
    ppo_config: dict[str, Any],
    model_path: Path,
    obs_dim: int,
    action_dim: int,
    device: str,
):
    hidden_dim = int(ppo_config.get("hidden_dim", 128))

    torch_device = torch.device(device)
    model = ppo_module.ActorCritic(obs_dim, action_dim, hidden_dim=hidden_dim).to(torch_device)
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
    env,
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
    env,
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
    modules = load_training_modules(config.run_dir, config.code_dir)
    run_dir = modules["run_dir"]
    code_dir = modules["code_dir"]
    ppo_module = modules["ppo_module"]
    env_module = modules["env_module"]
    hydraulic_simulator = modules["simulation_module"].hydraulic_simulator

    env_config = load_env_config(run_dir, env_module.WaterAllocationConfig)
    env = env_module.WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)
    manual_demands = _validate_manual_demands(config.manual_demands, env_config.num_channels)
    model_path = find_model_file(run_dir, config.model_name)
    ppo_config = _load_json(run_dir / "ppo_config.json")
    model, device, ppo_config = load_model(
        ppo_module,
        ppo_config,
        model_path,
        env.obs_dim,
        env.action_dim,
        config.device,
    )

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
        "code_dir": code_dir,
        "model_path": model_path,
        "ppo_agent_path": modules["ppo_agent_path"],
        "water_allocation_env_path": modules["env_path"],
        "simulation_path": modules["simulation_path"],
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
