from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from ppo_agent import ActorCritic, PPOAgent, PPOConfig, RolloutBuffer
from simulation import hydraulic_simulator
from water_allocation_env import WaterAllocationConfig, WaterAllocationEnv

import time


def _init_rollout_metrics() -> dict:
    return {
        "reward_sum": 0.0,
        "unmet_ratio_sum": 0.0,
        "oversupply_ratio_sum": 0.0,
        "smoothness_cost_sum": 0.0,
        "q_up_sum": 0.0,
        "valid_step_count": 0,
        "valid_episode_count": 0,
        "unmet_penalty_sum": 0.0,
        "oversupply_penalty_sum": 0.0,
        "smoothness_penalty_sum": 0.0,
        "safety_penalty_sum": 0.0,
        "completion_bonus_sum": 0.0,
        "episode_length_sum": 0.0,
        "early_finished_count": 0.0,
        "all_satisfied_count": 0.0,
        "step_count": 0,
        "episode_count": 0,
    }


def _update_rollout_metrics(metrics: dict, reward: float, info: dict) -> None:
    metrics["reward_sum"] += float(reward)
    metrics["unmet_ratio_sum"] += float(info["unmet_ratio"])
    metrics["oversupply_ratio_sum"] += float(info["oversupply_ratio"])
    metrics["smoothness_cost_sum"] += float(info["smoothness_cost"])
    metrics["q_up_sum"] += float(info["q_up"])
    metrics["valid_step_count"] += int(not bool(info.get("simulation_failed", False)))
    metrics["unmet_penalty_sum"] += float(info["unmet_penalty"])
    metrics["oversupply_penalty_sum"] += float(info["oversupply_penalty_value"])
    metrics["smoothness_penalty_sum"] += float(info["smoothness_penalty_value"])
    metrics["safety_penalty_sum"] += float(info["safety_penalty"])
    metrics["completion_bonus_sum"] += float(info["completion_bonus"])
    metrics["early_finished_count"] += float(info["early_finished"])
    metrics["all_satisfied_count"] += float(info["all_demands_satisfied"])
    metrics["step_count"] += 1


def _finalize_rollout_metrics(metrics: dict) -> dict:
    step_count = max(metrics["step_count"], 1)
    episode_count = max(metrics["episode_count"], 1)
    return {
        "avg_step_reward": metrics["reward_sum"] / step_count,
        "avg_unmet_ratio": metrics["unmet_ratio_sum"] / step_count,
        "avg_oversupply_ratio": metrics["oversupply_ratio_sum"] / step_count,
        "avg_smoothness_cost": metrics["smoothness_cost_sum"] / step_count,
        "avg_q_up": metrics["q_up_sum"] / step_count,
        "valid_step_ratio": metrics["valid_step_count"] / step_count,
        "valid_episode_ratio": metrics["valid_episode_count"] / episode_count,
        "avg_unmet_penalty": metrics["unmet_penalty_sum"] / step_count,
        "avg_oversupply_penalty": metrics["oversupply_penalty_sum"] / step_count,
        "avg_smoothness_penalty": metrics["smoothness_penalty_sum"] / step_count,
        "avg_safety_penalty": metrics["safety_penalty_sum"] / step_count,
        "avg_completion_bonus": metrics["completion_bonus_sum"] / step_count,
        "avg_episode_length": metrics["episode_length_sum"] / episode_count,
        "early_finished_rate": metrics["early_finished_count"] / episode_count,
        "all_satisfied_rate": metrics["all_satisfied_count"] / episode_count,
        "step_count": metrics["step_count"],
        "episode_count": metrics["episode_count"],
    }


def _build_detailed_log_record(
    iteration: int,
    avg_reward: float,
    avg_unmet: float,
    rollout_metrics: dict,
    loss_stats: dict,
) -> dict:
    return {
        "iteration": iteration,
        "summary": {
            "avg_reward": avg_reward,
            "avg_unmet_ratio": avg_unmet,
            "avg_episode_length": rollout_metrics["avg_episode_length"],
            "early_finished_rate": rollout_metrics["early_finished_rate"],
            "all_satisfied_rate": rollout_metrics["all_satisfied_rate"],
            "avg_q_up": rollout_metrics["avg_q_up"],
            "valid_step_ratio": rollout_metrics["valid_step_ratio"],
            "valid_episode_ratio": rollout_metrics["valid_episode_ratio"],
        },
        "reward_parts": {
            "avg_step_reward": rollout_metrics["avg_step_reward"],
            "unmet_penalty": rollout_metrics["avg_unmet_penalty"],
            "oversupply_penalty": rollout_metrics["avg_oversupply_penalty"],
            "smoothness_penalty": rollout_metrics["avg_smoothness_penalty"],
            "safety_penalty": rollout_metrics["avg_safety_penalty"],
            "completion_bonus": rollout_metrics["avg_completion_bonus"],
        },
        "rollout_parts": {
            "oversupply_ratio": rollout_metrics["avg_oversupply_ratio"],
            "smoothness_cost": rollout_metrics["avg_smoothness_cost"],
            "q_up": rollout_metrics["avg_q_up"],
            "valid_step_ratio": rollout_metrics["valid_step_ratio"],
            "valid_episode_ratio": rollout_metrics["valid_episode_ratio"],
            "steps": rollout_metrics["step_count"],
            "episodes": rollout_metrics["episode_count"],
        },
        "losses": {
            "policy_loss": loss_stats["policy_loss"],
            "value_loss": loss_stats["value_loss"],
            "entropy": loss_stats["entropy"],
            "total_loss": loss_stats["total_loss"],
        },
    }


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _create_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_run_metadata(run_dir: Path, args: argparse.Namespace, ppo_config: PPOConfig, env_config: WaterAllocationConfig) -> None:
    files = {
        "args.json": vars(args),
        "ppo_config.json": asdict(ppo_config),
        "env_config.json": asdict(env_config),
    }
    for filename, payload in files.items():
        (run_dir / filename).write_text(
            json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _load_resume_state(agent: PPOAgent, resume_path: Path) -> dict:
    checkpoint = torch.load(resume_path, map_location=agent.device)

    # Support both full checkpoints and plain model state_dict files.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {
            "start_iteration": int(checkpoint.get("iteration", 0)) + 1,
            "resume_type": "checkpoint",
        }

    if isinstance(checkpoint, dict):
        agent.model.load_state_dict(checkpoint)
        return {
            "start_iteration": 1,
            "resume_type": "model_only",
        }

    raise ValueError(f"Unsupported resume file format: {resume_path}")


def _save_checkpoint(run_dir: Path, agent: PPOAgent, iteration: int) -> None:
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }
    torch.save(checkpoint, run_dir / "checkpoint_latest.pt")


def _save_periodic_checkpoint(run_dir: Path, agent: PPOAgent, iteration: int) -> Path:
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }
    checkpoint_path = run_dir / f"checkpoint_iter_{iteration:04d}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _run_rollout_worker(
    env_config: WaterAllocationConfig,
    model_state_dict: dict,
    hidden_dim: int,
    num_episodes: int,
    base_seed: int,
) -> dict:
    env = WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)
    model = ActorCritic(env.obs_dim, env.action_dim, hidden_dim=hidden_dim)
    model.load_state_dict(model_state_dict)
    model.eval()

    trajectories = {
        "obs": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }
    episode_rewards = []
    unmet_ratios = []
    metrics = _init_rollout_metrics()

    for episode_offset in range(num_episodes):
        obs = env.reset(seed=base_seed + episode_offset)
        done = False
        ep_reward = 0.0
        ep_unmet = []
        ep_steps = 0
        episode_valid = True

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                alpha, beta, value = model(obs_t)
                dist = torch.distributions.Beta(alpha, beta)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            action_np = action.squeeze(0).cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.squeeze(0).cpu().numpy()

            next_obs, reward, done, info = env.step(action_np)

            trajectories["obs"].append(obs)
            trajectories["actions"].append(action_np)
            trajectories["log_probs"].append(log_prob_np)
            trajectories["rewards"].append(reward)
            trajectories["dones"].append(float(done))
            trajectories["values"].append(value_np)

            ep_reward += reward
            ep_unmet.append(info["unmet_ratio"])
            ep_steps += 1
            episode_valid = episode_valid and (not bool(info.get("simulation_failed", False)))
            _update_rollout_metrics(metrics, reward, info)
            obs = next_obs

        episode_rewards.append(ep_reward)
        unmet_ratios.append(float(np.mean(ep_unmet)))
        metrics["episode_length_sum"] += ep_steps
        metrics["episode_count"] += 1
        metrics["valid_episode_count"] += int(episode_valid)

    return {
        "trajectories": trajectories,
        "episode_rewards": episode_rewards,
        "unmet_ratios": unmet_ratios,
        "metrics": metrics,
    }


def _merge_worker_results(worker_results: list[dict]) -> tuple[RolloutBuffer, float, float, dict]:
    buffer = RolloutBuffer()
    episode_rewards = []
    unmet_ratios = []
    merged_metrics = _init_rollout_metrics()

    for result in worker_results:
        traj = result["trajectories"]
        buffer.obs.extend(traj["obs"])
        buffer.actions.extend(traj["actions"])
        buffer.log_probs.extend(traj["log_probs"])
        buffer.rewards.extend(traj["rewards"])
        buffer.dones.extend(traj["dones"])
        buffer.values.extend(traj["values"])
        episode_rewards.extend(result["episode_rewards"])
        unmet_ratios.extend(result["unmet_ratios"])
        for key, value in result["metrics"].items():
            merged_metrics[key] += value

    return (
        buffer,
        float(np.mean(episode_rewards)),
        float(np.mean(unmet_ratios)),
        _finalize_rollout_metrics(merged_metrics),
    )


def collect_rollouts(
    env: WaterAllocationEnv,
    agent: PPOAgent,
    rollout_episodes: int,
    num_workers: int = 1,
    base_seed: int = 0,
) -> tuple[RolloutBuffer, float, float, dict]:
    if num_workers <= 1:
        buffer = RolloutBuffer()
        episode_rewards = []
        unmet_ratios = []
        metrics = _init_rollout_metrics()

        for episode_idx in range(rollout_episodes):
            obs = env.reset(seed=base_seed + episode_idx)
            done = False
            ep_reward = 0.0
            ep_unmet = []
            ep_steps = 0
            episode_valid = True

            while not done:
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)

                buffer.obs.append(obs)
                buffer.actions.append(action)
                buffer.log_probs.append(log_prob)
                buffer.rewards.append(reward)
                buffer.dones.append(float(done))
                buffer.values.append(value)

                ep_reward += reward
                ep_unmet.append(info["unmet_ratio"])
                ep_steps += 1
                episode_valid = episode_valid and (not bool(info.get("simulation_failed", False)))
                _update_rollout_metrics(metrics, reward, info)
                obs = next_obs

            episode_rewards.append(ep_reward)
            unmet_ratios.append(float(np.mean(ep_unmet)))
            metrics["episode_length_sum"] += ep_steps
            metrics["episode_count"] += 1
            metrics["valid_episode_count"] += int(episode_valid)

        return (
            buffer,
            float(np.mean(episode_rewards)),
            float(np.mean(unmet_ratios)),
            _finalize_rollout_metrics(metrics),
        )

    episodes_per_worker = [rollout_episodes // num_workers] * num_workers
    for worker_idx in range(rollout_episodes % num_workers):
        episodes_per_worker[worker_idx] += 1
    episodes_per_worker = [count for count in episodes_per_worker if count > 0]

    model_state_dict = {
        key: value.detach().cpu()
        for key, value in agent.model.state_dict().items()
    }

    worker_results = []
    next_seed = base_seed
    with ProcessPoolExecutor(max_workers=len(episodes_per_worker)) as executor:
        futures = []
        for num_episodes in episodes_per_worker:
            futures.append(
                executor.submit(
                    _run_rollout_worker,
                    env.config,
                    model_state_dict,
                    agent.config.hidden_dim,
                    num_episodes,
                    next_seed,
                )
            )
            next_seed += num_episodes

        for future in futures:
            worker_results.append(future.result())

    return _merge_worker_results(worker_results)


def build_env_config(num_channels: int) -> WaterAllocationConfig:
    return WaterAllocationConfig(
        num_channels=num_channels,
        horizon=5,
        gate_open_min=0.0,
        gate_open_max=0.35,
        q_up_min=1.0,
        q_up_max=3.5,
        demand_low=5000,
        demand_high=40000,
        demand_noise_std=3.0,
        smoothness_penalty=0.02,
        oversupply_penalty=0.7,
        demand_satisfied_tolerance=1e-3,
        channel_weights=np.array([1.0, 1.5, 2.0], dtype=np.float32),
        safe_h_max=3.0,
        safe_q_max=3.5,
        safe_qf_max=np.array([1.0, 1.2, 1.1], dtype=np.float32),
        safety_penalty=5.0,
        early_completion_bonus=1.0,
        nan_penalty=1e6,
    )


def evaluate_policy(
    env: WaterAllocationEnv,
    agent: PPOAgent,
    episodes: int = 5,
    seed_offset: int = 0,
    verbose: bool = True,
) -> dict:
    if verbose:
        print("\n=== Evaluation ===")

    episode_records = []
    total_rewards = []
    total_unmet_ratios = []
    total_lengths = []
    total_q_ups = []
    early_finished_count = 0
    all_satisfied_count = 0

    for episode in range(episodes):
        obs = env.reset(seed=seed_offset + episode)
        done = False
        total_reward = 0.0
        step_id = 0
        unmet_ratios = []
        episode_steps = []
        episode_early_finished = False
        episode_all_satisfied = False
        episode_q_up = float(env.current_q_up)
        while not done:
            current_demand = env.current_demands.copy()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                alpha, beta, _ = agent.model(obs_t)
                action = (alpha / (alpha + beta)).squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            unmet_ratios.append(float(info["unmet_ratio"]))
            episode_early_finished = episode_early_finished or bool(info["early_finished"])
            episode_all_satisfied = episode_all_satisfied or bool(info["all_demands_satisfied"])
            episode_steps.append(
                {
                    "step": step_id + 1,
                    "demand": np.round(current_demand, 2).tolist(),
                    "gate": np.round(info["gate_action"], 3).tolist(),
                    "supply": np.round(info["actual_supply"], 2).tolist(),
                    "unmet": np.round(info["unmet_demand"], 2).tolist(),
                    "reward": float(reward),
                    "unmet_ratio": float(info["unmet_ratio"]),
                    "oversupply_ratio": float(info["oversupply_ratio"]),
                    "smoothness_cost": float(info["smoothness_cost"]),
                    "safety_penalty": float(info["safety_penalty"]),
                    "completion_bonus": float(info["completion_bonus"]),
                    "q_up": float(info["q_up"]),
                }
            )
            if verbose:
                print(
                    f"episode={episode + 1}, step={step_id + 1}, "
                    f"demand={np.round(current_demand, 2)}, "
                    f"Q_up={info['q_up']:.3f}, "
                    f"gate={np.round(info['gate_action'], 3)}, "
                    f"supply={np.round(info['actual_supply'], 2)}, "
                    f"unmet={np.round(info['unmet_demand'], 2)}"
                )
            step_id += 1
        total_rewards.append(total_reward)
        total_unmet_ratios.append(float(np.mean(unmet_ratios)) if unmet_ratios else 0.0)
        total_lengths.append(step_id)
        total_q_ups.append(episode_q_up)
        early_finished_count += float(episode_early_finished)
        all_satisfied_count += float(episode_all_satisfied)
        episode_records.append(
            {
                "episode": episode + 1,
                "q_up": episode_q_up,
                "total_reward": float(total_reward),
                "avg_unmet_ratio": float(np.mean(unmet_ratios)) if unmet_ratios else 0.0,
                "length": step_id,
                "early_finished": episode_early_finished,
                "all_demands_satisfied": episode_all_satisfied,
                "steps": episode_steps,
            }
        )
        if verbose:
            print(f"episode={episode + 1}, total_reward={total_reward:.3f}")

    return {
        "episodes": episodes,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_unmet_ratio": float(np.mean(total_unmet_ratios)) if total_unmet_ratios else 0.0,
        "avg_episode_length": float(np.mean(total_lengths)) if total_lengths else 0.0,
        "avg_q_up": float(np.mean(total_q_ups)) if total_q_ups else 0.0,
        "early_finished_rate": early_finished_count / max(episodes, 1),
        "all_satisfied_rate": all_satisfied_count / max(episodes, 1),
        "episode_results": episode_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO for 5-step water allocation")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--train-iterations", type=int, default=20, help="Training rounds")
    parser.add_argument("--rollout-episodes", type=int, default=64, help="Episodes per rollout")
    parser.add_argument("--num-workers", type=int, default=12, help="Parallel rollout workers")
    parser.add_argument("--model-path", type=str, default="ppo_water_model.pt", help="Model file")
    parser.add_argument("--log-file", type=str, default="training_details.jsonl", help="Detailed training log file")
    parser.add_argument("--eval-log-file", type=str, default="evaluation_details.jsonl", help="Detailed evaluation log file")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory for timestamped training outputs")
    parser.add_argument("--resume", type=str, default="", help="Resume from model .pt or checkpoint .pt")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save a named checkpoint every N iterations")
    parser.add_argument("--eval-interval", type=int, default=10, help="Run evaluation every N iterations")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes per run")
    args = parser.parse_args()

    env_config = build_env_config(args.num_channels)
    env = WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)

    ppo_config = PPOConfig(rollout_episodes=args.rollout_episodes)
    agent = PPOAgent(env.obs_dim, env.action_dim, ppo_config)
    run_dir = _create_run_dir(Path(args.output_dir))
    log_path = run_dir / Path(args.log_file).name
    eval_log_path = run_dir / Path(args.eval_log_file).name
    model_path = run_dir / Path(args.model_path).name
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    _save_run_metadata(run_dir, args, ppo_config, env_config)
    start_iteration = 1
    resume_info = None
    if args.resume:
        resume_path = Path(args.resume)
        resume_info = _load_resume_state(agent, resume_path)
        start_iteration = resume_info["start_iteration"]
        (run_dir / "resume_info.json").write_text(
            json.dumps(
                {
                    "resume_path": str(resume_path),
                    "resume_type": resume_info["resume_type"],
                    "start_iteration": start_iteration,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(
        f"Start training: channels={args.num_channels}, horizon={env.horizon}, "
        f"obs_dim={env.obs_dim}, action_dim={env.action_dim}, workers={args.num_workers}"
    )
    if resume_info is not None:
        print(
            f"Resume training from: {Path(args.resume).resolve()} "
            f"(type={resume_info['resume_type']}, start_iteration={start_iteration})"
        )
    log_path.write_text("", encoding="utf-8")
    eval_log_path.write_text("", encoding="utf-8")

    for iteration in range(start_iteration, args.train_iterations + 1):

        # print("=====start")
        # t1 = time.time()
        buffer, avg_reward, avg_unmet, rollout_metrics = collect_rollouts(
            env,
            agent,
            args.rollout_episodes,
            num_workers=args.num_workers,
            base_seed=iteration * 1000,
        )

        # t2 = time.time()
        # print(t2-t1)
        # print("=====update")
        
        
        stats = agent.update(buffer)

        

        # t3 = time.time()
        # print(t3-t2)
        # print("=====log")

        if iteration % 1 == 0 or iteration == 1:
            print(
                f"iter={iteration:04d} "
                f"avg_reward={avg_reward:.4f} "
                f"avg_unmet_ratio={avg_unmet:.4f} "
                f"policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} "
                f"valid_step_ratio={rollout_metrics['valid_step_ratio']:.4f} "
                f"valid_episode_ratio={rollout_metrics['valid_episode_ratio']:.4f}"
            )
        with log_path.open("a", encoding="utf-8") as log_file:
            log_record = _build_detailed_log_record(
                    iteration,
                    avg_reward,
                    avg_unmet,
                    rollout_metrics,
                    stats,
                )
            log_file.write(
                json.dumps(_to_jsonable(log_record), ensure_ascii=False) + "\n"
            )
        _save_checkpoint(run_dir, agent, iteration)
        if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
            _save_periodic_checkpoint(run_dir, agent, iteration)
        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
            eval_results = evaluate_policy(
                env,
                agent,
                episodes=args.eval_episodes,
                seed_offset=iteration * 10000,
                verbose=False,
            )
            eval_record = {
                "iteration": iteration,
                "summary": {
                    "avg_reward": eval_results["avg_reward"],
                    "avg_unmet_ratio": eval_results["avg_unmet_ratio"],
                    "avg_episode_length": eval_results["avg_episode_length"],
                    "avg_q_up": eval_results["avg_q_up"],
                    "early_finished_rate": eval_results["early_finished_rate"],
                    "all_satisfied_rate": eval_results["all_satisfied_rate"],
                },
                "episode_results": eval_results["episode_results"],
            }
            with eval_log_path.open("a", encoding="utf-8") as eval_log_file:
                eval_log_file.write(
                    json.dumps(_to_jsonable(eval_record), ensure_ascii=False) + "\n"
                )
            print(
                f"eval@iter={iteration:04d} "
                f"avg_reward={eval_results['avg_reward']:.4f} "
                f"avg_unmet_ratio={eval_results['avg_unmet_ratio']:.4f}"
            )
        

    agent.save(str(model_path))
    print(f"Model saved to: {model_path.resolve()}")
    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")
    print(f"Evaluation log saved to: {eval_log_path.resolve()}")

    final_eval = evaluate_policy(
        env,
        agent,
        episodes=args.eval_episodes,
        seed_offset=args.train_iterations * 10000 + 1,
        verbose=True,
    )
    final_eval_record = {
        "iteration": args.train_iterations,
        "phase": "final",
        "summary": {
            "avg_reward": final_eval["avg_reward"],
            "avg_unmet_ratio": final_eval["avg_unmet_ratio"],
            "avg_episode_length": final_eval["avg_episode_length"],
            "avg_q_up": final_eval["avg_q_up"],
            "early_finished_rate": final_eval["early_finished_rate"],
            "all_satisfied_rate": final_eval["all_satisfied_rate"],
        },
        "episode_results": final_eval["episode_results"],
    }
    with eval_log_path.open("a", encoding="utf-8") as eval_log_file:
        eval_log_file.write(
            json.dumps(_to_jsonable(final_eval_record), ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
