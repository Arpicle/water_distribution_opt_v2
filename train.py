from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ppo_agent import ActorCritic, PPOAgent, PPOConfig, RolloutBuffer
from simulation import hydraulic_simulator
from water_allocation_env import WaterAllocationConfig, WaterAllocationEnv


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

    for episode_offset in range(num_episodes):
        obs = env.reset(seed=base_seed + episode_offset)
        done = False
        ep_reward = 0.0
        ep_unmet = []

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
            obs = next_obs

        episode_rewards.append(ep_reward)
        unmet_ratios.append(float(np.mean(ep_unmet)))

    return {
        "trajectories": trajectories,
        "episode_rewards": episode_rewards,
        "unmet_ratios": unmet_ratios,
    }


def _merge_worker_results(worker_results: list[dict]) -> tuple[RolloutBuffer, float, float]:
    buffer = RolloutBuffer()
    episode_rewards = []
    unmet_ratios = []

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

    return buffer, float(np.mean(episode_rewards)), float(np.mean(unmet_ratios))


def collect_rollouts(
    env: WaterAllocationEnv,
    agent: PPOAgent,
    rollout_episodes: int,
    num_workers: int = 1,
    base_seed: int = 0,
) -> tuple[RolloutBuffer, float, float]:
    if num_workers <= 1:
        buffer = RolloutBuffer()
        episode_rewards = []
        unmet_ratios = []

        for episode_idx in range(rollout_episodes):
            obs = env.reset(seed=base_seed + episode_idx)
            done = False
            ep_reward = 0.0
            ep_unmet = []

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
                obs = next_obs

            episode_rewards.append(ep_reward)
            unmet_ratios.append(float(np.mean(ep_unmet)))

        return buffer, float(np.mean(episode_rewards)), float(np.mean(unmet_ratios))

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
        gate_open_min=0.03,
        gate_open_max=0.5,
        demand_low=5000,
        demand_high=40000,
        demand_noise_std=3.0,
        smoothness_penalty=0.05,
        oversupply_penalty=0.02,
        demand_satisfied_tolerance=1e-3,
        channel_weights=np.array([2.0, 1.0, 1.5], dtype=np.float32),
        safe_z_max=3.0,
        safe_q_max=3.0,
        safe_qf_max=np.array([1.2, 1.2, 1.2], dtype=np.float32),
        safety_penalty=5.0,
    )


def evaluate_policy(env: WaterAllocationEnv, agent: PPOAgent, episodes: int = 5) -> None:
    print("\n=== Evaluation ===")
    for episode in range(episodes):
        obs = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        step_id = 0
        while not done:
            current_demand = env.current_demands.copy()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                alpha, beta, _ = agent.model(obs_t)
                action = (alpha / (alpha + beta)).squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(
                f"episode={episode + 1}, step={step_id + 1}, "
                f"demand={np.round(current_demand, 2)}, "
                f"gate={np.round(info['gate_action'], 3)}, "
                f"supply={np.round(info['actual_supply'], 2)}, "
                f"unmet={np.round(info['unmet_demand'], 2)}"
            )
            step_id += 1
        print(f"episode={episode + 1}, total_reward={total_reward:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO for 5-step water allocation")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--train-iterations", type=int, default=200, help="Training rounds")
    parser.add_argument("--rollout-episodes", type=int, default=64, help="Episodes per rollout")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel rollout workers")
    parser.add_argument("--model-path", type=str, default="ppo_water_model.pt", help="Model file")
    args = parser.parse_args()

    env_config = build_env_config(args.num_channels)
    env = WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)

    ppo_config = PPOConfig(rollout_episodes=args.rollout_episodes)
    agent = PPOAgent(env.obs_dim, env.action_dim, ppo_config)

    print(
        f"Start training: channels={args.num_channels}, horizon={env.horizon}, "
        f"obs_dim={env.obs_dim}, action_dim={env.action_dim}, workers={args.num_workers}"
    )

    for iteration in range(1, args.train_iterations + 1):
        buffer, avg_reward, avg_unmet = collect_rollouts(
            env,
            agent,
            args.rollout_episodes,
            num_workers=args.num_workers,
            base_seed=iteration * 1000,
        )
        stats = agent.update(buffer)

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"iter={iteration:04d} "
                f"avg_reward={avg_reward:.4f} "
                f"avg_unmet_ratio={avg_unmet:.4f} "
                f"policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f}"
            )

    model_path = Path(args.model_path)
    agent.save(str(model_path))
    print(f"Model saved to: {model_path.resolve()}")

    evaluate_policy(env, agent)


if __name__ == "__main__":
    main()
