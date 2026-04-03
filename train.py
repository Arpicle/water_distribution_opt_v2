from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from simulation import hydraulic_simulator
from water_allocation_env import WaterAllocationConfig, WaterAllocationEnv


def collect_rollouts(
    env: WaterAllocationEnv,
    agent: PPOAgent,
    rollout_episodes: int,
) -> tuple[RolloutBuffer, float, float]:
    buffer = RolloutBuffer()
    episode_rewards = []
    unmet_ratios = []

    for _ in range(rollout_episodes):
        obs = env.reset()
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
    parser.add_argument("--model-path", type=str, default="ppo_water_model.pt", help="Model file")
    args = parser.parse_args()



    env_config = WaterAllocationConfig(
        num_channels=3,
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


    env = WaterAllocationEnv(env_config, hydraulic_simulator=hydraulic_simulator)

    ppo_config = PPOConfig(rollout_episodes=args.rollout_episodes)
    agent = PPOAgent(env.obs_dim, env.action_dim, ppo_config)

    print(
        f"Start training: channels={args.num_channels}, horizon={env.horizon}, "
        f"obs_dim={env.obs_dim}, action_dim={env.action_dim}"
    )

    for iteration in range(1, args.train_iterations + 1):
        buffer, avg_reward, avg_unmet = collect_rollouts(
            env, agent, args.rollout_episodes
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
