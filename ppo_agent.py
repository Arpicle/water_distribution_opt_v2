from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim * 2)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        actor_out = self.actor_head(h)
        alpha_raw, beta_raw = torch.chunk(actor_out, 2, dim=-1)

        # Keep alpha/beta > 1 to avoid unstable extreme actions near 0 and 1.
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0
        value = self.critic_head(h)
        return alpha, beta, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    rollout_episodes: int = 64
    update_epochs: int = 10
    minibatch_size: int = 128
    max_grad_norm: float = 0.5
    hidden_dim: int = 128


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[np.ndarray] = []

    def clear(self) -> None:
        self.__init__()


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def select_action(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta, value = self.model(obs_t)
            dist = Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.cpu().numpy(),
            value.squeeze(0).cpu().numpy(),
        )

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha, beta, values = self.model(obs)
        dist = Beta(alpha, beta)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy, values

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        obs = np.asarray(buffer.obs, dtype=np.float32)
        actions = np.asarray(buffer.actions, dtype=np.float32)
        old_log_probs = np.asarray(buffer.log_probs, dtype=np.float32).reshape(-1, 1)
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32).reshape(-1)

        advantages = self._compute_gae(rewards, dones, values, last_value)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(-1)

        dataset_size = obs_t.size(0)
        indices = np.arange(dataset_size)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                batch_idx = indices[start:end]

                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                log_probs, entropy, values_pred = self.evaluate_actions(batch_obs, batch_actions)
                ratios = torch.exp(log_probs - batch_old_log_probs)

                unclipped = ratios * batch_advantages
                clipped = torch.clamp(
                    ratios, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
                ) * batch_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (batch_returns - values_pred).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_bonus.item())
                updates += 1

        if updates > 0:
            for key in stats:
                stats[key] /= updates
        return stats

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> np.ndarray:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = last_value

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]
        return advantages

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
