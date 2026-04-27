from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from simulation import get_initial_gate_state, hydraulic_simulator


HydraulicSimulator = Callable[[np.ndarray], np.ndarray]


@dataclass
class WaterAllocationConfig:
    num_channels: int
    horizon: int = 5
    gate_open_min: float = 0.03
    gate_open_max: float = 0.5
    max_flow_per_channel: float = 40.0
    demand_low: float = 5000.0
    demand_high: float = 40000.0
    demand_noise_std: float = 3.0
    smoothness_penalty: float = 0.05
    oversupply_penalty: float = 0.1
    demand_satisfied_tolerance: float = 1e-3
    channel_weights: np.ndarray | None = None
    safe_h_max: float | None = None
    safe_q_max: float | None = None
    safe_qf_max: float | np.ndarray | None = None
    safety_penalty: float = 5.0
    early_completion_bonus: float = 1.0


class WaterAllocationEnv:
    """
    5-step water allocation environment.

    State:
        [current_demands(N), masked Z/Q/e history(3 * horizon * N), current_step_ratio(1)]

    Action:
        gate openings for the next period, each in [0, 1].
    """

    def __init__(
        self,
        config: WaterAllocationConfig,
        hydraulic_simulator: HydraulicSimulator | None = None,
    ):
        self.config = config
        self.num_channels = config.num_channels
        self.horizon = config.horizon
        self.channel_capacity = np.full(
            self.num_channels, config.max_flow_per_channel, dtype=np.float32
        )
        if config.gate_open_min > config.gate_open_max:
            raise ValueError("gate_open_min must be less than or equal to gate_open_max.")
        self.channel_weights = self._build_channel_weights(config.channel_weights)
        self.hydraulic_simulator = hydraulic_simulator or self.default_hydraulic_simulator
        if self.hydraulic_simulator is hydraulic_simulator and self.num_channels != 3:
            raise ValueError(
                "simulation.hydraulic_simulator currently expects exactly 3 gate openings, "
                f"but num_channels={self.num_channels}."
            )

        self.current_step = 0
        self.current_demands = np.zeros(self.num_channels, dtype=np.float32)
        self.previous_action = np.zeros(self.num_channels, dtype=np.float32)
        self.z_history = np.zeros((self.horizon, self.num_channels), dtype=np.float32)
        self.q_history = np.zeros((self.horizon, self.num_channels), dtype=np.float32)
        self.e_history = np.zeros((self.horizon, self.num_channels), dtype=np.float32)
        self.hydraulic_state = None

    @property
    def obs_dim(self) -> int:
        return self.num_channels + self.horizon * self.num_channels * 3 + 1

    @property
    def action_dim(self) -> int:
        return self.num_channels

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.previous_action = np.zeros(self.num_channels, dtype=np.float32)
        self.current_demands = self._sample_initial_demand()
        self.z_history.fill(0.0)
        self.q_history.fill(0.0)
        self.e_history.fill(0.0)
        self.hydraulic_state = None
        if self.hydraulic_simulator is hydraulic_simulator:
            gate_z, gate_q = get_initial_gate_state()
            self.hydraulic_state = {"gate_Z": gate_z, "gate_Q": gate_q}
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        normalized_action = np.asarray(action, dtype=np.float32)
        normalized_action = np.clip(normalized_action, 0.0, 1.0)
        gate_action = self._scale_action_to_gate_range(normalized_action)

        requested_supply = gate_action * self.channel_capacity
        actual_supply = self.simulate_supply(gate_action)

        unmet_demand = np.maximum(self.current_demands - actual_supply, 0.0)
        oversupply = np.maximum(actual_supply - self.current_demands, 0.0)

        weighted_demand = self.channel_weights * self.current_demands
        weighted_unmet = self.channel_weights * unmet_demand
        weighted_oversupply = self.channel_weights * oversupply

        unmet_ratio = weighted_unmet.sum() / (weighted_demand.sum() + 1e-6)
        oversupply_ratio = weighted_oversupply.sum() / (weighted_demand.sum() + 1e-6)
        smoothness_cost = float(np.mean(np.abs(gate_action - self.previous_action)))
        safety_violation, safety_penalty_value = self._compute_safety_penalty()

        reward = (
            1.0
            - unmet_ratio
            - self.config.oversupply_penalty * oversupply_ratio
            - self.config.smoothness_penalty * smoothness_cost
            - safety_penalty_value
        )
        unmet_penalty_value = unmet_ratio
        oversupply_penalty_value = self.config.oversupply_penalty * oversupply_ratio
        smoothness_penalty_value = self.config.smoothness_penalty * smoothness_cost

        self.previous_action = gate_action.copy()
        self.current_step += 1
        next_demand = self._transition_demand(
            previous_demand=self.current_demands,
            actual_supply=actual_supply,
        )
        all_demands_satisfied = np.all(
            next_demand <= self.config.demand_satisfied_tolerance
        )
        done = (self.current_step >= self.horizon) or bool(all_demands_satisfied)
        remaining_steps = max(self.horizon - self.current_step, 0)
        early_finished = bool(all_demands_satisfied) and (self.current_step < self.horizon)
        completion_bonus = (
            self.config.early_completion_bonus * remaining_steps
            if early_finished
            else 0.0
        )
        reward += completion_bonus

        info = {
            "normalized_action": normalized_action,
            "gate_action": gate_action,
            "requested_supply": requested_supply,
            "actual_supply": actual_supply,
            "unmet_demand": unmet_demand,
            "oversupply": oversupply,
            "channel_weights": self.channel_weights.copy(),
            "unmet_ratio": unmet_ratio,
            "oversupply_ratio": oversupply_ratio,
            "smoothness_cost": smoothness_cost,
            "unmet_penalty": unmet_penalty_value,
            "oversupply_penalty_value": oversupply_penalty_value,
            "smoothness_penalty_value": smoothness_penalty_value,
            "safety_violation": safety_violation,
            "safety_penalty": safety_penalty_value,
            "completion_bonus": completion_bonus,
            "early_finished": early_finished,
            "all_demands_satisfied": bool(all_demands_satisfied),
        }

        if not done:
            self.current_demands = next_demand
        else:
            self.current_demands = next_demand

        return self._get_obs(), float(reward), done, info

    def simulate_supply(self, gate_openings: np.ndarray) -> np.ndarray:
        """
        External hydraulic interface.

        Input:
            gate_openings: gate openings of N channels in [0, 1]

        Output:
            actual water supply of N channels within one time period
        """
        if self.hydraulic_simulator is hydraulic_simulator:
            result = self.hydraulic_simulator(
                gate_openings.astype(np.float32),
                previous_state=self.hydraulic_state,
                use_h00=self.current_step == 0,
            )
        else:
            result = self.hydraulic_simulator(gate_openings.astype(np.float32))

        if isinstance(result, tuple):
            supply, final_state = result
            self.hydraulic_state = final_state
        else:
            supply = result

        supply = np.asarray(supply, dtype=np.float32)

        if supply.shape != (self.num_channels,):
            raise ValueError(
                f"Hydraulic simulator must return shape {(self.num_channels,)}, got {supply.shape}"
            )
        return np.maximum(supply, 0.0)

    def default_hydraulic_simulator(self, gate_openings: np.ndarray) -> np.ndarray:
        """
        Placeholder hydraulic simulator.

        Replace this function, or pass `hydraulic_simulator=your_function`
        when constructing the environment.
        """
        return gate_openings * self.channel_capacity

    def _scale_action_to_gate_range(self, normalized_action: np.ndarray) -> np.ndarray:
        gate_min = self.config.gate_open_min
        gate_max = self.config.gate_open_max
        return gate_min + normalized_action * (gate_max - gate_min)

    def _build_channel_weights(self, channel_weights: np.ndarray | None) -> np.ndarray:
        if channel_weights is None:
            return np.ones(self.num_channels, dtype=np.float32)

        weights = np.asarray(channel_weights, dtype=np.float32)
        if weights.shape != (self.num_channels,):
            raise ValueError(
                f"channel_weights must have shape {(self.num_channels,)}, got {weights.shape}"
            )
        if np.any(weights < 0):
            raise ValueError("channel_weights must be non-negative.")
        if np.all(weights == 0):
            raise ValueError("channel_weights cannot be all zeros.")
        return weights

    def _compute_safety_penalty(self) -> tuple[Dict[str, float], float]:
        if not isinstance(self.hydraulic_state, dict):
            return {"h": 0.0, "q": 0.0, "qf": 0.0}, 0.0

        h_violation = 0.0
        q_violation = 0.0
        qf_violation = 0.0

        if self.config.safe_h_max is not None:
            h_key = "h_max_over_time" if "h_max_over_time" in self.hydraulic_state else "h"
            h_values = np.asarray(self.hydraulic_state[h_key], dtype=np.float32)
            h_excess = np.maximum(h_values - self.config.safe_h_max, 0.0)
            h_violation = float(h_excess.max()) if h_excess.size > 0 else 0.0

        if self.config.safe_q_max is not None:
            q_key = "Q_max_over_time" if "Q_max_over_time" in self.hydraulic_state else "Q"
            q_values = np.asarray(self.hydraulic_state[q_key], dtype=np.float32)
            q_excess = np.maximum(q_values - self.config.safe_q_max, 0.0)
            q_violation = float(q_excess.max()) if q_excess.size > 0 else 0.0

        if self.config.safe_qf_max is not None:
            qf_key = "Qf_max_over_time" if "Qf_max_over_time" in self.hydraulic_state else "Qf"
            qf_values = np.asarray(self.hydraulic_state[qf_key], dtype=np.float32)
            safe_qf = np.asarray(self.config.safe_qf_max, dtype=np.float32)
            if safe_qf.ndim == 0:
                safe_qf = np.full(self.num_channels, float(safe_qf), dtype=np.float32)
            if safe_qf.shape != (self.num_channels,):
                raise ValueError(
                    f"safe_qf_max must have shape {(self.num_channels,)}, got {safe_qf.shape}"
                )
            qf_excess = np.maximum(qf_values - safe_qf, 0.0)
            qf_violation = float(qf_excess.max()) if qf_excess.size > 0 else 0.0

        total_violation = h_violation + q_violation + qf_violation
        safety_penalty_value = self.config.safety_penalty * total_violation
        return (
            {"h": h_violation, "q": q_violation, "qf": qf_violation},
            float(safety_penalty_value),
        )

    def _sample_initial_demand(self) -> np.ndarray:
        base = np.random.uniform(
            self.config.demand_low,
            self.config.demand_high,
            size=self.num_channels,
        )
        noise = np.random.normal(
            0.0, self.config.demand_noise_std, size=self.num_channels
        )
        demand = np.maximum(base + noise, 0.0)
        return demand.astype(np.float32)

    def _transition_demand(
        self,
        previous_demand: np.ndarray,
        actual_supply: np.ndarray,
    ) -> np.ndarray:
        """
        Demand evolution for the next time period.

        In the current formulation, the next-period demand is the unmet part
        of the previous period demand.
        """
        next_demand = np.maximum(previous_demand - actual_supply, 0.0)
        return next_demand.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        demand_scale = max(self.config.demand_high, 1.0)
        self._update_state_history()
        history_features = self._get_masked_history_features()
        step_ratio = np.array(
            [self.current_step / max(self.horizon, 1)],
            dtype=np.float32,
        )
        obs_parts = [
            self.current_demands / demand_scale,
            history_features,
            step_ratio,
        ]
        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32)

    def _update_state_history(self) -> None:
        history_idx = min(max(self.current_step, 0), self.horizon - 1)
        gate_z, gate_q = self._get_gate_hydraulic_features()
        self.z_history[history_idx] = gate_z
        self.q_history[history_idx] = gate_q
        self.e_history[history_idx] = self.previous_action

    def _get_masked_history_features(self) -> np.ndarray:
        mask = self._build_history_mask()
        z_features = self.z_history.reshape(-1) * mask
        q_features = self.q_history.reshape(-1) * mask
        e_features = self.e_history.reshape(-1) * mask
        return np.concatenate([z_features, q_features, e_features]).astype(np.float32)

    def _build_history_mask(self) -> np.ndarray:
        if self.horizon != 5 or self.num_channels != 3:
            mask = np.zeros((self.horizon, self.num_channels), dtype=np.float32)
            history_idx = min(max(self.current_step, 0), self.horizon - 1)
            mask[history_idx] = 1.0
            return mask.reshape(-1)

        masks = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        history_idx = min(max(self.current_step, 0), self.horizon - 1)
        return masks[history_idx]

    def _get_gate_hydraulic_features(self) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(self.hydraulic_state, dict):
            zeros = np.zeros(self.num_channels, dtype=np.float32)
            return zeros, zeros.copy()

        gate_z = np.asarray(
            self.hydraulic_state.get(
                "gate_Z",
                np.zeros(self.num_channels, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        gate_q = np.asarray(
            self.hydraulic_state.get(
                "gate_Q",
                np.zeros(self.num_channels, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if gate_z.shape != (self.num_channels,) or gate_q.shape != (self.num_channels,):
            raise ValueError(
                "Hydraulic gate-state features must match num_channels: "
                f"expected {(self.num_channels,)}, got gate_Z={gate_z.shape}, gate_Q={gate_q.shape}."
            )
        return gate_z, gate_q
