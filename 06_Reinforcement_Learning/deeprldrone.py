from dataclasses import dataclass
import math
import os
import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Bernoulli
from reinforcement_learning_101.delivery_drone.game.socket_client import DroneGameClient, DroneState

def inverse_quadratic(x, decay=20, scaler=10.0, shifter=0):
    """reward decreases quadratically with distance"""
    return scaler / (1 + decay * (x - shifter)**2)

def scaled_shifted_negative_sigmoid(x, scaler=10.0, shift=0, steepness=10):
    """sigmoid function scaled and shifted"""
    return scaler / (1 + np.exp(steepness * (x - shift)))

def calc_velocity_alignment(state: DroneState):
    """
    Calculate how well the drone's velocity is aligned with optimal direction to platform.
    Returns cosine similarity: 1.0 = perfect alignment, -1.0 = opposite direction
    """
    # optimal direction: from drone to platform
    optimal_dx = state.dx_to_platform
    optimal_dy = state.dy_to_platform
    optimal_norm = math.sqrt(optimal_dx**2 + optimal_dy**2)

    if optimal_norm < 1e-6: # already at platform
        return 1.0

    optimal_dx /= optimal_norm
    optimal_dy /= optimal_norm

    # current velocity direction
    velocity_norm = state.speed
    if velocity_norm < 1e-6:  # not moving
        return 0.0

    velocity_dx = state.drone_vx / velocity_norm
    velocity_dy = state.drone_vy / velocity_norm

    # cosine similarity
    return velocity_dx * optimal_dx + velocity_dy * optimal_dy

def calc_reward(state: DroneState):
    """
    time_penality: -0.3 -> -1.0 (discourage slow completion)
    distance: 0 -> ~4 (reward moving toward platform)
    velocity_alignment: 0 or 0.5 (reward correct velocity direction)
    angle: -inf -> 0 (penalize excessive tilt)
    speed: -inf -> 0 (penalize excessive speed)
    vertical_position: -inf -> 0 strongly penalize being below
    terminal: -300 -> 600+ (success/failure outcome)
    """
    rewards = {}
    total_reward = 0.0

    # time penalty
    min_time_penality = 0.3
    max_time_penality = 1.0
    rewards["time_penality"] = -inverse_quadratic(
            state.distance_to_platform,
            decay=50,
            scaler=max_time_penality - min_time_penality
    ) - min_time_penality
    total_reward += rewards["time_penality"]

    # distance and velocity alignment
    velocity_alignment = calc_velocity_alignment(state)
    dist = state.distance_to_platform

    rewards["distance"] = 0
    rewards["velocity_alignment"] = 0

    # drone must be above platform (dy > 0)
    if dist > 0.065 and state.dy_to_platform > 0:
        if velocity_alignment > 0:
            rewards["distance"] = state.speed * scaled_shifted_negative_sigmoid(dist, scaler=4.5)
            rewards["velocity_alignment"] = 0.5

    total_reward += rewards["distance"]
    total_reward += rewards["velocity_alignment"]

    # angle penalty
    abs_angle = abs(state.drone_angle)
    max_angle = 0.20
    max_permissible_angle = ((max_angle - 0.111) * dist) + 0.111
    excess = abs_angle - max_permissible_angle
    rewards["angle"] = -max(excess, 0)
    total_reward += rewards["angle"]

    # speed penalty
    rewards["speed"] = 0.0
    speed = state.speed
    max_speed = 0.4
    if dist < 1:
        rewards["speed"] = -2 * max(speed - 0.1, 0)
    else:
        rewards["speed"] = -1 * max(speed - max_speed, 0)
    total_reward += rewards["speed"]

    # vertical position penalty
    rewards["vertical_position"] = 0.0
    if state.dy_to_platform > 0:
        rewards["vertical_position"] = 0.0
    else:
        rewards["vertical_position"] = state.dy_to_platform * 4.0
    total_reward += rewards["vertical_position"]

    # terminal rewards
    rewards["terminal"] = 0.0
    if state.landed:
        rewards["terminal"] = 500.0 + state.drone_fuel * 100.0
    elif state.crashed:
        rewards["terminal"] = -200.0
        if state.distance_to_platform > 0.3:
            rewards["terminal"] -= 100.0
        total_reward += rewards["terminal"]

    rewards["total"] = total_reward
    return rewards

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns (G_t) for each timestep based on the Bellman equation

    G_t = r_t + y*r_{t+1} + y^2 * r_{t+2} + ...
    """
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return returns

def state_to_array(state):
    """Convert DroneState dataclass to torch tensor"""
    return torch.as_tensor([
        state.drone_x,
        state.drone_y,
        state.drone_vx,
        state.drone_vy,
        state.drone_angle,
        state.drone_angular_vel,
        state.drone_fuel,
        state.platform_x,
        state.platform_y,
        state.distance_to_platform,
        state.dx_to_platform,
        state.dy_to_platform,
        state.speed,
        float(state.landed),
        float(state.crashed)
    ], dtype=torch.float32)

class DroneGamePolicy(nn.Module):
    def __init__(self, state_dim=15):
        super().__init__()

        # output is (activate main thruster, left thruster, right thruster)
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, state):
        if isinstance(state, DroneState): state = state_to_array(state)
        return self.network(state)

def collect_episodes(client: DroneGameClient, policy: nn.Module, max_steps=300):
    num_games = client.num_games

    all_episodes = [{"states": [], "actions": [], "log_probs": [], "rewards": [], "done": False} for _ in range(num_games)]

    game_states = [client.reset(game_id) for game_id in range(num_games)]
    step_counts = [0] * num_games

    while not all(ep["done"] for ep in all_episodes):
        batch_states = []
        active_game_ids = []

        for game_id in range(num_games):
            if not all_episodes[game_id]["done"]:
                batch_states.append(state_to_array(game_states[game_id]))
                active_game_ids.append(game_id)

        if len(batch_states) == 0: break

        batch_states_tensor = torch.stack(batch_states)
        batch_action_probs = policy(batch_states_tensor)
        batch_dist = Bernoulli(probs=batch_action_probs)
        batch_actions = batch_dist.sample()
        batch_log_probs = batch_dist.log_prob(batch_actions).sum(dim=1)

        for i, game_id in enumerate(active_game_ids):
            action = batch_actions[i]
            log_prob = batch_log_probs[i]

            next_state, _, done, _ = client.step({
                "main_thrust": int(action[0]),
                "left_thrust": int(action[1]),
                "right_thrust": int(action[2]),
            }, game_id)

            reward = calc_reward(next_state)

            all_episodes[game_id]['states'].append(batch_states[i])
            all_episodes[game_id]['actions'].append(action)
            all_episodes[game_id]['log_probs'].append(log_prob)
            all_episodes[game_id]['rewards'].append(reward['total'])

            game_states[game_id] = next_state
            step_counts[game_id] += 1

            if done or step_counts[game_id] >= max_steps:
                if step_counts[game_id] >= max_steps and not next_state.landed:
                    all_episodes[game_id]['rewards'][-1] -= 500 # timeout penalty

                all_episodes[game_id]['done'] = True

    return [(ep['states'], ep['actions'], ep['log_probs'], ep['rewards'])
            for ep in all_episodes]

if __name__ == "__main__":
    state = DroneState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0)
    s = np.array([
        state.drone_x, state.drone_y,
        state.drone_vx, state.drone_vy,
        state.drone_angle, state.drone_angular_vel,
        state.drone_fuel,
        state.platform_x, state.platform_y,
        state.distance_to_platform,
        state.dx_to_platform, state.dy_to_platform,
        state.speed,
        float(state.landed), float(state.crashed)
    ])

    client = DroneGameClient()
    client.connect()
