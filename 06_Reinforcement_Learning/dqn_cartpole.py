from collections import deque
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 batch_size: int=64,
                 learning_rate: float=0.0005,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=10_000)
        self.batch_size = batch_size

        self.gamma = 0.99

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size: return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        curr_q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(curr_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(env: gym.Env, agent: DQNAgent, episodes: int, ep_update: int):
    rewards_per_ep = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward, steps = 0.0, 0
        while not done:
            steps += 1
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # position penality to encourage staying centered
            cart_pos = next_state[0]
            position_penalty = abs(cart_pos) * 0.1
            modified_reward = reward - position_penalty

            position_done = abs(cart_pos) > 2.4
            done = terminated or truncated or position_done

            agent.remember(state, action, modified_reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        if (episode) % ep_update == 0: agent.update_target_model()
        rewards_per_ep.append(total_reward)
        if (episode) % ep_update == 0:
            print(f"episode: {episode}, reward: {total_reward:.2f}, steps: {steps}, eps: {agent.epsilon:.2f}")

    return rewards_per_ep

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    rewards = train_dqn(env, agent, 500, 10)
    #torch.save(agent.model.state_dict(), "dqn_cartpole_centered.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Deep Q-Learning on CartPole with Position Constraints")
    plt.grid()
    plt.savefig("dqn_cartpole_training_curve.png")
    plt.show()

    test_episodes, episode_rewards = 100, []
    for episode in range(test_episodes):
        state, _ = env.reset()
        episode_reward, done = 0, False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        episode_rewards.append(episode_reward)
    print(f"average reward over {test_episodes} test episodes: {(sum(episode_rewards) / test_episodes):.2f}")
    env.close()

    test_env = gym.make("CartPole-v1", render_mode="human")
    state, _ = test_env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        state = next_state
    test_env.close()
