from collections import deque
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(69)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
  def __init__(self, state_dim, action_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(state_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, action_dim),
    )
  def forward(self, x): return self.net(x)

class DQNAgent:
  def __init__(self,
               state_dim: int,
               action_dim: int,
               episodes: int,
               replay_size: int=20_000,
               batch_size: int=64,
               lr :float=1e-3
  ):
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.policy = DQN(state_dim, action_dim)
    self.behavior_policy = DQN(state_dim, action_dim)
    self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    self.criterion = nn.MSELoss()

    # with prob of eps, choose a random action (not from policy)
    #   this is done at the beginning of training to build in random
    epsilon_start = 1.0
    self.epsilon_min = 0.05
    self.epsilon = epsilon_start
    self.epsilon_decay = (self.epsilon_min / epsilon_start) ** (1.0 / (episodes // 2))
    print(f"epsilon_decay: {self.epsilon_decay:.3f}")

    # (state, action, reward, next_state, done)
    self.replay_memory = deque(maxlen=replay_size)
    self.batch_size = batch_size

    # discount factor for future rewards
    # the agent values immediate reward most, but still values future reward
    #   almost as much (99% per step).
    self.gamma = 0.99

    self.update_behavior_model()

  def update_behavior_model(self):
    self.behavior_policy.load_state_dict(self.policy.state_dict())

  def remember(self, state, action, reward, next_state, done):
    self.replay_memory.append((state, action, reward, next_state, done))

  def act(self, state, testing: bool=False):
    if not testing and np.random.rand() < self.epsilon:
      return np.random.choice(self.action_dim)
    state = torch.FloatTensor(state).unsqueeze(0)
    q_values = self.policy(state)
    return torch.argmax(q_values).item()

  def replay(self):
    if len(self.replay_memory) < self.batch_size: return

    batch = random.sample(self.replay_memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states), device=device)
    actions = torch.LongTensor(np.array(actions), device=device)
    rewards = torch.FloatTensor(np.array(rewards), device=device)
    next_states = torch.FloatTensor(np.array(next_states), device=device)
    dones = torch.FloatTensor(np.array(dones), device=device)

    # q
    curr_q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q = self.behavior_policy(next_states).max(1)[0]
    expected_q = rewards + (1 - dones) * self.gamma * next_q

    loss = self.criterion(curr_q, expected_q.detach())
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def update_eps(self):
    if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

def train_dqn(env: gym.Env, agent: DQNAgent, episodes: int, ep_update: int):
  steps = 0
  for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
      steps += 1
      action = agent.act(state)
      next_state, reward, terminated, truncated, _ = env.step(action)

      # position penalty to encourage staying centered
      cart_pos = next_state[0]
      position_penatly = abs(cart_pos) * 0.1
      modified_reward = reward - position_penatly

      position_done = abs(cart_pos) > 2.4
      done = terminated or truncated or position_done

      agent.remember(state, action, modified_reward, next_state, done)
      agent.replay()

      state = next_state
      total_reward += reward

    agent.update_eps()

    if episode % ep_update == 0:
      print(f"episode: {episode+1}, reward: {total_reward:.2f}, steps: {steps}, eps: {agent.epsilon:.2f}")
      agent.update_behavior_model()

def test_dqn(env: gym.Env, agent: DQNAgent, episodes: int):
  print("testing agent...")
  episode_rewards = []
  for _ in range(episodes):
    state, _ = env.reset()
    episode_reward, done = 0, False
    while not done:
      action = agent.act(state, testing=True)
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      episode_reward += reward
      state = next_state
    episode_rewards.append(episode_reward)
  return episode_rewards

if __name__ == "__main__":
  env = gym.make("CartPole-v1")
  state_dim = env.observation_space.shape[0] # 4
  action_dim = env.action_space.n # 2
  print(f"state_dim: {state_dim}, action_dim: {action_dim}")
  episodes = 1000
  up_episodes = 20

  agent = DQNAgent(state_dim, action_dim, episodes, lr=0.0001)
  train_dqn(env, agent, episodes, up_episodes)

  test_episodes = 100
  test_episode_rewards = test_dqn(env, agent, test_episodes)
  print(f"average reward over {test_episodes} test episodes: {(sum(test_episode_rewards) / test_episodes):.2f}")
  env.close()

  test_env = gym.make("CartPole-v1", render_mode="human")
  state, _ = test_env.reset()
  done = False
  while not done:
    action = agent.act(state, testing=True)
    next_state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    state = next_state
  test_env.close()
