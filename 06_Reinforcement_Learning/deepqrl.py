import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import trange
from collections import deque

device = "cpu"

class DQN(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Outputs Q-values for Left and Right
        )

    def forward(self, x):
        return self.fc(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train(env: gym.Env,
          policy_net: nn.Module,
          target_net: nn.Module,
          memory: ReplayMemory,
          epsilon: float,
          optimizer: torch.optim.Optimizer,
):
    for episode in (t := trange(EPISODES)):
        state, _ = env.reset()
        total_reward = 0
        done = False

        epsilon = max(EPSILON_END, EPSILON_START - (episode / EPSILON_DECAY_WINDOW))

        while not done:
            # action selection (Epsilon-Greedy)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_t).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # save to memory
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # optimize the Model
            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                # unzip the batch
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

                s_batch = torch.FloatTensor(np.array(s_batch))
                a_batch = torch.LongTensor(a_batch).unsqueeze(1)
                r_batch = torch.FloatTensor(r_batch)
                ns_batch = torch.FloatTensor(np.array(ns_batch))
                d_batch = torch.FloatTensor(d_batch)

                # current Q values
                current_q = policy_net(s_batch).gather(1, a_batch)

                # target Q values (using the Target Network)
                with torch.no_grad():
                    max_next_q = target_net(ns_batch).max(1)[0]
                    target_q = r_batch + (GAMMA * max_next_q * (1 - d_batch))

                loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # update Target Network
        if episode % TARGET_UPDATE_EPISODES == 0:
            target_net.load_state_dict(policy_net.state_dict())

        t.set_description(f"score: {total_reward}, epsilon: {epsilon:.2f}")

EPISODES = 500
BATCH_SIZE = 128
LR = 0.0005
GAMMA = 0.99
TARGET_UPDATE_EPISODES = 6

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_WINDOW = 350

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 256
    memory = ReplayMemory(50_000)
    epsilon = EPSILON_START

    policy_net = DQN(state_dim, hidden_dim, action_dim).to(device) # the one we train
    target_net = DQN(state_dim, hidden_dim, action_dim).to(device) # the one used for targets
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    train(env, policy_net, target_net, memory, epsilon, optimizer)

    env.close()

    test_env = gym.make("CartPole-v1", render_mode="human")
    state, _ = test_env.reset()
    done = False
    total_test_reward = 0

    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_t).argmax().item()

        state, reward, terminated, truncated, _ = test_env.step(action)
        total_test_reward += reward
        done = terminated or truncated

    print(f"Test Game Score: {total_test_reward}")
    test_env.close()
