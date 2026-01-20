import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import trange
from collections import deque

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

# Replay Buffer (Memory)
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

EPISODES = 550
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 10
LR = 0.0005

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_WINDOW = 350

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 256

# Initialize Networks
policy_net = DQN(state_dim, hidden_dim, action_dim)  # The one we train
target_net = DQN(state_dim, hidden_dim, action_dim)  # The one used for targets
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(50_000)
epsilon = EPSILON_START

for episode in (t := trange(EPISODES)):
    state, _ = env.reset()
    total_reward = 0
    done = False

    epsilon = max(EPSILON_END, EPSILON_START - (episode / EPSILON_DECAY_WINDOW))

    while not done:
        # Action selection (Epsilon-Greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_t).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Save to memory
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Optimize the Model
        if len(memory) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            # Unzip the batch
            s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

            s_batch = torch.FloatTensor(np.array(s_batch))
            a_batch = torch.LongTensor(a_batch).unsqueeze(1)
            r_batch = torch.FloatTensor(r_batch)
            ns_batch = torch.FloatTensor(np.array(ns_batch))
            d_batch = torch.FloatTensor(d_batch)

            # Current Q values
            current_q = policy_net(s_batch).gather(1, a_batch)

            # Target Q values (using the Target Network)
            with torch.no_grad():
                max_next_q = target_net(ns_batch).max(1)[0]
                target_q = r_batch + (GAMMA * max_next_q * (1 - d_batch))

            # Loss calculation (MSE)
            loss = nn.MSELoss()(current_q.squeeze(), target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update Target Network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    t.set_description(f"score: {total_reward}, epsilon: {epsilon:.2f}")

print("Training complete!")

env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=100_000)
state, _ = env.reset()
done = False
total_test_reward = 0

while not done:
    # 1. Turn state into a Torch Tensor
    state_t = torch.FloatTensor(state).unsqueeze(0)

    # 2. Get the action from the network (No gradient needed for testing)
    with torch.no_grad():
        action = policy_net(state_t).argmax().item()

    # 3. Apply the action
    state, reward, terminated, truncated, _ = env.step(action)
    total_test_reward += reward

    # 4. End game if pole falls or time limit reached
    done = terminated or truncated

print(f"Test Game Score: {total_test_reward}")
env.close()
