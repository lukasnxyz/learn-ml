import gymnasium as gym
import numpy as np
import math
import random

env = gym.make("CartPole-v1", render_mode="rgb_array")

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0 # exploration rate
EPSILON_DECAY = 0.995 # decay per episode
MIN_EPSILIN = 0.01 # minimum exploration prob
EPISODES = 5000

# CartPole gives us 4 continuous values: [Cart Position, Cart Velocity, Pole Angle, Pole Velocity]
# We need to turn these into bucket indices (0, 1, 2...) for our Q-Table.

POSITION_BINS = np.linspace(-2.4, 2.4, 24)
VELOCITY_BINS = np.linspace(-4, 4, 24)
ANGLE_BINS = np.linspace(-.2095, .2095, 24)
ANGLE_VELOCITY_BINS = np.linspace(-4, 4, 24)

def discretize_state(state):
    # state is array: [pos, vel, angle, angular_vel]

    if isinstance(state, tuple):
        state = state[0]

    p_idx = np.digitize(state[0], POSITION_BINS)
    v_idx = np.digitize(state[1], VELOCITY_BINS)
    a_idx = np.digitize(state[2], ANGLE_BINS)
    av_idx = np.digitize(state[3], ANGLE_VELOCITY_BINS)

    return (p_idx, v_idx, a_idx, av_idx)

q_table = np.zeros((len(POSITION_BINS)+1,
                    len(VELOCITY_BINS)+1,
                    len(ANGLE_BINS)+1,
                    len(ANGLE_VELOCITY_BINS)+1,
                    env.action_space.n))

for episode in range(EPISODES):
    state = env.reset()
    current_state_discrete = discretize_state(state)
    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state_discrete])

        next_state, reward, terminated, truncated, _ = env.step(action)

        next_state_discrete = discretize_state(next_state)

        if terminated or truncated:
            q_target = reward
        else:
            q_target = reward + DISCOUNT_FACTOR * np.max(q_table[next_state_discrete])

        current_q = q_table[current_state_discrete + (action,)]

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * q_target
        q_table[current_state_discrete + (action,)] = new_q

        current_state_discrete = next_state_discrete
        total_reward += 1

    if EPSILON > MIN_EPSILIN:
        EPSILON *= EPSILON_DECAY

    if episode % 500 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {EPSILON:.2f}")

print("Training finished!")

# run one episode with epsilon = 0 (pure exploitation)
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()
current_state_discrete = discretize_state(state)
terminated = False
truncated = False

while not terminated and not truncated:
    action = np.argmax(q_table[current_state_discrete]) # always pick best action
    state, reward, terminated, truncated, _ = env.step(action)
    current_state_discrete = discretize_state(state)
    env.render()

env.close()
