## Reinforcement Learning Notes
- Agent/actor: the robot, or the drone
- Environment: a simulated world the agent acts in
- Policy: what actions should be taken when given a certain situation
    - in this case, the neural network is our policy
- State: what the agent sees/knows about its current situation
- Action: the next thing to do given a specific state

- Advantage = Actual Return - Baseline(average)
    - Collect multiple episodes and calculate their returns (total discounted rewards)
    - Compute the baseline as the mean return across all episodes
    - Calculate advantage = return - baseline for each time step
    - Normalize advantages to have mean=0 and std=1

- Should we update the policy after every single action, or wait and see how
the whole episode plays out?
    - Three methods: learning after every action (per-step updates),
    learning after one complete attempt (per-episode updates),
    learning from multiple attempts (multi-episode batch updates)

### Extra
- for drone game in `deeprldrone.py`: `https://github.com/vedant-jumle/reinforcement-learning-101/tree/main`
    - `https://towardsdatascience.com/deep-reinforcement-learning-for-dummies/`
