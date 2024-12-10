# File: rainbow_highway.py

import gymnasium as gym
import numpy as np
import highway_env
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Replay Memory with Prioritized Experience Replay
class ReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Input layer
        self.fc2 = nn.Linear(128, 128)        # Hidden layer

        # Advantage and Value streams
        self.advantage_fc = nn.Linear(128, action_dim)
        self.value_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        advantage = self.advantage_fc(x)
        value = self.value_fc(x)
        return value + advantage - advantage.mean()



# Rainbow DQN Agent
class RainbowAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, batch_size=64, memory_size=100000, multi_step=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.multi_step = multi_step

        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to("cuda")
        self.target_net = DuelingDQN(state_dim, action_dim).to("cuda")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = ReplayMemory(memory_size)

        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state, epsilon=0.01):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()
        else:
            return random.randrange(self.action_dim)

    def compute_loss(self, samples, indices, weights):
        # Convert samples to a single numpy array first
        states = np.array([s for s, _, _, _, _ in samples])
        actions = np.array([a for _, a, _, _, _ in samples])
        rewards = np.array([r for _, _, r, _, _ in samples])
        next_states = np.array([ns for _, _, _, ns, _ in samples])
        dones = np.array([d for _, _, _, _, d in samples])

        # Convert to tensors more efficiently
        states = torch.from_numpy(states).float().to("cuda")
        actions = torch.from_numpy(actions).long().to("cuda")
        rewards = torch.from_numpy(rewards).float().to("cuda")
        next_states = torch.from_numpy(next_states).float().to("cuda")
        dones = torch.from_numpy(dones).float().to("cuda")
        weights = torch.tensor(weights, dtype=torch.float32).to("cuda")

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = (weights * (current_q - target_q.detach()).pow(2)).mean()
        return loss

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        samples, indices, weights = self.memory.sample(self.batch_size)
        loss = self.compute_loss(samples, indices, weights)

        # Update weights and compute TD error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute priorities
        with torch.no_grad():
            states, actions, rewards, next_states, dones = zip(*samples)
            states = torch.tensor(states, dtype=torch.float32).view(self.batch_size, -1).to("cuda")
            actions = torch.tensor(actions).to("cuda")
            next_states = torch.tensor(next_states, dtype=torch.float32).view(self.batch_size, -1).to("cuda")

            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = self.target_net(next_states).max(1)[0]
            target_q = torch.tensor(rewards, dtype=torch.float32).to("cuda") + \
                (1 - torch.tensor(dones, dtype=torch.float32).to("cuda")) * self.gamma * next_q
            td_errors = (current_q - target_q).abs().cpu().numpy()

        self.memory.update_priorities(indices, td_errors.tolist())


# Training Loop
def train_rainbow(env, agent, episodes=500, max_steps=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten the observation
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = next_state.flatten()  # Flatten the next observation

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        agent.update_target()
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")


# Main function
if __name__ == "__main__":
    env = gym.make("highway-v0", config = {
        "observation": {"type": "Kinematics", "features": ["x", "y", "vx", "vy"]},
        "action": {"type": "DiscreteMetaAction"},
        "reward_speed_range": [20, 30]
    })

    # state_dim = env.observation_space.shape[0]
    state_dim = np.prod(env.observation_space.shape)

    action_dim = env.action_space.n

    agent = RainbowAgent(state_dim, action_dim)
    print(env.observation_space.shape)


    train_rainbow(env, agent)
