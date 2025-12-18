import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
episodes = 500
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

episode_rewards = []
for episode in range(episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    for time in range(500):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state)
            action = torch.argmax(q_values).item()
        
        # Take action in the environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        total_reward += reward
        
        # Compute target Q-value
        with torch.no_grad():
            target = reward
            if not done:
                target += gamma * torch.max(q_network(next_state)).item()
        
        # Compute current Q-value and update
        q_values = q_network(state)
        target_f = q_values.clone()
        target_f[0][action] = target
        
        # Backpropagate loss
        loss = criterion(q_values, target_f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        if done:
            break
    episode_rewards.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode: {episode + 1}, Reward: {total_reward}")
env.close()


torch.save(q_network.state_dict(), "q_network_cartpole.pth")

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Rewards Over Time")
plt.legend()
plt.grid()
plt.show()

