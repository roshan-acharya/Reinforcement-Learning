import gymnasium as gym
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

episode_reward=[]

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(4,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=-1)
        )
    def forward(self,x):
        return self.fc(x)
    
#initialize env and hyperparameters
env=gym.make("CartPole-v1")
episodes = 500
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

#Network and optimizer

q_network=PolicyNetwork()
optimizer=optim.Adam(q_network.parameters(),lr=learning_rate)
criterion = nn.MSELoss()

#Training an agent

episode_reward=[]
for episode in range(episodes):
    state, _ =env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward=0

    for time in range(500):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values=q_network(state)
            action = torch.argmax(q_values).item()
    
    #Take action in environment
    next_state, reward,done,_,_= env.step(action)



