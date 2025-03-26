import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import gymnasium as gym

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# parameters
gamma = 0.99
memory_size = 1_000_000
batch_size = 32

replay_buffer = deque(maxlen=memory_size)

# Network class
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # q values for all states
    

def select_action(state, epsilon, policy_net):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
        return action
    else:
        state = torch.FloatTensor(state).unsqueeze(0) 
        action = policy_net(state) # returns q values for current state
        return action.argmax(dim=1).item()

def store_transition(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state ,done))

def train(policy_net, target_net, optimizer, loss_fn):
    if len(replay_buffer) < batch_size:
        return
    
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones).unsqueeze(1)

    # computing q values
    q_values = policy_net(states).gather(1, actions)

    # computing max q s'a' for next states using target network
    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
    next_q_values[dones==1] = 0.0

    # computing target q values
    target_q_values = rewards + gamma * next_q_values * (1-dones)
    
    #computing loss 
    loss = loss_fn(q_values, target_q_values)

    # optimizing model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

