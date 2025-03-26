import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from DQNetwork import DQNetwork, select_action, store_transition, train

# parameters
num_episodes = 1000
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995
episode_rewards = []
lr = 0.001
      
# environmental stuff
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQNetwork(state_size, action_size)
target_net = DQNetwork(state_size, action_size)

policy_net = DQNetwork(state_size, action_size)
target_net = DQNetwork(state_size, action_size)
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def correct_shape(state):
    return np.reshape(state, [1,env.observation_space.shape[0]]) # correcting shape to be fitting for NN

if __name__ == "__main__":

    try:
        policy_net.load_state_dict(torch.load("weights.pth")) # loading available weights if possible
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("State dict not found. Training from scratch...")

    # copying state dict from policy to target net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # training loop
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = select_action(state, epsilon, policy_net)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            store_transition(state, action, reward, next_state, done)
            train(policy_net, target_net, optimizer, loss_fn)
            state = next_state
        
        episode_rewards.append(total_reward)
        # decaying epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict()) # updating target net every 10 episodes

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-49:])
            print(f"Episode {episode}, Avg reward ladt 50 episodes: {avg_reward:.2f}, Epsilon: {epsilon:.3f}") # logging mean reward over last 50 steps

    print(f"Completed training over {num_episodes} episodes. Avg Reward last 100: {np.mean(episode_rewards[-100:])}")
    torch.save(policy_net.state_dict(), "weights.pth")

    plt.plot(episode_rewards, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Performance ober Training")
    plt.legend()
    plt.show

