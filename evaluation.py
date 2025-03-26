import gymnasium as gym
import torch
from DQNetwork import DQNetwork, select_action
from agent import correct_shape

env = gym.make("CartPole-v1", render_mode="human")
epsilon = 0
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
policy_net = DQNetwork(state_size, action_size)
policy_net.load_state_dict(torch.load("weights.pth"))
policy_net.eval()

for episode in range(3):
    state, _ = env.reset()
    done = False
    while not done:
        action = select_action(state, epsilon, policy_net)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        state = next_state
env.close()