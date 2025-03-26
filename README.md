# 🎯 CartPole-v1 – Deep Q-Learning Agent

This project implements a **Deep Q-Network (DQN)** to solve the classic `CartPole-v1` control problem using the Gymnasium library.  
The agent uses experience replay, target networks, and epsilon-greedy exploration to learn a stable policy.

---

## 🛠️ Environment Overview

- **Environment:** `CartPole-v1`
- **Goal:** Balance a pole on a moving cart for as long as possible
- **Observation space:** 4 continuous variables (cart position, velocity, pole angle, and angular velocity)
- **Actions:** Discrete → `0 = push left`, `1 = push right`
- **Reward:** +1 for every timestep the pole remains balanced

---

## 🧠 Algorithm

- **Method:** Deep Q-Learning (DQN)
- **Network architecture:**
  - Fully connected (16 → 8 → output)
- **Techniques used:**
  - Experience replay buffer
  - Target network updates every 10 episodes
  - Epsilon-greedy exploration with exponential decay

---

## ⚙️ Hyperparameters

| Parameter        | Value      |
|------------------|------------|
| Episodes         | `1,000`    |
| Learning rate    | `0.001`    |
| Batch size       | `32`       |
| Discount factor  | `0.99`     |
| Initial epsilon  | `1.0`      |
| Minimum epsilon  | `0.1`      |
| Epsilon decay    | `0.995`    |
| Replay buffer    | `1,000,000` transitions |

---

## 📁 Project Structure

| File           | Description |
|----------------|-------------|
| `DQNetwork.py` | Defines the DQNetwork class and all logic for action selection, training, and replay buffer |
| `agent.py`     | Main training script; trains the DQN agent and saves model weights to `weights.pth` |
| `evaluation.py`| Loads the trained model and runs the agent for 3 episodes with rendering enabled |
| `weights.pth`  | (Generated) Trained neural network weights |
