# Multi-Agent_Reinforcement_Learning


This project implements a **Multi-Agent Reinforcement Learning (MARL)** system for optimizing warehouse logistics. The system uses the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm to train multiple agents to collaborate in a simulated warehouse environment.

---

## Core Functionality

- Simulates a warehouse environment with multiple agents.
- Implements the MADDPG algorithm for multi-agent learning.
- Supports agent communication and cooperation.
- Provides visualization tools for training progress and agent behavior.

---

## Project Structure

- **`environment.py`**: Defines the warehouse environment using OpenAI Gym.
- **`maddpg.py`**: Implements the MADDPG algorithm and agent architecture.
- **`train.py`**: Contains the main training loop and evaluation functions.
- **`visualization.py`**: Provides tools for visualizing the environment and training progress.

---

## Requirements

This project requires the following dependencies:

- Python 3.7+
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib
- Seaborn

To install the required dependencies, run:

---

## How to Run

### 1. Clone this repository

git clone https://github.com/your_username/your_repository_name.git
cd your_repository_name

### 2. Train the Agents

To start training, run the following command:

python train.py

This will start the training process with default parameters (300 episodes). You can modify the number of episodes and other hyperparameters in the `train.py` file.

---

## Visualization

The training process generates several visualizations in the `logs/` directory:

1. **`training_results.png`**: Shows training rewards, steps per episode, and reward distribution.
2. **`movement_heatmap.png`**: Displays a heatmap of agent movements in the warehouse.
3. **`warehouse_animation.gif`**: Animates the agents' movements during training.

These visualizations help analyze agent behavior and learning progress.

---

## Customization

You can customize the warehouse environment by modifying parameters in `environment.py`, such as:

- Warehouse size (`width`, `height`)
- Number of agents (`n_agents`)
- Maximum steps per episode (`max_steps`)

For example, you can change the environment initialization in `train.py`:

env = WarehouseEnvWithCommunication(width=15, height=15, n_agents=3, max_steps=300)

---

## Results

After training, performance metrics are saved in `logs/metrics.txt`. This file contains:

- Final average reward over episodes.
- Best reward achieved during training.
- Average steps per episode.
- Number of successful episodes.
- Total number of episodes trained.

Example metrics file:

final_avg_reward: 15.23
best_reward: 25.0
avg_steps: 180.5
successful_episodes: 120
total_episodes: 300

---

## Future Work

Possible improvements and extensions include:

1. Implementing more advanced MARL algorithms (e.g., QMIX, PPO).
2. Enhancing the warehouse environment with additional features like dynamic obstacles or item pickups.
3. Optimizing performance for larger-scale simulations with more agents.

---
