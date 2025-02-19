import torch
from environment import WarehouseEnvWithCommunication
from maddpg import MADDPGAgent
from visualization import Visualizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import psutil
import os
import time
import logging
import seaborn as sns
import random

def train_step(env, agents, state, episode):
    total_reward = 0
    done = False
    step_count = 0
    episode_rewards = []
    
    actions = [0] * len(agents)
    
    while not done:
        for i, agent in enumerate(agents):
            actions[i] = agent.act(state[i])
        
        next_state, rewards, done, _ = env.step(actions)
        step_reward = sum(rewards)
        total_reward += step_reward
        episode_rewards.append(step_reward)
        
        for i, agent in enumerate(agents):    # storing experiences
            agent.remember(state[i], actions[i], rewards[i], next_state[i], done)
        
        if episode > 10:    # updating agents if enough experience (>10 episodes)
            for agent in agents: 
                agent.replay()
        
        state = next_state
        step_count += 1
    
    return total_reward, step_count, episode_rewards

def evaluate_agents(env, agents, n_episodes=5):
    eval_rewards = []
    eval_steps = []
    
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            actions = [agent.act(state[i]) for i, agent in enumerate(agents)]
            next_state, rewards, done, _ = env.step(actions)
            total_reward += sum(rewards)
            steps += 1
            state = next_state
            
        eval_rewards.append(total_reward)
        eval_steps.append(steps)
    
    return np.mean(eval_rewards), np.mean(eval_steps)

def plot_results(rewards_history, step_history):
    plt.figure(figsize=(15, 5))
    
    # plotting rewards
    plt.subplot(1, 3, 1)
    sns.lineplot(data=rewards_history, label='Episode Reward')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # plotting steps
    plt.subplot(1, 3, 2)
    sns.lineplot(data=step_history, label='Steps per Episode')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # plotting reward distribution
    plt.subplot(1, 3, 3)
    sns.histplot(rewards_history, kde=True)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    
    plt.tight_layout()
    plt.savefig('../warehouse_rl/logs/training_results.png')  # saving in the same folder
    plt.close()

def save_metrics(rewards_history, step_history):     # text file will be saved containing all the performance metrics
    metrics = {
        'final_avg_reward': np.mean(rewards_history[-100:]),
        'best_reward': max(rewards_history),
        'avg_steps': np.mean(step_history),
        'successful_episodes': sum(1 for r in rewards_history if r > 0),
        'total_episodes': len(rewards_history)
    }
    
    with open('../warehouse_rl/logs/metrics.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def train(episodes=300, eval_interval=10):  # can change the number to episodes to train for
    os.makedirs('../warehouse_rl/logs', exist_ok=True)   # creates a log folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initializing environment and agents
    env = WarehouseEnvWithCommunication(max_steps=200)
    agents = [MADDPGAgent(9, 5, device) for _ in range(env.n_agents)]
    
    visualizer = Visualizer(env, agents)

    best_reward = -float('inf')
    rewards_history = []
    step_history = []
    eval_rewards = []
    
    for episode in tqdm(range(episodes)):
        if episode % 50 == 0:
            torch.cuda.empty_cache()   # clears cache time to time
        
        state = env.reset()
        episode_start = time.time()
        total_reward, steps, episode_rewards = train_step(env, agents, state, episode)
        
        visualizer.update(total_reward, steps)

        rewards_history.append(total_reward)
        step_history.append(steps)
        
        # evaluate periodically
        if episode % eval_interval == 0:
            eval_reward, eval_step = evaluate_agents(env, agents)
            eval_rewards.append(eval_reward)
            
            avg_reward = np.mean(rewards_history[-eval_interval:])
            avg_steps = np.mean(step_history[-eval_interval:])
            
            if eval_reward > best_reward:            
                best_reward = eval_reward
        
        # if threashold reached, we can stop early
        if episode >= 100 and np.mean(eval_rewards[-5:]) >= 8.0:
            print("Reached performance threshold, stopping training")
            break
                
    visualizer.plot_heatmap()
    visualizer.save_animation()

    # saving final results
    plot_results(rewards_history, step_history)
    save_metrics(rewards_history, step_history)
        
    return rewards_history, step_history, eval_rewards

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    rewards, steps, eval_rewards = train()