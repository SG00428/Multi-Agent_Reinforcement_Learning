import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging

class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128): 
        super().__init__()
        self.network = nn.Sequential(   #using neural networks here
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        for m in self.modules(): # initializing weights
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)  # forward pass

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        
    def push(self, transition):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.memory.append(transition)
            self.priorities[self.size] = max_priority
            self.size += 1
        else:
            self.memory[self.pos] = transition
            self.priorities[self.pos] = max_priority
            
        self.pos = (self.pos + 1) % self.capacity
            
    def sample(self, batch_size):
        if self.size < batch_size:
            return None, None, None
        
        valid_priorities = self.priorities[:self.size]
        
        # calculating probabilities
        probs = valid_priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # calculating importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        samples = [self.memory[idx] for idx in indices]
        
        return samples, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item()  # Converting tensor to scalar

class MADDPGAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # initializing networks
        self.actor = DQNAgent(state_size, action_size, hidden_dim=128).to(device)
        self.actor_target = DQNAgent(state_size, action_size, hidden_dim=128).to(device)
        self.critic = DQNAgent(state_size + 1, 1, hidden_dim=128).to(device)
        self.critic_target = DQNAgent(state_size + 1, 1, hidden_dim=128).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # initializing optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)

        
        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(capacity=100000)        
        # these are hyperparameters
        self.batch_size = 256  # increased batch size
        self.gamma = 0.995  # higher discount
        self.epsilon = 1.0
        self.epsilon_decay = 0.997  # slower decay
        self.epsilon_min = 0.05  # higher minimum
        self.tau = 0.005  # faster target updates

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.actor(state)
            return torch.argmax(q_values).cpu().item()

    def remember(self, state, action, reward, next_state, done):  # stores
        self.memory.push((state, action, reward, next_state, done))

    def replay(self):
        result = self.memory.sample(self.batch_size)
        if result is None:
            return
            
        batch, indices, weights = result
        batch = list(zip(*batch))
        
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(np.array(batch[1])).to(self.device)
        rewards = torch.FloatTensor(np.array(batch[2])).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(np.array(batch[4])).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device)

        # computing targets for the Q function
        with torch.no_grad():
            next_actions = self.actor_target(next_states).max(1)[0].unsqueeze(1)
            next_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            targets = rewards + (1 - dones) * self.gamma * next_values
# current Q-values
        current_values = self.critic(torch.cat([states, actions.unsqueeze(1).float()], dim=1))
        
        td_errors = torch.abs(targets - current_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        critic_loss = (weights * F.mse_loss(current_values, targets, reduction='none')).mean()   # weighted MSE loss
        
        # updating critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # actor loss
        actor_loss = -(weights * self.critic(torch.cat([
            states,
            self.actor(states).max(1)[0].unsqueeze(1)
        ], dim=1))).mean()
        
        # update the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # soft update targets
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        # update the epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)