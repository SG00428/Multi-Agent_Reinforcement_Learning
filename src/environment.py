import gym
import numpy as np

class WarehouseEnvWithCommunication(gym.Env):
    def __init__(self, width=10, height=10, n_agents=2, max_steps=200):
        super().__init__()  # environment dimensions and parameters
        self.width = width 
        self.height = height
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.episodes_completed = 0 
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(n_agents, 9),  # 9 features per agent
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)  # 5 possible actions
        
        # making arrays
        self.agents = np.zeros((n_agents, 2), dtype=np.float32)  # agent position
        self.messages = np.zeros((n_agents, 2), dtype=np.float32) # agent messages for communication
        self.rewards = np.zeros(n_agents, dtype=np.float32)  # stores rewards
        self.current_messages = np.zeros((n_agents, 2), dtype=np.float32)  
        self.target = np.zeros(2, dtype=np.float32)   # position of target
        self.obs = np.zeros((n_agents, 9), dtype=np.float32)  # prev distances to target
        self.prev_distances = None
        
        self.reset() # initializes

    def reset(self, difficulty=None):
        self.current_step = 0
        if difficulty is None:
            difficulty = min(1.0, self.episodes_completed / 1000)
            
        # Adjusting target distance based on difficulty
        max_dist = self.width * (0.3 + 0.7 * difficulty)
        while True:
            self.agents = np.random.randint(0, [self.width, self.height], size=(self.n_agents, 2)).astype(np.float32) # randomly position agents and targets
            self.target = np.random.randint(0, [self.width, self.height], size=2).astype(np.float32)
            if np.all(np.linalg.norm(self.agents - self.target, axis=1) <= max_dist):
                break
                
        self.prev_distances = np.linalg.norm(self.agents - self.target, axis=1) # initial distance
        return self._get_obs()

    def step(self, actions):
        self.current_step += 1
        self.rewards.fill(0) # restes rewards
        done = False
        self.current_messages.fill(0)  # resets msg
        
        # agent's action
        for i, action in enumerate(actions):
            if action == 1:  # up
                self.agents[i, 1] = max(0, self.agents[i, 1] - 1)
            elif action == 2:  # right
                self.agents[i, 0] = min(self.width - 1, self.agents[i, 0] + 1)
            elif action == 3:  # down
                self.agents[i, 1] = min(self.height - 1, self.agents[i, 1] + 1)
            elif action == 4:  # left
                self.agents[i, 0] = max(0, self.agents[i, 0] - 1)
            
            # current distance to target
            current_distance = np.linalg.norm(self.agents[i] - self.target)
            
            if np.all(self.agents[i] == self.target):
                self.rewards[i] = 20.0  # higher success reward
                done = True
            else:
                # reward structure
                distance_reward = (self.prev_distances[i] - current_distance)
                self.rewards[i] += distance_reward * 2.0
                
                self.rewards[i] -= 0.05 #  penalty for steps
                
                if current_distance < self.prev_distances[i]:  # can explore more
                    self.rewards[i] += 0.5
            
            # communication reward
            if np.linalg.norm(self.agents[i] - self.target) < 3:
                self.current_messages[i] = self.target
                self.rewards[i] += 0.5
            
            # Updating distance
            self.prev_distances[i] = current_distance

        if self.current_step >= self.max_steps:  # Checking if maximum steps reached
            done = True
            self.rewards -= 1.0

        if done:
            self.episodes_completed += 1 

        return self._get_obs(self.current_messages), self.rewards, done, {}

    def _get_obs(self, messages=None):
        if messages is None:
            messages = np.zeros((self.n_agents, 2), dtype=np.float32)
            
        for i in range(self.n_agents):
            self.obs[i] = np.array([  # normalizing here
                self.agents[i, 0] / self.width,
                self.agents[i, 1] / self.height,
                self.target[0] / self.width,
                self.target[1] / self.height,
                float(np.all(self.agents[i] == self.target)),
                messages[i, 0] / self.width,
                messages[i, 1] / self.height,
                np.linalg.norm(self.agents[i] - self.target) / np.sqrt(self.width**2 + self.height**2),
                self.current_step / self.max_steps
            ], dtype=np.float32)
            
        return self.obs.copy()