import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.reward_history = []
        self.step_history = []
        self.agent_positions = [[] for _ in range(len(agents))]

    def update_plot(self, frame):
        self.ax.clear()
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_title(f"Warehouse Environment - Step {frame}")

        # plots agents
        for i, agent_pos in enumerate(self.env.agents):
            self.ax.plot(agent_pos[0], agent_pos[1], 'bo', markersize=10, label=f'Agent {i+1}')

        # plots target
        self.ax.plot(self.env.target[0], self.env.target[1], 'r*', markersize=15, label='Target')

        # plots trajectories
        for i, positions in enumerate(self.agent_positions):
            if len(positions) > 1:
                x, y = zip(*positions)
                self.ax.plot(x, y, '--', linewidth=1, alpha=0.5)

        self.ax.legend()
        self.ax.grid(True)

    def save_animation(self, filename='warehouse_animation.gif'):
        frames = len(self.agent_positions[0])
        anim = FuncAnimation(self.fig, self.update_plot, frames=frames, interval=200, blit=False)
        anim.save(filename, writer='pillow', fps=5)
        plt.close()

    def update(self, reward, steps):
        self.reward_history.append(reward)
        self.step_history.append(steps)
        for i, agent_pos in enumerate(self.env.agents):
            self.agent_positions[i].append(agent_pos.tolist())

    def plot_heatmap(self):
        plt.figure(figsize=(10, 10))
        heatmap_data = np.zeros((self.env.height, self.env.width))
        for agent_positions in self.agent_positions:
            for x, y in agent_positions:
                heatmap_data[int(y), int(x)] += 1
        sns.heatmap(heatmap_data, cmap='YlOrRd')
        plt.title('Agent Movement Heatmap')
        plt.savefig('movement_heatmap.png')
        plt.close()
