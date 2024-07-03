import numpy as np
from scipy.stats import norm

def simulate_trajectory_data(num_points=100, num_trajectories=3):
    np.random.seed(42)
    trajectories = []
    for _ in range(num_trajectories):
        x = np.cumsum(norm.rvs(size=num_points))
        y = np.cumsum(norm.rvs(size=num_points))
        trajectories.append(np.vstack((x, y)).T)
    return trajectories


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

class SimulatedTrajectoryData:
    def __init__(self, num_trajectories=10, num_steps=100):
        self.num_trajectories = num_trajectories
        self.num_steps = num_steps
    
    def brownian_motion(self, diffusion_coefficient=1.0):
        trajectories = []
        for _ in range(self.num_trajectories):
            steps = np.random.normal(0, np.sqrt(diffusion_coefficient), self.num_steps)
            trajectory = np.cumsum(steps)
            trajectories.append(trajectory)
        return np.array(trajectories)
    
    def levy_flight(self, alpha=1.5, beta=0):
        trajectories = []
        for _ in range(self.num_trajectories):
            steps = levy_stable.rvs(alpha, beta, size=self.num_steps)
            trajectory = np.cumsum(steps)
            trajectories.append(trajectory)
        return np.array(trajectories)
    
    def spiral_trajectory(self, r=1.0, omega=0.1):
        trajectories = []
        t = np.linspace(0, 10, self.num_steps)
        for _ in range(self.num_trajectories):
            x = r * t * np.cos(omega * t)
            y = r * t * np.sin(omega * t)
            trajectories.append(np.vstack((x, y)).T)
        return np.array(trajectories)
    
    def circular_trajectory(self, R=1.0, omega=0.1):
        trajectories = []
        t = np.linspace(0, 10, self.num_steps)
        for _ in range(self.num_trajectories):
            x = R * np.cos(omega * t)
            y = R * np.sin(omega * t)
            trajectories.append(np.vstack((x, y)).T)
        return np.array(trajectories)

