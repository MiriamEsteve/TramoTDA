from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

class SimulatedTrajectoryData:
    def __init__(self, num_trajectories=10, num_steps=100):
        self.num_trajectories = num_trajectories
        self.num_steps = num_steps
    
    
    def simulate_trajectory_data(self):
        np.random.seed(42)
        trajectories = []
        for _ in range(self.num_trajectories):
            x = np.cumsum(norm.rvs(size=self.num_steps))
            y = np.cumsum(norm.rvs(size=self.num_steps))
            trajectories.append(np.vstack((x, y)).T)
        return trajectories

    def brownian_motion(self, diffusion_coefficient=1.0):
        trajectories = []
        for _ in range(self.num_trajectories):
            #steps = np.random.normal(0, np.sqrt(diffusion_coefficient), self.num_steps)
            #trajectory = np.cumsum(steps)
            #trajectories.append(trajectory)
            x = np.cumsum(np.random.normal(0, np.sqrt(diffusion_coefficient), self.num_steps))
            y = np.cumsum(np.random.normal(0, np.sqrt(diffusion_coefficient), self.num_steps))
            trajectories.append(np.vstack((x, y)).T)
        #return np.array(trajectories)
        return trajectories
    
    def levy_flight(self, alpha=1.5, beta=0):
        trajectories = []
        for _ in range(self.num_trajectories):
            #steps = levy_stable.rvs(alpha, beta, size=self.num_steps)
            #trajectory = np.cumsum(steps)
            #trajectories.append(trajectory)
            x = np.cumsum(levy_stable.rvs(alpha, beta, size=self.num_steps))
            y = np.cumsum(levy_stable.rvs(alpha, beta, size=self.num_steps))
            trajectories.append(np.vstack((x, y)).T)
        return trajectories
    
    def spiral_trajectory(self, r=1.0, omega=0.1):
        trajectories = []
        # Linear time steps with noise
        t_base = np.linspace(0, 10, self.num_steps)
        noise = np.random.normal(0, 0.1, self.num_steps)  # Adjust 0.1 to change noise level
        t = np.sort(t_base + noise)
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

