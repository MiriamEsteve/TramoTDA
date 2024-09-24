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
        """Generate multiple 2D Brownian motion trajectories."""
        trajectories = []
        for _ in range(self.num_trajectories):
            x = np.cumsum(np.random.normal(0, np.sqrt(diffusion_coefficient), self.num_steps))
            y = np.cumsum(np.random.normal(0, np.sqrt(diffusion_coefficient), self.num_steps))
            trajectories.append(np.vstack((x, y)).T)
        return trajectories

    def levy_flight(self, alpha=1.5, beta=0):
        """Generate multiple 2D Lévy flight trajectories."""
        trajectories = []
        for _ in range(self.num_trajectories):
            x = np.cumsum(levy_stable.rvs(alpha, beta, size=self.num_steps))
            y = np.cumsum(levy_stable.rvs(alpha, beta, size=self.num_steps))
            trajectories.append(np.vstack((x, y)).T)
        return trajectories
    
    def spiral_trajectory(self, r=1.0, omega=0.1, noise_level=0.05):
        """Generate spiral trajectories with added dispersion."""
        trajectories = []
        t_base = np.linspace(0, 10, self.num_steps)
        noise_t = np.random.normal(0, 0.1, self.num_steps)  # Noise in time steps
        t = np.sort(t_base + noise_t)

        for _ in range(self.num_trajectories):
            x = r * t * np.cos(omega * t) + np.random.normal(0, noise_level, self.num_steps)
            y = r * t * np.sin(omega * t) + np.random.normal(0, noise_level, self.num_steps)
            trajectories.append(np.vstack((x, y)).T)
        return np.array(trajectories)
    
    def circular_trajectory(self, R=1.0, omega=0.1, noise_level=0.05):
        """Generate circular trajectories with added dispersion."""
        trajectories = []
        t = np.linspace(0, 10, self.num_steps)
        
        for _ in range(self.num_trajectories):
            x = R * np.cos(omega * t) + np.random.normal(0, noise_level, self.num_steps)
            y = R * np.sin(omega * t) + np.random.normal(0, noise_level, self.num_steps)
            trajectories.append(np.vstack((x, y)).T)
        return np.array(trajectories)
