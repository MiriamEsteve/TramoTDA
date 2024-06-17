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
