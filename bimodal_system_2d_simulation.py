import torch
import barriers as barriers
import grid_generation as grid
import bounds_common as bounds
import propagation_methods as propag
import probability_mass_computation as proba
import total_variation_bound as tv
import monte_carlo

import tv_bound_algorithm as algorithm

from dynamics import LinearDynamics
from distributions import GaussianMixture, Gaussian

import numpy as np

import matplotlib.pyplot as plt


# Steps ahead for prediction
n_steps_ahead = 3

#Dynamics parameters
A = torch.Tensor(
        [
            [0.84, 0.1],
            [0.05, 0.72]
        ])

# Initial distribution parameters
initial_weights = torch.Tensor([0.5, 0.5])
initial_means = torch.Tensor([[8, 10], [6, 10]])
sigma = 0.005
cov = sigma * torch.eye(2)
initial_covariances = torch.stack((cov, cov), dim=0)

# Noise
mean_noise = torch.Tensor([0, 0])
sigma_noise = 0.01
cov_noise = sigma_noise * torch.eye(2)  # Assumes uncorrelation (this could be relaxed in further upgrades)

# Unbounded region (arbitrary definition)
unbounded_region = torch.Tensor([[torch.inf, torch.inf], [torch.inf, torch.inf]])  # a representation choice for the unbounded region

# Barrier (unsafe set)
barrier = torch.Tensor([[3.5, 2.0], [4.5, 3.0]])

if __name__ == "__main__":

    torch.manual_seed(0) # for reproducibility

    f = LinearDynamics(A)

    initial_distribution = GaussianMixture(initial_means, initial_covariances, initial_weights)
    noise_distribution = Gaussian(mean_noise, cov_noise)

    monte_carlo.monte_carlo_simulation(f, initial_distribution, noise_distribution, barrier, 5, 10000)

    #algorithm.tv_bound_algorithm(f, initial_distribution, noise_distribution, barrier)




