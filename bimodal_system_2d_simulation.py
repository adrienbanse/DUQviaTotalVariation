import torch
import barriers as barriers
import grid_generation as grid
import bounds_common as bounds
import propagation_methods as propag
import probability_mass_computation as proba
import monte_carlo

from dynamics import LinearDynamics
from distributions import GaussianMixture, Gaussian

import numpy as np

import matplotlib.pyplot as plt


# Steps ahead for prediction
n_steps_ahead = 10

#Dynamics parameters
A = torch.Tensor(
        [
            [0.84, 0.1],
            [0.05, 0.72]
        ])

# Initial distribution parameters
initial_weights = torch.Tensor([0.5, 0.5])
initial_means = torch.Tensor([[8, 10], [6, 10]])
sigma = 0.1
cov = sigma * torch.eye(2)
initial_covariances = torch.stack((cov, cov), dim=0)

# Noise
mean_noise = torch.Tensor([0, 0])
sigma_noise = 0.03
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

    #hitting_probs_monte_carlo = monte_carlo.monte_carlo_simulation(f, initial_distribution, noise_distribution, barrier, 10, 10000)

    samples = initial_distribution(1000)

    macro = grid.identify_high_prob_region(samples)
    print(macro)
    print(macro.shape)

    print(grid.get_vertices(macro))

    print(samples.shape)

    regions = grid.subdivide_region(macro, samples, 0.01, 0.2)
    regions = torch.stack([torch.stack([r[0], r[1]]) for r in regions])
    print(regions)
    print(regions.shape)

    signatures = grid.place_signatures(regions)

    print(signatures)

    #compute_signature_probabilities_in_parallel(regions, means, cov, weights):

    p = proba.compute_signature_probabilities(regions, initial_means, cov_noise, initial_weights)
    print(p)
    print(p.sum())


    def draw_square(x_min, y_min, x_max, y_max):
        plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color='blue')


    def plot_regions(regions):
        for region in regions:
            x_min, y_min = region[0]
            x_max, y_max = region[1]
            draw_square(x_min, y_min, x_max, y_max)
        plt.xlabel('State[0]')
        plt.ylabel('State[1]')
        plt.grid(True)
        plt.axis('equal')  # Set equal aspect ratio
        plt.show()

    plot_regions(regions)
