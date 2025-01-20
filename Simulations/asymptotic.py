import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dynamics import LinearDynamics
from distributions import Gaussian, GaussianMixture
from copy import deepcopy
import grid_generation as grid
import matplotlib.pyplot as plt

A = torch.Tensor(
    [
        [0.84, 0.10],
        [0.05, 0.72]
    ])
n_samples = int(10e5)

# Initial distribution
mean_initial = torch.Tensor([0, 0])
sigma_initial = 0.005
cov_initial = sigma_initial * torch.eye(2)

# Noise distribution
mean_noise = torch.Tensor([0, 0])
sigma_noise = 0.03
cov_noise = sigma_noise * torch.eye(2)

# TODO: To include in the codesource maybe
def generate_uniform_partitions(n):
    step = 2 / n  # Step size for partitioning along each axis
    regions = []
    for i in range(n):
        for j in range(n):
            # Calculate corners
            xmin, ymin = -1 + i * step, -1 + j * step
            xmax, ymax = -1 + (i + 1) * step, -1 + (j + 1) * step
            regions.append([[xmin, ymin], [xmax, ymax]])

    return torch.tensor(regions)

# Generates 2^n u
n = 3
regions = generate_uniform_partitions(n)

if __name__ == "__main__":
    torch.manual_seed(0) 

    # Prepare fixed grid
    signatures = grid.place_signatures(regions)
    unbounded_region = torch.full((1, 2, 2), torch.inf)
    regions = torch.cat((regions, unbounded_region))
    outer_point = grid.outer_point(torch.tensor([[-1, -1], [1, 1]]))
    signatures = torch.cat((signatures, outer_point.unsqueeze(0)))

    # Dynamics definition
    f = LinearDynamics(A)
    initial_distribution = GaussianMixture(mean_initial.unsqueeze(0), cov_initial.unsqueeze(0), torch.Tensor([1.]))
    noise_distribution = Gaussian(mean_noise, cov_noise)
    
    # Initilizations
    current_distribution = deepcopy(initial_distribution)
    approximation_distribution = deepcopy(initial_distribution)
    means_gmm = f(signatures)
    tv_approx_list = []

    for t in range(1, 50):
        # Update true current
        cov_current = torch.matmul(torch.t(A), current_distribution.covariances[0, :, :])
        cov_current = torch.matmul(cov_current, A) + noise_distribution.covariance
        mean_current = torch.matmul(A, current_distribution.means[0, :]) + noise_distribution.mean
        current_distribution = GaussianMixture(mean_current.unsqueeze(0), cov_current.unsqueeze(0), torch.Tensor([1]))

        # Update approx
        double_hat_probs = approximation_distribution.compute_regions_probabilities(regions[:-1])
        covs_noise = noise_distribution.covariance.unsqueeze(0).expand(means_gmm.size(0), -1, -1)
        approximation_distribution = GaussianMixture(means_gmm, covs_noise, double_hat_probs)

        mc_samples = current_distribution(n_samples)
        tv_approx = 0.5 * torch.mean(torch.abs(
            1 - approximation_distribution.density(mc_samples) / current_distribution.density(mc_samples)
        ))
        tv_approx_list.append(tv_approx)

    plt.plot(tv_approx_list)
    plt.show()


