import torch
import parameters
import monte_carlo
import tv_bound_algorithm as algorithm

from dynamics import DubinsDynamics
from distributions import GaussianMixture, Gaussian


# Dynamics parameters

# Initial distribution parameters
initial_weights = torch.Tensor([1.0, 0.0])
initial_means = torch.Tensor([[1.0, 1.0, 0.0], [0, 0, 0]])
sigma = 0.002
cov = sigma * torch.eye(3)
initial_covariances = torch.stack((cov, cov), dim=0)

# Noise
mean_noise = torch.Tensor([0, 0, 0])
sigma_noise = 0.1
cov_noise = sigma_noise * torch.eye(3)  # Assumes uncorrelation (this could be relaxed in further upgrades)

# Barrier (unsafe set)
barrier = torch.Tensor([[3.5, 2.0, -torch.inf], [4.5, 3.0, torch.inf]])

if __name__ == "__main__":

    torch.manual_seed(0) # for reproducibility

    f = DubinsDynamics(h = 0.3, v = 0.1, u = 1/0.5)

    initial_distribution = GaussianMixture(initial_means, initial_covariances, initial_weights)
    noise_distribution = Gaussian(mean_noise, cov_noise)

    monte_carlo.monte_carlo_simulation(f, initial_distribution, noise_distribution, barrier, parameters.n_steps_ahead, parameters.n_samples)

    tv_bounds_time_step, gmms = algorithm.tv_bound_algorithm(f, initial_distribution, noise_distribution, parameters.grid_type)
    tv_bounds = torch.cumsum(tv_bounds_time_step, dim=0)
    tv_bounds = torch.minimum(tv_bounds, torch.tensor(1))
    print(f"Final TV bounds: {tv_bounds}")

    gmm_hitting_probs = monte_carlo.gmm_approximation_monte_carlo(gmms, barrier, parameters.n_samples)
    print(f"GMM hitting probs: {gmm_hitting_probs}")




