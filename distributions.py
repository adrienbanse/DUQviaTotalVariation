import torch
from torch.distributions import MultivariateNormal

class _Distributions:

    def __call__(self, *args, **kwargs):
        pass


class Gaussian(_Distributions):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.mean = mean
        self.covariance = cov

    def __call__(self, n_samples: int):
        mvn = MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance)
        return mvn.sample((n_samples,))


class GaussianMixture(_Distributions):
    def __init__(self, means: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor):
        self.means = means
        self.covariances = covs
        self.weights = weights

    def __call__(self, n_samples: int):

        chosen_components = torch.multinomial(self.weights, n_samples, replacement=True)

        gaussian_distributions = MultivariateNormal(self.means, self.covariances)
        samples = gaussian_distributions.sample((n_samples,))

        return samples[torch.arange(n_samples), chosen_components]