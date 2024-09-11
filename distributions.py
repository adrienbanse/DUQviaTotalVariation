import torch
from torch.distributions import MultivariateNormal
from abc import ABC, abstractmethod
import probability_mass_computation as proba

class _Distributions:

    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_regions_probabilities(self, regions: torch.Tensor):
        pass


class Gaussian(_Distributions):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.mean = mean
        self.covariance = cov

    def __call__(self, n_samples: int):
        mvn = MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance)
        return mvn.sample((n_samples,))

    def compute_regions_probabilities(self, regions):
        raise NotImplementedError("Not yet implemented")


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

    def compute_regions_probabilities(self, regions):
        signature_probas = proba.gaussian_mixture_proba_mass_inside_hypercubes(self.means, self.covariances[0], self.weights, regions)

        # the remaining probability is attributed to the whole unbounded region
        unbounded_proba = torch.Tensor([1 - signature_probas.sum()])
        signature_probas = torch.cat((signature_probas, unbounded_proba))

        return signature_probas

class Uniform(_Distributions):
    def __init__(self, center: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
        self.center = center
        self.low = low
        self.high = high

    def __call__(self, n_samples: int):
        uniform_dist = torch.distributions.Uniform(self.center + self.low, self.center + self.high)
        samples = uniform_dist.sample((n_samples,))

        return samples

    def compute_regions_probabilities(self, regions):
        raise NotImplementedError("Not yet implemented")

class UniformMixture(_Distributions):
    def __init__(self, centers: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor, weights: torch.Tensor):
        self.centers = centers
        self.lows = lows
        self.highs = highs
        self.weights = weights

    def __call__(self, n_samples: int):

        chosen_components = torch.multinomial(self.weights, n_samples, replacement=True)

        uniform_distributions = torch.distributions.Uniform(self.centers + self.lows, self.centers + self.highs)
        samples = uniform_distributions.sample((n_samples,))

        return samples[torch.arange(n_samples), chosen_components]

    def compute_regions_probabilities(self, regions):
        signature_probas = proba.uniform_mixture_proba_mass_inside_hypercubes(self.centers + self.lows, self.centers + self.highs, self.weights, regions)

        # the remaining probability is attributed to the whole unbounded region
        unbounded_proba = torch.Tensor([1 - signature_probas.sum()])
        signature_probas = torch.cat((signature_probas, unbounded_proba))

        return signature_probas