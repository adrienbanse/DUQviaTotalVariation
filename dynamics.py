import torch
import math
import torch.linalg as linalg
import grid_generation as grid
from abc import ABC, abstractmethod

from distributions import _Distributions


class _Dynamics(ABC):

    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_hypercube_envelopes(self, regions: torch.Tensor):
        pass

    def compute_envelopes_transform(self, distribution: _Distributions, signatures: torch.Tensor, envelopes: torch.Tensor):

        cov = distribution.covariance
        inv_cov = linalg.inv(cov)
        L = linalg.cholesky(inv_cov)

        propagated_signatures = self(signatures)
        centered_envelopes = grid.get_centered_region(envelopes, propagated_signatures)

        return torch.matmul(centered_envelopes, L)

    def compute_h(self, envelopes: torch.Tensor):

        vertices = grid.get_vertices(envelopes)
        norms = torch.norm(vertices, dim=2)  # Calculate norms across the last dimension
        max_values = torch.max(norms, dim=1).values  # Get the maximum value across the vertices

        return max_values / (2 * math.sqrt(2))


class LinearDynamics(_Dynamics):
    def __init__(self, A: torch.Tensor):
        self.A = A

    def __call__(self, x: torch.Tensor):
        return torch.matmul(x, self.A.T)

    def compute_hypercube_envelopes(self, regions):

        vertices = grid.get_vertices(regions)

        #TODO: Is there a better way to compute this below?
        n, d = vertices.shape[0], vertices.shape[-1]
        propagated_vertices = torch.stack([self(vert) for vert in vertices.reshape(-1, d)]).reshape(n, 2**d, d)

        min_vals, _ = torch.min(propagated_vertices, dim=1)
        max_vals, _ = torch.max(propagated_vertices, dim=1)

        envelopes = torch.stack([min_vals, max_vals], dim=1)
        return envelopes


class PolynomialDynamics(_Dynamics):
    def __init__(self, h: float):
        self.h = h

    def __call__(self, x: torch.Tensor):
        component_1 = x[:, 0] + 1.25 * self.h * x[:, 1]
        component_2 = 1.4 * x[:, 1] + self.h * 0.3 * (0.25 * x[:, 0] ** 2 - 0.4 * x[:, 0] * x[:, 1] + 0.25 * x[:, 1] ** 2)

        return torch.stack((component_1, component_2), dim=1)

    def compute_hypercube_envelopes(self, regions):

        vertices = grid.get_vertices(regions)

        # TODO: Is there a better way to compute this below?
        n, d = vertices.shape[0], vertices.shape[-1]
        propagated_vertices = torch.stack([self(vert) for vert in vertices.reshape(-1, d)]).reshape(n, 2 ** d, d)

        min_vals, _ = torch.min(propagated_vertices, dim=1)
        max_vals, _ = torch.max(propagated_vertices, dim=1)

        envelopes = torch.stack([min_vals, max_vals], dim=1)
        return envelopes


class DubinsDynamics(_Dynamics):
    def __init__(self, h: float, v: float, u: float):
        self.h = h
        self.v = v
        self.u = u

    def __call__(self, x: torch.Tensor):
        component_1 = x[:, 0] + self.h * self.v * torch.sin(x[:, 2])
        component_2 = x[:, 1] + self.h * self.v * torch.cos(x[:, 2])
        component_3 = x[:, 2] + self.h * self.u

        return torch.stack((component_1, component_2, component_3), dim=1)

    def compute_hypercube_envelopes(self, regions):
        raise NotImplementedError("Subclasses of _Dynamics must implement this method!")
