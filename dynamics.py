import torch
import math
import grid_generation as grid
import torch.linalg as linalg

from distributions import _Distributions


class _Dynamics:

    def __call__(self, *args, **kwargs):
        pass

    def compute_hypercube_envelopes(self, regions):

        vertices = grid.get_vertices(regions)
        propagated_vertices = self(vertices)

        min_vals, _ = torch.min(propagated_vertices, dim=1)
        max_vals, _ = torch.max(propagated_vertices, dim=1)

        # Stack the min and max values along a new dimension
        return torch.stack([min_vals, max_vals], dim=1)

    def compute_envelope_transform(self, distribution: _Distributions, signatures: torch.Tensor, envelopes: torch.Tensor):

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
