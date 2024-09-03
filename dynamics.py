import torch
import grid_generation as grid

class _Dynamics:

    def __call__(self, *args, **kwargs):
        pass


class LinearDynamics(_Dynamics):
    def __init__(self, A: torch.Tensor):
        self.A = A

    def __call__(self, x: torch.Tensor):
        return torch.matmul(x, self.A.T)

    def hypercube_envelope(self, region):

        vertices = grid.get_vertices(region)

        propagated_vertices = self(vertices)

        min_vals, _ = torch.min(propagated_vertices, dim=0)
        max_vals, _ = torch.max(propagated_vertices, dim=0)

        result = torch.stack([min_vals, max_vals])

        return result
