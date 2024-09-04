import torch
import torch.linalg as linalg
from torch.special import erf
import math

import grid_generation as grid

def print_bound(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"TV bound at propag step: {result[0]}")
        return result
    return wrapper


def check_if_region_is_bounded(region):
    is_inf = torch.logical_or(torch.isinf(region[0]), torch.isinf(region[1]))
    return not torch.any(is_inf)

# ----------------------------------------------------------------------------------------- #
# ------------------- TV upper bound (using the maximization approach) -------------------- #
# ----------------------------------------------------------------------------------------- #

@print_bound
def compute_upper_bound_for_TV(dynamics, noise_distribution, signatures, probs, regions):

    tv = torch.tensor(0.0)
    contributions = []

    # Preprocessing to vectorize operations
    bounded_regions = regions[:-1]
    bounded_signatures = signatures[:-1]

    envelopes = dynamics.compute_hypercube_envelopes(bounded_regions)


    transformed_envelopes = dynamics.compute_envelope_transform(noise_distribution, bounded_signatures, envelopes)


    h = dynamics.compute_h(transformed_envelopes)

    max_values = erf(h)

    # Initialize max_values with 1 for unbounded regions
    max_values = torch.stack((max_values, torch.ones(1)), dim=1)
    max_values = max_values[0]

    # Compute contributions and TV value
    #contributions = max_values * probs
    contributions = torch.dot(max_values, probs)
    tv = torch.sum(contributions)

    return tv, contributions.tolist()