import torch
from torch.special import erf

def print_bound(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"TV bound at propag step: {result[0]}")
        return result
    return wrapper

# ----------------------------------------------------------------------------------------- #
# ------------------- TV upper bound (using the maximization approach) -------------------- #
# ----------------------------------------------------------------------------------------- #

@print_bound
def compute_upper_bound_for_TV(dynamics, noise_distribution, signatures, probs, regions):

    # Preprocessing to vectorize operations
    bounded_regions = regions[:-1]
    bounded_signatures = signatures[:-1]

    max_values = dynamics.compute_max_s(noise_distribution, bounded_regions, bounded_signatures)

    # Compute contributions and TV value
    contributions = max_values * probs
    tv = torch.sum(contributions)

    return tv, contributions