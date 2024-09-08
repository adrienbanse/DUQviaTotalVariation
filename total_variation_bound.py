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

    tv = torch.tensor(0.0)
    contributions = []

    # Preprocessing to vectorize operations
    bounded_regions = regions[:-1]
    bounded_signatures = signatures[:-1]

    envelopes = dynamics.compute_hypercube_envelopes(bounded_regions)
    transformed_envelopes = dynamics.compute_envelopes_transform(noise_distribution, bounded_signatures, envelopes)

    h = dynamics.compute_h(transformed_envelopes)

    max_values = erf(h)

    max_values = torch.cat((max_values, torch.ones(1)), dim=0)

    # Compute contributions and TV value
    contributions = max_values * probs
    tv = torch.sum(contributions)

    return tv, contributions