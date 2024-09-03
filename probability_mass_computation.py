import numpy as np
import multiprocessing
from scipy import special
from functools import partial
import torch
 
from time import time
def timer_func(func):
    # This function shows the execution time of  
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
 
 
# ----------------------------------------------------------------------------------------- #
# ------------------------ Gaussian probability mass inside hypercube --------------------- #
# ----------------------------------------------------------------------------------------- #
 
def erf_factor(mean: torch.Tensor, variance: torch.Tensor, region_min: torch.Tensor, region_max: torch.Tensor):
    # Compute the square root of 2 * variance
    sqrt2var = torch.sqrt(2 * variance)
    # Ensure that mean, region_min, and region_max are broadcasted correctly
    # mean shape: (num_components, dimension)
    # variance shape: (num_components, dimension)
    # region_min/max shape: (num_regions, dimension)

    # Expand dimensions for broadcasting
    # mean shape will become: (1, num_components, dimension) -> (num_regions, num_components, dimension)
    mean_expanded = mean.unsqueeze(0)  # shape: (1, num_components, dimension)
    variance_expanded = variance.unsqueeze(0)  # shape: (1, num_components, dimension)

    # region_min/max shape will become: (num_regions, 1, dimension) -> (num_regions, num_components, dimension)
    region_min_expanded = region_min.unsqueeze(1)  # shape: (num_regions, 1, dimension)
    region_max_expanded = region_max.unsqueeze(1)  # shape: (num_regions, 1, dimension)

    # Broadcast and compute scaled_min and scaled_max
    scaled_min = (mean_expanded - region_min_expanded) / torch.sqrt(2 * variance_expanded)
    scaled_max = (mean_expanded - region_max_expanded) / torch.sqrt(2 * variance_expanded)

    return torch.special.erf(scaled_min) - torch.special.erf(scaled_max)
 
 
def gaussian_proba_mass_inside_hypercube(means, var, regions):

    factor = 1/(2 ** means.shape[-1])

    region_min = regions[:, 0, :]  # Extracting min bounds for all regions
    region_max = regions[:, 1, :]  # Extracting max bounds for all regions

    erfs = erf_factor(means, var, region_min, region_max)
 
    return factor * erfs.prod(dim=-1)
 
 
def gaussian_mixture_proba_mass_inside_hypercube(weights, means, var, regions):

    proba_segment = gaussian_proba_mass_inside_hypercube(means, var, regions)
    proba_region = (weights * proba_segment).sum(dim=-1)
    
    return proba_region
 
 
@timer_func
def compute_signature_probabilities(regions, means, cov, weights):

    signature_probas = gaussian_mixture_proba_mass_inside_hypercube(weights, means, cov, regions)

    signature_probas = torch.Tensor(signature_probas)

    # the remaining probability is attributed to the whole unbounded region
    unbounded_proba = torch.Tensor([1 - signature_probas.sum()])
    signature_probas = torch.cat((signature_probas, unbounded_proba))

    return signature_probas


