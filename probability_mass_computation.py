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
 
def erf_factor(means: torch.Tensor, covs: torch.Tensor, regions_min: torch.Tensor, regions_max: torch.Tensor):

    means_broadcast = means.unsqueeze(0)  # Shape: (1, num_components, dimension)

    variances = covs.diagonal(dim1=-2, dim2=-1)  # Take diagonal values of covariance
    variances_broadcast = variances.unsqueeze(0)  # shape: (1, num_components, dimension)

    regions_min_broadcast = regions_min.unsqueeze(1)  # Shape: (num_regions, 1, dimension)
    regions_max_broadcast = regions_max.unsqueeze(1)  # Shape: (num_regions, 1, dimension)

    # Broadcast and compute scaled_min and scaled_max
    scaled_mins = (means_broadcast - regions_min_broadcast) / torch.sqrt(2 * variances_broadcast)
    scaled_maxs = (means_broadcast - regions_max_broadcast) / torch.sqrt(2 * variances_broadcast)

    return torch.special.erf(scaled_mins) - torch.special.erf(scaled_maxs)
 
 
def gaussian_proba_mass_inside_hypercubes(means, covs, regions):

    factor = 1/(2 ** means.shape[-1])

    regions_min = regions[:, 0, :]  # Extracting min bounds for all regions
    regions_max = regions[:, 1, :]  # Extracting max bounds for all regions

    erfs = erf_factor(means, covs, regions_min, regions_max)
 
    return factor * erfs.prod(dim=-1)
 
 
def gaussian_mixture_proba_mass_inside_hypercubes(means, cov, weights, regions):

    covs = cov.unsqueeze(0).expand(means.size(0), -1, -1)

    proba_segment = gaussian_proba_mass_inside_hypercubes(means, covs, regions)
    proba_region = (weights * proba_segment).sum(dim=-1)
    
    return proba_region
 
 
@timer_func
def compute_signature_probabilities(means, cov, weights, regions):

    signature_probas = gaussian_mixture_proba_mass_inside_hypercubes(means, cov, weights, regions)

    # the remaining probability is attributed to the whole unbounded region
    unbounded_proba = torch.Tensor([1 - signature_probas.sum()])
    signature_probas = torch.cat((signature_probas, unbounded_proba))

    return signature_probas


