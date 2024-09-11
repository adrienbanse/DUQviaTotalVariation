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
# ----------------------------------- Gaussian Mixtures ----------------------------------- #
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


# ----------------------------------------------------------------------------------------- #
# ----------------------------------- Uniform Mixtures ------------------------------------ #
# ----------------------------------------------------------------------------------------- #

def compute_intersection(lows_extremities, highs_extremities, regions):

    # Extract the min and max bounds for both sets of regions
    min_n, max_n = regions[:, 0, :], regions[:, 1, :]
    min_m, max_m = lows_extremities, highs_extremities

    # Compute the intersection bounds
    intersection_min = torch.max(min_n[:, None, :], min_m[None, :, :])
    intersection_max = torch.min(max_n[:, None, :], max_m[None, :, :])

    intersection_length = torch.clamp(intersection_max - intersection_min, min=0)
    intersection_volume = intersection_length.prod(dim=-1)

    return intersection_volume


def uniform_proba_mass_inside_hypercubes(lows_extremities, highs_extremities, regions):

    support_size = highs_extremities - lows_extremities
    area = support_size.prod(dim=-1)
    height = 1 / area

    intersection_volume = compute_intersection(lows_extremities, highs_extremities, regions)

    return intersection_volume * height

def uniform_mixture_proba_mass_inside_hypercubes(lows_extremities, highs_extremities, weights, regions):

    proba_segment = uniform_proba_mass_inside_hypercubes(lows_extremities, highs_extremities, regions)
    proba_region = (weights * proba_segment).sum(dim=-1)

    return proba_region



