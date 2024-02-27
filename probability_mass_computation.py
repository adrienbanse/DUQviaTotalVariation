import numpy as np
import multiprocessing
from scipy import special
from functools import partial
 
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
 
def erf_factor(mean, variance, region_min, region_max):
    sqrt2var = np.sqrt(2 * variance)
    scaled_min = (mean - region_min) / sqrt2var
    scaled_max = (mean - region_max) / sqrt2var
    
    # Be careful! The difference between two erfs is not
    # numerically stable like this. But it is good for perf.
    return special.erf(scaled_min) - special.erf(scaled_max)
 
 
def gaussianProbaMassInsideHypercube(means, var, cube):
    # Assuming cube = [x_lower, x_upper] where each is an n-dimensional vector
    # Assume mean and var are n-dimensional vectors
 
    factor = 1/(2**means.shape[-1])
    #factor = 1/4
    # erf_factor works vectors (all the functions take vectors as input)
    # It also works if mean or var are scalar while min and max 
    # are vectors. See more about broadcast semantics here:
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    erfs = erf_factor(means, var, cube[0], cube[1])
 
    return factor * erfs.prod(axis=-1)
 
 
def gmmProbabilityMassInsideRegion(weights, means, var, region):
    # Assume means has size [m, n] where n is the dimensionality of
    # the region and m is the number of segments#
    # Assume weights has size [m]
    # Assume var has size [n] - could also have [m, n], the code is the same
    # Assume region is a tuple of ([n], [n])
    
    proba_segment = gaussianProbaMassInsideHypercube(means, var, region)
    
    proba_region = (weights[: np.newaxis] * proba_segment).sum()
    print(proba_region)
    
    return proba_region
 
 
@timer_func
def computeSignatureProbabilitiesInParallel(regions, means, cov, weights):
    #Assumes a GMM where the components are described by means and covs(initial step becomes a GMM w/ 1 component)
    #RECALL THAT THE COVARIANCE IS THE SAME FOR ALL CONDITIONAL DIST IN THIS CASE
    #This can be more generic for other cases
 
    partial_function = partial(gmmProbabilityMassInsideRegion, weights, means, cov)
 
    # multiprocessing pool object
    pool = multiprocessing.Pool()
 
    # pool object with number of element
    pool = multiprocessing.Pool(processes = 10)
 
    signature_probas = pool.map(partial_function, regions)
      
    return signature_probas


def computeSignatureProbabilities(regions, means, cov, weights):
    #Assumes a GMM where the components are described by means and covs(initial step becomes a GMM w/ 1 component)
    #RECALL THAT THE COVARIANCE IS THE SAME FOR ALL CONDITIONAL DIST IN THIS CASE
    #This can be more generic for other cases 
    
    signature_probas = []

    for region in regions:
        
        proba_segment = gaussianProbaMassInsideHypercube(means, cov, region)
    
        proba_region = (weights[: np.newaxis] * proba_segment).sum()
        
        signature_probas.append(proba_region)
    
    return signature_probas

