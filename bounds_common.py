import probability_mass_computation as proba
import bounds_linear_system_2d as bounds_linear
import bounds_polynomial_system_2d as bounds_polynomial
import bounds_dubin_3d as bounds_dubin
import grid_generation as grid
import propagation_methods as propag

import numpy as np
from scipy import special
from scipy.stats import norm

from stable_trunc_gaussian import TruncatedGaussian as TG
from torch import tensor as t



def print_bound(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"TV bound at propag step: {result[0]}")
        return result
    return wrapper

# ----------------------------------------------------------------------------------------- #
# ------------------------------ Auxiliary computing methods ------------------------------ #
# ----------------------------------------------------------------------------------------- #

Pi = np.pi

def sqrt(x):
    return np.sqrt(x)

def exp(x):
    return np.exp(x)

def erf(x):
    return special.erf(x)


# ----------------------------------------------------------------------------------------- #
# ----------------------------- Auxiliary maximization methods ---------------------------- #
# ----------------------------------------------------------------------------------------- #


def verifyIfRegionIsBounded(region):
    
    is_inf = np.logical_or(np.isinf(region[0]), np.isinf(region[1]))
    return not np.any(is_inf)



def findHyperCubicEnvelope(region, method, params):
 
    vertices = grid.getVertices(region)

    if method == 'linear':
 
        propagated_vertices = propag.systemDynamics(vertices, method, params)
 
        #Compute min and max for each component
        min_f1, min_f2 = np.min(propagated_vertices, axis = 0)
        max_f1, max_f2 = np.max(propagated_vertices, axis = 0)
 
        return np.array([[min_f1, min_f2], [max_f1, max_f2]])
 
    elif method == 'polynomial':
 
        propagated_vertices = propag.systemDynamics(vertices, method, params)
 
        #Compute min and max for each component
        min_f1, min_f2 = np.min(propagated_vertices, axis = 0)
        max_f1, max_f2 = np.max(propagated_vertices, axis = 0)
 
        if grid.checkIfPointIsInRegion(np.array([0, 0]), region):
            min_f2 = min(min_f2, 0)
 
        return np.array([[min_f1, min_f2], [max_f1, max_f2]])
    
    elif method == 'dubin':

        h = params[0]
        v = params[1]
        u = params[2]

        der_zero_max_sin = np.pi/2 + 2*np.pi*np.arange(-10, 10)
        der_zero_min_sin = np.pi/2 + np.pi*np.arange(-10, 10)

        der_zero_max_cos = 0 + 2*np.pi*np.arange(-10, 10)
        der_zero_min_cos = 0 + np.pi*np.arange(-10, 10)

        propagated_vertices = propag.systemDynamics(vertices, method, params)

        #Compute min and max for each component
        min_f1, min_f2, min_f3 = np.min(propagated_vertices, axis = 0)
        max_f1, max_f2, max_f3 = np.max(propagated_vertices, axis = 0)

        for v in der_zero_max_sin:
            if region[0, 2] <= v <= region[1, 2]:
                for vertice in vertices:
                    max_f1 = max(max_f1, vertice[0] + h*v*1)

        for v in der_zero_min_sin:
            if region[0, 2] <= v <= region[1, 2]:
                for vertice in vertices:
                    min_f1 = min(min_f1, vertice[0] + h*v*(-1))

        for v in der_zero_max_cos:
            if region[0, 2] <= v <= region[1, 2]:
                for vertice in vertices:
                    max_f2 = max(max_f2, vertice[1] + h*v*(1))

        for v in der_zero_min_cos:
            if region[0, 2] <= v <= region[1, 2]:
                for vertice in vertices:
                    min_f2 = min(min_f2, vertice[1] + h*v*(-1))

        return np.array([[min_f1, min_f2, min_f3], [max_f1, max_f2, max_f3]])


def transformEnvelope(envelope, var_noise, signature, method, params):

    cov = np.diag(var_noise)
    inv_cov = np.linalg.inv(cov)
    L = np.linalg.cholesky(inv_cov)

    components = propag.systemDynamics(np.array([signature]), method, params)

    centered_envelope = grid.getCenteredRegion(envelope, components)

    return np.dot(centered_envelope, L)


def computeMax(envelope):
    vertices = grid.getVertices(envelope)
    norms = np.linalg.norm(vertices, axis=1)
    max_value = np.max(norms)
    return max_value


# ----------------------------------------------------------------------------------------- #
# ------------------- TV upper bound (using the maximization approach) -------------------- #
# ----------------------------------------------------------------------------------------- #

@print_bound
def computeUpperBoundForTVWithMax(signatures, regions, probas, var_noise, method, params):

    tv = 0

    contributions = []

    for cont, (signature, region) in enumerate(zip(signatures, regions)):
        
        max_value = 0
        prob_gmm_region = 0

        if verifyIfRegionIsBounded(region):
            envelope = findHyperCubicEnvelope(region, method, params)
            envelope = transformEnvelope(envelope, var_noise, signature, method, params)

            max_value = erf(computeMax(envelope)/(2*sqrt(2)))

        # if verifyIfRegionIsBounded(region):
        #     if method == 'linear':
        #         A = params[0]
        #         max_value = erf(bounds_linear.maxValueInsideRegion(A, signature, region, var_noise)/(2*sqrt(2)))

        #     elif method == 'polynomial':
        #         h = params[0]
        #         #max_value = erf(bounds_polynomial.maxValueInsideRegion(h, signature, region, var_noise)/(2*sqrt(2)))

        #     elif method == 'dubin':
        #         h = params[0]
        #         v = params[1]
        #         u = params[2]
        #         #max_value = erf(bounds_dubin.maxValueInsideRegion(h, v, u, signature, region, var_noise)/(2*sqrt(2)))
        else:
            max_value = 1

        
        prob_gmm_region = probas[cont]

        contributions.append(max_value*prob_gmm_region)

        tv += max_value*prob_gmm_region

    return tv, contributions