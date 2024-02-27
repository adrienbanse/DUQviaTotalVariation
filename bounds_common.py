import probability_mass_computation as proba
import bounds_linear_system_2d as bounds_linear
import bounds_polynomial_system_2d as bounds_polynomial

import numpy as np
from scipy import special

# ----------------------------------------------------------------------------------------- #
# ------------------------------ Auxiliary computing methods ------------------------------ #
# ----------------------------------------------------------------------------------------- #

def sqrt(x):
    return np.sqrt(x)

def erf(x):
    return special.erf(x)


# ----------------------------------------------------------------------------------------- #
# ----------------------------- Auxiliary maximization methods ---------------------------- #
# ----------------------------------------------------------------------------------------- #


def verifyIfRegionIsBounded(region):
    if np.isinf(region[0][0]) or np.isinf(region[0][1]) or np.isinf(region[1][0]) or np.isinf(region[1][1]):
        return False
    return True


# ----------------------------------------------------------------------------------------- #
# -------------------------- Measure of Sqrt[KL divergence] - ALTERNATIVE ------------------------------- #
# ----------------------------------------------------------------------------------------- #

def computeUpperBoundForTV(signatures, regions, gmm, cov_noise, method, params):

    tv = 0
    for signature, region in zip(signatures, regions):
        
        max_value = 0
        prob_gmm_region = 0

        if verifyIfRegionIsBounded(region):
            if method == 'linear':
                A = params[0]
                max_value = erf(bounds_linear.maxValueInsideRegion(A, signature, region, cov_noise)/(2*sqrt(2)))

            elif method == 'polynomial':
                h = params[0]
                max_value = erf(bounds_polynomial.maxValueInsideRegion(h, signature, region, cov_noise)/(2*sqrt(2)))
        else:
            max_value = 1

        
        prob_gmm_region = proba.gmmProbaMassInsideHypercube(gmm, region)

        tv += max_value*prob_gmm_region

    return tv



def computeUpperBoundForTVWithMax(signatures, regions, probas, cov_noise, method, params):


    tv = 0

    for cont, (signature, region) in enumerate(zip(signatures, regions)):
        
        max_value = 0
        prob_gmm_region = 0

        if verifyIfRegionIsBounded(region):
            if method == 'linear':
                A = params[0]
                max_value = erf(bounds_linear.maxValueInsideRegion(A, signature, region, cov_noise)/(2*sqrt(2)))

            elif method == 'polynomial':
                h = params[0]
                max_value = erf(bounds_polynomial.maxValueInsideRegion(h, signature, region, cov_noise)/(2*sqrt(2)))
        else:
            max_value = 1

        
        prob_gmm_region = probas[cont]

        tv += max_value*prob_gmm_region

    return tv