import probability_mass_computation as proba
import bounds_linear_system_2d as bounds_linear
import bounds_polynomial_system_2d as bounds_polynomial
import grid_generation as grid

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
            if method == 'linear':
                A = params[0]
                max_value = erf(bounds_linear.maxValueInsideRegion(A, signature, region, var_noise)/(2*sqrt(2)))

            elif method == 'polynomial':
                h = params[0]
                max_value = erf(bounds_polynomial.maxValueInsideRegion(h, signature, region, var_noise)/(2*sqrt(2)))
        else:
            max_value = 1

        
        prob_gmm_region = probas[cont]

        contributions.append(max_value*prob_gmm_region)

        tv += max_value*prob_gmm_region

    return tv, contributions



# ----------------------------------------------------------------------------------------- #
# ------------------- TV upper bound (using the maximization approach) -------------------- #
# ----------------------------------------------------------------------------------------- #

def cholesky(A):
    return np.linalg.cholesky(A)

def choleskyTransposed(A):
    return np.linalg.cholesky(A).T

def computeTransformation(Sigma, A):
    return np.dot(A.T, np.dot(np.linalg.inv(Sigma), A))

def getMaxAbsoluteEigenvalue(M):
    
    eigenvalues = np.linalg.eigvals(M)
    return np.max(np.abs(eigenvalues))


def truncatedGaussianMomentZero(mean, stdev, inf, sup):
    
    normalized_sup = (sup - mean)/stdev
    normalized_inf = (inf - mean)/stdev

    return norm.cdf(normalized_sup) - norm.cdf(normalized_inf)

def truncatedGaussianMomentOne(mean, stdev, inf, sup):
    normalized_sup = (sup - mean)/stdev
    normalized_inf = (inf - mean)/stdev
    
    return stdev*(norm.pdf(normalized_inf) - norm.pdf(normalized_sup)) + mean*(norm.cdf(normalized_sup) - norm.cdf(normalized_inf))


def truncatedGaussianMomentTwo(mean, stdev, inf, sup):
    normalized_sup = (sup - mean)/stdev
    normalized_inf = (inf - mean)/stdev
    
    return (stdev**2)*(normalized_inf*norm.pdf(normalized_inf) - normalized_sup*norm.pdf(normalized_sup)) + (stdev**2 + mean**2)*(norm.cdf(normalized_sup) - norm.cdf(normalized_inf))


def euclideanDistance(point1, point2):
    return sqrt(np.sum(point1 - point2) ** 2)

def cubeSize(region):
    
    vertices = grid.getVertices(region)
    distance = euclideanDistance(vertices[0], vertices[-1])

    return distance


def probaUnivariateGaussian(mean, stdev, inf, sup):
    
    normalized_sup = (sup - mean)/stdev
    normalized_inf = (inf - mean)/stdev

    return norm.cdf(normalized_sup) - norm.cdf(normalized_inf)


def computeDistanceMomentWrtGmm(signature, region, gmm, p_hat_in_region):

    weights_gmm = gmm[0]
    means_gmm = gmm[1]
    var_gmm = gmm[2]

    result = 0

    for cont, (weight, mean_gmm) in enumerate(zip(weights_gmm, means_gmm)):

        if(region[0, 0] < region[1, 0] and region[0, 1] < region[1, 1]):

            dist_0 = TG(t(mean_gmm[0]), t(var_gmm[0]), t(region[0, 0]), t(region[1, 0]))
            prob_0 = probaUnivariateGaussian(mean_gmm[0], var_gmm[0]**0.5, region[0, 0], region[1, 0])
            variance_cond_0 = dist_0.variance
            expect_cond_0 = dist_0.mean
            int_square_0 = (variance_cond_0 + expect_cond_0**2 - 2*signature[0]*expect_cond_0 + signature[0]**2)*prob_0

            dist_1 = TG(t(mean_gmm[1]), t(var_gmm[1]), t(region[0, 1]), t(region[1, 1]))
            prob_1 = probaUnivariateGaussian(mean_gmm[1], var_gmm[1]**0.5, region[0, 1], region[1, 1])
            variance_cond_1 = dist_1.variance
            expect_cond_1 = dist_1.mean
            int_square_1 = (variance_cond_1 + expect_cond_1**2 - 2*signature[1]*expect_cond_1 + signature[1]**2)*prob_1

            result += weight * (int_square_0 * prob_1 + int_square_1 * prob_0)
        
        #else:
            #print('see region')

    return result



@print_bound
def computeUpperBoundForTVWithLinearApprox(A_upper, signatures, regions, gmm, var_noise, double_hat_proba):

    weights_gmm = gmm[0]

    tv_unbounded, tv_bounded = 0, 0

    cov_noise = np.diag(var_noise)
    norm_B = getMaxAbsoluteEigenvalue(choleskyTransposed(computeTransformation(cov_noise, A_upper)))

    print(norm_B)

    factor_ct = 1/(2*sqrt(2))

    for cont, (signature, region) in enumerate(zip(signatures, regions)):       

        if verifyIfRegionIsBounded(region):
            
            tv_bounded += computeDistanceMomentWrtGmm(signature, region, gmm, double_hat_proba[cont])

    
    
    tv_unbounded = double_hat_proba[-1]

    return erf(factor_ct * norm_B * sqrt(tv_bounded)) + tv_unbounded