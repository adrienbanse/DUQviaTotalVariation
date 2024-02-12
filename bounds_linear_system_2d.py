import propagation_methods as propag

import numpy as np
from scipy import special

# ----------------------------------------------------------------------------------------- #
# ------------------------------ Auxiliary computing methods ------------------------------ #
# ----------------------------------------------------------------------------------------- #

Pi = np.pi

def exp(x):
    return np.exp(x)

def sqrt(x):
    return np.sqrt(x)

def erf(x):
    return special.erf(x)

def erfc(x):
    return special.erfc(x)

def log(x):
    return np.log(x)


# ----------------------------------------------------------------------------------------- #
# ----------------------------- Auxiliary maximization methods ---------------------------- #
# ----------------------------------------------------------------------------------------- #

def KLTerm(A, x, signature, cov_noise, i):
    return 1/cov_noise[i][i]*(np.matmul(A, x)[i] - np.matmul(A, signature)[i])**2

def getVertices(region):
    
    vertice1 = [region[0][0], region[1][0]]
    vertice2 = [region[0][0], region[1][1]]
    vertice3 = [region[0][1], region[1][0]]
    vertice4 = [region[0][1], region[1][1]]

    return [vertice1, vertice2, vertice3, vertice4]

def verifyIfRegionIsBounded(region):
    if np.isinf(region[0][0]) or np.isinf(region[0][1]) or np.isinf(region[1][0]) or np.isinf(region[1][1]):
        return False
    return True


def maxValueInsideRegion(A, signature, region, cov_noise):

    max_value_0, max_value_1 = 0, 0

    vertices = getVertices(region)

    #print(signature)
    #print(vertices)
    
    for vertice in vertices:
        max_value_0 = max(max_value_0, KLTerm(A, vertice, signature, cov_noise, 0))
        max_value_1 = max(max_value_1, KLTerm(A, vertice, signature, cov_noise, 1))

    return max_value_0 + max_value_1


# ----------------------------------------------------------------------------------------- #
# -------------------------- Measure of Sqrt[KL divergence] - ALTERNATIVE ------------------------------- #
# ----------------------------------------------------------------------------------------- #

def computeTV_withMax(A, signatures, regions, gmm, cov_noise):

    #prob_check = 0

    tv = 0
    for signature, region in zip(signatures, regions):
        
        max_value = 0
        prob_gmm_region = 0

        if verifyIfRegionIsBounded(region):
            max_value = erf(maxValueInsideRegion(A, signature, region, cov_noise)**0.5)/(2*np.sqrt(2))
        else:
            max_value = 1

        
        #print(max_value)
        for (weight, gmm_mean) in zip(gmm[0], gmm[1]):
            prob_gmm_region += weight*propag.gaussianProbaMassInsideHypercube(gmm_mean, gmm[2], region)

        #prob_check += prob_gmm_region

        tv += max_value*prob_gmm_region


    #print(prob_check)

    return tv






# ----------------------------------------------------------------------------------------- #
# -------------------------- Measure of Sqrt[KL divergence] ------------------------------- #
# ----------------------------------------------------------------------------------------- #

def computeFoldedMomentum(sigCompon, meanX, covX, region):

    factor_constant = 0.5*sqrt(Pi/2)*sqrt(covX[0][0]*covX[1][1])

    factor_exponential = 2*sqrt(covX[0][0])*(np.exp(-0.5*(region[0][1] - meanX[0])**2/covX[0][0]) - 2*np.exp(-0.5*(sigCompon - meanX[0])**2/covX[0][0]) + np.exp(-0.5*(meanX[0] - region[0][0])**2/covX[0][0]))

    factor_erf = sqrt(2*Pi)*(sigCompon - meanX[0])*(erf((region[0][1] - meanX[0])/(sqrt(2*covX[0][0]))) + erf((region[0][0] - meanX[0])/(sqrt(2*covX[0][0]))) - 2*erf((sigCompon - meanX[0])/(sqrt(2*covX[0][0]))))

    factor_x2 = erf((region[1][0] - meanX[1])/(sqrt(2*covX[1][1]))) + erf((meanX[1] - region[1][1])/(sqrt(2*covX[1][1])))
    
    return factor_constant*(factor_exponential + factor_erf)*factor_x2


def computeMeasureSqrtKLWrtGMMComponent(A, cov_noise, signature, meanX, covX, region):

    factor_constant_x1 = abs(A[0][0])/sqrt(cov_noise[0][0]) + abs(A[1][0])/sqrt(cov_noise[1][1])
    factor_integral_x1 = computeFoldedMomentum(signature[0], meanX, covX, region)

    #Use symmetry
    inverted_meanX = [meanX[1], meanX[0]]
    inverted_covX = [[covX[1][1], 0.0],
                     [0.0, covX[0][0]]]
    inverted_region = [region[1], region[0]]
    
    factor_constant_x2 = abs(A[0][1])/sqrt(cov_noise[0][0]) + abs(A[1][1])/sqrt(cov_noise[1][1])
    factor_integral_x2 = computeFoldedMomentum(signature[1], inverted_meanX, inverted_covX, inverted_region)

    result = factor_constant_x1*factor_integral_x1 + factor_constant_x2*factor_integral_x2

    return result/sqrt(2)


def computeMeasureSqrtKLWrtGMM(A, signature, region, cov_noise, gmm):

    measure = 0

    weights = gmm[0]
    gmm_means = gmm[1]
    gmm_cov = gmm[2]

    for (weight, mean_x) in zip(weights, gmm_means):

        measure += weight*computeMeasureSqrtKLWrtGMMComponent(A, cov_noise, signature, mean_x, gmm_cov, region)
    
    return measure


# ----------------------------------------------------------------------------------------- #
# --------------------------------- Total Variation bound --------------------------------- #
# ----------------------------------------------------------------------------------------- #

def computeTVbound(A, signatures, regions, cov_noise, gmm):

    tv_bound = 0

    for k, region in enumerate(regions):
    
        sqrtKL_term_region = computeMeasureSqrtKLWrtGMM(A, signatures[k], region, cov_noise, gmm)
        tv_bound += sqrtKL_term_region

    return sqrt(2)*tv_bound