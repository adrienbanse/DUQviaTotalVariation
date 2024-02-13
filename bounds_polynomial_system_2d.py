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

def KLTerm(x, signature, cov_noise, i):

    if i==0:
        return 1/(cov_noise[i][i]**0.5)*abs(x[0] + 0.1*x[1] - (signature[0] + 0.1*signature[1]))
    elif i==1:
        return 1/(cov_noise[i][i]**0.5)*abs(x[1] + 0.1*(1/3*x[0]**3 - x[0] - x[1]) - (signature[1] + 0.1*(1/3*signature[0]**3 - signature[0] - signature[1])))
    
    return 0

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


def maxValueInsideRegion(signature, region, cov_noise):

    max_value_0, max_value_1 = 0, 0

    vertices = getVertices(region)

    if region[0][0] < 1 and 1 < region[0][1]:
        vertices.append([1, region[1][0]])
        vertices.append([1, region[1][1]])
    elif region[0][0] < -1 and -1 < region[0][1]:
        vertices.append([-1, region[1][0]])
        vertices.append([-1, region[1][1]])

    #print(signature)
    #print(vertices)
    
    for vertice in vertices:
        max_value_0 = max(max_value_0, KLTerm(vertice, signature, cov_noise, 0))
        max_value_1 = max(max_value_1, KLTerm(vertice, signature, cov_noise, 1))

    return max_value_0 + max_value_1


# ----------------------------------------------------------------------------------------- #
# -------------------------- Measure of Sqrt[KL divergence] - ALTERNATIVE ------------------------------- #
# ----------------------------------------------------------------------------------------- #

def computeTV_withMax(signatures, regions, gmm, cov_noise):

    #prob_check = 0

    tv = 0
    for signature, region in zip(signatures, regions):
        
        max_value = 0
        prob_gmm_region = 0

        if verifyIfRegionIsBounded(region):
            max_value = erf(maxValueInsideRegion(signature, region, cov_noise))/(2*np.sqrt(2))
        else:
            max_value = 1

        
        #print(max_value)
        for (weight, gmm_mean) in zip(gmm[0], gmm[1]):
            prob_gmm_region += weight*propag.gaussianProbaMassInsideHypercube(gmm_mean, gmm[2], region)

        #prob_check += prob_gmm_region

        tv += max_value*prob_gmm_region


    #print(prob_check)

    return tv