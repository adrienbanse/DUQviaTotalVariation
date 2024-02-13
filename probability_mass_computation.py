import numpy as np
from scipy import special


# ----------------------------------------------------------------------------------------- #
# ------------------------ Gaussian probability mass inside hypercube --------------------- #
# ----------------------------------------------------------------------------------------- #

def erf_factor(mean, variance, region_min, region_max):
    return special.erf((mean - region_min)/(np.sqrt(2*variance))) - special.erf((mean - region_max)/(np.sqrt(2*variance)))


def gaussianProbaMassInsideHypercube(mean, cov, cube):
    #Assuming cube = [[x_min, x_max], [y_min, y_max]]

    factor = 1/4
    x_erf = erf_factor(mean[0], cov[0][0], cube[0][0], cube[0][1])
    y_erf = erf_factor(mean[1], cov[1][1], cube[1][0], cube[1][1])

    return factor*x_erf*y_erf

def computeSignatureProbabilities(regions, means, cov, weights):
    #Assumes a GMM where the components are described by means and covs(initial step becomes a GMM w/ 1 component)
    #RECALL THAT THE COVARIANCE IS THE SAME FOR ALL CONDITIONAL DIST IN THIS CASE
    #This can be more generic for other cases 
    
    signature_probas = []

    for region in regions:
        
        proba_region = 0
        
        for (mean, weight) in zip(means, weights):
            proba_region += weight*gaussianProbaMassInsideHypercube(mean, cov, region)
        
        signature_probas.append(proba_region)
    
    return signature_probas