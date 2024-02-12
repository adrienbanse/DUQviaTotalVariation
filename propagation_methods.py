import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy import special


def getInitialState(mean, cov):
    
    multivariate_gaussian = multivariate_normal(mean = mean, cov = cov)
    initial_state_samples = multivariate_gaussian.rvs(size = 1)

    return np.array(initial_state_samples)


def systemDynamics(A, x):

    return np.matmul(A, x)


def systemNoise(mean, cov):
    
    multivariate_gaussian = multivariate_normal(mean = mean, cov = cov)
    noise_samples = multivariate_gaussian.rvs(size = 1)

    return np.array(noise_samples)


def stateEvolution(A, initial_state, mean_noise, cov_noise, n_steps_ahead):
    
    state = initial_state

    for t in range(n_steps_ahead):

        noise = systemNoise(mean_noise, cov_noise)
        
        state = systemDynamics(A, state) + noise
    
    return state


def monteCarloSimulationSystem(n_simulations, A, mean_initial_state, cov_initial_state, mean_noise, cov_noise, n_steps_ahead):
    
    initial_states, final_states = [], []
    
    for i in range(n_simulations):
    
        initial_state = getInitialState(mean_initial_state, cov_initial_state)
        final_state = stateEvolution(A, initial_state, mean_noise, cov_noise, n_steps_ahead)

        initial_states.append(initial_state)
        final_states.append(final_state)

    return initial_states, final_states


def erf_factor(mean, variance, region_min, region_max):
    return special.erf((mean - region_min)/(np.sqrt(2*variance))) - special.erf((mean - region_max)/(np.sqrt(2*variance)))


def gaussianProbaMassInsideHypercube(mean, cov, cube):
    #Assuming cube = [[x_min, x_max], [y_min, y_max]]

    factor = 1/4
    x_erf = erf_factor(mean[0], cov[0][0], cube[0][0], cube[0][1])
    y_erf = erf_factor(mean[1], cov[1][1], cube[1][0], cube[1][1])

    return factor*x_erf*y_erf


def generateIntervalsPerAxis(bounds, n_partitions):

    center = bounds[1] - bounds[0]
    sigma = center/6

    factor = 8

    lim0 = bounds[0]
    lim1 = bounds[0] + sigma
    lim2 = bounds[0] + 2*sigma
    lim3 = bounds[0] + 3*sigma
    lim4 = bounds[0] + 4*sigma
    lim5 = bounds[0] + 5*sigma
    lim6 = bounds[1]

    factor_high = 0.68
    factor_medium = 0.27/2

    n_partitions_center = math.floor(factor_high*n_partitions)
    n_partitions_medium_left = math.floor(factor_medium*n_partitions)
    n_partitions_medium_right = math.floor(factor_medium*n_partitions)

    partial_sum = n_partitions_center + n_partitions_medium_left + n_partitions_medium_right
    remaining_partitions = n_partitions - partial_sum

    if remaining_partitions <= 0:
        n_partitions_low_left, n_partitions_low_right = 0, 0
    else:
        if remaining_partitions % 2 == 0:
            n_partitions_low_left = int(remaining_partitions/2)
            n_partitions_low_right = int(remaining_partitions/2)
        else:
            n_partitions_center += 1
            remaining_partitions -= 1
            n_partitions_low_left = int(remaining_partitions/2)
            n_partitions_low_right = int(remaining_partitions/2)

    points = np.linspace(lim0, lim1, n_partitions_low_left)
    points = np.append(points, np.linspace(lim1, lim2, n_partitions_medium_left))
    points = np.append(points, np.linspace(lim2, lim4, n_partitions_center))
    points = np.append(points, np.linspace(lim4, lim5, n_partitions_medium_right))
    points = np.append(points, np.linspace(lim5, lim6, n_partitions_low_right))

    #points = np.linspace(bounds[0], bounds[1], n_partitions)
    points = np.append(points, -np.inf)
    points = np.append(points, np.inf)
    points = np.sort(points)

    return points


def createRegions(bounds_x, bounds_y, n_partitions):

    x_points = generateIntervalsPerAxis(bounds_x, n_partitions)
    y_points = generateIntervalsPerAxis(bounds_y, n_partitions)

    regions = []

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
                
            if i > 0 and j > 0:
                
                region = [[x_points[i-1], x_points[i]],
                        [y_points[j-1], y_points[j]]]
            
                regions.append(region)

    return regions


def computePosition(min, max, delta):

    if np.isinf(min):
            position = max - delta
    elif np.isinf(max):
            position = min + delta
    else:
            position = (min + max)/2

    return position


def placeSignatures(regions, delta):

    signatures = []

    for region in regions:
        
        signature_point = [
                            computePosition(region[0][0], region[0][1], delta), 
                            computePosition(region[1][0], region[1][1], delta)
                          ]
        
        signatures.append(signature_point)
    
    return signatures


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


def propagateSignatures(A, signatures):
    
    #Returns the propagation of the signature points (this is useful because we are fixating the signatures)

    gaussian_means = []

    for point in signatures:
        
        propagated_point = systemDynamics(A, point)
        gaussian_means.append(propagated_point)
    
    return gaussian_means


def sampleFromGMM(A, n_samples, weights, signatures, cov):
                
    sampled_points = []

    means_gmm = propagateSignatures(A, signatures) #See equation to see the need of this propagation (THINK: IT SEEMS TO DEPEND ON HOW WE BUILD PROPAGATION)
    #means_gmm = signatures

    for i in range(n_samples):
        
        index_normal = np.random.choice(len(weights), p = weights)

        rv = multivariate_normal(mean = means_gmm[index_normal], cov = cov)

        sample_from_component = np.array(rv.rvs(size = 1))

        sampled_points.append(sample_from_component)

    return sampled_points


def defineRegionParameters(samples):

    coord_x0 = [sample[0] for sample in samples]
    coord_x1 = [sample[1] for sample in samples]

    mean_x0 = np.mean(coord_x0)
    stdev_x0 = np.std(coord_x0)

    mean_x1 = np.mean(coord_x1)
    stdev_x1 = np.std(coord_x1)

    return [mean_x0, stdev_x0], [mean_x1, stdev_x1]


def propagateUncertaintyOneStep(A, regions, signatures, probas, cov_noise):

    #Propagate signature points (signatures are fixed)
    #Later we can think about moving the signature points each time, so this will have to be adapted
    #propagated_signatures = propagateSignatures(signatures)

    #This has to come before the definition of the new signatures!!!! Nice :)
    means_gmm = propagateSignatures(A, signatures) #OR SHOULD IT BE JUST SIGNATURES?

    samples = sampleFromGMM(A, 1000, probas, signatures, cov_noise)
        
    params_x0, params_x1 = defineRegionParameters(samples)


    n_signatures = math.floor(6*params_x0[1]/0.05) #proxy
    print(n_signatures)

    #regions = createRegions([params_x0[0] - 3*params_x0[1], params_x0[0] + 3*params_x0[1]], [params_x1[0] - 3*params_x1[1], params_x1[0] + 3*params_x1[1]], math.floor(math.sqrt(len(signatures)))-1)
    regions = createRegions([params_x0[0] - 4*params_x0[1], params_x0[0] + 4*params_x0[1]], [params_x1[0] - 4*params_x1[1], params_x1[0] + 4*params_x1[1]], n_signatures)
    signatures = placeSignatures(regions, 0.5)

    #means_gmm = signatures

    #means_gmm = propagateMeans(means_gmm)
    probas = computeSignatureProbabilities(regions, means_gmm, cov_noise, probas) 

    return probas, signatures, regions