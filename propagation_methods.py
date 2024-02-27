import grid_generation as grid
import probability_mass_computation as proba

import numpy as np
import math
from scipy.stats import multivariate_normal


def getInitialState(mean, cov):
    
    multivariate_gaussian = multivariate_normal(mean = mean, cov = cov)
    initial_state_samples = multivariate_gaussian.rvs(size = 1)

    return np.array(initial_state_samples)


def systemNoise(mean, cov):
    
    multivariate_gaussian = multivariate_normal(mean = mean, cov = cov)
    noise_samples = multivariate_gaussian.rvs(size = 1)

    return np.array(noise_samples)


def systemDynamics(x, method, params):

    if method == 'linear':
        A = params[0]
        return np.matmul(A, x)
    
    elif method == 'polynomial':
        h = params[0]

        component_1 = x[0] + h*x[1]
        component_2 = x[1] + h*(1/3*x[0]**3 - x[0] - x[1])

        return np.array([component_1, component_2])
    
    else:
        raise Exception("Please refer to the method's signature")


def stateEvolution(initial_state, mean_noise, cov_noise, n_steps_ahead, method, params):
    
    state = initial_state

    for t in range(n_steps_ahead):

        noise = systemNoise(mean_noise, cov_noise)
        
        state = systemDynamics(state, method, params) + noise
    
    return state


def monteCarloSimulationSystem(n_simulations, mean_initial_state, cov_initial_state, mean_noise, cov_noise, n_steps_ahead, method, params):
    
    initial_states, final_states = [], []
    
    for i in range(n_simulations):
    
        initial_state = getInitialState(mean_initial_state, cov_initial_state)
        final_state = stateEvolution(initial_state, mean_noise, cov_noise, n_steps_ahead, method, params)

        initial_states.append(initial_state)
        final_states.append(final_state)

    return initial_states, final_states


def propagateSignatures(signatures, method, params):
    
    #Returns the propagation of the signature points (this is useful because we are fixating the signatures)

    gaussian_means = []

    for point in signatures:
        
        propagated_point = systemDynamics(point, method, params)
        gaussian_means.append(propagated_point)
    
    return gaussian_means


def sampleFromGMM(n_samples, weights, signatures, cov, method, params):
                
    sampled_points = []

    means_gmm = propagateSignatures(signatures, method, params) #See equation to see the need of this propagation (THINK: IT SEEMS TO DEPEND ON HOW WE BUILD PROPAGATION)
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


def defineNumberOfSignatures(coverage, st_dev, desired_region_size):
    return math.floor(coverage*st_dev/desired_region_size)


def updateGrid(desired_region_size, regions, signatures, probas, cov_noise, method, params):

    #Propagate signature points (signatures are fixed)
    #Later we can think about moving the signature points each time, so this will have to be adapted
    #propagated_signatures = propagateSignatures(signatures)

    #This has to come before the definition of the new signatures!!!! Nice :)
    #means_gmm = propagateSignatures(signatures, method, params)

    samples = sampleFromGMM(1000, probas, signatures, cov_noise, method, params)
        
    params_x0, params_x1 = defineRegionParameters(samples)

    n_signatures = defineNumberOfSignatures(8, max(params_x0[1], params_x1[1]), desired_region_size)

    #regions = grid.createRegions([params_x0[0], params_x0[1]], [params_x1[0], params_x1[1]], 4, n_signatures)
    regions = grid.createRegionsAlternative([params_x0[0], params_x0[1]], [params_x1[0], params_x1[1]], [120, 90, 40, 20, 10])
    signatures = grid.placeSignatures(regions, 0.5)

    return signatures, regions


def propagateUncertaintyOneStep(desired_region_size, regions, signatures, probas, cov_noise, method, params):

    #Propagate signature points (signatures are fixed)
    #Later we can think about moving the signature points each time, so this will have to be adapted
    #propagated_signatures = propagateSignatures(signatures)

    #This has to come before the definition of the new signatures!!!! Nice :)
    means_gmm = propagateSignatures(signatures, method, params)

    samples = sampleFromGMM(1000, probas, signatures, cov_noise, method, params)
        
    params_x0, params_x1 = defineRegionParameters(samples)

    n_signatures = defineNumberOfSignatures(6, params_x0[1], desired_region_size)

    regions = grid.createRegions([params_x0[0], params_x0[1]], [params_x1[0], params_x1[1]], 4, n_signatures)
    signatures = grid.placeSignatures(regions, 0.5)

    #means_gmm = signatures

    #means_gmm = propagateMeans(means_gmm)
    probas = proba.computeSignatureProbabilitiesInParallel(regions, means_gmm, cov_noise, probas) 

    return probas, signatures, regions