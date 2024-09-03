import grid_generation as grid
import probability_mass_computation as proba

import numpy as np
import math
from scipy.stats import multivariate_normal


def get_initial_state(mean, var, size):

    cov = np.diag(var)
    
    multivariate_gaussian = multivariate_normal(mean = mean, cov = cov)
    initial_state_samples = multivariate_gaussian.rvs(size = size)

    return np.array(initial_state_samples)


def systemNoise(mean, var, size):

    cov = np.diag(var)
    
    multivariate_gaussian = multivariate_normal(mean = mean, cov = cov)
    noise_samples = multivariate_gaussian.rvs(size = size)

    return np.array(noise_samples)


def systemDynamics(x, method, params):
    if method == 'linear':
        A = params[0]
        return np.matmul(x, A.T)  #Due to shape of arrays
    
    elif method == 'polynomial':
        h = params[0]
        #component_1 = x[:, 0] + h * x[:, 1]
        #component_2 = x[:, 1] + h * (1/3 * x[:, 0]**3 - x[:, 0] - x[:, 1])

        component_1 = x[:, 0] + 1.25 * h * x[:, 1]
        component_2 = 1.4 * x[:, 1] + h * 0.3 * ( 0.25*x[:, 0]**2 - 0.4 * x[:, 0] * x[:, 1] + 0.25*x[:, 1]**2 )
        
        return np.column_stack((component_1, component_2))
    
    elif method == 'dubin':
        h = params[0]
        v = params[1]
        u = params[2]

        component_1 = x[:, 0] + h * v * np.sin(x[:, 2])
        component_2 = x[:, 1] + h * v * np.cos(x[:, 2])
        component_3 = x[:, 2] + h * u

        return np.column_stack((component_1, component_2, component_3))
    
    else:
        raise Exception("Please refer to the method's signature")
    

def propagateSignatures(signatures, method, params):
    return systemDynamics(signatures, method, params)


def stateOneStepEvolution(current_state, mean_noise, cov_noise, method, params):
    
    size_samples = current_state.shape[0]

    noise = systemNoise(mean_noise, cov_noise, size_samples)

    return systemDynamics(current_state, method, params) + noise



def stateEvolution(initial_state, mean_noise, cov_noise, n_steps_ahead, method, params):
    
    state = initial_state

    for t in range(n_steps_ahead):

        noise = systemNoise(mean_noise, cov_noise)
        
        state = systemDynamics(state, method, params) + noise
    
    return state


def monteCarloSimulationSystem(n_simulations, mean_initial_state, cov_initial_state, mean_noise, cov_noise, n_steps_ahead, method, params):
    
    initial_states, final_states = [], []
    
    for i in range(n_simulations):
    
        initial_state = get_initial_state(mean_initial_state, cov_initial_state, )
        final_state = stateEvolution(initial_state, mean_noise, cov_noise, n_steps_ahead, method, params)

        initial_states.append(initial_state)
        final_states.append(final_state)

    return initial_states, final_states


# def sampleFromGMM(n_samples, weights, means, var):
#
#     sampled_points = []
#
#     cov = np.diag(var)
#
#     for i in range(n_samples):
#
#         index_normal = np.random.choice(len(weights), p = weights)
#
#         rv = multivariate_normal(mean = means[index_normal], cov = cov)
#
#         sample_from_component = rv.rvs(size = 1)
#
#         sampled_points.append(sample_from_component)
#
#     return np.array(sampled_points)