import numpy as np


def print_proportion(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"Hitting proba: {result}")
        return result
    return wrapper



def generate_barriers(*lists):
    if len(lists) == 0:
        return []

    # Get the lengths of each list
    lengths = [len(lst) for lst in lists]

    # Create slices for each list
    slices = [slice(None)] * len(lists)
    regions = []

    # Recursive function to generate regions
    def generate_region(slice_indices):
        if len(slice_indices) == len(lists):
            region = np.array([[lst[i-1], lst[i]] for lst, i in zip(lists, slice_indices)])
            regions.append(region.T)
        else:
            for i in range(1, lengths[len(slice_indices)]):
                new_slice_indices = slice_indices + [i]
                generate_region(new_slice_indices)

    generate_region([])

    return np.array(regions)


def createBarrier(region_limits):
    
    dimension = region_limits.shape[-1]

    separations = []

    #Generate separations for non-infinite limits
    separations = [
        np.array([region_limits[0][n], region_limits[1][n]])
        for n in range(dimension)
    ]

    regions = generate_barriers(*separations)

    return regions


def verifyIfInsideBarrier(state, barrier):

    for dim in range(len(state)):
        if not (barrier[0][dim] <= state[dim] <= barrier[1][dim]):
            return False
    return True

@print_proportion
def hittingProbabilityMC(states, barrier):
    
    states_inside_barrier = sum(verifyIfInsideBarrier(state, barrier) for state in states)
    proportion = states_inside_barrier / len(states)
    return proportion

