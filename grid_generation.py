import numpy as np
import math
from scipy.stats import norm

import propagation_methods as propag

def print_array_size(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"Number of signatures: {len(result)}")
        return result
    return wrapper

# ----------------------------------------------------------------------------------------- #
# ---------------------------------- Grid generation -------------------------------------- #
# ----------------------------------------------------------------------------------------- #

def generateEquallySpacedIntervalsPerAxis(start, stop, n_partitions):

    return np.linspace(start, stop, n_partitions)


def generateUnequallyLinearSpacedIntervalsPerAxis(start, stop, n_partitions):

    sigma = (stop - start)/8

    mark1 = start + 1*sigma
    mark2 = start + 2*sigma
    mark3 = start + 3*sigma
    mark5 = start + 5*sigma
    mark6 = start + 6*sigma
    mark7 = start + 7*sigma

    # x1 = np.linspace(start, mark1, int(np.ceil(0.1*n_partitions)))
    # x2 = np.linspace(mark1, mark2, int(np.ceil(0.1*n_partitions)))
    # x3 = np.linspace(mark2, mark3, int(np.ceil(0.2*n_partitions)))
    # x4 = np.linspace(mark3, mark5, int(np.ceil(0.6*n_partitions))) #center
    # x5 = np.linspace(mark5, mark6, int(np.ceil(0.2*n_partitions)))
    # x6 = np.linspace(mark6, mark7, int(np.ceil(0.1*n_partitions)))
    # x7 = np.linspace(mark7, stop, int(np.ceil(0.1*n_partitions)))

    x1 = np.linspace(start, mark1, 5)
    x2 = np.linspace(mark1, mark2, 15)
    x3 = np.linspace(mark2, mark3, 40)
    x4 = np.linspace(mark3, mark5, 110) #center
    x5 = np.linspace(mark5, mark6, 40)
    x6 = np.linspace(mark6, mark7, 15)
    x7 = np.linspace(mark7, stop, 5)

    #TODO: maybe latter set a minimum nb of signatures in each partition to 2, so both starts and ends are taken into account

    return np.concatenate((x1, x2, x3, x4, x5, x6, x7))



def generate_regions(*lists):
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



def createRegionPartitions(region_limits, n_partitions, type_partition):
    
    dimension = region_limits.shape[-1]

    separations = []

    is_inf = np.logical_or(np.isinf(region_limits[0]), np.isinf(region_limits[1]))

    #Generate separations for non-infinite limits
    if type_partition == 'equally':
        separations = [
                        generateEquallySpacedIntervalsPerAxis(region_limits[0][n], region_limits[1][n], n_partitions[n])
                        if not is_inf[n] else np.array([region_limits[0][n], region_limits[1][n]])
                        for n in range(dimension)
                      ]
    elif type_partition == 'unequally_linear':
        separations = [
                        generateUnequallyLinearSpacedIntervalsPerAxis(region_limits[0][n], region_limits[1][n], n_partitions[n])
                        if not is_inf[n] else np.array([region_limits[0][n], region_limits[1][n]])
                        for n in range(dimension)
                      ]

    regions = generate_regions(*separations)

    return regions


def compute_outer_point(cubes, factor):
#TODO: this method is only a heuristics. We should improve it to define an outer signature that makes sense somehow
    
    if len(cubes) == 1:
        # If there's only one cube, find a point outside it by extending one of its corners
        cube = cubes[0]
        outer_point = np.array(cube[0]) - factor * np.abs(np.array(cube[1]) - np.array(cube[0]))
        return outer_point
        
    else:
        # If there are multiple cubes, find a central point guaranteed to be outside all of them
        all_corners = np.array([corner for cube in cubes for corner in cube])
        min_corner = np.min(all_corners, axis=0)
        max_corner = np.max(all_corners, axis=0)
        return (min_corner + max_corner + 0.5) / 2 #added this 0.5 to the case where this is zero


def computePosition(minimum, maximum):
    return minimum + (maximum - minimum) / 2


@print_array_size
def placeSignatures(bounded_regions):
    
    # Extract dimensions of regions
    num_regions, _, num_dimensions = bounded_regions.shape

    # Compute signature points for all bounded regions
    signature_points = np.empty((num_regions, num_dimensions))
    for dim in range(num_dimensions):
        signature_points[:, dim] = computePosition(bounded_regions[:, 0, dim], bounded_regions[:, 1, dim])

    return signature_points


def addUnboundedRepresentations(regions, unbounded_region, signatures, outer_signature):

    regions = np.concatenate((regions, [unbounded_region]))
    signatures = np.concatenate((signatures, [outer_signature]))

    return regions, signatures



def defineRegionParameters(samples):
    
    mean = np.mean(samples, axis=0)
    stdev = np.std(samples, axis=0)
    
    lower_bound = mean - 4 * stdev
    upper_bound = mean + 4 * stdev
    
    return np.array([lower_bound, upper_bound])



def updateGrid(samples, n_partitions, type_partition):
        
    hpr = defineRegionParameters(samples)

    regions = createRegionPartitions(hpr, n_partitions, type_partition)
    signatures = placeSignatures(regions)

    return signatures, regions


# ----------------------------------------------------------------------------------------- #
# --------------------------------- Vertices of the grid ---------------------------------- #
# ----------------------------------------------------------------------------------------- #

def getVertices(cube):
    
    dimensions = len(cube[0])  # Get the dimension of the cube
    vertices = []

    for i in range(2 ** dimensions):
        vertex = []
        for j in range(dimensions):
            if (i >> j) & 1:
                vertex.append(cube[1][j])  # Use max value for this dimension
            else:
                vertex.append(cube[0][j])  # Use min value for this dimension
        vertices.append(vertex)

    return np.array(vertices)