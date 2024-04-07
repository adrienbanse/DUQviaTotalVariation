import numpy as np
import math
from scipy.stats import norm

import propagation_methods as propag
import bounds_linear_system_2d as bounds_linear

def print_array_size(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"Number of signatures: {len(result)}")
        return result
    return wrapper

# ----------------------------------------------------------------------------------------- #
# ---------------------------------- Grid generation -------------------------------------- #
# ----------------------------------------------------------------------------------------- #

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


def computeOuterPoint(region):
    dimensions = len(region[0])  # Get the number of dimensions
    max_coords = [max(region[i][d] for i in range(len(region))) for d in range(dimensions)]
    min_coords = [min(region[i][d] for i in range(len(region))) for d in range(dimensions)]
    
    # Choose a point close to the upper face
    outer_point = [(min_coords[d] + max_coords[d]) / 2 if d != dimensions - 1 else max_coords[d] + 1e-2 for d in range(dimensions)]
    
    return np.array(outer_point)


def computePosition(minimum, maximum):
    return minimum + (maximum - minimum) / 2


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


def findMinMaxPoints(samples):

    min_point = np.min(samples, axis=0)
    max_point = np.max(samples, axis=0)

    return min_point, max_point


def checkIfPointIsInRegion(point, region):
    return np.all(point >= region[0]) and np.all(point <= region[1])

def getCenteredRegion(region, point):
    return region - point


# ----------------------------------------------------------------------------------------- #
# ------------------------------------ Recursive grid ------------------------------------- #
# ----------------------------------------------------------------------------------------- #
def euclideanDistance(point1, point2):
    return np.sqrt(np.sum(point1 - point2) ** 2)

def regionSize(region):
    
    vertices = getVertices(region)
    distance = euclideanDistance(vertices[0], vertices[-1])

    return distance

def pointInsideRegion(point, cube):    
    min_bound, max_bound = cube
    if np.any(point < min_bound) or np.any(point > max_bound):
        return False
    return True

def checkProportionInsideRegion(points, cube):
    min_bound, max_bound = cube
    return np.mean((np.all(points >= min_bound, axis=1)) & (np.all(points <= max_bound, axis=1)))


def check_condition(region, samples, min_proportion, min_size):
    condition_proportion = checkProportionInsideRegion(samples, region) > min_proportion
    condition_size = regionSize(region) > min_size

    return condition_proportion & condition_size


def subdivideRegion(region, samples, min_proportion, min_size):
    subregions = []
    if check_condition(region, samples, min_proportion, min_size):
        
        # If condition is true, subdivide the region in half
        num_dimensions = len(region[0])
        midpoints = np.mean(region, axis=0)

        # Generate binary sequences for all possible subdivisions
        for i in range(2 ** num_dimensions):
            coords = np.empty_like(region)
            for j in range(num_dimensions):
                min_val = region[0][j]
                max_val = region[1][j]
                mid_val = midpoints[j]
                if i & (1 << j):
                    coords[:, j] = np.array([mid_val, max_val])
                else:
                    coords[:, j] = np.array([min_val, mid_val])
            subregions.extend(subdivideRegion(coords, samples, min_proportion, min_size))

    else:
        # If condition is false, append the region to the list of subregions
        subregions.append(region)
    return subregions


def refineRegions(regions, signatures, contributions, threshold):
    new_regions = []
    new_signatures = []

    for i, contribution in enumerate(contributions):
        if contribution > threshold and not np.isinf(regions[i][0][0]):
            num_dimensions = regions[i].shape[1]  # Get the number of dimensions

            # Calculate midpoints for each dimension
            midpoints = np.mean(regions[i], axis=0)

            # Generate binary sequences for all possible subdivisions
            for j in range(2 ** num_dimensions):
                coords = np.empty_like(regions[i])
                for k in range(num_dimensions):
                    if j & (1 << k):
                        coords[:, k] = np.array([midpoints[k], regions[i][1][k]])
                    else:
                        coords[:, k] = np.array([regions[i][0][k], midpoints[k]])
                new_regions.append(coords)
                
                # Calculate signatures for each subregion
                sub_sig = np.mean(coords, axis=0)
                new_signatures.append(sub_sig)
        else:
            new_regions.append(regions[i])
            new_signatures.append(signatures[i])

    return np.array(new_regions), np.array(new_signatures)



def subdivideRegionUniformly(region, n):
    x_min = min(region[0][0], region[1][0])
    x_max = max(region[0][0], region[1][0])
    y_min = min(region[0][1], region[1][1])
    y_max = max(region[0][1], region[1][1])
    
    x_interval = (x_max - x_min) / n
    y_interval = (y_max - y_min) / n
    
    partitions = []
    
    for i in range(n):
        for j in range(n):
            x_start = x_min + i * x_interval
            x_end = x_start + x_interval
            y_start = y_min + j * y_interval
            y_end = y_start + y_interval
            partitions.append([[x_start, y_start], [x_end, y_end]])
    
    return partitions