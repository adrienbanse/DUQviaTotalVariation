import numpy as np
import torch

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


def outer_point(macro_region: torch.Tensor):
    dimensions = macro_region.size(1)

    # Calculate max and min coordinates along each dimension
    max_coords, _ = torch.max(macro_region, dim=0)
    min_coords, _ = torch.min(macro_region, dim=0)

    # Choose a point close to an arbitrary face
    outer_point = torch.where(
        torch.arange(dimensions) != dimensions - 1,
        (min_coords + max_coords) / 2,
        max_coords + 1e-1
    )
    
    return outer_point


def place_signatures(bounded_regions: torch.Tensor):

    r_min = bounded_regions[:, 0, :]
    r_max = bounded_regions[:, 1, :]

    return r_min + (r_max - r_min) / 2


def add_unbounded_representations(regions, unbounded_region, signatures, outer_signature):

    regions = torch.cat((regions, unbounded_region))
    signatures = torch.cat((signatures, outer_signature))

    return regions, signatures



# ----------------------------------------------------------------------------------------- #
# --------------------------------- Vertices of the grid ---------------------------------- #
# ----------------------------------------------------------------------------------------- #

def get_vertices(hypercube):
    dimensions = hypercube.size(1)  # Get the dimension of the cube

    # Generate all possible combinations of 0s and 1s
    combinations = torch.stack(torch.meshgrid(*[torch.arange(2)] * dimensions, indexing='ij'), dim=-1).reshape(-1, dimensions)

    # Determine which dimensions are min or max
    min_values = hypercube[0]  # Min values for each dimension
    max_values = hypercube[1]  # Max values for each dimension

    # Compute vertices
    vertices = combinations * max_values + (1 - combinations) * min_values

    return vertices


def identify_high_prob_region(samples: torch.Tensor):

    min_point = torch.min(samples, dim=0).values
    max_point = torch.max(samples, dim=0).values

    return torch.stack((min_point, max_point), dim=0)


def check_if_point_is_in_region(point, region):
    lower_extremities = region[0]
    upper_extremities = region[1]

    inside = (point >= lower_extremities) & (point <= upper_extremities)
    return inside.all(dim=1)

def get_centered_region(region, point):
    return region - point


# ----------------------------------------------------------------------------------------- #
# ------------------------------------ Recursive grid ------------------------------------- #
# ----------------------------------------------------------------------------------------- #
def euclidean_distance(x: torch.Tensor, y: torch.Tensor):
    return torch.sqrt(torch.sum((x - y) ** 2))

def region_size(region):
    
    vertices = get_vertices(region)
    distance = euclidean_distance(vertices[0], vertices[-1])

    return distance

def point_inside_region(point, cube):
    min_bound, max_bound = cube
    if np.any(point < min_bound) or np.any(point > max_bound):
        return False
    return True

def check_proportion_inside_region(points, cube):
    min_bound, max_bound = cube
    inside = (points >= min_bound).all(dim=1) & (points <= max_bound).all(dim=1)
    return torch.mean(inside.float())


def check_condition(region, samples, min_proportion, min_size):
    condition_proportion = check_proportion_inside_region(samples, region) > min_proportion
    condition_size = region_size(region) > min_size

    return condition_proportion and condition_size


def subdivide_region(region, samples, min_proportion, min_size):
    min_corner, max_corner = region
    d = min_corner.size(0)

    # If depth exceeds max_depth or check_condition is false, return this region
    if not check_condition(region, samples, min_proportion, min_size):
        return [region]

    # Subdivide the region into smaller subregions
    mid_corner = (min_corner + max_corner) / 2
    subregions = []

    for i in range(2 ** d):
        mask = torch.tensor([(i >> j) & 1 for j in range(d)], dtype=torch.bool)
        new_min_corner = torch.where(mask, mid_corner, min_corner)
        new_max_corner = torch.where(mask, max_corner, mid_corner)
        new_region =  torch.stack((new_min_corner, new_max_corner), dim=0)
        subregions.extend(subdivide_region(new_region, samples, min_proportion, min_size))

    return subregions


def refine_regions(regions, signatures, contributions, threshold):
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



def subdivide_region_uniformly(region, n):
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