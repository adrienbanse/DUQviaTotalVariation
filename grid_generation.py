import numpy as np
import math
from scipy.stats import norm

# ----------------------------------------------------------------------------------------- #
# ---------------------------------- Grid generation -------------------------------------- #
# ----------------------------------------------------------------------------------------- #

def non_linear_spacing_gaussian(start, stop, num_points, mean, std_dev):
    """
    Generate non-linearly spaced samples using a Gaussian density as reference.

    Parameters:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num_points (int): Number of samples to generate.
        mean (float): The mean of the Gaussian distribution.
        std_dev (float): The standard deviation of the Gaussian distribution.

    Returns:
        array: A 1-D array containing the non-linearly spaced samples.
    """
    # Generate Gaussian PDF
    x = np.linspace(start, stop, num_points)
    pdf = norm.pdf(x, mean, 2*std_dev)

    # Adjust spacing using Gaussian PDF
    spacing = 1 / pdf
    spacing /= np.max(spacing)
    cum_spacing = np.cumsum(spacing)
    cum_spacing /= np.max(cum_spacing)
    samples = start + (stop - start) * cum_spacing

    samples = np.append(samples, start) #Need to add the start value to guarantee the symmetry
    samples = np.sort(samples)

    return samples

def getStartStop(mean, std_dev, sigma_factor):
    
    start = mean - sigma_factor*std_dev
    stop = mean + sigma_factor*std_dev

    return start, stop


def generateIntervalsPerAxis(distrib_params, sigma_factor, n_partitions):

    mean = distrib_params[0]
    std_dev = distrib_params[1]

    start, stop = getStartStop(mean, std_dev, sigma_factor)

    points = non_linear_spacing_gaussian(start, stop, n_partitions, mean, std_dev)

    #points = np.append(points, -np.inf)
    #points = np.append(points, np.inf)
    #points = np.sort(points)

    return points

def generateEquallySpacedIntervalsPerAxis(start, stop, n_partitions):

    return np.linspace(start, stop, n_partitions)


def createBoundedRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions):
    
    x_points = generateIntervalsPerAxis(distrib_params_1, sigma_factor, n_partitions)
    y_points = generateIntervalsPerAxis(distrib_params_2, sigma_factor, n_partitions)

    regions = []

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
                
            if i > 0 and j > 0:              
                region = [[x_points[i-1], x_points[i]],
                        [y_points[j-1], y_points[j]]]
            
                regions.append(region)

    return regions


def createUnboundedRegions(distrib_params_1, distrib_params_2, sigma_factor):
     

    a1, a2 = getStartStop(distrib_params_1[0], distrib_params_1[1], sigma_factor)
    b1, b2 = getStartStop(distrib_params_2[0], distrib_params_2[1], sigma_factor)

    r1 = [[-np.inf, a1], [b2, np.inf]]
    r2 = [[a1, a2], [b2, np.inf]]
    r3 = [[a2, np.inf], [b2, np.inf]]
    r4 = [[a2, np.inf], [b1, b2]]
    r5 = [[a2, np.inf], [-np.inf, b1]]
    r6 = [[a1, a2], [-np.inf, b1]]
    r7 = [[-np.inf, a1], [-np.inf, b1]]
    r8 = [[-np.inf, a1], [b1, b2]]

    return [r1, r2, r3, r4, r5, r6, r7, r8]



def createCenterRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions):
    
    x_points = generateIntervalsPerAxis(distrib_params_1, sigma_factor, n_partitions)
    y_points = generateIntervalsPerAxis(distrib_params_2, sigma_factor, n_partitions)

    regions = []

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
                
            if i > 0 and j > 0:              
                region = [[x_points[i-1], x_points[i]],
                        [y_points[j-1], y_points[j]]]
            
                regions.append(region)

    return regions

def createLeftLateralRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions):
     
    x_points = generateEquallySpacedIntervalsPerAxis(distrib_params_1[0] - sigma_factor*distrib_params_1[1], distrib_params_1[0] - (sigma_factor-1)*distrib_params_1[1], n_partitions//2)
    y_points = generateEquallySpacedIntervalsPerAxis(distrib_params_2[0] - sigma_factor*distrib_params_2[1], distrib_params_2[0] + sigma_factor*distrib_params_2[1], n_partitions*2)

    regions = []

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
                
            if i > 0 and j > 0:              
                region = [[x_points[i-1], x_points[i]],
                        [y_points[j-1], y_points[j]]]
            
                regions.append(region)
    
    return regions


def createRightLateralRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions):
     
    x_points = generateEquallySpacedIntervalsPerAxis(distrib_params_1[0] + (sigma_factor-1)*distrib_params_1[1], distrib_params_1[0] + sigma_factor*distrib_params_1[1], n_partitions//2)
    y_points = generateEquallySpacedIntervalsPerAxis(distrib_params_2[0] - sigma_factor*distrib_params_2[1], distrib_params_2[0] + sigma_factor*distrib_params_2[1], n_partitions*2)

    regions = []

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
                
            if i > 0 and j > 0:              
                region = [[x_points[i-1], x_points[i]],
                        [y_points[j-1], y_points[j]]]
            
                regions.append(region)
    
    return regions


def createUpperRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions):
     
    x_points = generateEquallySpacedIntervalsPerAxis(distrib_params_1[0] - (sigma_factor-1)*distrib_params_1[1], distrib_params_1[0] + (sigma_factor-1)*distrib_params_1[1], n_partitions*2)
    y_points = generateEquallySpacedIntervalsPerAxis(distrib_params_2[0] + (sigma_factor-1)*distrib_params_2[1], distrib_params_2[0] + sigma_factor*distrib_params_2[1], n_partitions//2)

    regions = []

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
                
            if i > 0 and j > 0:              
                region = [[x_points[i-1], x_points[i]],
                        [y_points[j-1], y_points[j]]]
            
                regions.append(region)
    
    return regions


def createUnderRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions):
     
    x_points = generateEquallySpacedIntervalsPerAxis(distrib_params_1[0] - (sigma_factor-1)*distrib_params_1[1], distrib_params_1[0] + (sigma_factor-1)*distrib_params_1[1], n_partitions*2)
    y_points = generateEquallySpacedIntervalsPerAxis(distrib_params_2[0] - sigma_factor*distrib_params_2[1], distrib_params_2[0] - (sigma_factor-1)*distrib_params_2[1], n_partitions//2)

    regions = []

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
                
            if i > 0 and j > 0:              
                region = [[x_points[i-1], x_points[i]],
                        [y_points[j-1], y_points[j]]]
            
                regions.append(region)
    
    return regions


def createRegionsAlternative(distrib_params_1, distrib_params_2, number_signatures):

    n_center = number_signatures[0]
    n_2sigma = number_signatures[1]
    n_3sigma = number_signatures[2]
    n_4sigma = number_signatures[3]
    n_5sigma = number_signatures[4]
     
    center = createCenterRegions(distrib_params_1, distrib_params_2, 1, n_center)

    lateral_left = createLeftLateralRegions(distrib_params_1, distrib_params_2, 2, n_2sigma)
    lateral_right = createRightLateralRegions(distrib_params_1, distrib_params_2, 2, n_2sigma)
    upper = createUpperRegions(distrib_params_1, distrib_params_2, 2, n_2sigma)
    under = createUnderRegions(distrib_params_1, distrib_params_2, 2, n_2sigma)


    lateral_left_after = createLeftLateralRegions(distrib_params_1, distrib_params_2, 3, n_3sigma)
    lateral_right_after = createRightLateralRegions(distrib_params_1, distrib_params_2, 3, n_3sigma)
    upper_after = createUpperRegions(distrib_params_1, distrib_params_2, 3, n_3sigma)
    under_after = createUnderRegions(distrib_params_1, distrib_params_2, 3, n_3sigma)


    lateral_left_after_later = createLeftLateralRegions(distrib_params_1, distrib_params_2, 4, n_4sigma)
    lateral_right_after_later = createRightLateralRegions(distrib_params_1, distrib_params_2, 4, n_4sigma)
    upper_after_later = createUpperRegions(distrib_params_1, distrib_params_2, 4, n_4sigma)
    under_after_later = createUnderRegions(distrib_params_1, distrib_params_2, 4, n_4sigma)

    lateral_left_after_later_then = createLeftLateralRegions(distrib_params_1, distrib_params_2, 5, n_5sigma)
    lateral_right_after_later_then = createRightLateralRegions(distrib_params_1, distrib_params_2, 5, n_5sigma)
    upper_after_later_then = createUpperRegions(distrib_params_1, distrib_params_2, 5, n_5sigma)
    under_after_later_then = createUnderRegions(distrib_params_1, distrib_params_2, 5, n_5sigma)


    unbounded_regions = createUnboundedRegions(distrib_params_1, distrib_params_2, 5)

    regions = center + lateral_left + lateral_right + upper + under + lateral_left_after + lateral_right_after + upper_after + under_after + lateral_left_after_later + lateral_right_after_later + upper_after_later + under_after_later + lateral_left_after_later_then + lateral_right_after_later_then + upper_after_later_then + under_after_later_then + unbounded_regions

    return regions
     
          


def createRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions):
     
    bounded_regions = createBoundedRegions(distrib_params_1, distrib_params_2, sigma_factor, n_partitions)

    unbounded_regions = createUnboundedRegions(distrib_params_1, distrib_params_2, sigma_factor)

    regions = bounded_regions + unbounded_regions

    return regions


def computePosition(min, max, delta):

    if np.isinf(min):
            position = max - delta
    elif np.isinf(max):
            position = min + delta
    else:
            position = min + (max - min)/2

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


# ----------------------------------------------------------------------------------------- #
# --------------------------------- Vertices of the grid ---------------------------------- #
# ----------------------------------------------------------------------------------------- #

def getVertices(region):
    
    vertice1 = [region[0][0], region[1][0]]
    vertice2 = [region[0][0], region[1][1]]
    vertice3 = [region[0][1], region[1][0]]
    vertice4 = [region[0][1], region[1][1]]

    return [vertice1, vertice2, vertice3, vertice4]