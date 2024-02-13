import numpy as np
import math

# ----------------------------------------------------------------------------------------- #
# ---------------------------------- Grid generation -------------------------------------- #
# ----------------------------------------------------------------------------------------- #

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


# ----------------------------------------------------------------------------------------- #
# --------------------------------- Vertices of the grid ---------------------------------- #
# ----------------------------------------------------------------------------------------- #

def getVertices(region):
    
    vertice1 = [region[0][0], region[1][0]]
    vertice2 = [region[0][0], region[1][1]]
    vertice3 = [region[0][1], region[1][0]]
    vertice4 = [region[0][1], region[1][1]]

    return [vertice1, vertice2, vertice3, vertice4]