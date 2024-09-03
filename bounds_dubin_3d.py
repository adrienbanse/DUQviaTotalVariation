import propagation_methods as propag
import grid_generation as grid

import numpy as np
from scipy import special

# ----------------------------------------------------------------------------------------- #
# ------------------------------ Auxiliary computing methods ------------------------------ #
# ----------------------------------------------------------------------------------------- #

def sqrt(x):
    return np.sqrt(x)

def erf(x):
    return special.erf(x)


# ----------------------------------------------------------------------------------------- #
# ----------------------------- Auxiliary maximization methods ---------------------------- #
# ----------------------------------------------------------------------------------------- #

def insideSumTerm(h, v, u, x, signature, var_noise):

    sum = 0
    sum += 1/(var_noise[0]) * (x[0] - signature[0] + h*v* (np.sin(x[2]) - np.sin(signature[2])))**2
    sum += 1/(var_noise[1]) * (x[1] - signature[1] + h*v* (np.cos(x[2]) - np.cos(signature[2])))**2
    sum += 1/(var_noise[2]) * (x[2] - signature[2])**2
    
    return sum


def maxValueInsideRegion(h, v, u, signature, region, var_noise):

    max_value = 0

    vertices = grid.get_vertices(region)
    
    for vertice in vertices:
        max_value = max(max_value, sqrt(insideSumTerm(h, v, u, vertice, signature, var_noise)))

    return max_value