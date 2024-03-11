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
    sum += 1/(var_noise[0]**0.5) * (abs(x[0] - signature[0]) + h*v*abs(x[2] - signature[2]))
    sum += 1/(var_noise[1]**0.5) * (abs(x[1] - signature[1]) + h*v*abs(x[2] - signature[2]))
    sum += 1/(var_noise[2]**0.5) * abs(x[2] - signature[2])
    
    return sum


def maxValueInsideRegion(h, v, u, signature, region, var_noise):

    max_value = 0

    vertices = grid.getVertices(region)
    
    for vertice in vertices:
        max_value = max(max_value, insideSumTerm(h, v, u, vertice, signature, var_noise))

    return max_value