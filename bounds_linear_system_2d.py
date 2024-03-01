import propagation_methods as propag
import bounds_common as common
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

def insideSumTerm(A, x, signature, var_noise):

    inverse_noise = np.linalg.inv(np.diag(var_noise))
    
    diff_f = np.dot(A, x) - np.dot(A, signature)

    return np.dot(diff_f, np.dot(inverse_noise, diff_f))


def maxValueInsideRegion(A, signature, region, var_noise):

    max_value = 0

    vertices = grid.getVertices(region)
    
    for vertice in vertices:

        max_value = max(max_value, sqrt(insideSumTerm(A, vertice, signature, var_noise)))

    return max_value
