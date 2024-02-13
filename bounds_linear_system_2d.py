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

def insideSumTerm(A, x, signature, cov_noise, i):
    return 1/(cov_noise[i][i]**0.5)*abs(np.matmul(A, x)[i] - np.matmul(A, signature)[i])


def maxValueInsideRegion(A, signature, region, cov_noise):

    max_value_0, max_value_1 = 0, 0

    vertices = grid.getVertices(region)
    
    for vertice in vertices:
        max_value_0 = max(max_value_0, insideSumTerm(A, vertice, signature, cov_noise, 0))
        max_value_1 = max(max_value_1, insideSumTerm(A, vertice, signature, cov_noise, 1))

    return max_value_0 + max_value_1