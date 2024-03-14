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

def insideSumTerm(h, x, signature, var_noise, i):

    if i==0:
        return 1/(var_noise[i]**0.5)*abs(x[0] + h*x[1] - (signature[0] + h*signature[1]))
    elif i==1:
        return 1/(var_noise[i]**0.5)*abs(x[1] + h*(1/3*x[0]**3 - x[0] - x[1]) - (signature[1] + h*(1/3*signature[0]**3 - signature[0] - signature[1])))
    
    return 0


def insideSqrt(h, x, signature, var_noise):

    sum = 0
    sum += 1/(var_noise[0]) * (x[0] + h*x[1] - signature[0] - h*signature[1]) ** 2
    sum += 1/(var_noise[1]) * (x[1] + h*(x[0] + x[1])**2 - signature[1] - h*(signature[0] + signature[1])**2) ** 2
    
    return sum


def maxValueInsideRegion(h, signature, region, var_noise):

    #max_value_0, max_value_1 = 0, 0
    max_value = 0

    vertices = grid.getVertices(region)

    #Add other points where the max value can be (see math)
    # if region[0][0] < 1 < region[1][0]:
    #     vertices = np.vstack((vertices, np.array([1, region[0][1]])))
    #     vertices = np.vstack((vertices, np.array([1, region[1][1]])))
    # elif region[0][0] < -1 < region[1][0]:
    #     vertices = np.vstack((vertices, np.array([-1, region[0][1]])))
    #     vertices = np.vstack((vertices, np.array([-1, region[0][1]])))

    
    for vertice in vertices:
        # max_value_0 = max(max_value_0, insideSumTerm(h, vertice, signature, var_noise, 0))
        # max_value_1 = max(max_value_1, insideSumTerm(h, vertice, signature, var_noise, 1))
        max_value = max(max_value, sqrt(insideSqrt(h, vertice, signature, var_noise)))

    #return max_value_0 + max_value_1
    return max_value