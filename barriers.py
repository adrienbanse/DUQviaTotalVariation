def createBarrier(point_lower_left, point_upper_right):
    
    cube = [[point_lower_left[0], point_upper_right[0]], [point_lower_left[1], point_upper_right[1]]]

    return cube


def verifyIfInsideBarrier(state, barrier):
    
    if barrier[0][0] <= state[0] and state[0] <= barrier[0][1] and barrier[1][0] <= state[1] and state[1] <= barrier[1][1]:
        return True
    
    return False