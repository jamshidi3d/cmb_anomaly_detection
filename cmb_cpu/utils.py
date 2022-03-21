from numba import njit

@njit
def clamp(x, min = -1, max = 1):
    if x >= 1:
        return 0.99999999
    if x <= -1:
        return -0.99999999
    return x