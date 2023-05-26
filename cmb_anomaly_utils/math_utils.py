import numpy as np
from scipy.interpolate import CubicSpline


def clamp(x, min_val = -1.0, max_val = 1.0):
    if x >= max_val:
        return max_val - 0.0000001
    if x <= min_val:
        return min_val + 0.0000001
    return x

def legendre(n, x):
    '''Legendre Polynomials'''
    if(n == 0): # P0 = 1
        return np.ones(len(x)) if type(x) == np.ndarray else 1 
    elif(n == 1): # P1 = x
        return x
    else:
        return (((2*n)-1) * x * legendre(n-1, x) - (n-1) * legendre(n-2, x)) / float(n)

def integrate_curve(x, y):
    dx = x[1:] - x[:-1]
    mean_y = 0.5 * (y[1:] + y[:-1])
    return np.sum(mean_y * dx)

def get_legendre_coefficients(theta, y, max_l):
    '''theta has to be in radians'''
    a_l = np.zeros(max_l + 1)
    for l in range(0, max_l + 1):
        a_l[l] = (2*l + 1)/2 * integrate_curve(theta, y * np.sin(theta) * legendre(l, np.cos(theta)))
    return a_l

def extrapolate_curve(x, y, extended_x):
    '''gives extrapolated y-values'''
    curve = CubicSpline(x, y, bc_type='not-a-knot')
    return curve(extended_x)





