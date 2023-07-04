import numpy as np
from scipy.interpolate import CubicSpline

def integrate_curve(x, y):
    dx = x[1:] - x[:-1]
    mean_y = 0.5 * (y[1:] + y[:-1])
    return np.sum(mean_y * dx)

#--------------legendre utils----------------

def legendre(n, x):
    '''Legendre Polynomials'''
    if(n == 0): # P0 = 1
        return np.ones(len(x)) if type(x) == np.ndarray else 1 
    elif(n == 1): # P1 = x
        return x
    else:
        return (((2*n)-1) * x * legendre(n-1, x) - (n-1) * legendre(n-2, x)) / float(n)

def get_single_legendre_coef(theta, y, l):
    return (2*l + 1)/2 * integrate_curve(theta, y * np.sin(theta) * legendre(l, np.cos(theta)))

def get_all_legendre_coefs(theta, y, max_l):
    '''theta has to be in radians \n
    output is of size (max_l + 1)'''
    a_l = np.zeros(max_l + 1)
    for l in range(0, max_l + 1):
        a_l[l] = get_single_legendre_coef(theta, y, l)
    return a_l

def get_single_legendre_modulation(theta, y, l):
    y_mean = np.mean(y)
    y_mod = y if y_mean == 0 else y / y_mean
    return get_single_legendre_coef(theta, y_mod, l)

def get_all_legendre_modulation(theta, y, max_l):
    y_mean = np.mean(y)
    y_mod = y if y_mean == 0 else y / y_mean
    return get_all_legendre_coefs(theta, y_mod, max_l)

#--------------extrapolation----------------

def extrapolate_curve(x, y, extended_x, curve_type='clamped', deriv=0):
    '''gives extrapolated y-values\n
    types are: 'not-a-knot' - 'natural' - 'clamped' '''
    curve = CubicSpline(x, y, bc_type=curve_type)
    if curve_type in ('clamped', 'natural'):
        add_boundary_knots(curve)
    return curve(extended_x, nu = deriv)

def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.
    
    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])



