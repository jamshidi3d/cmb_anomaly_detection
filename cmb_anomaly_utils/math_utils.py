import numpy as np
from scipy.interpolate import CubicSpline

def integrate_curve(x, y):
    dx = x[1:] - x[:-1]
    mean_y = 0.5 * (y[1:] + y[:-1])
    return np.sum(mean_y * dx)

def get_normalized_to_mean(arr):
    _mean = np.mean(arr)
    _norm = arr if _mean == 0 else arr / _mean
    return _norm

# --------- legendre utils ---------

def legendre(n, x):
    return np.polynomial.Legendre.basis(n)(x)

def get_single_legendre_coef(theta_arr, y_arr, l):
    _integral = integrate_curve(theta_arr,
                                y_arr * np.sin(theta_arr) * legendre(l, np.cos(theta_arr)))
    return (2*l + 1)/2 * _integral

def get_all_legendre_coefs(theta_arr, y_arr, max_l):
    '''theta has to be in radians \n
    output is of size (max_l + 1)'''
    a_l = np.zeros(max_l + 1)
    for l in range(0, max_l + 1):
        a_l[l] = get_single_legendre_coef(theta_arr, y_arr, l)
    return a_l

def get_single_legendre_modulation(theta_arr, y_arr, l):
    y_norm = get_normalized_to_mean(y_arr)
    return get_single_legendre_coef(theta_arr, y_norm, l)

def get_all_legendre_modulation(theta_arr, y_arr, max_l):
    y_norm = get_normalized_to_mean(y_arr)
    return get_all_legendre_coefs(theta_arr, y_norm, max_l)

def create_legendre_modulation_factor(pos_arr, a_l):
    '''Generates the factor to be multiplied by map'''
    # in legendre polynomials z = cos(theta) is used
    z = pos_arr[:, 2]
    legendre_on_pix = np.array([a_l[i] * legendre(i, z) for i in range(1, len(a_l))])
    return (1 + np.sum(legendre_on_pix, axis = 0))

#--------- extrapolation ---------
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



