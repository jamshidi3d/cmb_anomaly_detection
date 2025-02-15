import numpy as np
from scipy.interpolate import CubicSpline

_lmax = None
_theta_domain = None
_sin_theta = None      
_P_l = None      
        

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

def get_all_coefs_having_legendre_values(y_arr, theta_arr = None, max_l = None, do_reset_domain = False):
    # Check if it needs to recalculate P_l
    global _lmax
    global _theta_domain
    global _sin_theta
    global _P_l
    if do_reset_domain or _lmax != max_l or _lmax is None :
        _theta_domain = theta_arr
        _sin_theta = np.sin(theta_arr)
        _lmax = max_l
        _P_l = np.array([legendre(ell, np.cos(theta_arr)) for ell in range(_lmax + 1)])
    # Actual calculation
    a_l = np.zeros(max_l + 1)
    for ell in range(0, max_l + 1):
        a_l[ell] = integrate_curve( _theta_domain, 
                                    y_arr * _sin_theta * _P_l[ell] )
    return a_l

def get_single_legendre_modulation(theta_arr, y_arr, l):
    y_norm = get_normalized_to_mean(y_arr)
    return get_single_legendre_coef(theta_arr, y_norm, l)

def get_all_legendre_modulation(theta_arr, y_arr, max_l):
    a_l = get_all_legendre_coefs(theta_arr, y_arr, max_l)
    a_l = a_l / a_l[0]
    return a_l

def create_legendre_modulation_factor(pos_arr, a_l):
    '''Generates the factor to be multiplied by map'''
    # in legendre polynomials z = cos(theta) is used
    z = pos_arr[:, 2]
    legendre_on_pix = np.array([a_l[i] * legendre(i, z) for i in range(1, len(a_l))])
    return (1 + np.sum(legendre_on_pix, axis = 0))

# def spherical_psuedo_gaussian(sigma, pole_vec, pos_array, )

def normalized_spherical_gaussian_emulator(sharpness, pole_vec, pos_array, area = 4 * np.pi):
    _s = sharpness
    # This is the normalized value of the amplitude,
    # if we integrate over the whole sphere:
    # a = 1/2pi * s/(e^(-2s) - 1)
    amplitude = _s/ (2*np.pi) / (1 - np.exp(-2*_s))
    dot_arr = np.dot(np.transpose(pole_vec), pos_array)
    return amplitude * np.exp(sharpness * (dot_arr - 1))



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



