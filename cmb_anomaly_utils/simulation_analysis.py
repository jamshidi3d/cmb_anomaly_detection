import numpy as np
# from scipy.integrate import simpson as simpson_integrate
from .dtypes import pix_data
from .utils import legendre, integrate_curve
from .measure import get_stripe_std

def add_multipole_to_map(sky_data:pix_data, a_l):
    y = sky_data.data
    legendre_on_pix = np.array([a_l[i] * legendre(i,y) for i in range(len(a_l))])
    sky_data.data *= (1 + np.sum(legendre_on_pix, axis = 0)) 
    return sky_data

