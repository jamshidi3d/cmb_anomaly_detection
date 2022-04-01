import numpy as np
from numba import njit
# import healpy

# @njit
def get_top_bottom_caps(pix_temp, pix_pos, cap_angle):
    _cos = np.sign(cap_angle) * np.cos(cap_angle * np.pi / 180)
    indices = pix_pos[:, 2] > _cos
    top_cap = pix_temp[indices], pix_pos[indices]
    indices = pix_pos[:, 2] <= _cos
    bottom_cap = pix_temp[indices], pix_pos[indices]
    return top_cap, bottom_cap
