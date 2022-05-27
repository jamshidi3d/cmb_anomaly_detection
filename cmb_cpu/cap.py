import numpy as np
from numba import njit
# import healpy

def get_z(angle):
    return np.cos(angle * np.pi / 180)

# @njit
def get_top_bottom_caps(pix_temp, pix_pos, cap_angle):
    z_border = get_z(cap_angle)
    indices = pix_pos[:, 2] > z_border
    top_cap = pix_temp[indices], pix_pos[indices]
    indices = pix_pos[:, 2] <= z_border
    bottom_cap = pix_temp[indices], pix_pos[indices]
    return top_cap, bottom_cap

def get_ring(pix_temp, pix_pos, start_angle, stop_angle):
    z_start = get_z(start_angle)
    z_stop = get_z(stop_angle)
    ring_indices = (z_start >= pix_pos[:, 2]) * (pix_pos[:, 2] >= z_stop)
    return pix_temp[ring_indices]