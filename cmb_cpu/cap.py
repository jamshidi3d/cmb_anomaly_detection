import numpy as np
from numba import njit


def get_z(angle):
    return np.cos(angle * np.pi / 180)


def get_masked(pix_obj, filter, mask = None):
    if mask is None:
        return tuple(d[filter] for d in pix_obj)
    _mask = mask[filter]
    return tuple(d[filter][_mask] for d in pix_obj)


def get_top_bottom_caps(pix_data, cap_angle, sky_mask = None):
    pix_temp, pix_pos = pix_data[0], pix_data[1]
    z_border = get_z(cap_angle)
    # top cap
    top_filter = pix_pos[:, 2] > z_border
    top_cap = get_masked(pix_data, top_filter, sky_mask)
    # bottom cap
    bottom_filter = pix_pos[:, 2] <= z_border
    bottom_cap = get_masked(pix_data, bottom_filter, sky_mask)
    return top_cap, bottom_cap


def get_stripe(pix_data, start_angle, stop_angle, sky_mask = None):
    pix_temp, pix_pos = pix_data[0], pix_data[1]
    z_start = get_z(start_angle)
    z_stop  = get_z(stop_angle)
    stripe_filter = (z_start >= pix_pos[:, 2]) * (pix_pos[:, 2] >= z_stop)
    stripe = get_masked(pix_data, stripe_filter, sky_mask)
    rest_of_sky_filter = np.array([not i for i in stripe_filter])
    rest_of_sky = get_masked(pix_data, rest_of_sky_filter, sky_mask)
    return stripe, rest_of_sky
