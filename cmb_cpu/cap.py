import numpy as np
from numba import njit


def get_z(angle):
    return np.cos(angle * np.pi / 180)


# parallel functions
@njit
def get_masked(pix_arr, indices, mask = None):
    '''used for both temperature and position'''
    if not mask is None:
        _mask = mask[indices]
        masked_obj = pix_arr[indices][_mask]
        return masked_obj
    return pix_arr

@njit
def get_cap_temp(pix_temp, pix_pos, cap_angle, section = 'top', sky_mask = None):
    z_border = get_z(cap_angle)
    # top cap
    if section == 'top':
        top_indices = pix_pos[:, 2] > z_border
        top_temp = get_masked(pix_temp, top_indices, sky_mask)
        return top_temp
    # bottom cap
    if section == 'bottom': 
        bottom_indices = pix_pos[:, 2] <= z_border
        bottom_cap = get_masked(pix_temp, bottom_indices, sky_mask)
        return bottom_cap

@njit
def get_cap_pos(pix_pos, cap_angle, section = 'top', sky_mask = None):
    z_border = get_z(cap_angle)
    # top cap
    if section == 'top':
        top_indices = pix_pos[:, 2] > z_border
        top_temp = get_masked(pix_pos, top_indices, sky_mask)
        return top_temp
    # bottom cap
    if section == 'bottom': 
        bottom_indices = pix_pos[:, 2] <= z_border
        bottom_cap = get_masked(pix_pos, bottom_indices, sky_mask)
        return bottom_cap



# linear calculation
def get_masked(pix_obj, indices, mask = None):
    if not mask is None:
        _mask = mask[indices]
        masked_obj = tuple(d[indices][_mask] for d in pix_obj)
        return masked_obj
    return tuple(d[indices] for d in pix_obj)


def get_top_bottom_caps(pix_data, cap_angle, sky_mask = None):
    pix_temp, pix_pos = pix_data[0], pix_data[1]
    z_border = get_z(cap_angle)
    # top cap
    top_indices = pix_pos[:, 2] > z_border
    top_cap = get_masked(pix_data, top_indices, sky_mask)
    # bottom cap
    bottom_indices = pix_pos[:, 2] <= z_border
    bottom_cap = get_masked(pix_data, bottom_indices, sky_mask)
    return top_cap, bottom_cap


def get_stripe(pix_data, start_angle, stop_angle, sky_mask = None):
    pix_temp, pix_pos = pix_data[0], pix_data[1]
    z_start = get_z(start_angle)
    z_stop  = get_z(stop_angle)
    stripe_indices = (pix_pos[:, 2] >= z_stop) * (z_start >= pix_pos[:, 2])
    stripe = get_masked(pix_data, stripe_indices, sky_mask)
    rest_of_sky_indices = np.array(list(set(range(len(pix_temp))) - set(stripe_indices)))
    rest_of_sky = get_masked(pix_data, rest_of_sky_indices, sky_mask)
    return stripe, rest_of_sky