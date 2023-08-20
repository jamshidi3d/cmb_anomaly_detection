import numpy as np

from . import coords
from .dtypes import PixMap

def get_top_bottom_caps(pix_map: PixMap, cap_angle):
    z_cap_border     = coords.angle_to_z(cap_angle)
    # top cap
    top_selection    = pix_map.raw_pos[:, 2] > z_cap_border
    top_cap          = pix_map.extract_selection(top_selection)
    # bottom cap
    bottom_selection = np.logical_not(top_selection)
    bottom_cap       = pix_map.extract_selection(bottom_selection)
    return top_cap, bottom_cap

def get_strip(pix_map:PixMap, start_angle, stop_angle):
    '''returns a strip between given angles and the rest of sky\n
    start and stop angles have to be in degrees'''
    z_start         = coords.angle_to_z(start_angle)
    z_stop          = coords.angle_to_z(stop_angle)
    strip_selection = (z_start >= pix_map.raw_pos[:, 2]) * (pix_map.raw_pos[:, 2] >= z_stop)
    strip           = pix_map.extract_selection(strip_selection)
    r_o_s_selection = np.logical_not(strip_selection)
    rest_of_sky     = pix_map.extract_selection(r_o_s_selection)
    return strip, rest_of_sky

def get_caps_on_range(sky_pix_map, geom_range):
    pass

def get_strips_on_range(sky_pix_map, strip_thickness, geom_range):
    pass