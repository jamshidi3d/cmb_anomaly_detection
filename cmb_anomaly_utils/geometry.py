import numpy as np
import healpy as hp

from . import const, coords, stat_utils as su
from .dtypes import PixMap

def get_top_bottom_caps_selection_filters(pix_map:PixMap, cap_angle):
    z_cap_border     = coords.angle_to_z(cap_angle)
    top_selection    = pix_map.raw_pos[:, 2] > z_cap_border
    bottom_selection = np.logical_not(top_selection)
    return top_selection, bottom_selection

def get_top_bottom_caps_by_filters(pix_map: PixMap, top_sel, bottom_sel):
    top_cap    = pix_map.extract_selection(top_sel)
    bottom_cap = pix_map.extract_selection(bottom_sel)
    return top_cap, bottom_cap

def get_top_bottom_caps(pix_map: PixMap, cap_angle):
    top_sel, bottom_sel = get_top_bottom_caps_selection_filters(pix_map, cap_angle)
    top_cap, bottom_cap = get_top_bottom_caps_by_filters(pix_map, top_sel, bottom_sel)
    return top_cap, bottom_cap

def get_stripe_rest_selection_filters(pix_map: PixMap, start_angle, stop_angle):
    z_start         = coords.angle_to_z(start_angle)
    z_stop          = coords.angle_to_z(stop_angle)
    stripe_selection = (z_start >= pix_map.raw_pos[:, 2]) * (pix_map.raw_pos[:, 2] >= z_stop)
    r_o_s_selection = np.logical_not(stripe_selection)
    return stripe_selection, r_o_s_selection

def get_stripe_rest(pix_map: PixMap, start_angle, stop_angle):
    '''returns a stripe between given angles and the rest of sky\n
    start and stop angles have to be in degrees'''
    stripe_sel, r_o_s_sel = get_stripe_rest_selection_filters(  pix_map,
                                                                start_angle,
                                                                stop_angle)
    stripe      = pix_map.extract_selection(stripe_sel)
    rest_of_sky = pix_map.extract_selection(r_o_s_sel)
    return stripe, rest_of_sky

def get_stripe_limits(stripe_thickness, geom_range):
    def clamp_to_sphere_degree(value):
        return 180 / np.pi * np.arccos(np.clip(value, -1, 1))
    height          = 1 - np.cos(stripe_thickness * np.pi / 180)
    stripe_mid_locs  = np.cos(geom_range * np.pi / 180)
    # stripe starts
    top_lim         = stripe_mid_locs + height / 2
    stripe_starts    = clamp_to_sphere_degree(top_lim)
    # stripe ends
    bottom_lim      = stripe_mid_locs - height / 2
    stripe_ends      = clamp_to_sphere_degree(bottom_lim)
    stripe_centers   = np.copy(geom_range)
    return stripe_starts, stripe_centers, stripe_ends
