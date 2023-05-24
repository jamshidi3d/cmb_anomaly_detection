import numpy as np
import json

from .coords import angle_to_z
from . import const


class pix_data:
    def __init__(self, data:np.ndarray, pos:np.ndarray, mask:np.ndarray = None):
        self.data = data
        self.pos = pos
        self.mask = mask
    
    def copy(self):
        return pix_data(np.copy(self.data), np.copy(self.pos), np.copy(self.mask))

    def get_filtered(self, filter) -> "pix_data":
        if self.mask is None:
            return pix_data(self.data[filter], self.pos[filter])
        _mask = self.mask[filter]
        return pix_data(self.data[filter][_mask], self.pos[filter][_mask])

    def get_top_bottom_caps(self, cap_angle):
        z_border = angle_to_z(cap_angle)
        # top cap
        top_filter = self.pos[:, 2] > z_border
        top_cap = self.get_filtered(top_filter)
        # bottom cap
        bottom_filter = self.pos[:, 2] <= z_border
        bottom_cap = self.get_filtered(bottom_filter)
        return top_cap, bottom_cap

    def get_stripe(self, start_angle, stop_angle):
        '''returns a stripe between given angles and the rest of sky\n
        start and stop angles have to be in degrees'''
        z_start = angle_to_z(start_angle)
        z_stop  = angle_to_z(stop_angle)
        stripe_filter = (z_start >= self.pos[:, 2]) * (self.pos[:, 2] >= z_stop)
        stripe = self.get_filtered(stripe_filter)
        rest_of_sky_filter = np.array([not i for i in stripe_filter])
        rest_of_sky = self.get_filtered(rest_of_sky_filter)
        return stripe, rest_of_sky