import numpy as np
import json

from .coords import angle_to_z
from . import math_utils as mu


class pix_data:
    def __init__(self, data:np.ndarray, pos:np.ndarray, raw_mask:np.ndarray = None):
        self.raw_data = data
        self.raw_pos = pos
        self.raw_mask = None if raw_mask is None else np.array(raw_mask, dtype=bool)
    
    @property
    def data(self):
        if not self.raw_mask is None:
            screen = self.get_pixel_screen()
            return self.raw_data[screen]
        return self.raw_data
    
    @data.setter
    def data(self, value):
        self.raw_data = value

    @property
    def pos(self):
        if not self.raw_mask is None:
            screen = self.get_pixel_screen()
            return self.raw_pos[screen]
        return self.raw_pos

    @pos.setter
    def pos(self, value):
        self.raw_pos = value

    def copy(self):
        return pix_data(np.copy(self.raw_data), np.copy(self.raw_pos), np.copy(self.raw_mask))
    
    def extract_selection(self, selection) -> "pix_data":
        _raw_mask = None if self.raw_mask is None else self.raw_mask[selection]
        return pix_data(self.raw_data[selection],
                        self.raw_pos[selection],
                        _raw_mask)

    def get_pixel_screen(self):
        _mask = np.logical_not(self.raw_mask)
        # swapping ON and OFF, because _mask is True in masked areas and False in data area
        screen = (_mask == False)
        return screen

    def get_valid_pixel_ratio(self):
        if self.raw_mask is None:
            return 1.0
        screen = self.get_pixel_screen()
        return np.sum(screen) / len(screen)

    def get_top_bottom_caps(self, cap_angle):
        z_cap_border     = angle_to_z(cap_angle)
        # top cap
        top_selection    = self.raw_pos[:, 2] > z_cap_border
        top_cap          = self.extract_selection(top_selection)
        # bottom cap
        bottom_selection = np.logical_not(top_selection)
        bottom_cap       = self.extract_selection(bottom_selection)
        return top_cap, bottom_cap

    def get_strip(self, start_angle, stop_angle):
        '''returns a strip between given angles and the rest of sky\n
        start and stop angles have to be in degrees'''
        z_start         = angle_to_z(start_angle)
        z_stop          = angle_to_z(stop_angle)
        strip_selection = (z_start >= self.raw_pos[:, 2]) * (self.raw_pos[:, 2] >= z_stop)
        strip           = self.extract_selection(strip_selection)
        r_o_s_selection = np.logical_not(strip_selection)
        rest_of_sky     = self.extract_selection(r_o_s_selection)
        return strip, rest_of_sky
    
    def add_legendre_modulation(self, a_l):
        # in legendre polynomials z = cos(theta) is used
        z = self.raw_pos[:, 2]
        legendre_on_pix = np.array([a_l[i] * mu.legendre(i, z) for i in range(1, len(a_l))])
        self.raw_data *= (1 + np.sum(legendre_on_pix, axis = 0))