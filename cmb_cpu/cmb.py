import numpy as np
# import healpy
from numba import njit, jit, prange

from cmb_cpu.measure import correlation_tt

from . import coords
from . import utils
from . import measure

class CMB_Map:
    def __init__(self, pix_temp, pix_pos):
        self.pole = np.array([[0,0,1]])
        self.pix_temp = pix_temp
        self.pix_pos = pix_pos
        self.e_polarizarion = None
        self.b_polarization = None

    # def reset_pole(self):
    #     self.pix_pos = coords.rotate_pole_to_north(90, 0, self.pix_pos)

    # def get_pole(self):
    #     return self.pole[0]

    def set_pole(self, pole_lat, pole_lon):
        self.pix_pos = coords.rotate_pole_to_north(pole_lat, pole_lon, self.pix_pos)
        self.pole = coords.convert_polar_to_xyz(np.array([pole_lat]), np.array([pole_lon]))

    def get_top_bottom_caps(self, cap_angle):
        _cos = np.sign(cap_angle) * np.cos(cap_angle * np.pi / 180)
        indices = self.pix_pos[:, 2] > _cos
        top_cap = CMB_Map(self.pix_temp[indices], self.pix_pos[indices])
        indices = self.pix_pos[:, 2] <= _cos
        bottom_cap = CMB_Map(self.pix_temp[indices], self.pix_pos[indices])
        return top_cap, bottom_cap

    def correlation_tt(self, n_samples):
        return measure.correlation_tt(self.pix_temp, self.pix_pos, n_samples)

    def std_t(self):
        return np.std(self.pix_temp)