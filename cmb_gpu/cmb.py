import cupy as cp
# import healpy
from numba import njit, jit, prange

@njit
def clamp(x, min = -1, max = 1):
    if x >= 1:
        return 0.99999999
    if x <= -1:
        return -0.99999999
    return x


class CMB_Map:
    def __init__(self, sky_temp, sky_pos):
        self.pole = cp.array([0,0,1])
        self.temp = sky_temp
        self.pos = sky_pos
        self.e_polarizarion = None
        self.b_polarization = None
    
    def create_using_lat_lon():
        pass

    def get_top_bottom_caps(self, cap_angle):
        _cos = cp.sign(cap_angle) * cp.cos(cap_angle * cp.pi / 180)
        indices = self.pos[:, 2] > _cos
        top_cap = CMB_Map(self.temp[indices], self.pos[indices])
        indices = self.pos[:, 2] <= _cos
        bottom_cap = CMB_Map(self.temp[indices], self.pos[indices])
        return top_cap, bottom_cap
    
    @njit(parallel=True)
    def correlation_tt(self, n_samples = 180):
        c_tt = cp.zeros(n_samples)
        n_tt = cp.zeros(n_samples, dtype = cp.int_)
        temp = self.temp - cp.mean(self.temp)
        for i in prange(len(self.temp)):
            for j in prange(i, len(self.temp)):
                tt = temp[i] * temp[j]
                _cos_th = cp.dot(self.pos[i], self.pos[j])
                angle = cp.arccos(clamp(_cos_th))
                index = int(n_samples * angle / cp.pi)
                c_tt[index] += tt
                n_tt[index] += 1
        n_tt[n_tt == 0] = 1
        return c_tt / n_tt
