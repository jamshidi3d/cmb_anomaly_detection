import numpy as np
from numba import njit, prange

from .utils import clamp

@njit(parallel=True)
def correlation_tt(pix_temp, pix_pos, n_samples = 180):
    c_tt = np.zeros(n_samples)
    n_tt = np.zeros(n_samples, dtype = np.int_)
    temp = pix_temp - np.mean(pix_temp)
    for i in prange(len(temp)):
        for j in prange(i, len(temp)):
            _cos_th = np.dot(pix_pos[i], pix_pos[j])
            angle = np.arccos(clamp(_cos_th))
            index = int(n_samples * angle / np.pi)
            c_tt[index] += temp[i] * temp[j]
            n_tt[index] += 1
    n_tt[n_tt == 0] = 1
    return c_tt / n_tt