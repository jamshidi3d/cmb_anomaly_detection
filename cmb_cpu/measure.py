from unittest import result
import numpy as np
from numba import njit, prange

import concurrent.futures

from .utils import clamp


# Parallel
def get_block(pix_temp, pix_pos, block_size, block_num):
    start_i = block_num * block_size
    end_i   = (block_num + 1) * block_size
    _temp   = pix_temp[start_i : end_i]
    _pos    = pix_pos[start_i : end_i]
    return _temp, _pos

@njit
def single_correlation(temp1, pos1, temp2, pos2, n_samples, is_same):
    data = np.zeros((2, n_samples))
    for i in range(len(temp1)):
        start = i if is_same else 0
        for j in range(start, len(temp2)):
            _cos_th = np.dot(pos1[i], pos2[j])
            angle = np.arccos(clamp(_cos_th))
            index = int(n_samples * angle / np.pi)
            data[0, index] += temp1[i] * temp2[j]
            data[1, index] += 1
    return data

def parallel_correlation_tt(pix_data, n_samples = 180, nblocks = 1):
    pix_temp, pix_pos = pix_data[0], pix_data[1]
    block_size = round(len(pix_temp) / nblocks)
    print("- Block size: {}".format(block_size))
    processes = []
    _temp = pix_temp - np.mean(pix_temp)
    with concurrent.futures.ProcessPoolExecutor() as exec:
        for i in range(nblocks):
            temp1, pos1 = get_block(_temp, pix_pos, block_size, i)
            for j in range(i, nblocks):
                temp2, pos2 = get_block(_temp, pix_pos, block_size, j)
                is_same = i==j
                processes.append(\
                    exec.submit(\
                        single_correlation, temp1, pos1, temp2, pos2, n_samples, is_same))
                print("- Process for blocks \"{}\" and \"{}\" queued".format(i,j))
        results = np.array([proc.result() for proc in processes])
        ctt_tiles = results[:, 0]
        ntt_tiles = results[:, 1]
    ctt = np.sum(ctt_tiles, axis= 0)
    ntt = np.array(np.sum(ntt_tiles, axis= 0), dtype = np.int_)
    ntt[ntt == 0] = 1
    return ctt / ntt

# Linear
@njit(fastmath = True)
def correlation_tt(pix_obj, n_samples = 180):
    pix_temp, pix_pos = pix_obj[0], pix_obj[1]
    ctt = np.zeros(n_samples)
    ntt = np.zeros(n_samples, dtype = np.int_)
    temp = pix_temp - np.mean(pix_temp)
    for i in prange(len(temp)):
        for j in prange(i, len(temp)):
            _cos_th = np.dot(pix_pos[i], pix_pos[j])
            angle = np.arccos(clamp(_cos_th))
            index = int(n_samples * angle / np.pi)
            ctt[index] += temp[i] * temp[j]
            ntt[index] += 1
    ntt[ntt == 0] = 1
    return ctt / ntt


# @njit
def std_t(pix_obj):
    pix_temp = pix_obj[0]
    return np.std(pix_temp)