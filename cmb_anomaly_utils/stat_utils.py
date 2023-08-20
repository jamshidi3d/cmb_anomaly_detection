import numpy as np
import concurrent.futures
from numba import njit, prange

from .dtypes import PixMap
from . import const

@njit(fastmath = True)
def clamp(x, x_min = -1, x_max = 1):
    if x <= x_min:
        return x_min + const.THRESHOLD
    elif x >= x_max:
        return x_max - const.THRESHOLD
    return x

def get_range(start, stop, dsamples):
    nsamples = 1 + int(np.abs((stop - start) / dsamples))
    return np.linspace(start, stop, nsamples)

def get_extended_range(sampling_range, new_start = 0, new_stop = 180):
    new_len = new_stop - new_start
    ext_nsamples = new_len/(np.max(sampling_range) - np.min(sampling_range)) * len(sampling_range)
    ext_range = np.linspace(new_start, new_stop, int(ext_nsamples) )
    return ext_range

def find_nearest_index(arr, val):
    return np.nanargmin(np.abs(arr - val))

#----------- Parallel -----------
def get_chunk(pix_map:PixMap, block_size, block_num):
    start_i = block_num * block_size
    end_i   = (block_num + 1) * block_size
    _data   = pix_map.data[start_i : end_i]
    _pos    = pix_map.pos[start_i : end_i]
    return _data, _pos

@njit(fastmath = True)
def two_blocks_correlation(data1:np.ndarray, pos1:np.ndarray,
                           data2:np.ndarray, pos2:np.ndarray,
                           n_samples:int,
                           is_same:bool):
    '''internal function for two data blocks(portions of total data)'''
    corr_n = np.zeros((2, n_samples))
    for i in range(len(data1)):
        start = i if is_same else 0
        for j in range(start, len(data2)):
            cos_th = np.dot(pos1[i], pos2[j])
            cos_th = clamp(cos_th, -1, 1)
            angle = np.arccos(cos_th)
            index = int(n_samples * angle / np.pi)
            corr_n[0, index] += data1[i] * data2[j]
            corr_n[1, index] += 1
    return corr_n

def parallel_correlation(pix_map:PixMap, **kwargs):
    ndata_chunks        = kwargs.get('ndata_chunks', 4)
    nmeasure_samples    = kwargs.get('nmeasure_samples', 181)
    mode                = kwargs.get('tpcf_mode', const.TT_2PCF)
    if len(pix_map.data) == 0:
        return 0
    _pix_map = pix_map.copy()
    if mode == const.TT_2PCF:
        _pix_map.data = _pix_map.data - np.mean(_pix_map.data)
    chunk_size = round(len(_pix_map.data) / ndata_chunks)
    # print("- Chunk size: {}".format(chunk_size))
    processes = []
    with concurrent.futures.ProcessPoolExecutor() as exec:
        for i in range(ndata_chunks):
            data1, pos1 = get_chunk(_pix_map, chunk_size, i)
            for j in range(i, ndata_chunks):
                data2, pos2 = get_chunk(_pix_map, chunk_size, j)
                is_same = i==j
                processes.append(\
                    exec.submit(\
                        two_blocks_correlation, data1, pos1, data2, pos2, nmeasure_samples, is_same))
                # print("- Process for data chunks \"{}\" and \"{}\" queued".format(i,j))
        results = np.array([proc.result() for proc in processes])
        corr_chunkpair   = results[:, 0]
        count_chunkpair  = results[:, 1]
    _corr   = np.sum(corr_chunkpair, axis= 0)
    _count  = np.array(np.sum(count_chunkpair, axis= 0), dtype = np.int_)
    _count[_count == 0] = 1
    return _corr / _count



#------------- Linear -------------
def correlation(pix_map:PixMap, n_samples = 180, mode = const.TT_2PCF):
    _pix_map = pix_map.copy()
    corr = np.zeros(n_samples)
    count = np.zeros(n_samples, dtype = np.int_)
    if mode == const.TT_2PCF:
        _pix_map.data = _pix_map.data - np.mean(_pix_map.data)
    for i in range(len(_pix_map.data)):
        for j in range(i, len(_pix_map.data)):
            cos_th = np.dot(_pix_map.pos[i], _pix_map.pos[j])
            angle = np.arccos(np.clip(cos_th, -1, 1))
            index = int(n_samples * angle / np.pi)
            corr[index] += _pix_map.data[i] * _pix_map.data[j]
            count[index] += 1
    count[count == 0] = 1
    return corr / count


@njit(fastmath = True)
def fast_std(arr):
    return np.std(arr)

@njit(fastmath = True)
def fast_mean(arr):
    return np.mean(arr)

def std_map(pix_map:PixMap):
    _data = pix_map.data
    if len(_data) == 0:
        return 0
    return fast_std(_data)

def mean_map(pix_map:PixMap):
    _data = pix_map.data
    if len(_data) == 0:
        return 0
    return np.mean(_data)