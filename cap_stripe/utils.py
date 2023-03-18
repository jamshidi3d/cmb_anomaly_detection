import numpy as np
from numba import njit, prange
import concurrent.futures

from .dtypes import pix_data


@njit
def clamp(x, min = -1, max = 1):
    if x >= 1:
        return 0.99999999
    if x <= -1:
        return -0.99999999
    return x

@njit
def P(n, x):
    '''Legendre Polynomials'''
    if(n == 0):
        return 1 # P0 = 1
    elif(n == 1):
        return x # P1 = x
    else:
        return (((2 * n)-1)*x * P(n-1, x)-(n-1)*P(n-2, x))/float(n)

############# Parallel #############
def get_block(pdata:pix_data, block_size, block_num):
    start_i = block_num * block_size
    end_i   = (block_num + 1) * block_size
    _temp   = pdata.data[start_i : end_i]
    _pos    = pdata.pos[start_i : end_i]
    return pix_data(_temp, _pos)

@njit
def two_blocks_correlation(pdata1:pix_data, pdata2:pix_data, n_samples, is_same):
    '''internal function for two data blocks(portions of total data)'''
    data1, pos1 = pdata1.data, pdata1.pos
    data2, pos2 = pdata2.data, pdata2.pos
    corr_n = np.zeros((2, n_samples))
    for i in range(len(data1)):
        start = i if is_same else 0
        for j in range(start, len(data2)):
            cos_th = np.dot(pos1[i], pos2[j])
            angle = np.arccos(clamp(cos_th))
            index = int(n_samples * angle / np.pi)
            corr_n[0, index] += data1[i] * data2[j]
            corr_n[1, index] += 1
    return corr_n

def parallel_correlation(pdata:pix_data, nsamples = 180, nblocks = 1, mode = 'TT'):
    _pdata = pdata.copy()
    block_size = round(len(_pdata.data) / nblocks)
    print("- Block size: {}".format(block_size))
    processes = []
    if mode == 'TT':
        _pdata.data = _pdata.data - np.mean(_pdata.data)
    with concurrent.futures.ProcessPoolExecutor() as exec:
        for i in range(nblocks):
            pd1 = get_block(_pdata, block_size, i)
            for j in range(i, nblocks):
                pd2 = get_block(_pdata, block_size, j)
                is_same = i==j
                processes.append(\
                    exec.submit(\
                        two_blocks_correlation, pd1, pd2, nsamples, is_same))
                print("- Process for blocks \"{}\" and \"{}\" queued".format(i,j))
        results = np.array([proc.result() for proc in processes])
        corr_tilepair = results[:, 0]
        count_tilepair = results[:, 1]
    _corr = np.sum(corr_tilepair, axis= 0)
    _count = np.array(np.sum(count_tilepair, axis= 0), dtype = np.int_)
    _count[_count == 0] = 1
    return _corr / _count



############# Linear #############
@njit(fastmath = True)
def correlation(pdata:pix_data, n_samples = 180, mode = 'TT'):
    _pdata = pdata.copy()
    corr = np.zeros(n_samples)
    count = np.zeros(n_samples, dtype = np.int_)
    if mode == 'TT':
        _pdata.data = _pdata.data - np.mean(_pdata.data)
    for i in prange(len(_pdata.data)):
        for j in prange(i, len(_pdata.data)):
            _cos_th = np.dot(_pdata.pos[i], _pdata.pos[j])
            angle = np.arccos(clamp(_cos_th))
            index = int(n_samples * angle / np.pi)
            corr[index] += _pdata.data[i] * _pdata.data[j]
            count[index] += 1
    count[count == 0] = 1
    return corr / count


# @njit
def std_pix_data(pdata:pix_data):
    _data = pdata.data
    return np.std(_data)

def mean_pix_data(pdata:pix_data):
    _data = pdata.data
    return np.mean(_data)