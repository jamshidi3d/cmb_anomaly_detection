import numpy as np
import concurrent.futures

from .dtypes import pix_data
from . import const


def clamp(x, x_min = -1, x_max = 1):
    if x <= x_min:
        return x_min + const.THRESHOLD
    elif x >= x_max:
        return x_max - const.THRESHOLD
    return x

def get_sampling_range(**kwargs):
    sampling_start, sampling_stop, nsamples = \
        kwargs['sampling_start'], kwargs['sampling_stop'], kwargs['nsamples']
    return np.linspace(sampling_start, sampling_stop, nsamples)

def get_extended_range(sampling_range, new_start = 0, new_stop = 180):
    new_len = new_stop - new_start
    ext_nsamples = new_len/(np.max(sampling_range) - np.min(sampling_range)) * len(sampling_range)
    ext_range = np.linspace(new_start, new_stop, int(ext_nsamples) )
    return ext_range

#----------- Parallel -----------
def get_block(pdata:pix_data, block_size, block_num):
    start_i = block_num * block_size
    end_i   = (block_num + 1) * block_size
    _temp   = pdata.data[start_i : end_i]
    _pos    = pdata.pos[start_i : end_i]
    return pix_data(_temp, _pos)

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
            angle = np.arccos(clamp(cos_th, -1, 1))
            index = int(n_samples * angle / np.pi)
            corr_n[0, index] += data1[i] * data2[j]
            corr_n[1, index] += 1
    return corr_n

def parallel_correlation(pdata:pix_data, **kwargs):
    nblocks, nsamples = \
        kwargs['nblocks'], kwargs['nsamples']
    try:
        mode = kwargs['2pcf_mode']
    except:
        mode = 'TT'
    _pdata = pdata.copy()
    block_size = round(len(_pdata.data) / nblocks)
    # print("- Block size: {}".format(block_size))
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
                        two_blocks_correlation, pd1.data, pd1.pos, pd2.data, pd2.pos, nsamples, is_same))
                # print("- Process for blocks \"{}\" and \"{}\" queued".format(i,j))
        results = np.array([proc.result() for proc in processes])
        corr_tilepair = results[:, 0]
        count_tilepair = results[:, 1]
    _corr = np.sum(corr_tilepair, axis= 0)
    _count = np.array(np.sum(count_tilepair, axis= 0), dtype = np.int_)
    _count[_count == 0] = 1
    return _corr / _count



#------------- Linear -------------
def correlation(pdata:pix_data, n_samples = 180, mode = 'TT'):
    _pdata = pdata.copy()
    corr = np.zeros(n_samples)
    count = np.zeros(n_samples, dtype = np.int_)
    if mode == 'TT':
        _pdata.data = _pdata.data - np.mean(_pdata.data)
    for i in range(len(_pdata.data)):
        for j in range(i, len(_pdata.data)):
            cos_th = np.dot(_pdata.pos[i], _pdata.pos[j])
            angle = np.arccos(clamp(cos_th, -1, 1))
            index = int(n_samples * angle / np.pi)
            corr[index] += _pdata.data[i] * _pdata.data[j]
            count[index] += 1
    count[count == 0] = 1
    return corr / count



def std_pix_data(pdata:pix_data):
    _data = pdata.data
    return np.std(_data)

def mean_pix_data(pdata:pix_data):
    _data = pdata.data
    return np.mean(_data)