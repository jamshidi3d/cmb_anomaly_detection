import numpy as np

from .dtypes import pix_data

from . import const, stat_utils as su, coords

def get_corr_full_integral(sky_pix:pix_data, **kwargs):
    '''full integral of 2pcf for later use\n
    -> keyword arguments:\n
    nblocks - nsamples
    '''
    full_int = 1
    if kwargs['measure_flag'] in (const.CORR_FLAG, const.D_CORR2_FLAG):
        fullsky_corr = su.parallel_correlation(sky_pix, **kwargs)
        full_int = np.sum(fullsky_corr ** 2)
    return full_int

def calc_cap_anomaly_in_all_dir(cmb_pd: pix_data, dir_lat_arr, dir_lon_arr, **_inputs):
    ndir        = len(dir_lat_arr)
    nsamples    = _inputs['nsamples']
    all_dir_anomaly = np.zeros((ndir, nsamples))
    pix_pos     = np.copy(cmb_pd.pos)
    for i in range(ndir):
        print(f"{i}/{ndir - 1} \r", end="")
        cmb_pd.pos = coords.rotate_pole_to_north(pix_pos, dir_lat_arr[i], dir_lon_arr[i])
        _result = get_cap_anomaly(cmb_pd, **_inputs)
        all_dir_anomaly[i] = _result
    return all_dir_anomaly

#------------ Cap functions ------------
def get_cap_dcorr2(top:pix_data, bottom:pix_data, **kwargs):
    '''-> keyword arguments: \n
    cap_angle - cutoff_ratio -\n
    nblocks - nsamples'''
    cap_angle, cutoff_ratio, nsamples = \
        kwargs['cap_angle'], kwargs['cutoff_ratio'], kwargs['nsamples']
    tctt = su.parallel_correlation(top, **kwargs)
    bctt = su.parallel_correlation(bottom, **kwargs)
    max_index = int(cutoff_ratio * 2 * min(cap_angle, 180-cap_angle) / 180 * nsamples)
    return np.sum((tctt[:max_index] - bctt[:max_index])**2)

def get_cap_corr(top:pix_data, bottom:pix_data, **kwargs):
    '''-> keyword arguments: \n
    full_integral - nsamples'''
    f_int, nsamples = kwargs['full_integral'], kwargs['nsamples']
    tctt = su.parallel_correlation(top, **kwargs)
    return np.sum(tctt ** 2) / f_int - 1

def get_cap_dstd2(top:pix_data, bottom:pix_data, **kwargs):
    return (su.std_pix_data(top) - su.std_pix_data(bottom))**2

def get_cap_std(top:pix_data, bottom:pix_data, **kwargs):
    return su.std_pix_data(top)

def get_cap_mean(top:pix_data, bottom:pix_data, **kwargs):
    return su.mean_pix_data(top)


cap_func_dict = {
    const.D_CORR2_FLAG: get_cap_dcorr2,
    const.CORR_FLAG: get_cap_corr,
    const.MEAN_FLAG: get_cap_mean,
    const.STD_FLAG: get_cap_std,
    const.D_STD2_FLAG: get_cap_dstd2
}
def get_cap_anomaly(sky_pix:pix_data, **kwargs):
    '''-> keyword arguments: \n
    sampling_range - measure_flag -\n
    nsamples - nblocks'''
    sampling_range, measure_flag, nsamples, nblocks = \
        kwargs['sampling_range'], kwargs['measure_flag'], kwargs['nsamples'], kwargs['nblocks']
    measure_results = np.zeros(len(sampling_range))
    f_int = get_corr_full_integral(sky_pix, **kwargs)
    cap_angles = sampling_range
    _kwargs = {'nsamples': nsamples,
              'nblocks': nblocks,
              'cap_angle': 0,
              'cutoff_ratio': kwargs['cutoff_ratio'],
              'full_integral': f_int}
    measure_func = cap_func_dict[measure_flag]
    for i in range(len(cap_angles)):
        ca = _kwargs['cap_angle'] = cap_angles[i]
        # print("++ Cap of size {} degrees\r".format(ca), end = "")
        top, bottom = sky_pix.get_top_bottom_caps(ca)
        measure_results[i] = measure_func(top, bottom, **_kwargs)
    # print()
    return measure_results


#---------- Stripe functions ----------
def get_strip_limits(strip_thickness, sampling_range):
    height = 1 - np.cos(strip_thickness * np.pi / 180)
    strip_centers = sampling_range
    # strip starts
    top_lim = np.cos(strip_centers * np.pi / 180) + height / 2
    strip_starts  = 180 / np.pi * np.arccos(np.clip(top_lim, -1, 1))
    # strip ends
    bottom_lim = np.cos(strip_centers * np.pi / 180) - height / 2
    strip_ends    = 180 / np.pi * np.arccos(np.clip(bottom_lim, -1, 1))
    return strip_starts, strip_centers, strip_ends

def get_strip_dcorr2(strip, rest_of_sky, **kwargs):
    '''-> keyword arguments: \n
    strip_thickness - \n
    nsamples - nblocks - cutoff_ratio'''
    strip_thickness, cutoff_ratio, nsamples, nblocks = \
        kwargs['strip_thickness'], kwargs['cutoff_ratio'], kwargs['nsamples'], kwargs['nblocks']
    sctt = su.parallel_correlation(strip, nsamples, nblocks)
    rctt = su.parallel_correlation(rest_of_sky, nsamples, nblocks)
    max_index = int(cutoff_ratio * 2 * strip_thickness / 180 * nsamples)
    return np.sum((sctt[:max_index] - rctt[:max_index])**2)

def get_strip_corr(strip:pix_data, rest_of_sky:pix_data, **kwargs):
    '''keyword arguments: \n
    nsamples - full_integral'''
    nsamples, f_int = kwargs['nsamples'], kwargs['full_integral']
    tctt = su.parallel_correlation(strip, nsamples, 4)
    return np.sum(tctt ** 2) / f_int - 1

def get_strip_dstd2(strip:pix_data, rest_of_sky:pix_data, **kwargs):
    return (su.std_pix_data(strip) - su.std_pix_data(rest_of_sky))**2

def get_strip_std(strip:pix_data, rest_of_sky:pix_data = None, **kwargs):
    return su.std_pix_data(strip)

def get_strip_mean(strip:pix_data, rest_of_sky:pix_data, **kwargs):
    return su.mean_pix_data(strip)


strip_func_dict = {
    const.D_CORR2_FLAG: get_strip_dcorr2,
    const.CORR_FLAG: get_strip_corr,
    const.MEAN_FLAG: get_strip_mean,
    const.STD_FLAG: get_strip_std,
    const.D_STD2_FLAG: get_strip_dstd2
}
def get_strip_anomaly(sky_pix:pix_data, **kwargs):
    '''keyword arguments: \n
    sampling_range - strip_thickness - measure_flag -\n
    nsamples - cutoff_ratio - nblocks
    '''
    sampling_range, strip_thickness =\
        kwargs['sampling_range'], kwargs['strip_thickness']
    measure_flag, nsamples, nblocks =\
        kwargs['measure_flag'], kwargs['nsamples'], kwargs['nblocks']
    measure_results = np.zeros(len(sampling_range))
    f_int = get_corr_full_integral(sky_pix, **kwargs)
    strip_starts, strip_centers, strip_ends = get_strip_limits(strip_thickness, sampling_range)
    _kwargs = {'nsamples': nsamples, 'cutoff_ratio': kwargs['cutoff_ratio'], 'full_integral': f_int, 'nblocks': nblocks}
    measure_func = strip_func_dict[measure_flag]
    # measure
    for i in range(len(strip_centers)):
        # print("++ Stripe center {} degrees".format(strip_centers[i])+" " * 20+"\r", end="")
        start = strip_starts[i]
        end   = strip_ends[i]
        strip, rest_of_sky = sky_pix.get_strip(start, end)
        measure_results[i] = measure_func(strip, rest_of_sky, **_kwargs)
    # print()
    return measure_results