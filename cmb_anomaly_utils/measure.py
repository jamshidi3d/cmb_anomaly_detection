import numpy as np

from .dtypes import pix_data

from . import const, stat_utils as su

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
def get_stripe_limits(stripe_thickness, sampling_range):
    height = 1 - np.cos(stripe_thickness * np.pi / 180)
    stripe_centers = sampling_range
    # stripe starts
    top_lim = np.cos(stripe_centers * np.pi / 180) + height / 2
    stripe_starts  = 180 / np.pi * np.arccos(np.clip(top_lim, -1, 1))
    # stripe ends
    bottom_lim = np.cos(stripe_centers * np.pi / 180) - height / 2
    stripe_ends    = 180 / np.pi * np.arccos(np.clip(bottom_lim, -1, 1))
    return stripe_starts, stripe_centers, stripe_ends

def get_stripe_dcorr2(stripe, rest_of_sky, **kwargs):
    '''-> keyword arguments: \n
    stripe_thickness - \n
    nsamples - nblocks - cutoff_ratio'''
    stripe_thickness, cutoff_ratio, nsamples, nblocks = \
        kwargs['stripe_thickness'], kwargs['cutoff_ratio'], kwargs['nsamples'], kwargs['nblocks']
    sctt = su.parallel_correlation(stripe, nsamples, nblocks)
    rctt = su.parallel_correlation(rest_of_sky, nsamples, nblocks)
    max_index = int(cutoff_ratio * 2 * stripe_thickness / 180 * nsamples)
    return np.sum((sctt[:max_index] - rctt[:max_index])**2)

def get_stripe_corr(stripe:pix_data, rest_of_sky:pix_data, **kwargs):
    '''keyword arguments: \n
    nsamples - full_integral'''
    nsamples, f_int = kwargs['nsamples'], kwargs['full_integral']
    tctt = su.parallel_correlation(stripe, nsamples, 4)
    return np.sum(tctt ** 2) / f_int - 1

def get_stripe_dstd2(stripe:pix_data, rest_of_sky:pix_data, **kwargs):
    return (su.std_pix_data(stripe) - su.std_pix_data(rest_of_sky))**2

def get_stripe_std(stripe:pix_data, rest_of_sky:pix_data = None, **kwargs):
    return su.std_pix_data(stripe)

def get_stripe_mean(stripe:pix_data, rest_of_sky:pix_data, **kwargs):
    return su.mean_pix_data(stripe)


stripe_func_dict = {
    const.D_CORR2_FLAG: get_stripe_dcorr2,
    const.CORR_FLAG: get_stripe_corr,
    const.MEAN_FLAG: get_stripe_mean,
    const.STD_FLAG: get_stripe_std,
    const.D_STD2_FLAG: get_stripe_dstd2
}
def get_stripe_anomaly(sky_pix:pix_data, **kwargs):
    '''keyword arguments: \n
    sampling_range - stripe_thickness - measure_flag -\n
    nsamples - cutoff_ratio - nblocks
    '''
    sampling_range, stripe_thickness =\
        kwargs['sampling_range'], kwargs['stripe_thickness']
    measure_flag, nsamples, nblocks =\
        kwargs['measure_flag'], kwargs['nsamples'], kwargs['nblocks']
    measure_results = np.zeros(len(sampling_range))
    f_int = get_corr_full_integral(sky_pix, **kwargs)
    stripe_starts, stripe_centers, stripe_ends = get_stripe_limits(stripe_thickness, sampling_range)
    _kwargs = {'nsamples': nsamples, 'cutoff_ratio': kwargs['cutoff_ratio'], 'full_integral': f_int, 'nblocks': nblocks}
    measure_func = stripe_func_dict[measure_flag]
    # measure
    for i in range(len(stripe_centers)):
        # print("++ Stripe center {} degrees".format(stripe_centers[i])+" " * 20+"\r", end="")
        start = stripe_starts[i]
        end = stripe_ends[i]
        stripe, rest_of_sky = sky_pix.get_stripe(start, end)
        measure_results[i] = measure_func(stripe, rest_of_sky, **_kwargs)
    # print()
    return measure_results