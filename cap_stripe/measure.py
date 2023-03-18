import numpy as np


from .dtypes import pix_data, run_parameters
from . import const
from . import utils

def get_corr_full_integral(sky_pix:pix_data, params:run_parameters):
    '''full integral of 2p_correlation for later use'''
    mflag = params.measure_flag
    full_int = 1
    if mflag in (const.CORR_FLAG, const.D_CORR2_FLAG):
        fullsky_corr = utils.parallel_correlation(sky_pix, params.nsamples, 4)
        full_int = np.sum(fullsky_corr ** 2)
    else:
        full_int = utils.std_pix_data(sky_pix)
    return full_int

############ Cap functions ############
def get_cap_dcorr2(top:pix_data, bottom:pix_data, **kwargs):
    params:run_parameters = kwargs['params']
    cap_angle = kwargs['cap_angle']
    nblocks, nsamples = kwargs['nblocks'], kwargs['nsamples']
    tctt = utils.parallel_correlation(top, nsamples, nblocks)
    bctt = utils.parallel_correlation(bottom, nsamples, nblocks)
    max_index = int(params.cacr * 2 * min(cap_angle, 180-cap_angle) / 180 * nsamples)
    return np.sum((tctt[:max_index] - bctt[:max_index])**2)

def get_cap_corr(top:pix_data, bottom:pix_data, **kwargs):
    f_int, nsamples = kwargs['full_integral'], kwargs['nsamples']
    tctt = utils.parallel_correlation(top, nsamples, 4)
    return np.sum(tctt ** 2) / f_int - 1

def get_cap_dstd2(top:pix_data, bottom:pix_data, **kwargs):
    return (utils.std_pix_data(top) - utils.std_pix_data(bottom))**2

def get_cap_std(top:pix_data, bottom:pix_data, **kwargs):
    return utils.std_pix_data(top)

def get_cap_mean(top:pix_data, bottom:pix_data, **kwargs):
    return utils.mean_pix_data(top)


cap_func_dict = {
    const.D_CORR2_FLAG: get_cap_dcorr2,
    const.CORR_FLAG: get_cap_corr,
    const.MEAN_FLAG: get_cap_mean,
    const.STD_FLAG: get_cap_std,
    const.D_STD2_FLAG: get_cap_dstd2
}
def get_cap_anomaly(sky_pix:pix_data, params:run_parameters):
    measure_results = np.zeros(len(params.sampling_range))
    f_int = get_corr_full_integral(sky_pix, params)
    cap_angles = params.sampling_range
    kwargs = {'nsamples': params.nsamples,
              'params': params,
              'cap_angle':0,
              'full_integral': f_int}
    measure_func = cap_func_dict[params.measure_flag]
    for i in range(len(cap_angles)):
        ca = kwargs['cap_angle'] = cap_angles[i]
        print("++ Cap of size {} degrees".format(ca))
        top, bottom = sky_pix.get_top_bottom_caps(ca)
        measure_results[i] = measure_func(top, bottom, **kwargs)
    return measure_results


############ Stripe functions ############
def get_stripe_limits(params:run_parameters):
    height = 1 - np.cos(params.stripe_thickness * np.pi / 180)
    stripe_centers = params.sampling_range
    stripe_starts  = 180 / np.pi * np.arccos(np.cos(stripe_centers * np.pi / 180) + height/2)
    stripe_ends    = 180 / np.pi * np.arccos(np.cos(stripe_centers * np.pi / 180) - height/2)
    return stripe_starts, stripe_centers, stripe_ends

def get_stripe_dcorr2(stripe, rest_of_sky, **kwargs):
    nsamples, nblocks = kwargs['nsamples'], kwargs['nblocks']
    params:run_parameters = kwargs['params']
    sctt = utils.parallel_correlation(stripe, nsamples, nblocks)
    rctt = utils.parallel_correlation(rest_of_sky, nsamples, nblocks)
    max_index = int(params.cacr * 2 * params.stripe_thickness / 180 * nsamples)
    return np.sum((sctt[:max_index] - rctt[:max_index])**2)

def get_stripe_corr(stripe:pix_data, rest_of_sky:pix_data, **kwargs):
    nsamples, f_int = kwargs['nsamples', 'full_integral']
    tctt = utils.parallel_correlation(stripe, nsamples, 4)
    return np.sum(tctt ** 2) / f_int - 1

def get_stripe_dstd2(stripe:pix_data, rest_of_sky:pix_data, **kwargs):
    return (utils.std_pix_data(stripe) - utils.std_pix_data(rest_of_sky))**2

def get_stripe_std(stripe:pix_data, rest_of_sky:pix_data, **kwargs):
    return utils.std_pix_data(stripe)

def get_stripe_mean(stripe:pix_data, rest_of_sky:pix_data, **kwargs):
    return utils.mean_pix_data(stripe)


stripe_func_dict = {
    const.D_CORR2_FLAG: get_stripe_dcorr2,
    const.CORR_FLAG: get_stripe_corr,
    const.MEAN_FLAG: get_stripe_mean,
    const.STD_FLAG: get_stripe_std,
    const.D_STD2_FLAG: get_stripe_dstd2
}
def get_stripe_anomaly(sky_pix:pix_data, params:run_parameters):
    measure_results = np.zeros(len(params.sampling_range))
    f_int = get_corr_full_integral(sky_pix, params)
    stripe_starts, stripe_centers, stripe_ends = get_stripe_limits(params)
    kwargs = {'nsamples':params.nsamples, 'full_integral': f_int, 'nblocks': params.nblocks}
    measure_func = stripe_func_dict[params.measure_flag]
    # measure
    for i in range(len(stripe_centers)):
        print("++ Stripe center {} degrees".format(stripe_centers[i]))
        start = stripe_starts[i]
        end = stripe_ends[i]
        stripe, rest_of_sky = sky_pix.get_stripe(start, end)
        measure_results[i] = measure_func(stripe, rest_of_sky, **kwargs)
    return measure_results