import numpy as np


from .dtypes import pix_data, run_parameters
from . import const
from . import utils

def get_corr_full_integral(sky_pix, params:run_parameters):
    '''full integral of 2p_correlation for later use'''
    mflag = params.measure_flag
    full_int = 1
    if mflag in (const.CORR_FLAG, const.D_CORR2_FLAG):
        fullsky_corr = utils.parallel_correlation(sky_pix, params.nsamples, 4)
        full_int = np.sum(fullsky_corr ** 2)
    else:
        full_int = utils.std_pix_data(sky_pix)
    return full_int


def get_cap_anomaly(sky_pix:pix_data, params:run_parameters):
    measure_result = np.zeros(len(params.sampling_range))
    if params.measure_flag in (const.CORR_FLAG, const.D_CORR2_FLAG):
        f_int = get_corr_full_integral(sky_pix, params)
    mflag, nsamples = params.measure_flag, params.nsamples
    cap_angles = params.sampling_range
    for i in range(len(cap_angles)):
        ca = cap_angles[i]
        print("++ Cap of size {} degrees".format(ca))
        top, bottom = sky_pix.get_top_bottom_caps(ca)
        if mflag == const.D_CORR2_FLAG:
            tctt = utils.parallel_correlation(top, nsamples, 4)
            bctt = utils.parallel_correlation(bottom, nsamples, 4)
            max_index = int(params.cacr * 2 * min(ca, 180-ca) / 180 * nsamples)
            measure_result[i] = np.sum((tctt[:max_index] - bctt[:max_index])**2)
        elif mflag == const.D_STD2_FLAG:
            measure_result[i] = (utils.std_pix_data(top) - utils.std_pix_data(bottom))**2
        elif mflag == const.STD_FLAG:
            measure_result[i] = utils.std_pix_data(top)
        elif mflag == const.CORR_FLAG:
            tctt = utils.parallel_correlation(top, nsamples, 4)
            measure_result[i] = np.sum(tctt ** 2) / f_int - 1
    return measure_result

def get_stripe_anomaly(sky_pix:pix_data, params:run_parameters):
    measure_result = np.zeros(len(params.sampling_range))
    if params.measure_flag in (const.CORR_FLAG, const.D_CORR2_FLAG):
        f_int = get_corr_full_integral(sky_pix, params)
    mflag, nsamples = params.measure_flag, params.nsamples
    # creating stripes' limits
    height = 1 - np.cos(params.stripe_thickness * np.pi / 180)
    stripe_centers = params.sampling_range
    stripe_starts  = 180 / np.pi * np.arccos(np.cos(stripe_centers * np.pi / 180) + height/2)
    stripe_ends    = 180 / np.pi * np.arccos(np.cos(stripe_centers * np.pi / 180) - height/2)
    # measure
    for i in range(len(stripe_centers)):
        print("++ Stripe center {} degrees".format(stripe_centers[i]))
        start = stripe_starts[i]
        end = stripe_ends[i]
        stripe, rest_of_sky = sky_pix.get_stripe(start, end)
        if mflag == const.D_CORR2_FLAG:
            sctt = utils.parallel_correlation(stripe, nsamples, 4)
            rctt = utils.parallel_correlation(rest_of_sky, nsamples, 4)
            max_index = int(params.cacr * 2 * params.stripe_thickness / 180 * nsamples)
            measure_result[i] = np.sum((sctt[:max_index] - rctt[:max_index])**2)
        elif mflag == const.D_STD2_FLAG:
            measure_result[i] = (utils.std_pix_data(stripe) - utils.std_pix_data(rest_of_sky))**2
        elif mflag == const.STD_FLAG:
            measure_result[i] = utils.std_pix_data(stripe)
        elif mflag == const.CORR_FLAG:
            tctt = utils.parallel_correlation(stripe, nsamples, 4)
            measure_result[i] = np.sum(tctt ** 2) / f_int - 1