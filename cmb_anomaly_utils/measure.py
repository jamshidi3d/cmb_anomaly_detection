import numpy as np

from .dtypes import PixMap

from . import (
    const,
    stat_utils as su,
    math_utils as mu,
    coords,
    geometry as geom
)

def calc_corr_full_integral(sky_pix:PixMap, **kwargs):
    '''- kwargs:\n
    ndata_chunks - measure_range'''
    full_int = 1
    measure_flag        = kwargs.get(const.KEY_MEASURE_FLAG, const.OBS_T)
    if measure_flag in (const.NORM_CORR_FLAG, ):
        fullsky_corr    = su.parallel_correlation_pix_map(sky_pix, **kwargs)
        measure_range   = kwargs.get(const.KEY_MEASURE_RANGE, su.get_range())
        full_int        = mu.integrate_curve(measure_range, fullsky_corr ** 2)
    return full_int

def calc_measure_in_all_dir(cmb_pd: PixMap, dir_lat_arr, dir_lon_arr, **kwargs):
    ndir            = len(dir_lat_arr)
    geom_range      = kwargs.get(const.KEY_GEOM_RANGE, su.get_range())
    nsamples        = len(geom_range)
    all_dir_measure = np.zeros((ndir, nsamples))
    pix_pos         = np.copy(cmb_pd.raw_pos)
    get_measure     = get_cap_measure if kwargs.get(const.KEY_GEOM_FLAG) == const.CAP_FLAG \
                        else get_stripe_measure
    for i in range(ndir):
        print(f"{i + 1}/{ndir} \r", end="")
        cmb_pd.raw_pos = coords.rotate_pole_to_north(pix_pos, dir_lat_arr[i], dir_lon_arr[i])
        _result = get_measure(cmb_pd, **kwargs)
        all_dir_measure[i] = _result
    return all_dir_measure

#------------ Measures ------------
def calc_dcorr2(patch1:PixMap, patch2:PixMap, **kwargs):
    '''- kwargs: \n
    max_valid_ang - cutoff_ratio\n
    ndata_chunks - measure_range'''
    max_valid_ang   = kwargs.get(const.KEY_MAX_VALID_ANG, 0)
    cutoff_ratio    = kwargs.get(const.KEY_MIN_PIX_RATIO, 2 / 3)
    measure_range   = kwargs.get(const.KEY_MEASURE_RANGE, su.get_range())
    nsamples        = len(measure_range)
    t_tpcf          = su.parallel_correlation_pix_map(patch1, **kwargs)
    b_tpcf          = su.parallel_correlation_pix_map(patch2, **kwargs)
    max_index       = int(cutoff_ratio * 2 * max_valid_ang / 180 * nsamples)
    if len(measure_range[:max_index]) == 0:
        return 0
    return mu.integrate_curve(measure_range[:max_index],
                              (t_tpcf[:max_index] - b_tpcf[:max_index])**2)

def calc_norm_corr(patch1:PixMap, patch2:PixMap = None, **kwargs):
    '''- kwargs: \n
    measure_range - corr_full_integral\n
    max_valid_ang - cutoff_ratio - ndata_chunks\n
    '''
    f_int           = kwargs.get(const.KEY_CORR_FULL_INT, 1)
    measure_range   = kwargs.get(const.KEY_MEASURE_RANGE, su.get_range())
    cutoff_ratio    = kwargs.get(const.KEY_CUTOFF_RATIO, 2 / 3)
    max_valid_ang   = kwargs.get(const.KEY_MAX_VALID_ANG, 0)
    nsamples        = len(measure_range)
    max_index       = int(cutoff_ratio * 2 * max_valid_ang / 180 * nsamples)
    tctt            = su.parallel_correlation_pix_map(patch1, **kwargs)
    if len(measure_range[:max_index]) == 0:
        return -1
    geom_int    = mu.integrate_curve(measure_range[:max_index], tctt[:max_index] ** 2)
    return geom_int / f_int - 1

def calc_var(patch1:PixMap, patch2:PixMap = None, **kwargs):
    return su.var_pix_map(patch1)

def calc_std(patch1:PixMap, patch2:PixMap = None, **kwargs):
    return su.std_pix_map(patch1)

def calc_norm_std(patch1:PixMap, patch2:PixMap = None, **kwargs):
    std_full = kwargs.get(const.KEY_STD_FULL, 1)
    return calc_std(patch1) / std_full - 1

def calc_dstd2(patch1:PixMap, patch2:PixMap, **kwargs):
    return (calc_std(patch1) - calc_std(patch2))**2

def calc_norm_dstd2(patch1:PixMap, patch2:PixMap, **kwargs):
    return (calc_norm_std(patch1, **kwargs) - calc_norm_std(patch2, **kwargs))**2

def calc_mean(patch1:PixMap, patch2:PixMap = None, **kwargs):
    return su.mean_pix_map(patch1)

func_dict = {
    const.MEAN_FLAG:        calc_mean,
    const.NORM_CORR_FLAG:   calc_norm_corr,
    const.D_CORR2_FLAG:     calc_dcorr2,
    const.STD_FLAG:         calc_std,
    const.NORM_STD_FLAG:    calc_norm_std,
    const.D_STD2_FLAG:      calc_dstd2,
    const.NORM_D_STD2_FLAG: calc_norm_dstd2
}

#------------ General Geometry ------------
def get_measure(sky_pix, **kwargs):
    geom_flag   = kwargs.get(const.KEY_GEOM_FLAG)
    if geom_flag == const.CAP_FLAG:
        return get_cap_measure(sky_pix, **kwargs)
    elif geom_flag == const.STRIPE_FLAG:
        return get_stripe_measure(sky_pix, **kwargs)
    else:
        print("Selected geometry is not supported yet!")

#------------ Cap ------------
def get_cap_measure(sky_pix:PixMap, **kwargs):
    '''- kwargs: \n
    measure_flag - measure_range\n
    ngeom_samples - geom_range\n
    ndata_chunks'''
    min_pix_ratio   = kwargs.get(const.KEY_MIN_PIX_RATIO, 1)
    measure_flag    = kwargs.get(const.KEY_MEASURE_FLAG, const.STD_FLAG)
    geom_range      = kwargs.get(const.KEY_GEOM_RANGE, su.get_range())
    _kwargs         = kwargs.copy()
    _kwargs[const.KEY_CORR_FULL_INT] = calc_corr_full_integral(sky_pix, **_kwargs)
    _kwargs[const.KEY_STD_FULL]      = calc_std(sky_pix, None, **_kwargs)
    # Measure
    measure_func    = func_dict[measure_flag]
    measure_results = np.zeros(len(geom_range))
    for i, ca in enumerate(geom_range):
        top, bottom = geom.get_top_bottom_caps(sky_pix, ca)
        _kwargs.setdefault(const.KEY_MAX_VALID_ANG,
                            np.minimum(ca, 180 - ca))
        # Remove invalid pixels
        if top.get_visible_pixels_ratio() < min_pix_ratio or \
                bottom.get_visible_pixels_ratio() < min_pix_ratio:
            measure_results[i] = np.nan
            continue
        measure_results[i] = measure_func(top, bottom, **_kwargs)
    return measure_results


#---------- Stripe ----------
def get_stripe_measure(sky_pix:PixMap, **kwargs):
    '''- kwargs: \n
    sampling_range - stripe_thickness - measure_flag\n
    cutoff_ratio - ndata_chunks
    '''
    min_pix_ratio   = kwargs.get(const.KEY_MIN_PIX_RATIO, 1)
    geom_range      = kwargs.get(const.KEY_GEOM_RANGE, su.get_range())
    stripe_thickness = kwargs.get(const.KEY_STRIPE_THICKNESS, 20)
    measure_flag    = kwargs.get(const.KEY_MEASURE_FLAG, const.STD_FLAG)
    geom_range      = kwargs.get(const.KEY_GEOM_RANGE, su.get_range())
    _kwargs         = kwargs.copy()
    stripe_starts, stripe_centers, stripe_ends = geom.get_stripe_limits(stripe_thickness, geom_range)
    _kwargs[const.KEY_CORR_FULL_INT] = calc_corr_full_integral(sky_pix, **_kwargs)
    _kwargs[const.KEY_STD_FULL]      = calc_std(sky_pix, None, **_kwargs)
    # Measure
    measure_func = func_dict[measure_flag]
    measure_results = np.zeros(len(geom_range))
    for i in range(len(stripe_centers)):
        start, end          = stripe_starts[i], stripe_ends[i]
        stripe, rest_of_sky  = geom.get_stripe(sky_pix, start, end)
        # Remove invalid pixels
        if stripe.get_visible_pixels_ratio() < min_pix_ratio or \
                rest_of_sky.get_visible_pixels_ratio() < min_pix_ratio:
            measure_results[i] = np.nan
            continue
        ang   = np.maximum(start, end)
        _kwargs.setdefault(const.KEY_MAX_VALID_ANG,
                           np.minimum(ang, 180 - ang))
        measure_results[i] = measure_func(stripe, rest_of_sky, **_kwargs)
    return measure_results