import numpy as np

from .dtypes import pix_data

from . import const, stat_utils as su, math_utils as mu, coords

# global value to be used in measures
default_range = np.arange(0, 180, 181)


def calc_corr_full_integral(sky_pix:pix_data, **kwargs):
    '''-> keyword arguments:\n
    ndata_chunks - nmeasure_samples - measure_range'''
    full_int = 1
    if kwargs['measure_flag'] in (const.CORR_FLAG, ):
        fullsky_corr    = su.parallel_correlation(sky_pix, **kwargs)
        measure_range   = kwargs.get('measure_range', default_range)
        full_int        = mu.integrate_curve(measure_range, fullsky_corr ** 2)
    return full_int

def calc_cap_measure_in_all_dir(cmb_pd: pix_data, dir_lat_arr, dir_lon_arr, **kwargs):
    ndir            = len(dir_lat_arr)
    nsamples        = kwargs.get('ngeom_samples', 181)
    all_dir_measure = np.zeros((ndir, nsamples))
    pix_pos         = np.copy(cmb_pd.pos)
    for i in range(ndir):
        print(f"{i}/{ndir - 1} \r", end="")
        cmb_pd.pos = coords.rotate_pole_to_north(pix_pos, dir_lat_arr[i], dir_lon_arr[i])
        _result = get_cap_measure(cmb_pd, **kwargs)
        all_dir_measure[i] = _result
    return all_dir_measure

#------------ Measures ------------
def calc_dcorr2(patch1:pix_data, patch2:pix_data, **kwargs):
    '''-> keyword arguments: \n
    max_valid_ang - cutoff_ratio -\n
    ndata_chunks - measure_range - nmeasure_samples'''
    max_valid_ang   = kwargs.get('max_valid_ang', 0)
    cutoff_ratio    = kwargs.get('cutoff_ratio', 2 / 3)
    measure_range   = kwargs.get('measure_range', default_range)
    nsamples        = kwargs.get('nmeasure_samples', len(measure_range))
    tctt        = su.parallel_correlation(patch1, **kwargs)
    bctt        = su.parallel_correlation(patch2, **kwargs)
    max_index   = int(cutoff_ratio * 2 * max_valid_ang / 180 * nsamples)
    if len(measure_range[:max_index]) == 0:
        return 0
    return mu.integrate_curve(measure_range[:max_index],
                              (tctt[:max_index] - bctt[:max_index])**2)

def calc_corr(patch1:pix_data, patch2:pix_data, **kwargs):
    '''-> keyword arguments: \n
    full_integral - nmeasure_samples\n
    ndata_chunks - measure_range - max_valid_ang - cutoff_ratio'''
    f_int           = kwargs.get('full_integral', 1)
    measure_range   = kwargs.get('measure_range', default_range)
    cutoff_ratio    = kwargs.get('cutoff_ratio', 2 / 3)
    max_valid_ang   = kwargs.get('max_valid_ang', 0)
    nsamples        = kwargs.get('nmeasure_samples', len(measure_range))
    max_index   = int(cutoff_ratio * 2 * max_valid_ang / 180 * nsamples)
    tctt        = su.parallel_correlation(patch1, **kwargs)
    if len(measure_range[:max_index]) == 0:
        return -1
    geom_int    = mu.integrate_curve(measure_range[:max_index], tctt[:max_index] ** 2)
    return geom_int / f_int - 1

def calc_dstd2(patch1:pix_data, patch2:pix_data, **kwargs):
    return (su.std_pix_data(patch1) - su.std_pix_data(patch2))**2

def calc_std(patch1:pix_data, patch2:pix_data, **kwargs):
    return su.std_pix_data(patch1)

def calc_mean(patch1:pix_data, patch2:pix_data, **kwargs):
    return su.mean_pix_data(patch1)

func_dict = {
    const.D_CORR2_FLAG: calc_dcorr2,
    const.CORR_FLAG:    calc_corr,
    const.MEAN_FLAG:    calc_mean,
    const.STD_FLAG:     calc_std,
    const.D_STD2_FLAG:  calc_dstd2
}

#------------ Cap ------------
def get_cap_measure(sky_pix:pix_data, **kwargs):
    '''-> keyword arguments: \n
    measure_flag - nmeasure_samples - measure_range\n
    ngeom_samples - geom_range\n
    ndata_chunks'''
    measure_flag  = kwargs.get('measure_flag', const.STD_FLAG)
    geom_range    = kwargs.get('geom_range', default_range)
    kwargs['full_integral'] = calc_corr_full_integral(sky_pix, **kwargs)
    measure_func            = func_dict[measure_flag]
    measure_results         = np.zeros(len(geom_range))
    for i, ca in enumerate(geom_range):
        print("- Cap size: {} degrees\r".format(ca), end = "")
        top, bottom = sky_pix.get_top_bottom_caps(ca)
        kwargs['max_valid_ang'] = np.minimum(ca, 180 - ca)
        measure_results[i] = measure_func(top, bottom, **kwargs)
    print()
    return measure_results


#---------- Strip ----------
def get_strip_limits(strip_thickness, geom_range):
    def clamp_to_sphere_degree(value):
        return 180 / np.pi * np.arccos(np.clip(value, -1, 1))
    height          = 1 - np.cos(strip_thickness * np.pi / 180)
    strip_mid_locs  = np.cos(geom_range * np.pi / 180)
    # strip starts
    top_lim         = strip_mid_locs + height / 2
    strip_starts    = clamp_to_sphere_degree(top_lim)
    # strip ends
    bottom_lim      = strip_mid_locs - height / 2
    strip_ends      = clamp_to_sphere_degree(bottom_lim)
    return strip_starts, geom_range, strip_ends


def get_strip_measure(sky_pix:pix_data, **kwargs):
    '''keyword arguments: \n
    sampling_range - strip_thickness - measure_flag -\n
    nmeasure_samples - cutoff_ratio - ndata_chunks
    '''
    geom_range      = kwargs.get('geom_range', default_range)
    strip_thickness = kwargs.get('strip_thickness', 20)
    measure_flag    = kwargs.get('measure_flag', const.STD_FLAG)
    measure_range   = kwargs.get('measure_range', default_range)
    strip_starts, strip_centers, strip_ends = get_strip_limits(strip_thickness, geom_range)
    kwargs['full_integral'] = calc_corr_full_integral(sky_pix, **kwargs)
    
    # measure
    measure_func = func_dict[measure_flag]
    measure_results = np.zeros(len(measure_range))
    for i in range(len(strip_centers)):
        print("- Strip center: {} degrees".format(strip_centers[i])+" " * 20+"\r", end="")
        start = strip_starts[i]
        end   = strip_ends[i]
        strip, rest_of_sky = sky_pix.get_strip(start, end)
        ang   = np.maximum(start, end)
        kwargs['max_valid_ang'] = np.minimum(ang, 180 - ang)
        measure_results[i] = measure_func(strip, rest_of_sky, **kwargs)
    print()
    return measure_results