import numpy as np
import healpy as hp
import os

from . import coords, const, output
from .dtypes import PixMap

def read_txt_attr(fpath):
    attr_arr  = np.loadtxt(fpath)
    return attr_arr * 10**6

def read_fits_attr(fpath, nside, field):
    map = hp.read_map(fpath, field = field)
    map = hp.ud_grade(map, nside_out=nside)
    return map * 10**6

def read_fits_mask(fpath, nside):
    mask = hp.read_map(fpath)
    mask = hp.ud_grade(mask, nside_out=nside)
    mask = np.logical_not(mask)
    return mask

def read_fits_temp(fpath, nside):
    '''returns inpainted temprature in mu.K units'''
    return read_fits_attr(fpath, nside, 5)

def read_fits_q(fpath, nside):
    '''returns inpainted Q_stokes in mu.K units'''
    return read_fits_attr(fpath, nside, 6)

def read_fits_u(fpath, nside):
    '''returns inpainted U_stokes in mu.K units'''
    return read_fits_attr(fpath, nside, 7)

def read_fits_p_stregth(fpath, nside):
    _u = read_fits_u(fpath, nside)
    _q = read_fits_q(fpath, nside)
    return np.sqrt(_u**2 + _q**2)

def read_fits_e_mode(fpath, nside):
    pass

def read_fits_b_mode(fpath, nside):
    pass
    

fits_func_dict = {
    const.OBS_U :      read_fits_u,
    const.OBS_T :      read_fits_temp,
    const.OBS_P :      read_fits_p_stregth,
    const.OBS_E_MODE : read_fits_e_mode,
    const.OBS_B_MODE : read_fits_b_mode,
}

# ------- Precalculated measures -------
def get_fnames_in_dir(path):
    return [fname for fname in os.listdir(path) \
            if os.path.isfile(os.path.join(path, fname))]

# These are not clean functions! need to be rewritten
def check_precalc_name(fname:str, dir_cap_size, cmb_or_sim_key:str, measure_or_a_l_key:str):
    dir_geom_check  = str(int(dir_cap_size)) + "cap" in fname.lower()
    data_check      = cmb_or_sim_key.lower() in fname.lower()
    res_type_check  = measure_or_a_l_key.lower() in fname.lower()
    return data_check and dir_geom_check and res_type_check

def read_geom_range_precalc(base_path, **kwargs):
    '''Provides and easy access to geom range file\n
    use cases are in computing a_l and plotting etc.'''
    path = output.get_output_path(base_path, **kwargs)
    if not output.does_path_exist(path):
        print("measure is not computed yet!")
        return None
    fnames = [fname for fname in os.listdir(path) if 'range' in fname]
    return np.loadtxt(path + fnames[0])

def read_cmb_precalc(base_path, dir_cap_size, **kwargs):
    path = output.get_output_path(base_path, **kwargs)
    if not output.does_path_exist(path):
        print("Measure is not computed yet!")
        return None
    fnames  =  [f_n for f_n in os.listdir(path) \
               if check_precalc_name(f_n, dir_cap_size, 'cmb', 'measure')]
    if len(fnames) == 0:
        print("Measure is not computed yet!")
        return None
    return np.loadtxt(path + fnames[0])

def read_cmb_a_l(base_path, dir_cap_size, **kwargs):
    path = output.get_output_path(base_path, **kwargs)
    if not output.does_path_exist(path):
        print("Measure is not computed yet!")
        return None
    fnames  =  [f_n for f_n in os.listdir(path) \
               if check_precalc_name(f_n, dir_cap_size, 'cmb', 'a_l')]
    if len(fnames) == 0:
        print("Measure is not computed yet!")
        return None
    return np.loadtxt(path + fnames[0])

# Note that these are generators not functions so 
# they must be used in a loop or so
def iter_read_sims_precalc(base_path, dir_cap_size, **kwargs):
    path = output.get_output_path(base_path, **kwargs)
    if not output.does_path_exist(path):
        print("Measure is not computed yet!")
        yield None
        return
    fnames =  [f_n for f_n in os.listdir(path) \
               if check_precalc_name(f_n, dir_cap_size, 'sim', 'measure')]
    for f_n in fnames:
        _result = np.loadtxt(path + f_n)
        yield _result

def iter_read_sims_a_l(base_path, dir_cap_size, **kwargs):
    path = output.get_output_path(base_path, **kwargs)
    if not output.does_path_exist(path):
        print("Measure is not computed yet!")
        yield None
        return
    fnames =  [f_n for f_n in os.listdir(path) \
               if check_precalc_name(f_n, dir_cap_size, 'sim', 'a_l')]
    for f_n in fnames:
        _result = np.loadtxt(path + f_n)
        yield _result
    


