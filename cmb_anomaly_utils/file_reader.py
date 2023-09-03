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

def check_precalc_name(fname, cmb_or_sim:bool, dir_cap_size):
    data_sign = "cmb" if cmb_or_sim else "sim"
    geom_sign = str(int(dir_cap_size)) + "cap"
    return geom_sign in fname and \
            data_sign in fname and \
                "measure" in fname

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
               if check_precalc_name(f_n, True, dir_cap_size)]
    if len(fnames) == 0:
        print("Measure is not computed yet!")
        return None
    return np.loadtxt(path + fnames[0])

# Note that this is a generator not a function so 
# it must be used in a loop or so
def iter_read_sims_precalc(base_path, dir_cap_size, **kwargs):
    path = output.get_output_path(base_path, **kwargs)
    if not output.does_path_exist(path):
        print("Measure is not computed yet!")
        yield None
        return
    fnames =  [f_n for f_n in os.listdir(path) \
               if check_precalc_name(f_n, False, dir_cap_size)]
    for f_n in fnames:
        _result = np.loadtxt(path + f_n)
        yield _result

    


