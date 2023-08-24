import numpy as np
import healpy as hp
import os

from . import coords, const
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



