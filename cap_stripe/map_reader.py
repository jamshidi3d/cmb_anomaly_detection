import numpy as np
import healpy as hp
# from astropy.io import fits

from .dtypes import pix_data, run_parameters
from . import coords
from . import const


def read_mask(fpath, nside):
    mask = hp.read_map(fpath)
    mask = hp.ud_grade(mask, nside_out=nside)
    mask = np.logical_not(mask)
    # swapping ON and OFF, because sky_mask is true in masked areas and false in data area
    mask = np.array([not off_pix for off_pix in mask])
    return mask

def read_param(fpath, nside, field):
    map = hp.read_map(fpath, field = field, nest=True)
    map = hp.ud_grade(map, nside_out=nside, order_in='NESTED')
    map = hp.reorder(map, inp='NESTED', out='RING')
    return map * 10**6

def read_temp(fpath, nside):
    '''returns inpainted temprature in mu.K units'''
    return read_param(fpath, nside, 5)

def read_q(fpath, nside):
    '''returns inpainted Q_stokes in mu.K units'''
    return read_param(fpath, nside, 6)

def read_u(fpath, nside):
    '''returns inpainted U_stokes in mu.K units'''
    return read_param(fpath, nside, 7)

def read_p_stregth(fpath, nside):
    _u = read_u(fpath, nside)
    _q = read_q(fpath, nside)
    return np.sqrt(_u**2 + _q**2)

def read_e_mode(fpath, nside):
    pass

def read_b_mode(fpath, nside):
    pass

def read_pos(nside = 64, pole_lat = 0, pole_lon = 0):
    npix     = np.arange(12 * nside **2)
    lon, lat = hp.pix2ang(nside, npix, lonlat = True)
    pos = coords.convert_polar_to_xyz(lat, lon)
    pos = coords.rotate_pole_to_north(pos, pole_lat, pole_lon)
    return pos


func_dict = {
    const.U : read_u,
    const.T : read_temp,
    const.P : read_p_stregth,
    const.E_MODE : read_e_mode,
    const.B_MODE : read_b_mode,
}
def get_data_pix(data_fpath, mask_fpath, params:run_parameters):
    read_data = func_dict[params.observable_flag]
    _data = read_data(data_fpath, params.nside)
    _pos = read_pos(params.nside, params.pole_lat, params.pole_lon)
    _mask = read_mask(mask_fpath) if params.is_masked else None
    return pix_data(_data, _pos, _mask)
