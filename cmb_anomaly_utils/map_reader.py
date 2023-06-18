import numpy as np
import healpy as hp
# from astropy.io import fits

from .dtypes import pix_data
from . import coords, const


fname_dict = {const.T: 'temp'}
def get_sim_attr(sims_path, observable = const.T, num = 0):
    fname = sims_path + "sim{:05}_".format(num) + fname_dict[observable] +'.txt'
    attr_list = np.loadtxt(fname)
    return attr_list

def read_mask(fpath, nside):
    mask = hp.read_map(fpath)
    mask = hp.ud_grade(mask, nside_out=nside)
    return mask

def read_attr(fpath, nside, field):
    map = hp.read_map(fpath, field = field, nest=True)
    map = hp.ud_grade(map, nside_out=nside, order_in='NESTED')
    map = hp.reorder(map, inp='NESTED', out='RING')
    return map * 10**6

def read_temp(fpath, nside):
    '''returns inpainted temprature in mu.K units'''
    return read_attr(fpath, nside, 5)

def read_q(fpath, nside):
    '''returns inpainted Q_stokes in mu.K units'''
    return read_attr(fpath, nside, 6)

def read_u(fpath, nside):
    '''returns inpainted U_stokes in mu.K units'''
    return read_attr(fpath, nside, 7)

def read_p_stregth(fpath, nside):
    _u = read_u(fpath, nside)
    _q = read_q(fpath, nside)
    return np.sqrt(_u**2 + _q**2)

def read_e_mode(fpath, nside):
    pass

def read_b_mode(fpath, nside):
    pass

def read_pos(nside = 64, pole_lat = 90, pole_lon = 0):
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
def get_data_pix_from_cmb(data_fpath, mask_fpath, **kwargs):
    pole_lat, pole_lon, nside, observable_flag, is_masked = \
        kwargs['pole_lat'], kwargs['pole_lon'], kwargs['nside'], kwargs['observable_flag'], kwargs['is_masked']
    read_data = func_dict[observable_flag]
    _data = read_data(data_fpath, nside)
    _pos = read_pos(nside, pole_lat, pole_lon)
    _mask = read_mask(mask_fpath, nside) if is_masked else None
    return pix_data(_data, _pos, _mask)
