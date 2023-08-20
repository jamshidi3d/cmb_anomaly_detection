import numpy as np
import healpy as hp
import os

from . import coords, const
from .dtypes import PixMap


def read_txt_attr(fpath):
    attr_arr  = np.loadtxt(fpath)
    return attr_arr * 10**6

def read_fits_mask(fpath, nside):
    mask = hp.read_map(fpath)
    mask = hp.ud_grade(mask, nside_out=nside)
    mask = np.logical_not(mask)
    return mask

def read_fits_attr(fpath, nside, field):
    map = hp.read_map(fpath, field = field, nest=True)
    map = hp.ud_grade(map, nside_out=nside, order_in='NESTED')
    map = hp.reorder(map, inp='NESTED', out='RING')
    return map * 10**6

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
    const.U :      read_fits_u,
    const.T :      read_fits_temp,
    const.P :      read_fits_p_stregth,
    const.E_MODE : read_fits_e_mode,
    const.B_MODE : read_fits_b_mode,
}


class MapManager:
    '''This class generates PixMap(s) that are used for cap and strip computation\n
    - kwargs: \n
    observable - nside - is_masked \n
    sims_fpath - cmb_fpath - mask_fpath'''
    def __init__(self, **kwargs):
        self.observable     = kwargs.get('observable', const.T)
        self.sims_fpath     = kwargs.get('sims_fpath', None)
        self.sims_fnames    = os.listdir(self.sims_fpath)
        self.cmb_fpath      = kwargs.get('cmb_fpath', None)
        self.mask_fpath     = kwargs.get('mask_fpath', None)
        self.nside          = kwargs.get('nside', 64)
        self.is_masked      = kwargs.get('is_masked', False)
        self.mask           = read_fits_mask(self.mask_fpath, self.nside) if self.is_masked else None
        self.pos            = coords.get_healpix_xyz(self.nside)
    
    def create_cmb_map(self):
        read_func   = fits_func_dict[self.observable]
        _data       = read_func(self.cmb_fpath, self.nside)
        return PixMap(_data, self.pos, self.mask)

    def create_sim_map_from_txt(self, num):
        fpathname  = self.sims_fpath + self.sims_fnames[num]
        _data      = read_txt_attr(fpathname)
        return PixMap(_data, self.pos, self.mask)
    
    # def create_sim_map_from_fits(num):
    #     pass


