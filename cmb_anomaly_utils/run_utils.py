import os
import numpy as np

from . import const, coords, file_reader as freader, stat_utils
from .dtypes import PixMap

class MapGenerator:
    '''This class generates PixMap(s) that are used for cap and strip computation\n
    - kwargs: \n
    observable - nside - is_masked \n
    sims_path - cmb_fpath - mask_fpath'''
    def __init__(self, **kwargs):
        self.observable  = kwargs.get(const.KEY_OBSERVABLE, const.OBS_T)
        self.nside       = kwargs.get(const.KEY_NSIDE, 64)
        self.is_masked   = kwargs.get(const.KEY_IS_MASKED, False)
        self.sims_path   = kwargs.get(const.KEY_SIMS_PATH, None)
        self.sims_fnames = os.listdir(self.sims_path)
        self.cmb_fpath   = kwargs.get(const.KEY_CMB_FPATH, None)
        self.mask_fpath  = kwargs.get(const.KEY_MASK_FPATH, None)
        self.mask        = freader.read_fits_mask(self.mask_fpath, self.nside) if self.is_masked else None
        self.pos         = coords.get_healpix_xyz(self.nside)
    
    def create_cmb_map(self):
        read_func   = freader.fits_func_dict[self.observable]
        _data       = read_func(self.cmb_fpath, self.nside)
        return PixMap(_data, self.pos, self.mask)

    def create_sim_map_from_txt(self, num):
        fpathname  = self.sims_path + self.sims_fnames[num]
        _data      = freader.read_txt_attr(fpathname)
        if self.observable == const.OBS_T:
            _data -= np.nanmean(_data)
        return PixMap(_data, self.pos, self.mask)
    
    # def create_sim_map_from_fits(num):
    #     pass


class RunInputs:
    def __init__(self, **kwargs) -> None:
        self.observable         = kwargs.get(const.KEY_OBSERVABLE,       const.OBS_T)
        self.sims_path          = kwargs.get(const.KEY_SIMS_PATH,        None)
        self.cmb_fpath          = kwargs.get(const.KEY_CMB_FPATH,        None)
        self.mask_fpath         = kwargs.get(const.KEY_MASK_FPATH,       None)
        self.nside              = kwargs.get(const.KEY_NSIDE,            64)
        self.dir_nside          = kwargs.get(const.KEY_DIRNSIDE,         16)
        self.sims_dir_anom_path = kwargs.get(const.KEY_SIMS_ANOM_PATH,   None)
        self.cmb_dir_anom_fpath = kwargs.get(const.KEY_CMB_ANOM_FPATH,   None)
        self.is_masked          = kwargs.get(const.KEY_IS_MASKED,        False)
        self.tpcf_mode          = kwargs.get(const.KEY_TPCF_MODE,        const.TPCF_TT)
        self.pole_lon           = kwargs.get(const.KEY_POLE_LAT,         90)
        self.pole_lat           = kwargs.get(const.KEY_POLE_LON,         0)
        self.min_pix_ratio      = kwargs.get(const.KEY_MIN_PIX_RATIO,    1)
        self.max_valid_ang      = kwargs.get(const.KEY_MAX_VALID_ANG,    0)
        self.corr_full_int      = kwargs.get(const.KEY_CORR_FULL_INT,    1)
        self.std_full           = kwargs.get(const.KEY_STD_FULL,         1)
        self.measure_flag       = kwargs.get(const.KEY_MEASURE_FLAG,     const.STD_FLAG)
        self.geom_flag          = kwargs.get(const.KEY_GEOM_FLAG,        const.CAP_FLAG)
        self._measure_start     = kwargs.get(const.KEY_MEASURE_START,    0)
        self._measure_stop      = kwargs.get(const.KEY_MEASURE_STOP,     180)
        self._dmeasure_samples  = kwargs.get(const.KEY_DMEASURE_SAMPLES, 1)
        self.cutoff_ratio       = kwargs.get(const.KEY_CUTOFF_RATIO,     2/3)
        self.ndata_chunks       = kwargs.get(const.KEY_NDATA_CHUNKS,     4)
        self._geom_start        = kwargs.get(const.KEY_GEOM_START,       0)
        self._geom_stop         = kwargs.get(const.KEY_GEOM_STOP,        180)
        self._dgeom_samples     = kwargs.get(const.KEY_DGEOM_SAMPLES,    5)
        self.strip_thickness    = kwargs.get(const.KEY_STRIP_THICKNESS,  20)
        self.set_geom_range()
        self.set_measure_range()

    def get_default():
        return RunInputs()

    def set_geom_range(self):
        self.geom_range     = stat_utils.get_range( self._geom_start,
                                                    self._geom_stop, 
                                                    self._dgeom_samples)

    def set_measure_range(self):
        self.measure_range  = stat_utils.get_range( self._measure_start,
                                                    self._measure_stop, 
                                                    self._dmeasure_samples)
    
    # ------ Geom range ------
    @property
    def geom_start(self):
        return self._geom_start

    @geom_start.setter
    def geom_start(self, value):
        self._geom_start = value
        self.set_geom_range()
    
    @property
    def geom_stop(self):
        return self._geom_stop

    @geom_stop.setter
    def geom_stop(self, value):
        self._geom_stop = value
        self.set_geom_range()

    @property
    def delta_geom_samples(self):
        return self._dgeom_samples
    
    @delta_geom_samples.setter
    def delta_geom_samples(self, value):
        self._dgeom_samples = value
        self.set_geom_range()

    # ------ Measure range ------
    @property
    def measure_start(self):
        return self._measure_start
    
    @measure_start.setter
    def measure_start(self, value):
        self._measure_start = value
        self.set_measure_range()

    @property
    def measure_stop(self):
        return self._measure_stop
    
    @measure_stop.setter
    def measure_stop(self, value):
        self._measure_stop = value
        self.set_measure_range()

    @property
    def delta_measure_samples(self):
        return self._dmeasure_samples
    
    @delta_measure_samples.setter
    def delta_measure_samples(self, value):
        self._dmeasure_samples = value
        self.set_measure_range()

    # ------ Utility ------
    @property
    def masked_txt(self):
        return 'masked' if self.is_masked else 'inpainted'

    # ------ Conversion ------
    def to_kwargs(self):
        kwargs = {}
        kwargs.setdefault(const.KEY_OBSERVABLE,       self.observable)
        kwargs.setdefault(const.KEY_SIMS_PATH,        self.sims_path)
        kwargs.setdefault(const.KEY_CMB_FPATH,        self.cmb_fpath)
        kwargs.setdefault(const.KEY_MASK_FPATH,       self.mask_fpath)
        kwargs.setdefault(const.KEY_SIMS_ANOM_PATH,   self.sims_dir_anom_path)
        kwargs.setdefault(const.KEY_CMB_ANOM_FPATH,   self.cmb_dir_anom_fpath)
        kwargs.setdefault(const.KEY_NSIDE,            self.nside)
        kwargs.setdefault(const.KEY_DIRNSIDE,         self.dir_nside)
        kwargs.setdefault(const.KEY_IS_MASKED,        self.is_masked)
        kwargs.setdefault(const.KEY_TPCF_MODE,        self.tpcf_mode)
        kwargs.setdefault(const.KEY_POLE_LON,         self.pole_lon)
        kwargs.setdefault(const.KEY_POLE_LAT,         self.pole_lat)
        kwargs.setdefault(const.KEY_MIN_PIX_RATIO,    self.min_pix_ratio)
        kwargs.setdefault(const.KEY_MAX_VALID_ANG,    self.max_valid_ang)
        kwargs.setdefault(const.KEY_CORR_FULL_INT,    self.corr_full_int)
        kwargs.setdefault(const.KEY_STD_FULL,         self.std_full)
        kwargs.setdefault(const.KEY_MEASURE_FLAG,     self.measure_flag)
        kwargs.setdefault(const.KEY_GEOM_FLAG,        self.geom_flag)
        kwargs.setdefault(const.KEY_GEOM_START,       self.geom_start)
        kwargs.setdefault(const.KEY_GEOM_STOP,        self.geom_stop)
        kwargs.setdefault(const.KEY_DGEOM_SAMPLES,    self.delta_geom_samples)
        kwargs.setdefault(const.KEY_GEOM_RANGE,       self.geom_range)
        kwargs.setdefault(const.KEY_MEASURE_RANGE,    self.measure_range)
        kwargs.setdefault(const.KEY_MEASURE_START,    self.measure_start)
        kwargs.setdefault(const.KEY_MEASURE_STOP,     self.measure_stop)
        kwargs.setdefault(const.KEY_DMEASURE_SAMPLES, self.delta_measure_samples)
        kwargs.setdefault(const.KEY_CUTOFF_RATIO,     self.cutoff_ratio)
        kwargs.setdefault(const.KEY_NDATA_CHUNKS,     self.ndata_chunks)
        kwargs.setdefault(const.KEY_STRIP_THICKNESS,  self.strip_thickness)
        return kwargs