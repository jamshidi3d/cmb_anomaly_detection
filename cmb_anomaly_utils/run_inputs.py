
from . import stat_utils
from . import const

class RunInputs:
    def __init__(self, **kwargs) -> None:
        self.observable         = kwargs.get('observable', const.T)
        self.sims_fpath         = kwargs.get('sims_fpath', None)
        self.cmb_fpath          = kwargs.get('cmb_fpath', None)
        self.mask_fpath         = kwargs.get('mask_fpath', None)
        self.nside              = kwargs.get('nside', 64)
        self.is_masked          = kwargs.get('is_masked', False)
        self.tpcf_mode          = kwargs.get("tpcf_mode", const.TT_2PCF)
        self.pole_lon           = kwargs.get("pole_lat", 90)
        self.pole_lat           = kwargs.get("pole_lon", 0)
        self.min_pix_ratio      = kwargs.get("min_pix_ratio", 1)
        self.measure_flag       = kwargs.get("measure_flag", const.NORM_STD_FLAG)
        self.geom_flag          = kwargs.get("geom_flag", const.CAP_FLAG)
        self._measure_start     = kwargs.get("measure_start", 0)
        self._measure_stop      = kwargs.get("measure_stop", 180)
        self._dmeasure_samples  = kwargs.get("dmeasure_samples", 1)
        self.cutoff_ratio       = kwargs.get("cutoff_ratio", 2/3)
        self.ndata_chunks       = kwargs.get("ndata_chunks", 4)
        self._geom_start        = kwargs.get("geom_start", 0)
        self._geom_stop         = kwargs.get("geom_stop", 180)
        self._dgeom_samples     = kwargs.get("dgeom_samples", 5)
        self.strip_thickness    = kwargs.get("strip_thickness", 20)
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
    
    # ------ geom range ------
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
    def dgeom_samples(self):
        return self._dgeom_samples
    
    @dgeom_samples.setter
    def dgeom_samples(self, value):
        self._dgeom_samples = value
        self.set_geom_range()

    # ------ measure range ------
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
    def dmeasure_samples(self):
        return self._dmeasure_samples
    
    @dmeasure_samples.setter
    def dmeasure_samples(self, value):
        self._dmeasure_samples = value
        self.set_measure_range()

    # ------ conversion ------
    def as_kwargs(self):
        kwargs = {}
        kwargs.setdefault("observable", self.observable)      
        kwargs.setdefault("sims_fpath", self.sims_fpath)
        kwargs.setdefault("cmb_fpath", self.cmb_fpath)
        kwargs.setdefault("mask_fpath", self.mask_fpath)
        kwargs.setdefault("nside", self.nside)
        kwargs.setdefault("is_masked", self.is_masked)
        kwargs.setdefault("tpcf_mode", self.tpcf_mode)
        kwargs.setdefault("pole_lon", self.pole_lon)
        kwargs.setdefault("pole_lat", self.pole_lat)  
        kwargs.setdefault("min_pix_ratio", self.min_pix_ratio)
        kwargs.setdefault("measure_flag", self.measure_flag)
        kwargs.setdefault("geom_flag", self.geom_flag)
        kwargs.setdefault("measure_start", self.measure_start)
        kwargs.setdefault("measure_stop", self.measure_stop) 
        kwargs.setdefault("dmeasure_samples", self.dmeasure_samples)
        kwargs.setdefault("cutoff_ratio", self.cutoff_ratio)
        kwargs.setdefault("ndata_chunks", self.ndata_chunks)   
        kwargs.setdefault("geom_start", self.geom_start)   
        kwargs.setdefault("geom_stop", self.geom_stop)   
        kwargs.setdefault("dgeom_samples", self.dgeom_samples) 
        kwargs.setdefault("strip_thickness", self.strip_thickness)
        return kwargs