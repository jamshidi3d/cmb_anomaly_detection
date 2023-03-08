import numpy as np
import json

from .coords import angle_to_z
from . import const


class pix_data:
    def __init__(self, data:np.ndarray, pos:np.ndarray, mask:np.ndarray = None):
        self.data = data
        self.pos = pos
        self.mask = mask
    
    def copy(self):
        return pix_data(np.copy(self.data), np.copy(self.pos), np.copy(self.mask))

    def get_filtered(self, filter) -> "pix_data":
        if self.mask is None:
            return pix_data(self.data[filter], self.pos[filter])
        _mask = self.mask[filter]
        return pix_data(self.data[filter][_mask], self.pos[filter][_mask])

    def get_top_bottom_caps(self, cap_angle):
        z_border = angle_to_z(cap_angle)
        # top cap
        top_filter = self.pos[:, 2] > z_border
        top_cap = self.get_filtered(top_filter)
        # bottom cap
        bottom_filter = self.pos[:, 2] <= z_border
        bottom_cap = self.get_filtered(bottom_filter)
        print(top_cap.data[-1])
        return top_cap, bottom_cap

    def get_stripe(self, start_angle, stop_angle):
        '''returns a stripe between given angles and the rest of sky\n
        start and stop angles have to be in degrees'''
        z_start = angle_to_z(start_angle)
        z_stop  = angle_to_z(stop_angle)
        stripe_filter = (z_start >= self.pos[:, 2]) * (self.pos[:, 2] >= z_stop)
        stripe = self.get_filtered(stripe_filter)
        rest_of_sky_filter = np.array([not i for i in stripe_filter])
        rest_of_sky = self.get_filtered(rest_of_sky_filter)
        return stripe, rest_of_sky



class run_parameters:
    def __init__(self,
                observable_flag = const.T,
                nside = None,
                is_masked = None,
                pole_lat = None,
                pole_lon = None,
                measure_flag = None,
                geom_flag = None,
                nsamples = None,
                nblocks = 4,
                stripe_thickness = None,
                dtheta = None,
                cacr = None,
                sampling_start = None,
                sampling_stop = None,):
        self.observable_flag    = observable_flag
        self.nside              = nside
        self.is_masked          = is_masked
        self.pole_lat           = pole_lat
        self.pole_lon           = pole_lon
        self.measure_flag       = measure_flag
        self.geom_flag          = geom_flag
        self.nsamples           = nsamples
        self.nblocks            = nblocks
        self.stripe_thickness   = stripe_thickness
        '''also top cap size'''
        self.dtheta             = dtheta
        self.cacr               = cacr
        '''correlation angle cutoff ratio'''
        self._sampling_start     = sampling_start
        self._sampling_stop      = sampling_stop
        self.redefine_sampling_range()
    
    @property
    def sampling_start(self):
        return self._sampling_start
    
    @sampling_start.setter
    def sampling_start(self, value):
        self.sampling_start = value
        self.redefine_sampling_range()
    
    @property
    def sampling_stop(self):
        return self._sampling_stop
    
    @sampling_stop.setter
    def sampling_stop(self, value):
        self.sampling_stop = value
        self.redefine_sampling_range()

    def redefine_sampling_range(self):
        self.sampling_range = np.arange(self.sampling_start, self.sampling_stop, self.dtheta)

    @staticmethod
    def create_from_json(fpath):
        with open(fpath,'r') as json_params_file:
            input_params = json.loads(json_params_file.read())
        _inputs = run_parameters(
        observable_flag     = input_params['observable_flag'],
        nside               = input_params['nside'],
        pole_lat            = input_params['pole_lat'],
        pole_lon            = input_params['pole_lon'],
        is_masked           = input_params['is_masked'],
        measure_flag        = input_params['measure_flag'],
        geom_flag           = input_params['geom_flag'],
        nsamples            = input_params['nsamples'],
        nblocks             = input_params['nblocks'],
        stripe_thickness    = input_params['stripe_thickness'],
        dtheta              = input_params['dtheta'],
        cacr                = input_params['corr_ang_cutoff_ratio'],
        sampling_start      = input_params['sampling_start'],
        sampling_stop       = input_params['sampling_stop'],
        )
        return _inputs
    
    def create_json(self, fpath):
        _inputs = {
            'observable_flag': self.observable_flag,
            'nside': self.nside,
            'pole_lat':self.pole_lat,
            'pole_lon':self.pole_lon,
            'is_masked' :        self.is_masked,
            'measure_flag': self.measure_flag,
            'geom_flag':self.geom_flag,
            'nsamples':self.nsamples,
            'nblocks':self.nblocks,
            'stripe_thickness':self.stripe_thickness,
            'dtheta':self.dtheta,
            'corr_ang_cutoff_ratio':self.cacr,
            'sampling_start':self.sampling_start,
            'sampling_stop':self.sampling_stop,
        }
        with open(fpath,'w') as file:
            file.write(json.dumps(_inputs, indent=4))