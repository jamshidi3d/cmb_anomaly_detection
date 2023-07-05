# 
# 
# This script checks modulation for a step function
#
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import cmb_anomaly_utils as cau
import read_cmb_maps_params as rmp
from cmb_anomaly_utils.dtypes import pix_data
from cmb_anomaly_utils import math_utils as mu

_inputs = rmp.get_inputs()

max_l = 20
_inputs['nside'] = 64
_inputs['sampling_start'] = 0
_inputs['sampling_stop'] = 180
_inputs['measure_flag'] = cau.const.MEAN_FLAG
_inputs['geom_flag'] = cau.const.STRIP_FLAG
_inputs['nsamples'] = 1 + 180

# modify sampling range to extrapolate curves
_inputs['sampling_range'] = cau.stat_utils.get_sampling_range(**_inputs)
sampling_range = _inputs['sampling_range']

_pos = cau.map_reader.read_pos(_inputs['nside'])

_filter = (np.cos(30 * np.pi / 180) < _pos[:,2]) * (_pos[:,2] < np.cos(20 * np.pi/180))
_data   = np.zeros(len(_pos))
_data[_filter] = 1

step_pix_data       = cau.dtypes.pix_data(_data, np.copy(_pos))
_measure_results    = cau.measure.get_strip_anomaly(step_pix_data , **_inputs)
step_a_l            = mu.get_all_legendre_modulation(sampling_range * np.pi / 180, _measure_results, max_l)

# store data
np.savetxt('./output/step_a_l.txt', step_a_l)