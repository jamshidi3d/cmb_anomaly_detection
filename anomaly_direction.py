import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import read_maps_params as rmp
import cmb_anomaly_utils as cau
from cmb_anomaly_utils.dtypes import pix_data

_inputs = rmp.get_inputs()

dir_nside                   = 16
'''nside for different pole directions'''
_inputs['nside']            = 64
_inputs['pole_lat']         = 90
_inputs['pole_lon']         = 0
_inputs['measure_flag']     = cau.const.STD_FLAG
_inputs['geom_flag']        = cau.const.CAP_FLAG
_inputs['nsamples']         = _inputs['sampling_stop'] - _inputs['sampling_start'] + 1
_inputs['sampling_range']   = cau.stat_utils.get_sampling_range(**_inputs)

npix     = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)
all_dir_anomaly = []

cmb_pd : pix_data = rmp.get_cmb_pixdata(**_inputs)
pix_pos = np.copy(cmb_pd.pos)
for i in range(npix):
    print(f"{i}/{npix} \r", end="")
    cmb_pd.pos = cau.coords.rotate_pole_to_north(pix_pos, dir_lat[i], dir_lon[i])
    _result = cau.measure.get_cap_anomaly(cmb_pd, **_inputs)
    all_dir_anomaly.append(_result)

all_dir_anomaly = np.array(all_dir_anomaly)
np.savetxt("./output/direction_data.txt", all_dir_anomaly)
