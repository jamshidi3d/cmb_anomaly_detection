#
#
# This script computes measures, for caps in different sizes and 
# different directions in CMB
#
#

import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import read_cmb_maps_params as rmp
import cmb_anomaly_utils as cau
from cmb_anomaly_utils.dtypes import pix_data

sims_path = './input/commander_sims/'

_inputs = rmp.get_inputs()

dir_nside                   = 16
max_sim_num                 = 1000
'''nside for different pole directions'''
_inputs['nside']            = 64
_inputs['pole_lat']         = 90
_inputs['pole_lon']         = 0
_inputs['measure_flag']     = cau.const.STD_FLAG
_inputs['geom_flag']        = cau.const.CAP_FLAG
_inputs['sampling_start']   = 10
_inputs['sampling_stop']    = 90
_inputs['nsamples']         = 1 + int((_inputs['sampling_stop'] - _inputs['sampling_start']) / 2)
_inputs['sampling_range']   = cau.stat_utils.get_sampling_range(**_inputs)

print(_inputs['sampling_range'])

# all directions that we look for
npix     = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)

# direction for cmb
print("finding direction for cmb")
cmb_pd : pix_data = rmp.get_cmb_pixdata(**_inputs)
cmb_all_dir_anom = cau.measure.calc_cap_anomaly_in_all_dir(cmb_pd, dir_lat, dir_lon, **_inputs)
np.savetxt("./output/cmb_all_dir_anomaly.txt", cmb_all_dir_anom)
