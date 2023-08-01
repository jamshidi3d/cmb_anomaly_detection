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

_inputs = rmp.get_inputs()

dir_nside                   = 16
_inputs['nside']            = 64
_inputs['is_masked']        = False
_inputs['pole_lat']         = 90
_inputs['pole_lon']         = 0
_inputs['measure_flag']     = cau.const.STD_FLAG
_inputs['measure_start']    = 0
_inputs['measure_stop']     = 180
_inputs['nmeasure_samples'] = 181
_inputs['geom_flag']        = cau.const.CAP_FLAG
_inputs['geom_start']       = 10
_inputs['geom_stop']        = 90
dtheta                      = 5
_inputs['ngeom_samples']    = 1 + int((_inputs['geom_stop'] - _inputs['geom_start']) / dtheta)
_inputs['measure_range']    = cau.stat_utils.get_measure_range(**_inputs)
_inputs['geom_range']       = cau.stat_utils.get_geom_range(**_inputs)

# all directions that we look for
npix     = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)

# direction for cmb
mask_txt = 'masked' if _inputs.get('is_masked') else 'inpainted'
print("finding direction for cmb")
cmb_pd : pix_data = rmp.get_cmb_pixdata(**_inputs)
cmb_all_dir_anom  = cau.measure.calc_measure_in_all_dir(cmb_pd, dir_lat, dir_lon, **_inputs)
np.savetxt(f"./output/cmb_{mask_txt}_all_dir_cap_anom.txt", cmb_all_dir_anom)
