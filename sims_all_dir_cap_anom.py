#
#
# This script computes stat_measure on 1000 simulations for different caps of
# different sizes and directions
#
#
import numpy as np
import healpy as hp
import concurrent.futures

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import read_cmb_maps_params as rmp
import cmb_anomaly_utils as cau
from cmb_anomaly_utils.dtypes import pix_data

sims_path = './input/commander_sims/'

_inputs = rmp.get_inputs()

dir_nside                   = 16
'''nside for different pole directions'''
max_sim_num                 = 1000

_inputs['nside']            = 64
_inputs['is_masked']        = True
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

# Direction for simulation poles
npix             = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)

# Positions and mask
print("finding direction for simulations")
sim_pos     = cau.map_reader.read_pos(_inputs['nside'])
sim_mask    = rmp.get_mask(**_inputs)

mask_txt    = "masked" if _inputs['is_masked'] else "inpainted"

for sim_num in range(max_sim_num):
    print(f'sim number {sim_num:04}')
    try:
        sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', sim_num)
    except:
        print("simulation number {:05} is currupted!".format(sim_num))
        continue
    sim_temp         *= 10**6
    sim_pix_data     = cau.dtypes.pix_data(sim_temp, np.copy(sim_pos), sim_mask)
    cmb_all_dir_anom = cau.measure.calc_measure_in_all_dir(sim_pix_data, dir_lat, dir_lon, **_inputs)
    
    fname   = f"sim{sim_num:04}_all_dir_cap_anom_{mask_txt}.txt"
    np.savetxt(f"./output/" + fname, cmb_all_dir_anom)
