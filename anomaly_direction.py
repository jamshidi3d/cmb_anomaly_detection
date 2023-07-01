import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import read_maps_params as rmp
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

def calc_anomaly_in_all_dir(cmb_pd: pix_data, dir_lat_arr, dir_lon_arr, **_inputs):
    npix        = len(dir_lat_arr)
    nsamples    = _inputs['nsamples']
    all_dir_anomaly = np.zeros((npix, nsamples))
    pix_pos     = np.copy(cmb_pd.pos)
    for i in range(npix):
        print(f"{i}/{npix - 1} \r", end="")
        cmb_pd.pos = cau.coords.rotate_pole_to_north(pix_pos, dir_lat_arr[i], dir_lon_arr[i])
        _result = cau.measure.get_cap_anomaly(cmb_pd, **_inputs)
        all_dir_anomaly[i] = _result
    return all_dir_anomaly

# all directions that we look for
npix     = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)

# direction for cmb
print("finding direction for cmb")
cmb_pd : pix_data = rmp.get_cmb_pixdata(**_inputs)
cmb_all_dir_anom = calc_anomaly_in_all_dir(cmb_pd, dir_lat, dir_lon, **_inputs)
np.savetxt("./output/cmb_all_dir_anomaly.txt", cmb_all_dir_anom)

# direction for simulations
# print("finding direction for simulations")
# sim_pos = cau.map_reader.read_pos(_inputs['nside'])
# for sim_num in range(800, max_sim_num):
#     print(f'sim number {sim_num:04}')
#     try:
#         sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', sim_num)
#     except:
#         print("simulation number {:05} is currupted!".format(sim_num))
#         continue
#     sim_temp         *= 10**6
#     sim_pix_data     = cau.dtypes.pix_data(sim_temp, np.copy(sim_pos))
#     cmb_all_dir_anom = calc_anomaly_in_all_dir(cmb_pd, dir_lat, dir_lon, **_inputs)
#     np.savetxt(f"./output/sim{sim_num:04}_all_dir_anomaly.txt", cmb_all_dir_anom)
