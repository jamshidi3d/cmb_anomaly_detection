#
#
# This script computes the internal legendre coefficients of measure, computed
# on strips in the MostAnomalyDirection(MAD).
# It is useful for computing p-values 
#
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import cmb_anomaly_utils as cau, read_cmb_maps_params as rmp

sims_anom_path  = './output/sims_all_dir_anomaly_2deg/'
sims_path       = './input/commander_sims/'

_inputs = rmp.get_inputs()

dir_nside                   = 16
max_sim_num                 = 1000
max_l                       = 10
selected_size_for_dir       = 30
_inputs['nside']            = 64
_inputs['pole_lat']         = 90
_inputs['pole_lon']         = 0
_inputs['measure_flag']     = cau.const.STD_FLAG
_inputs['geom_flag']        = cau.const.STRIP_FLAG
_inputs['sampling_start']   = 0
_inputs['sampling_stop']    = 180
_inputs['nsamples']         = 1 + 30
_inputs['sampling_range']   = cau.stat_utils.get_sampling_range(**_inputs)

theta = _inputs['sampling_range'] * np.pi / 180

cap_nsamples    = 1 + int((90 - 10) / 2)
cap_size_range  = cau.stat_utils.get_sampling_range(**{'sampling_start': 0, 'sampling_stop': 90, 'nsamples': cap_nsamples})

# all directions that we look for
npix     = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)
dir_lon *= 180 / np.pi
dir_lat *= 180 / np.pi

file_list   = os.listdir(sims_anom_path)
sim_pos     = cau.map_reader.read_pos(_inputs['nside'])
sims_a_l    = np.zeros((max_sim_num, max_l + 1))
cap_index   = cau.stat_utils.find_nearest_index(cap_size_range, selected_size_for_dir)
# find mad & read legendre coefs
for sim_num, fname in enumerate(file_list):
    print(f"{sim_num}/{max_sim_num - 1}\r", end='')
    fpath = sims_anom_path + fname
    all_measure_results = np.loadtxt(fpath)
    # maximum anomaly direction
    mad_i    =  np.argmax(all_measure_results[cap_index, :])
    _inputs['pole_lon'], _inputs['pole_lat'] =\
        dir_lon[mad_i], dir_lat[mad_i]
    mad_aligned_pos = cau.coords.rotate_pole_to_north(sim_pos, dir_lat[mad_i], dir_lon[mad_i])
    # create data
    try:
        sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', sim_num)
    except:
        print("simulation number {:05} is currupted!".format(sim_num))
        continue
    sim_temp -= np.mean(sim_temp)
    sim_pix_data = cau.dtypes.pix_data(sim_temp, mad_aligned_pos)
    _result = cau.measure.get_strip_anomaly(sim_pix_data, **_inputs)
    sims_a_l[sim_num] = cau.math_utils.get_all_legendre_coefs(theta, _result, max_l)

np.savetxt('./output/sims_internal_a_l.txt', sims_a_l)