# 
# 
# This script:
#   1. calculates measures for strips of the CMB in the MostAnomalyDirection(MAD)
#   2. reads the legendre coefficients from that measure-function (which is a function of theta)
#   3. applies the modulation on 1000 simulations
#   4. calculates measures for stripes on this modulated simulations
#   5. reads legendre coefficients from modulated simulations
# this would be helpful to find errorbars of legendre coefficients 
# (plots are involved in plotting.ipynb)
#
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import cmb_anomaly_utils as cau
import read_cmb_maps_params as rmp
from cmb_anomaly_utils.dtypes import pix_data
from cmb_anomaly_utils import math_utils as mu

sims_path = './input/commander_sims/'

_inputs = rmp.get_inputs()

max_l = 10
max_sim_num = 1000
_inputs['nside'] = 64
_inputs['sampling_start'] = 0
_inputs['sampling_stop'] = 180
_inputs['measure_flag'] = cau.const.STD_FLAG

# modify sampling range to extrapolate curves
_inputs['sampling_range'] = cau.stat_utils.get_sampling_range(**_inputs)
sampling_range = _inputs['sampling_range']

# take legendre expansion of cmb first
cmb_pix_data        = rmp.get_cmb_pixdata(**_inputs)
cmb_measure_results = cau.measure.get_strip_anomaly(cmb_pix_data , **_inputs)

# convert degree to radians to compute legendre coeffs
cmb_a_l = mu.get_all_legendre_modulation(sampling_range * np.pi / 180, cmb_measure_results, max_l)
np.savetxt('./output/cmb_a_l.txt', cmb_a_l)

# change probe after modulation
_inputs['measure_flag'] = cau.const.STD_FLAG

sims_a_l        = np.zeros((max_sim_num, max_l + 1))
sims_results    = np.zeros((max_sim_num, len(sampling_range)))
sim_pos = cau.map_reader.read_pos(_inputs['nside'])

# modulate for each l separately
for nonzero_l in range(max_l + 1):
    print(nonzero_l)
    single_cmb_a_l = np.copy(cmb_a_l)
    single_cmb_a_l[:nonzero_l] = 0
    single_cmb_a_l[nonzero_l + 1:] = 0

    modulation_factor = cau.map_filler.create_legendre_modulation_factor(sim_pos, single_cmb_a_l)

    for sim_num in range(max_sim_num):
        print(f'sim number {sim_num:04} \r', end='')
        try:
            sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', sim_num)
        except:
            print("simulation number {:05} is currupted!".format(sim_num))
            continue
        sim_pix_data        = cau.dtypes.pix_data(sim_temp, np.copy(sim_pos))
        sim_pix_data.data  *= modulation_factor
        sim_measure_results = cau.measure.get_strip_anomaly(sim_pix_data , **_inputs)
        s_a_l = mu.get_single_legendre_modulation(sampling_range * np.pi / 180, sim_measure_results, nonzero_l)
        sims_a_l[sim_num, nonzero_l] = s_a_l
        sims_results[sim_num] = sim_measure_results
        # del sim_pix_data


# store data
np.savetxt('./output/sims_a_l.txt', sims_a_l)
np.savetxt('./output/sims_{}.txt'.format(_inputs['measure_flag']), sims_results)

# print("cmb_a_l: \n",cmb_a_l)
# print("sim_a_l: \n",np.mean(sims_a_l, axis=0))
# print("sim_a_l_std: \n",np.std(sims_a_l, axis=0))

# matplotlib.use('Agg')
# plt.plot(sampling_range, cmb_measure_results, '-k')
# plt.plot(ext_range, sim_ext_results, '-r')
# plt.savefig('./test2.pdf', transparent = True)
