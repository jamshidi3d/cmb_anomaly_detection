# 
# 
# This script:
#   1. calculates measures for strips of the CMB in the MostAnomalousDirection(MAD)
#   2. reads the legendre coefficients from that measure-function (which is a function of theta)
#   3. applies the modulation on 1000 simulations
#   4. calculates measures for stripes on this modulated simulations
#   5. reads legendre coefficients from modulated simulations
# this would be used to find errorbars of legendre coefficients
# there are two ways for applying modulation on simulation:
#       [for reading errorbars]
#       1.applying l by l (from cmb to simluation, single l for each time)
#       [for reading measure values]
#       2.applying all l's at once 
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

# possible modulation modes
ALL_L_MODE = "ALL_L_MODE"
SINGLE_L_MODE = "SINGLE_L_MODE"

# running mode
mod_mode = ALL_L_MODE

# run parameters
max_l       = 10
max_sim_num = 1000

_inputs['nside']            = 64
_inputs['geom_start']       = 0
_inputs['geom_stop']        = 180
_inputs['nsamples']         = 1 + 180
_inputs['measure_flag']     = cau.const.STD_FLAG
_inputs['geom_flag']        = cau.const.STRIP_FLAG
_inputs['geom_range']       = cau.stat_utils.get_geom_range(**_inputs)
# This is the measure that we take from simulations
measure_to_look             = cau.const.STD_FLAG

geom_range = _inputs['geom_range']


def get_modulated_measure_from_sim(sim_num, modulation_factor):
    print(f'sim number {sim_num:04} \r', end='')
    try:
        sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', sim_num)
    except:
        print("simulation number {:05} is currupted!".format(sim_num))
        return None
    sim_pix_data        = cau.dtypes.pix_data(sim_temp, np.copy(sim_pos))
    sim_pix_data.data  *= modulation_factor
    sim_measure_results = cau.measure.get_strip_measure(sim_pix_data , **_inputs)
    return sim_measure_results


# take legendre expansion of cmb first
cmb_pix_data        = rmp.get_cmb_pixdata(**_inputs)
cmb_measure_results = cau.measure.get_strip_measure(cmb_pix_data , **_inputs)

# convert degree to radians to compute legendre coeffs
cmb_a_l = mu.get_all_legendre_modulation(geom_range * np.pi / 180, cmb_measure_results, max_l)
np.savetxt('./output/cmb_a_l.txt', cmb_a_l)

# change probe after modulation
_inputs['measure_flag'] = measure_to_look

sims_a_l        = np.zeros((max_sim_num, max_l + 1))
sims_results    = np.zeros((max_sim_num, len(geom_range)))
sim_pos         = cau.map_reader.read_pos(_inputs['nside'])

# modulate each l separately
if mod_mode == SINGLE_L_MODE:
    for nonzero_l in range(max_l + 1):
        print(nonzero_l)
        single_cmb_a_l = np.copy(cmb_a_l)
        single_cmb_a_l[:nonzero_l] = 0
        single_cmb_a_l[nonzero_l + 1:] = 0
        
        # single l factor
        modulation_factor = cau.map_filler.create_legendre_modulation_factor(sim_pos, single_cmb_a_l)
        for sim_num in range(max_sim_num):
            sim_measure_results = get_modulated_measure_from_sim(sim_num, modulation_factor)
            if sim_measure_results is None:
                continue
            s_a_l = mu.get_single_legendre_modulation(geom_range * np.pi / 180,
                                                    sim_measure_results,
                                                    nonzero_l)
            sims_a_l[sim_num, nonzero_l] = s_a_l
        print()
    np.savetxt('./output/sims_modulated_a_l.txt', sims_a_l)


# all l's at the same time
if mod_mode == ALL_L_MODE:
    modulation_factor = cau.map_filler.create_legendre_modulation_factor(sim_pos, cmb_a_l)
    for sim_num in range(max_sim_num):
        sim_measure_results = get_modulated_measure_from_sim(sim_num, modulation_factor)
        if sim_measure_results is None:
            continue
        sims_results[sim_num] = sim_measure_results
    print()

    np.savetxt('./output/sims_modulated_{}.txt'.format(_inputs['measure_flag'].lower()), sims_results)
