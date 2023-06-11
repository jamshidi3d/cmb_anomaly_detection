import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib

import cmb_anomaly_utils as cau
from cmb_anomaly_utils.dtypes import pix_data
from cmb_anomaly_utils import math_utils as mu

cmb_fpath           = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
mask_fpath          = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
input_params_fpath  = './input/run_parameters.json'
sims_path           = './input/commander_sims/'

json_params_file =  open(input_params_fpath,'r')
_inputs = json.loads(json_params_file.read())

max_l = 10
max_sim_num = 1000
_inputs['nside'] = 64

# modify sampling range to extrapolate curves
_inputs['sampling_range'] = cau.stat_utils.get_sampling_range(**_inputs)
sampling_range = _inputs['sampling_range']
ext_range = cau.stat_utils.get_extended_range(sampling_range, 0, 180)

# take legendre expansion of cmb first
cmb_pix_data        = cau.map_reader.get_data_pix_from_cmb(cmb_fpath, mask_fpath, **_inputs)
cmb_measure_results = cau.measure.get_stripe_anomaly(cmb_pix_data , **_inputs)
# cmb_ext_results     = mu.extrapolate_curve(sampling_range, cmb_measure_results, ext_range, 'clamped', 0)

# change probe after modulation
_inputs['measure_flag'] = cau.const.CORR_FLAG

# convert degree to radians to compute legendre coeffs
cmb_a_l = mu.get_legendre_modulation(ext_range * np.pi / 180, cmb_measure_results, max_l)

sims_a_l = []
sims_results = []
sim_pos = cau.map_reader.read_pos(_inputs['nside'])
modulation_factor = cau.map_filler.create_legendre_modulation_factor(sim_pos, cmb_a_l)

_inputs = _inputs.copy()
_inputs['measure_flag'] = cau.const.MEAN_FLAG
for i in range(max_sim_num):
    print(f'sim number {i:04}')
    try:
        sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', i)
    except:
        print("simulation number {:05} is currupted!".format(i))
        continue
    sim_temp *= 10**6
    sim_pix_data        = cau.dtypes.pix_data(sim_temp, np.copy(sim_pos))
    sim_pix_data.data  *= modulation_factor
    sim_measure_results = cau.measure.get_stripe_anomaly(sim_pix_data , **_inputs)
    sim_ext_results     = mu.extrapolate_curve(sampling_range, sim_measure_results, ext_range)
    sims_results.append(sim_ext_results)
    # _a_l                = mu.get_legendre_modulation(ext_range * np.pi / 180, sim_ext_results, max_l)
    # sims_a_l.append(_a_l)

# convert to numpy array
sims_results = np.array(sims_results)
np.savetxt('./output/sims_{}.txt'.format(input['measure_flag']), sims_results)


# convert to numpy array
# sims_a_l = np.array(sims_a_l)
# np.savetxt('./output/sims_a_l.txt', sims_a_l)

# print("cmb_a_l: \n",cmb_a_l)
# print("sim_a_l: \n",np.mean(sims_a_l, axis=0))
# print("sim_a_l_std: \n",np.std(sims_a_l, axis=0))

# matplotlib.use('Agg')
# plt.plot(sampling_range, cmb_measure_results, '-k')
# plt.plot(ext_range, sim_ext_results, '-r')
# plt.savefig('./test2.pdf', transparent = True)
