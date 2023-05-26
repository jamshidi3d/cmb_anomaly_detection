import numpy as np
import json

import cmb_anomaly_utils as cau
from cmb_anomaly_utils.dtypes import pix_data
from cmb_anomaly_utils import math_utils as mu

cmb_fpath           = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
mask_fpath          = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
input_params_fpath  = './input/run_parameters.json'
sims_path           = './input/commander_sims/'

json_params_file =  open(input_params_fpath,'r')
inputs = json.loads(json_params_file.read())

max_l = 6
max_sim_num = 1
inputs['nside'] = 64

cmb_a_l = []

# modify sampling range to extrapolate curves
inputs['sampling_range'] = cau.stat_utils.get_sampling_range(**inputs)
sampling_range = inputs['sampling_range']
ext_nsamples = 180/(np.max(sampling_range) - np.min(sampling_range)) * len(sampling_range)
ext_range = np.linspace(0, 180, int(ext_nsamples) )

# take legendre expansion of cmb first
cmb_pix_data        = cau.map_reader.get_data_pix_from_cmb(cmb_fpath, mask_fpath, **inputs)
cmb_measure_results = cau.measure.get_stripe_anomaly(cmb_pix_data , **inputs)
ext_curve           = mu.extrapolate_curve(sampling_range, cmb_measure_results, ext_range)
# convert degree to radians to compute legendre coeffs
print(ext_range * np.pi / 180)
cmb_a_l = mu.get_legendre_coefficients(ext_range * np.pi / 180, ext_curve, max_l)


sims_a_l = []
sim_pos = cau.map_reader.read_pos(inputs['nside'])

for i in range(max_sim_num):
    try:
        sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', i)
    except:
        print("simulation number {:05} is currupted!".format(i))
        continue
    sim_pix_data        = cau.dtypes.pix_data(sim_temp, np.copy(sim_pos))
    sim_pix_data.add_multipole_modulation(cmb_a_l)
    sim_measure_results = cau.measure.get_stripe_anomaly(sim_pix_data , **inputs)
    ext_curve           = mu.extrapolate_curve(sampling_range, sim_measure_results, ext_range)
    _a_l                = mu.get_legendre_coefficients(ext_range * np.pi / 180, ext_curve, max_l)
    sims_a_l.append(_a_l)
# convert to numpy array
sims_a_l = np.array(sims_a_l)

print("cmb_a_l: \n",cmb_a_l)
print("sim_a_l: \n",np.mean(sims_a_l, axis=0))