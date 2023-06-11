import numpy as np
import healpy as hp
import json

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import cmb_anomaly_utils as cau
from cmb_anomaly_utils.dtypes import pix_data

cmb_fpath           = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
mask_fpath          = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
input_params_fpath  = './input/run_parameters.json'

json_params_file =  open(input_params_fpath,'r')
_inputs = json.loads(json_params_file.read())

_inputs['pole_lat'] = 0
_inputs['pole_lon'] = 0
_inputs['geom_flag'] = cau.const.CAP_FLAG
_inputs['nsamples']         = _inputs['sampling_stop'] - _inputs['sampling_start'] + 1
_inputs['sampling_range']   = cau.stat_utils.get_sampling_range(**_inputs)

dir_nside = 16
'''nside for different pole directions'''
npix     = 12 * dir_nside **2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)
all_dir_anomaly = []

cmb_pd : pix_data = cau.map_reader.get_data_pix_from_cmb(cmb_fpath, mask_fpath, **_inputs)
pix_pos = np.copy(cmb_pd.pos)
for i in range(npix):
    print(f"{i}/{npix} \r", end="")
    cmb_pd.pos = cau.coords.rotate_pole_to_north(pix_pos, dir_lat[i], dir_lon[i])
    _result = cau.measure.get_cap_anomaly(cmb_pd, **_inputs)
    all_dir_anomaly.append(_result)

all_dir_anomaly = np.array(all_dir_anomaly)
np.savetxt("./output/direction_data.txt", all_dir_anomaly)
