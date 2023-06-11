import numpy as np
import healpy as hp
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import cmb_anomaly_utils as cau

input_params_fpath  = './input/run_parameters.json'

json_params_file =  open(input_params_fpath,'r')
_inputs = json.loads(json_params_file.read())

_inputs['nside'] = 64
_inputs['geom_flag'] = cau.const.CAP_FLAG
_inputs['nsamples']         = _inputs['sampling_stop'] - _inputs['sampling_start'] + 1
_inputs['sampling_range']   = cau.stat_utils.get_sampling_range(**_inputs)


all_dir_anomaly = np.loadtxt("./output/direction_data.txt")

dir_nside = 16
'''nside for different pole directions'''
npix     = 12 * dir_nside **2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)


fig, ax = plt.subplots()
plt.axes(ax)

cap_size = 90
cap_index = cau.stat_utils.get_nearest_index(_inputs['sampling_range'], cap_size)
dir_index = np.argmax(all_dir_anomaly[:, cap_index])
_title = r"$cap size = {}^\circ , lat|_{{max}} = {:0.1f}, lon|_{{max}} = {:0.1f}$".format(cap_size, dir_lat[dir_index], dir_lon[dir_index])
hp.mollview(all_dir_anomaly[:, cap_index], title = _title, hold=True)
fig.savefig(f"./output/dir_{cap_size}.jpg", transparent=True)