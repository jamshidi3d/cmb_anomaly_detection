#
#
# This script plots computed measures for the CMB (for caps of different sizes and directions)
#
#
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import cmb_anomaly_utils as cau
import read_cmb_maps_params as rmp

_inputs = rmp.get_inputs()

img_dpi                     = 150
dir_nside                   = 16
# map params
_inputs['nside']            = 64
_inputs['is_masked']        = False
_inputs['measure_flag']     = cau.const.STD_FLAG
_inputs['geom_flag']        = cau.const.CAP_FLAG
_inputs['geom_start']       = 10
_inputs['geom_stop']        = 90
dtheta                      = 5
_inputs['ngeom_samples']    = 1 + int((_inputs['geom_stop'] - _inputs['geom_start']) / dtheta)
_inputs['geom_range']       = cau.stat_utils.get_geom_range(**_inputs)
geom_range = _inputs['geom_range']

masked_txt = 'masked' if _inputs.get('is_masked') else 'inpainted'
all_dir_cap_anom  = np.loadtxt(f"./output/cmb_{masked_txt}_all_dir_cap_anom.txt")

'''nside for different pole directions'''
npix             = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)

def flatten_low_values_with_std(arr, nsigma = 1, flat_val = 0):
    copy_arr = np.copy(arr)
    top_val = np.nanmax(arr) - nsigma * np.std(arr)
    _filter = arr < top_val
    copy_arr[_filter] = flat_val
    return copy_arr

def flatten_low_values_with_precent(arr, top_percent, flat_val = 0):
    copy_arr = np.copy(arr)
    top_val = np.nanmax(arr) - top_percent / 100 * (np.nanmax(arr) - np.min(arr))
    _filter = arr < top_val
    copy_arr[_filter] = flat_val
    return copy_arr

def colorize_special_pix(arr, index, factor = 0.25, from_min = True):
    copy_arr = np.copy(arr)
    diff = factor * (np.nanmax(arr) - np.min(arr))
    copy_arr[index] = np.min(arr) - diff if from_min else np.nanmax(arr) + diff
    return copy_arr


# Stores the anomaly for selecting the direction
dir_pref = np.zeros(npix)
akrami_pix_index = hp.ang2pix(dir_nside, np.deg2rad(110), np.deg2rad(221))

for cap_index, cap_size in enumerate(geom_range):
    fig, ax = plt.subplots()
    plt.axes(ax)
    anom_arr = all_dir_cap_anom[:, cap_index]
    dir_index = np.nanargmax(anom_arr)
    _title = r"$cap size = {}^\circ , lat|_{{max}} = {:0.1f}, lon|_{{max}} = {:0.1f}$".format(
        cap_size,
        dir_lat[dir_index],
        dir_lon[dir_index]
        )
    f_anom_arr = anom_arr #flatten_low_values_with_std(anom_arr, nsigma = 1, flat_val = 0)
    plot_f_anom_arr = colorize_special_pix(f_anom_arr, akrami_pix_index, factor = 0.4, from_min = True)
    hp.mollview(plot_f_anom_arr, title = _title, xsize = 1600, hold=True)
    fig.savefig(f"./output/dir_{int(cap_size)}.jpg", transparent=True, dpi=img_dpi)
    plt.close(fig)
    dir_pref += f_anom_arr

# dir_pref = flatten_low_values_with_std(dir_pref, nsigma = 1, flat_val = np.min(dir_pref))
dir_pref = flatten_low_values_with_precent(dir_pref, top_percent= 10, flat_val = np.min(dir_pref))
dir_pref = colorize_special_pix(dir_pref, akrami_pix_index, factor = 0.4, from_min = True)
fig, ax = plt.subplots()
plt.axes(ax)
_title = "Direction of Anomaly"
# np.savetxt("./output/dir_preference.txt", dir_pref)
hp.mollview(dir_pref, title = _title, xsize = 1600, hold=True)
fig.savefig("./output/dir_preference.png", transparent=True, dpi=img_dpi)

