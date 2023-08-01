#
#
# This script simply computes the measure for given run parameters and plots them
#
#

import numpy as np
import healpy as hp
import cmb_anomaly_utils as cau
import read_cmb_maps_params as rmp

_inputs = rmp.get_inputs()

# map params
_inputs['nside']            = 64
_inputs['is_masked']        = True
dir_nside                   = 16
do_use_mac                  = True
selected_size_for_dir       = 30
_inputs['pole_lat']         = 90
_inputs['pole_lon']         = 0
# measure params
_inputs['measure_start']    = 0
_inputs['measure_stop']     = 180
_inputs['nmeasure_samples'] = 181
_inputs['measure_flag']     = cau.const.D_STD2_FLAG
# geometry params
_inputs['geom_start']       = 0
_inputs['geom_stop']        = 180
_inputs['ngeom_samples']    = 181
_inputs['geom_flag']        = cau.const.STRIP_FLAG

rmp.print_inputs(_inputs)

_inputs['measure_range']     = cau.stat_utils.get_measure_range(**_inputs)
_inputs['geom_range']        = cau.stat_utils.get_geom_range(**_inputs)

cap_geom_range = cau.stat_utils.get_geom_range(geom_start = 10, geom_stop = 90, ngeom_samples = 17)

if do_use_mac:
    mask_txt = 'masked' if _inputs.get('is_masked') else 'inpainted'
    cmb_all_dir_cap_anom = np.loadtxt(f'./output/cmb_{mask_txt}_all_dir_cap_anom.txt')
    
    plat, plon = cau.direction.find_dir_using_mac(
                                        cmb_all_dir_cap_anom,
                                        special_cap_size = selected_size_for_dir,
                                        geom_range = cap_geom_range)
    print(plon, plat)
    _inputs['pole_lat'], _inputs['pole_lon'] = plat, plon

sky_pix = rmp.get_cmb_pixdata(**_inputs)

# Measure
geom_flag = _inputs['geom_flag']
get_measure = cau.measure.get_cap_measure if geom_flag == cau.const.CAP_FLAG \
    else cau.measure.get_strip_measure if geom_flag == cau.const.STRIP_FLAG else None

if get_measure == None:
    raise ValueError("Geometry flag is not set correctly")

measure_results = get_measure(sky_pix, **_inputs)

# Save data
cau.output.save_data_to_txt(measure_results, **_inputs)

# Plot
# fig = cau.output.get_plot_fig(measure_results, **inputs)
# cau.output.save_fig_to_pdf(fig, **inputs)