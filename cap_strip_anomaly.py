#
#
# This script simply computes the measure for given run parameters and plots them
#
#

import cmb_anomaly_utils as cau
import read_cmb_maps_params as rmp

inputs = rmp.get_inputs()
# map params
inputs['nside']             = 64
inputs['is_masked']         = False
# measure params
inputs['measure_start']     = 0
inputs['measure_stop']      = 180
inputs['nmeasure_samples']  = 181
inputs['measure_flag']      = cau.const.D_STD2_FLAG
# geometry params
inputs['geom_start']        = 0
inputs['geom_stop']         = 180
inputs['ngeom_samples']     = 181
inputs['geom_flag']         = cau.const.STRIP_FLAG

rmp.print_inputs(inputs)

inputs['measure_range']     = cau.stat_utils.get_measure_range(**inputs)
inputs['geom_range']        = cau.stat_utils.get_geom_range(**inputs)


sky_pix = rmp.get_cmb_pixdata(**inputs)

# Measure
if inputs['geom_flag'] == cau.const.CAP_FLAG:
    measure_results = cau.measure.get_cap_measure(sky_pix, **inputs)
elif inputs['geom_flag'] == cau.const.STRIP_FLAG:
    measure_results = cau.measure.get_strip_measure(sky_pix, **inputs)
else:
    raise ValueError("Geometry flag is not set correctly")

# Save data
cau.output.save_data_to_txt(measure_results, **inputs)

# Plot
# fig = cau.output.get_plot_fig(measure_results, **inputs)
# cau.output.save_fig_to_pdf(fig, **inputs)