import cmb_anomaly_utils as cau
import json


cmb_fpath       = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
mask_fpath      = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
input_params_fpath    = './input/run_parameters.json'

json_inputs_file =  open(input_params_fpath,'r')
inputs = json.loads(json_inputs_file.read())
json_inputs_file.close()
# add sampling range
inputs['sampling_range'] = cau.stat_utils.get_sampling_range(**inputs)


print("=> Observable: {} | Measure: {} | Nside: {} | MapType: {} | Geometry: {}"\
        .format(inputs['observable_flag'],
                inputs['measure_flag'],
                inputs['nside'],
                "MASKED" if inputs['is_masked'] else "INPAINTED",
                inputs['geom_flag']
        )
    )

sky_pix = cau.map_reader.get_data_pix_from_cmb(cmb_fpath, mask_fpath, **inputs)

# Measure
if inputs['geom_flag'] == cau.const.CAP_FLAG:
    measure_results = cau.measure.get_cap_anomaly(sky_pix, **inputs)
elif inputs['geom_flag'] == cau.const.STRIPE_FLAG:
    measure_results = cau.measure.get_stripe_anomaly(sky_pix, **inputs)
else:
    raise ValueError("Geometry flag is not set correctly")

# Save data
cau.output.save_data_to_txt(measure_results, **inputs)

# Plot
fig = cau.output.get_plot_fig(measure_results, **inputs)
cau.output.save_fig_to_pdf(fig, **inputs)