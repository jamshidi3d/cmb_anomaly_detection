import cap_stripe as cac
import json

cmb_fpath       = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
mask_fpath      = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
input_params_fpath    = './input/run_parameters.json'

json_params_file =  open(input_params_fpath,'r')
input_params = json.loads(json_params_file.read())
json_params_file.close()
# add sampling range
input_params['sampling_range'] = cac.utils.get_sampling_range(**input_params)


print("=> Observable: {} | Measure: {} | Nside: {} | MapType: {} | Geometry: {}"\
        .format(input_params['observable_flag'],
                input_params['measure_flag'],
                input_params['nside'],
                "MASKED" if input_params['is_masked'] else "INPAINTED",
                input_params['geom_flag']
        )
    )

sky_pix = cac.map_reader.get_data_pix_from_cmb(cmb_fpath, mask_fpath, **input_params)

# Measure
if input_params['geom_flag'] == cac.const.CAP_FLAG:
    measure_results = cac.measure.get_cap_anomaly(sky_pix, **input_params)
elif input_params['geom_flag'] == cac.const.STRIPE_FLAG:
    measure_results = cac.measure.get_stripe_anomaly(sky_pix, **input_params)
else:
    raise ValueError("Geometry flag is not set correctly")

# Save data
cac.output.save_data_to_txt(measure_results, **input_params)

# Plot
fig = cac.output.get_plot_fig(measure_results, **input_params)
cac.output.save_fig_to_pdf(fig, **input_params)