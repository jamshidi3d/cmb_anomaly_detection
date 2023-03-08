import cmb_anomaly_cpu as cac

cmb_fpath       = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
mask_fpath      = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
input_params_fpath    = './input/run_parameters.json'

input_params = cac.dtypes.run_parameters.create_from_json(input_params_fpath)

print("=> Observable: {} | Measure: {} | Nside: {} | MapType: {} | Geometry: {}"\
        .format(input_params.observable_flag,
                input_params.measure_flag,
                input_params.nside,
                "MASKED" if input_params.is_masked else "INPAINTED",
                input_params.geom_flag
        )
    )

sky_pix = cac.map_reader.get_data_pix(cmb_fpath, mask_fpath, input_params)

# Measure
if input_params.geom_flag == cac.const.CAP_FLAG:
    measure_results = cac.measure.get_cap_anomaly(sky_pix, input_params)
elif input_params.geom_flag == cac.const.STRIPE_FLAG:
    measure_results = cac.measure.get_stripe_anomaly(sky_pix, input_params)
else:
    raise ValueError("Geometry flag is not set correctly")

# Save data
cac.output.save_data_to_txt(input_params, measure_results)

# Plot
fig = cac.output.get_plot_fig(input_params, measure_results)
cac.output.save_fig_to_pdf(input_params, fig)