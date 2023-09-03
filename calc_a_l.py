import os
import numpy as np
import cmb_anomaly_utils as cau

base_path                       = "./output/measure_results_mac_dir/"
max_sim_num                     = 1000
max_l                           = 10
run_inputs  = cau.run_utils.RunInputs()
run_inputs.mask_fpath           = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
run_inputs.cmb_fpath            = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
run_inputs.cmb_dir_anom_fpath   = "./output/cmb_inpainted_all_dir_cap_anom.txt"
run_inputs.sims_path            = "./input/commander_sims/"
run_inputs.sims_dir_anom_path   = "./output/sims_inpainted_all_dir_cap_anom_5deg/"
run_inputs.geom_flag            = cau.const.STRIP_FLAG
run_inputs.measure_flag         = cau.const.NORM_STD_FLAG
run_inputs.nside                = 64
run_inputs.dir_nside            = 16
run_inputs.geom_start           = 0
run_inputs.geom_stop            = 180
run_inputs.delta_geom_samples   = 5
run_inputs.strip_thickness      = 20
run_inputs.pole_lat             = -10
run_inputs.pole_lon             = 221

dir_cap_sizes   = cau.stat_utils.get_range(20, 70, 10)

output_path = cau.output.get_output_path(base_path, **run_inputs.to_kwargs())

geom_range  = cau.file_reader.read_geom_range_precalc(base_path, **run_inputs.to_kwargs())
theta       = np.radians(geom_range)

def print_sim_num(sim_num):
    print("{:03}\r".format(sim_num), end="")

for dcs in dir_cap_sizes:
    print(f"- Computing {int(dcs)}_cap measures")
    # CMB
    cmb_measure = cau.file_reader.read_cmb_precalc(base_path,
                                                   dcs,
                                                   **run_inputs.to_kwargs())
    a_l = cau.math_utils.get_all_legendre_modulation(theta, cmb_measure, max_l)
    np.savetxt(output_path + f"cmb_{int(dcs)}cap_a_l.txt", a_l)
    # Simulations
    iter_sim_measure = cau.file_reader.iter_read_sims_precalc(base_path,
                                                              dcs,
                                                              **run_inputs.to_kwargs())
    for sim_num, sim_measure in enumerate(iter_sim_measure):
        print_sim_num(sim_num)
        a_l = cau.math_utils.get_all_legendre_modulation(theta, sim_measure, max_l)
        fpath   = output_path + "sim{:03}_{}cap_a_l.txt".format(sim_num, int(dcs))
        np.savetxt(fpath, a_l)
print("- Done -")