import os
import numpy as np
import cmb_anomaly_utils as cau

max_sim_num                     = 1000
base_path                       = "./output/"
run_inputs  = cau.run_utils.RunInputs()
run_inputs.mask_fpath           = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
run_inputs.cmb_fpath            = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
run_inputs.cmb_dir_anom_fpath   = "./output/cmb_inpainted_all_dir_cap_anom.txt"
run_inputs.sims_path            = "./input/commander_sims/"
run_inputs.sims_dir_anom_path   = "./output/sims_inpainted_all_dir_cap_anom_5deg/"
run_inputs.geom_flag            = cau.const.CAP_FLAG
run_inputs.measure_flag         = cau.const.STD_FLAG
run_inputs.nside                = 64
run_inputs.dir_nside            = 16
run_inputs.geom_start           = 0
run_inputs.geom_stop            = 180
run_inputs.delta_geom_samples   = 5
run_inputs.strip_thickness      = 20


output_path = cau.output.ensure_path(base_path)
map_gen = cau.run_utils.MapGenerator(**run_inputs.to_kwargs())


np.savetxt(output_path + "inpainted_all_dir_geom_range.txt", run_inputs.geom_range)

all_dir_lat, all_dir_lon = cau.coords.get_healpix_latlon(run_inputs.dir_nside)

print("- Computing CMB all dir measures")
cmb_map     = map_gen.create_cmb_map()
_results    = cau.measure.calc_measure_in_all_dir(cmb_map,
                                                  all_dir_lat,
                                                  all_dir_lon,
                                                  **run_inputs.to_kwargs())
np.savetxt(run_inputs.cmb_dir_anom_fpath,
           _results)

print("- Computing Simulations all dir measures")
def print_sim_num(sim_num):
    print("{:03}\r".format(sim_num), end="")

for sim_num in range(max_sim_num):
    print_sim_num(sim_num)
    sim_map     = map_gen.create_sim_map_from_txt(sim_num)
    _results    = cau.measure.calc_measure_in_all_dir(sim_map,
                                                      all_dir_lat,
                                                      all_dir_lon,
                                                      **run_inputs.to_kwargs())
    fpath    = run_inputs.sims_dir_anom_path + "sim{:03}_all_dir_cap_anom.txt".format(sim_num)
    np.savetxt(fpath, _results)
