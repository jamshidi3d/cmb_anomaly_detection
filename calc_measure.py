import os
import numpy as np
import cmb_anomaly_utils as cau

base_path                       = "./output/measure_results_mac_dir/"
max_sim_num                     = 1000

run_inputs  = cau.run_utils.RunInputs()
run_inputs.mask_fpath           = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
run_inputs.cmb_fpath            = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
run_inputs.cmb_dir_anom_fpath   = "./output/cmb_inpainted_all_dir_cap_anom.txt"
run_inputs.sims_path            = "./input/commander_sims/"
run_inputs.sims_dir_anom_path   = "./output/sims_inpainted_all_dir_cap_anom_5deg/"
run_inputs.geom_flag            = cau.const.STRIPE_FLAG
run_inputs.measure_flag         = cau.const.STD_FLAG
run_inputs.nside                = 64
run_inputs.dir_nside            = 16
run_inputs.geom_start           = 0
run_inputs.geom_stop            = 180
run_inputs.delta_geom_samples   = 5
run_inputs.stripe_thickness     = 20
run_inputs.pole_lat             = -20
run_inputs.pole_lon             = 221

dir_cap_sizes       = cau.stat_utils.get_range(20, 70, 10)
pre_dir_cap_sizes   = cau.stat_utils.get_range(10, 90, 5)

output_path = cau.output.ensure_output_path(base_path, **run_inputs.to_kwargs())

map_gen = cau.run_utils.MapGenerator(**run_inputs.to_kwargs())

all_dir_lat, all_dir_lon = cau.coords.get_healpix_latlon(run_inputs.dir_nside)

ndir_caps, ngeom = len(dir_cap_sizes), len(run_inputs.geom_range)
_results = np.zeros((ndir_caps, ngeom))

print("- Computing CMB measures")

cmb_map     = map_gen.create_cmb_map()
all_dir_cap_anom = np.loadtxt(run_inputs.cmb_dir_anom_fpath)

np.savetxt(output_path + "geom_range.txt", run_inputs.geom_range)

for i, dcs in enumerate(dir_cap_sizes):
    plat, plon = cau.direction.align_pole_to_mac(cmb_map,
                                                 all_dir_cap_anom,
                                                 dcs,
                                                 pre_dir_cap_sizes,
                                                 all_dir_lat,
                                                 all_dir_lon)
    _results[i] = cau.measure.get_measure(cmb_map, **run_inputs.to_kwargs())
    fpath       = output_path + f"cmb_{int(dcs)}cap_measure.txt"
    np.savetxt(fpath, _results[i], header=f"lat = {plat}, lon = {plon}")

# Save accumulative result
np.savetxt( output_path + "cmb_acc_result.txt", np.sum(_results, axis=0))


print("- Computing Simulations measures")

def print_sim_num(sim_num):
    print("{:03}\r".format(sim_num), end="")

dir_anom_path   = run_inputs.sims_dir_anom_path
dir_anom_fnames = os.listdir(dir_anom_path)

for sim_num in range(max_sim_num):
    print_sim_num(sim_num)
    sim_map = map_gen.create_sim_map_from_txt(sim_num)
    all_dir_cap_anom = np.loadtxt(dir_anom_path + dir_anom_fnames[sim_num])
    for i, dcs in enumerate(dir_cap_sizes):
        plat, plon = cau.direction.align_pole_to_mac(sim_map,
                                                     all_dir_cap_anom,
                                                     dcs,
                                                     pre_dir_cap_sizes,
                                                     all_dir_lat,
                                                     all_dir_lon)
        _results[i] = cau.measure.get_measure(sim_map, **run_inputs.to_kwargs())
        fpath       = output_path + "sim{:03}_{}cap_measure.txt".format(sim_num, int(dcs))
        np.savetxt(fpath, _results[i], header=f"lat = {plat}, lon = {plon}")
    
    # Save accumulative result
    np.savetxt( output_path + "sim{:03}_acc_result.txt".format(sim_num),
                np.sum(_results, axis=0))

print("- Done -")