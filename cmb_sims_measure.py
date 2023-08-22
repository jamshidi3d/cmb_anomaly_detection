#
#
# This script provides a faster way to read maps and inputs!
#
#
import numpy as np
import cmb_anomaly_utils as cau

output_path                     = "./output/measure_results_mac_dir/"
max_sim_num                     = 1000

run_inputs  = cau.run_utils.RunInputs()
run_inputs.mask_fpath           = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
run_inputs.cmb_fpath            = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
run_inputs.sims_path            = "./input/commander_sims/"
run_inputs.sims_dir_anom_path   = "./output/sims_inpainted_all_dir_cap_anom_5deg"
run_inputs.cmb_dir_anom_fpath   = "./output/cmb_inpainted_all_dir_cap_anom.txt"
run_inputs.geom_flag            = cau.const.STRIP_FLAG
run_inputs.measure_flag         = cau.const.NORM_STD_FLAG
run_inputs.nside                = 64
run_inputs.dir_nside            = 16
run_inputs.geom_start           = 0
run_inputs.geom_stop            = 180
run_inputs.delta_geom_samples   = 1
run_inputs.strip_thickness      = 20
run_inputs.pole_lat             = -10
run_inputs.pole_lon             = 221

map_gen     = cau.run_utils.MapGenerator(**run_inputs.to_kwargs())

all_dir_lat, all_dir_lon = cau.coords.get_healpix_latlon(run_inputs.dir_nside)

dir_cap_sizes   = cau.stat_utils.get_range(20, 70, 10)
dir_geom_range  = cau.stat_utils.get_range(10, 90, 5)

print("Computing CMB measures")

cmb_map     = map_gen.create_cmb_map()
all_dir_cap_anom = np.loadtxt(run_inputs.cmb_dir_anom_fpath)
for i, dcs in enumerate(dir_cap_sizes):
    cau.direction.align_pole_to_mac(cmb_map,
                                    all_dir_cap_anom,
                                    dcs,
                                    dir_geom_range,
                                    all_dir_lat,
                                    all_dir_lon)
    _result = cau.measure.get_measure(cmb_map, **run_inputs.to_kwargs())
    fpath   = output_path + "cmb_{}_{}_{}_{}.txt".format(
                                                run_inputs.masked_txt,
                                                int(dcs),
                                                run_inputs.geom_flag.lower(),
                                                run_inputs.measure_flag.lower())
    np.savetxt(fpath, _result)



def print_sim_num(sim_num):
    print("{:03}\r".format(sim_num), end="")

print("Computing Simulations measures")
for sim_num in range(max_sim_num):
    print_sim_num(sim_num)
    sim_map = map_gen.create_sim_map_from_txt(sim_num)
    for i, dcs in enumerate(dir_cap_sizes):
        cau.direction.align_pole_to_mac(sim_map,
                                        all_dir_cap_anom,
                                        dcs,
                                        dir_geom_range,
                                        all_dir_lat,
                                        all_dir_lon)
        _result = cau.measure.get_measure(sim_map, **run_inputs.to_kwargs())
        fpath   = output_path + "sim{:03}_{}_{}cap_{}_{}.txt".format(
                                                    sim_num,
                                                    run_inputs.masked_txt,
                                                    int(dcs),
                                                    run_inputs.geom_flag.lower(),
                                                    run_inputs.measure_flag.lower())
        np.savetxt(fpath, _result)