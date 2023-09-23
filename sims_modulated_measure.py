import os
import numpy as np
import cmb_anomaly_utils as cau

base_path                       = "./output/measure_results_mac_dir/"
max_sim_num                     = 1000
max_l                           = 20
npoles                          = 3

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
run_inputs.delta_geom_samples   = 1
run_inputs.stripe_thickness      = 20
run_inputs.pole_lat             = -10
run_inputs.pole_lon             = 221

dir_cap_sizes       = [30]#cau.stat_utils.get_range(20, 70, 10)
pre_dir_cap_sizes   = cau.stat_utils.get_range(10, 90, 5)

map_gen     = cau.run_utils.MapGenerator(**run_inputs.to_kwargs())
cmb_a_l     = cau.file_reader.read_cmb_a_l(base_path, dir_cap_sizes[0], **run_inputs.to_kwargs())
sims_a_l    = np.zeros((max_sim_num * npoles, max_l + 1))

def print_sim_num(sim_num):
    print("{:03}\r".format(sim_num), end="")

for nonzero_l in range(max_l + 1):
    # modulate each l separately
    print(f"l = {nonzero_l}")
    single_cmb_a_l = np.copy(cmb_a_l)
    single_cmb_a_l[:nonzero_l] = 0
    single_cmb_a_l[nonzero_l + 1:] = 0
    
    for p_i, pole_lat, pole_lon  in \
        [[0, 0 , 0], [1, 0, -90], [2, 90, 0]]:
        print(f"plat = {pole_lat}, plon = {pole_lon}")
        # single l factor
        sim_pos = np.copy(map_gen.pos)
        sim_pos = cau.coords.rotate_pole_to_north(sim_pos, pole_lat, pole_lon)
        modulation_factor = cau.math_utils.create_legendre_modulation_factor(sim_pos, single_cmb_a_l)
        for sim_num in range(max_sim_num):
            print_sim_num(sim_num)
            sim_pix = map_gen.create_sim_map_from_txt(sim_num)
            sim_pix.raw_pos = sim_pos
            sim_pix.add_modulation(modulation_factor)
            sim_measure = cau.measure.get_measure(sim_pix, **run_inputs.to_kwargs())
            s_a_l = cau.math_utils.get_single_legendre_modulation(
                                                            run_inputs.geom_range * np.pi / 180,
                                                            sim_measure,
                                                            nonzero_l)
            sims_a_l[sim_num * npoles + p_i, nonzero_l] = s_a_l
np.savetxt('./output/sims_modulated_a_l.txt', sims_a_l)

