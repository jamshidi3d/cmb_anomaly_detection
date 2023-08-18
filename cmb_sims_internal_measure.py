#
#
# This script computes the internal legendre coefficients of measure, computed
# on strips in the MostAnomalousCap(MAC).
# It is useful for computing p-values 
#
import os
import numpy as np
import cmb_anomaly_utils as cau, read_cmb_maps_params as rmp


_inputs = rmp.get_inputs()

do_compute_legendre_exp     = False
# direction finding params
do_use_mac                  = True
dir_nside                   = 16
max_l                       = 10
cap_size_for_finding_dir    = 30
# simulations
do_compute_sims             = True
max_sim_num                 = 1000
# input params
_inputs['nside']            = 64
_inputs['is_masked']        = False
_inputs['min_pix_ratio']    = 0.7
_inputs['pole_lat']         = 90
_inputs['pole_lon']         = 0
_inputs['measure_flag']     = cau.const.D_STD2_FLAG
_inputs['geom_flag']        = cau.const.STRIP_FLAG
# geom range
_inputs['geom_start']       = 0
_inputs['geom_stop']        = 180
dtheta                      = 5
_inputs['ngeom_samples']    = 1 + int((_inputs['geom_stop'] - _inputs['geom_start']) / dtheta)
# measure range
_inputs['measure_start']    = 0
_inputs['measure_stop']     = 180
_inputs['nmeasure_samples'] = 1 + 180

rmp.print_inputs(_inputs)

_inputs['measure_range']    = cau.stat_utils.get_measure_range(**_inputs)
_inputs['geom_range']       = cau.stat_utils.get_geom_range(**_inputs)

dir_cap_geom_range          = np.linspace(10, 90, int((90 - 10)/5 + 1))

is_masked = _inputs['is_masked']
mask_txt  = 'masked' if is_masked else 'inpainted'
ratio_txt = "_{}ratio".format(_inputs["min_pix_ratio"]) if is_masked else ""

cmb_cap_anom_fpath  = './output/cmb_{}_all_dir_cap_anom{}_5deg.txt'.format(mask_txt, ratio_txt)
sims_cap_anom_path  = './output/sims_{}_all_dir_cap_anom{}_5deg/'.format(mask_txt, ratio_txt)
sims_path           = './input/commander_sims/'

geom_range  = _inputs['geom_range']
theta       = np.radians(_inputs['geom_range'])

measure_func = cau.measure.get_cap_measure if _inputs.get('geom_flag') == cau.const.CAP_FLAG \
                else cau.measure.get_strip_measure

def set_pole_along_mac(_inputs,
             all_dir_cap_anom,
             cap_size_for_finding_dir,
             dir_cap_geom_range):
    # function body
    _inputs['pole_lat'], _inputs['pole_lon'] = \
        cau.direction.find_dir_by_mac(
                                all_dir_cap_anom,
                                special_cap_size = cap_size_for_finding_dir,
                                geom_range  = dir_cap_geom_range,
                                all_dir_lat = dir_lat,
                                all_dir_lon = dir_lon)

# ---------- CMB -----------
print("Computing measure for CMB")
cmb_all_dir_cap_anom = np.loadtxt(cmb_cap_anom_fpath)
if do_use_mac:
    set_pole_along_mac( _inputs,
                        cmb_all_dir_cap_anom,
                        cap_size_for_finding_dir,
                        dir_cap_geom_range)
sky_pix = rmp.get_cmb_pixdata(**_inputs)
cmb_measure_results = measure_func(sky_pix, **_inputs)
cmb_a_l = np.zeros((max_sim_num, max_l + 1))
if do_compute_legendre_exp:
        cmb_a_l = cau.math_utils.get_all_legendre_modulation(theta, cmb_measure_results, max_l)

# Save CMB measure results
print("Saving CMB measure")
if do_compute_legendre_exp:
    np.savetxt('./output/cmb_measure_a_l.txt', cmb_a_l)
fpath = './output/cmb_{}_{}mac_{}_{}'.format(
    mask_txt,
    cap_size_for_finding_dir,
    _inputs['geom_flag'].lower(),
    _inputs['measure_flag'].lower())
np.savetxt(fpath + "_measure.txt", cmb_measure_results)
np.savetxt(fpath + "_sampling_range.txt", _inputs.get('geom_range'))


# ------ Simulations -------
if do_compute_sims:
    print("Computing measure for simulations")
    sim_pos     = cau.map_reader.read_pos(_inputs['nside'])
    sim_mask    = rmp.get_mask(**_inputs)

    sims_internal_measure = np.zeros((max_sim_num, len(geom_range)))
    sims_a_l    = np.zeros((max_sim_num, max_l + 1))

    # Find MAC -> read measure -> legendre coefs
    file_list   = os.listdir(sims_cap_anom_path)
    dir_lat, dir_lon = cau.direction.get_healpix_latlon(cau.direction.get_npix(dir_nside))

    for sim_num, fname in enumerate(file_list):
        print(f"{sim_num}/{max_sim_num - 1}\r", end='')
        fpath = sims_cap_anom_path + fname
        all_dir_cap_anom = np.loadtxt(fpath)
        # Set pole
        if do_use_mac:
            set_pole_along_mac( _inputs,
                                all_dir_cap_anom,
                                cap_size_for_finding_dir,
                                dir_cap_geom_range)
        # Create data
        try:
            sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', sim_num)
        except:
            print("simulation number {:05} is currupted!".format(sim_num))
            continue
        sim_temp            -= np.mean(sim_temp)
        sim_pd              = cau.dtypes.pix_data(sim_temp, sim_pos, sim_mask)
        # Align to MAC
        mac_aligned_pos     = cau.coords.rotate_pole_to_north(sim_pos, _inputs['pole_lat'], _inputs['pole_lon'])
        sim_pd.raw_pos      = mac_aligned_pos
        # Compute measure
        _result             = measure_func(sim_pd, **_inputs)
        sims_internal_measure[sim_num] = _result
        # Compute legendre expansion
        if do_compute_legendre_exp:
            sims_a_l[sim_num]   = cau.math_utils.get_all_legendre_modulation(theta, _result, max_l)


    print("Saving simulations measure")
    # Save legendre expansion
    if do_compute_legendre_exp:
        np.savetxt('./output/sims_internal_a_l.txt', sims_a_l)

    # Save measure results
    fpath = './output/sims_internal_{}_{}_{}_{}'.format(
        mask_txt,
        cap_size_for_finding_dir,
        _inputs['geom_flag'].lower(),
        _inputs['measure_flag'].lower())
    np.savetxt(fpath + "_measure.txt", sims_internal_measure)
