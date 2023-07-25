#
#
# This script computes the internal legendre coefficients of measure, computed
# on strips in the MostAnomalousDirection(MAD).
# It is useful for computing p-values 
#
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import cmb_anomaly_utils as cau, read_cmb_maps_params as rmp


_inputs = rmp.get_inputs()

dir_nside                   = 16
max_sim_num                 = 1000
max_l                       = 10
do_search_for_all_caps      = True
selected_size_for_dir       = 30
_inputs['is_masked']        = True
_inputs['nside']            = 64
_inputs['pole_lat']         = 90
_inputs['pole_lon']         = 0
_inputs['measure_flag']     = cau.const.STD_FLAG
_inputs['geom_flag']        = cau.const.STRIP_FLAG
_inputs['geom_start']       = 0
_inputs['geom_stop']        = 180
dtheta                      = 5
_inputs['ngeom_samples']    = 1 + int((_inputs['geom_stop'] - _inputs['geom_start']) / dtheta)
_inputs['measure_start']    = 0
_inputs['measure_stop']     = 180
_inputs['nmeasure_samples'] = 1 + 180
_inputs['measure_range']    = cau.stat_utils.get_measure_range(**_inputs)
_inputs['geom_range']       = cau.stat_utils.get_geom_range(**_inputs)
do_compute_legendre_exp     = False


mask_txt = 'masked' if _inputs.get('is_masked') else 'inpainted'

sims_anom_path  = './output/sims_{}_all_dir_anom_5deg/'.format(mask_txt)
sims_path       = './input/commander_sims/'

geom_range  = _inputs['geom_range']
theta       = _inputs['geom_range'] * np.pi / 180

# All directions that we look for
npix     = 12 * dir_nside ** 2
dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(npix), lonlat = True)
dir_lon *= 180 / np.pi
dir_lat *= 180 / np.pi

sim_pos     = cau.map_reader.read_pos(_inputs['nside'])
sim_mask    = rmp.get_mask(**_inputs)

sims_internal_measure = np.zeros((max_sim_num, len(geom_range)))
sims_a_l    = np.zeros((max_sim_num, max_l + 1))

get_measure = cau.measure.get_cap_measure if _inputs.get('geom_flag') == cau.const.CAP_FLAG \
                else cau.measure.get_strip_measure

cap_index   = cau.stat_utils.find_nearest_index(geom_range, selected_size_for_dir)

# Find MAD & read measure & legendre coefs
file_list   = os.listdir(sims_anom_path)
for sim_num, fname in enumerate(file_list):
    print(f"{sim_num}/{max_sim_num - 1}\r", end='')
    fbasepath = sims_anom_path + fname
    all_measure_results = np.loadtxt(fbasepath)
    # Maximum anomaly direction in specified cap size
    mad_i    =  np.nanargmax(all_measure_results[cap_index, :])
    # OR select the cap size in which the anomaly is the most
    if do_search_for_all_caps:
        index = np.unravel_index(np.nanargmax(all_measure_results),
                                 all_measure_results.shape)
        mad_i = index[1] # the second index determines direction
    _inputs['pole_lon'], _inputs['pole_lat'] =\
        dir_lon[mad_i], dir_lat[mad_i]
    # Create data
    try:
        sim_temp = cau.map_reader.get_sim_attr(sims_path, 'T', sim_num)
    except:
        print("simulation number {:05} is currupted!".format(sim_num))
        continue
    sim_temp            -= np.mean(sim_temp)
    sim_pix_data        = cau.dtypes.pix_data(sim_temp, sim_pos, sim_mask)
    # Align to MAD
    mad_aligned_pos     = cau.coords.rotate_pole_to_north(sim_pos, dir_lat[mad_i], dir_lon[mad_i])
    sim_pix_data.raw_pos    = mad_aligned_pos
    # Compute measure
    _result             = get_measure(sim_pix_data, **_inputs)
    sims_internal_measure[sim_num] = _result
    # Compute legendre expansion
    if do_compute_legendre_exp:
        sims_a_l[sim_num]   = cau.math_utils.get_all_legendre_modulation(theta, _result, max_l)

# Save legendre expansion
if do_compute_legendre_exp:
    np.savetxt('./output/sims_internal_a_l.txt', sims_a_l)

# Save measure results
fbasepath = './output/sims_internal_{}_{}_{}_{}'.format(
    mask_txt,
    'MAD' if do_search_for_all_caps else selected_size_for_dir,
    _inputs['geom_flag'].lower(),
    _inputs['measure_flag'].lower())
np.savetxt(fbasepath + "_measure.txt", sims_internal_measure)
np.savetxt(fbasepath + "_sampling_range.txt", _inputs.get('geom_range'))
