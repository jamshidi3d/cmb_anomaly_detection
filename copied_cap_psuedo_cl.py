#
#
# This script fills the map with a given cap (with other areas considered as masked) and 
# finds psuedo-Cls of it
#
import numpy as np
import healpy as hp
import pymaster as nmt
import astropy.units as units

import cmb_anomaly_utils as cau
import read_cmb_maps_params as rmp

_inputs = rmp.get_inputs()

_inputs['nside']     = 512
_inputs['is_masked'] = True
_inputs['pole_lat']  = 90
_inputs['pole_lon']  = 0

nside = _inputs['nside']

print('Reading CMB map')
cmb_pixdata = rmp.get_cmb_pixdata(**_inputs)

temp_map = cau.map_reader.read_attr(rmp.cmb_fpath, nside, 0)
mask = cau.map_reader.read_attr(rmp.cmb_fpath, nside, 0)

# real planck:
temp_map, mask_map = cmb_pixdata.data, cmb_pixdata.raw_mask

print('Filling map with caps')
fake_poles = np.loadtxt('./input/fake_cap_poles.txt')
# fake copied map:
temp_map, mask_map = \
    cau.map_filler.fill_map_with_cap(
        temp_map,
        -20, # pole_lat
        221, # pole_lon
        30, # cap size
        fake_poles)


# invert mask since healpy considers sky_mask is true in masked areas
mask_map = (mask_map == False)
print('Mask apodization')
mask    = nmt.mask_apodization(mask_map, 1., apotype="Smooth")
# get spin-0 field
print('Getting field')
f_0     = nmt.NmtField(mask, [temp_map])
# Initialize binning scheme with 4 ells per bandpower
print('Binning')
b       = nmt.NmtBin.from_nside_linear(nside, 4)
# compute spin-0 cl
print('Compute Cls')
cl_00   = nmt.compute_full_master(f_0, f_0, b)
ell_arr = b.get_effective_ells()
# applying beam function
print('Applying beam')
w_ell = hp.gauss_beam((5*units.arcmin).to_value(units.radian), lmax = ell_arr[-1])
# select the correct w_ell for each ell
all_ells = np.arange(len(w_ell))
nearest_index = cau.stat_utils.find_nearest_index
w_ell = np.array([w_ell[nearest_index(all_ells, ell)] for ell in ell_arr])
sky_fraction = 1 #(len(temp_map) - np.sum(mask_map)) / len(temp_map)
cl = cl_00[0] / sky_fraction / w_ell**2
psuedo_cl = np.column_stack((ell_arr, cl))
# np.savetxt('./output/psuedo_cl.txt', psuedo_cl)
np.savetxt('./output/copied_psuedo_cl.txt', psuedo_cl)