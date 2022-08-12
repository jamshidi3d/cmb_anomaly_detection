import numpy as np
import healpy as hp
# from astropy.io import fits
from numba import njit, prange 
import time
import matplotlib.pyplot as plt

from cmb_cpu.cap import *
from cmb_cpu.coords import *
from cmb_cpu.measure import *


t0 = time.time()

nside = 64
print("- NSide is set to {}".format(nside))

print("- Reading CMB map")
file_name = "smica.fits"
cmb_map = hp.read_map(file_name, field=0)
cmb_map = hp.ud_grade(cmb_map, nside_out=nside)

print("- Reading mask")
path = 'mask.fits'
mask = hp.read_map(path)
mask = hp.ud_grade(mask, nside_out=nside)

map_masked = hp.ma(cmb_map)
map_masked.mask = np.logical_not(mask)
# hp.mollview(map_masked, cmap='jet')

# temperature
sky_temp = cmb_map #map_masked.data
sky_temp *= 10**6

# mask
sky_mask = map_masked.mask
# swapping on and off, because sky_mask is true in masked areas and false in data area
sky_mask = np.array([not off_pix for off_pix in sky_mask])

# positions
print("- Converting Positions")
npix     = np.arange(len(sky_temp))
lon, lat = hp.pix2ang(nside, npix, lonlat = True)
sky_pos  = convert_polar_to_xyz(lat, lon)

# rotation to pole
print("- Rotating Positions")
pole_lat, pole_lon = -20, 221
sky_pos = rotate_pole_to_north(pole_lat, pole_lon, sky_pos)

# measuring correlation
sky_data = (sky_temp, sky_pos)
cap_angles = np.arange(10,170,1)
n_samples = 64*3
cutoff_ratio = 2/3
X2 = np.zeros(len(cap_angles))
top_percentage = np.zeros(len(cap_angles))
bottom_percentage = np.zeros(len(cap_angles))

print("- Correlation difference squared masked")
for i in range(len(cap_angles)):
    '''calculating masking percentage'''
    ca = cap_angles[i]
    top_m, bottom_m = get_top_bottom_caps(sky_data, ca, sky_mask)
    top_i, bottom_i = get_top_bottom_caps(sky_data, ca)
    top_percentage[i] = 1 - len(top_m[0])/len(top_i[0])
    bottom_percentage[i] = 1 - len(bottom_m[0])/len(bottom_i[0])
    
    ''' calculating correlations
    ca = cap_angles[i]
    print("++ Cap of size {} degrees".format(ca))
    top, bottom = get_top_bottom_caps(sky_data, ca, sky_mask)
    tctt = correlation_tt(top, n_samples)
    bctt = correlation_tt(bottom, n_samples)
    max_index = int(cutoff_ratio * 2 * min(ca, 180-ca) / 180 * n_samples)
    X2[i] = np.sum((tctt[:max_index] - bctt[:max_index])**2)
    '''


# plot
print("- Plotting")
fig, ax = plt.subplots(1,1)
fig.set_size_inches(8,5)

# title = r'$\int [C_{tt}^{top}(\theta) - C_{tt}^{bottom}(\theta)]^2 d\theta \,\,\,\, Vs \,\,\,\,$Top Cap Size - Masked Area'
title = r'Masked Area - Top Cap Size'
ax.set_title(title, fontsize = 12)
# label = r"$\int [C_{tt}^{top}(\theta) - C_{tt}^{bottom}(\theta)]^2 d\theta \>\>\>\> [\mu K]^2$"
label = "Masked area (%)"
ax.set_ylabel(label, fontsize=12)
ax.set_xlabel(r"Top Cap Size [$\degree$]", fontsize=11)

# ax.plot(cap_angles, X2, color = 'k', marker = '.', lw=1)
ax.plot(cap_angles, top_percentage, color = 'b', lw=1)
ax.plot(cap_angles, bottom_percentage, color = 'r', lw=1)

# maximum of sky
# _max = X2.argmax()
# ax.plot(cap_angles[_max], X2[_max], color = 'orange', marker = "o")

# max_point_label = "maximum at {}".format(cap_angles[_max])
# ax.legend(["CMB", max_point_label])
ax.legend(["top masked area (%)", "bottom masked area (%)"])

fig.savefig("./output/masking_percentage.pdf")

print("Total execution time: {} seconds".format(time.time() - t0))