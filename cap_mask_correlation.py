import numpy as np
import healpy as hp
# from astropy.io import fits
from numba import njit, prange 
import time
import matplotlib.pyplot as plt

from cmb_cpu.cap import *
from cmb_cpu.coords import *
from cmb_cpu.measure import *

correlation_flag    = 'CORRELATION'
variance_flag       = 'VARIANCE'
cap_flag            = 'CAP'
stripe_flag         = 'STRIPE'

t0 = time.time()

nside           = 1024
is_masked       = False
measure_mode    = correlation_flag # variance_flag
geom            = cap_flag # stripe_flag
n_samples       = 64*3
sampling_range  = np.arange(10,170, 1)
cacr = correlation_angle_cutoff_ratio = 2/3
stripe_size     = 5 # top cap thickness

print("- nSide: {}, Map: {}, Measure: {}, Geometry: {}".format(nside, "Masked" if is_masked else "Inpainted", measure_mode, geom))

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

# temperature
sky_temp = cmb_map #map_masked.data
sky_temp *= 10**6

# mask
sky_mask = map_masked.mask
# swapping on and off, because sky_mask is true in masked areas and false in data area
sky_mask = np.array([not off_pix for off_pix in sky_mask])

# positions
print("- Reading Positions")
npix     = np.arange(len(sky_temp))
lon, lat = hp.pix2ang(nside, npix, lonlat = True)
sky_pos  = convert_polar_to_xyz(lat, lon)
# rotation to pole
pole_lat, pole_lon = -20, 221
sky_pos = rotate_pole_to_north(pole_lat, pole_lon, sky_pos)


# measurement
sky_data = (sky_temp, sky_pos)
X2 = np.zeros(len(sampling_range))
_mask = None if not is_masked else sky_mask

if geom == cap_flag:
    cap_angles = sampling_range
    # measure
    for i in range(len(cap_angles)):
        ca = cap_angles[i]
        print("++ Cap of size {} degrees".format(ca))
        top, bottom = get_top_bottom_caps(sky_data, ca, _mask)
        if measure_mode == correlation_flag:
            tctt = parallel_correlation_tt(top, n_samples, 16)
            bctt = parallel_correlation_tt(bottom, n_samples, 16)
            max_index = int(cacr * 2 * min(ca, 180-ca) / 180 * n_samples)
            X2[i] = np.sum((tctt[:max_index] - bctt[:max_index])**2)
        elif measure_mode == variance_flag:
            X2[i] = (std_t(top) - std_t(bottom))**2

elif geom == stripe_flag:
    # creating stripes
    height = 1 - np.cos(stripe_size * np.pi / 180) 
    stripe_centers = sampling_range * np.pi / 180
    stripe_starts  = 180 / np.pi * np.arccos(np.cos(stripe_centers) + height/2)
    stripe_ends    = 180 / np.pi * np.arccos(np.cos(stripe_centers) - height/2)
    stripe_centers *= 180 / np.pi
    # measure
    for i in range(len(stripe_centers)):
        print("++ Stripe center {} degrees".format(stripe_centers[i]))
        start = stripe_starts[i]
        end = stripe_ends[i]
        stripe, rest_of_sky = get_stripe(sky_data, start, end, _mask)
        if measure_mode == correlation_flag:
            sctt = parallel_correlation_tt(stripe, 16)
            rctt = parallel_correlation_tt(rest_of_sky, 16)
            max_index = int(cacr * 2 * stripe_size / 180 * n_samples)
            X2[i] = np.sum((sctt[:max_index] - rctt[:max_index])**2)
        elif measure_mode == variance_flag:
            X2[i] = (std_t(stripe) - std_t(rest_of_sky))**2

# plot
print("- Plotting")
fig, ax = plt.subplots(1,1)
fig.set_size_inches(8,5)

if measure_mode == correlation_flag:
    captitle  = r'$\int [C_{tt}^{top}(\theta) - C_{tt}^{bottom}(\theta)]^2 d\theta$'
    strtitle  = r'$\int [C_{tt}^{stripe}(\theta) - C_{tt}^{rest\,of\,sky}(\theta)]^2 d\theta$'
elif measure_mode == variance_flag:
    captitle = r'$[\sigma_{top}(T) - \sigma_{bottom}(T)]^2$'
    strtitle = r'$[\sigma_{stripe}(T) - \sigma_{rest\,of\,sky}(T)]^2$'

# xlabel
capxlabel = r'Cap angle [$\degree$]'
strxlabel = r'Stripe Center [$\degree$]'
xlabel = capxlabel if geom == cap_flag else strxlabel
ax.set_xlabel(xlabel, fontsize=11)
# ylabel
ylabel = captitle if geom == cap_flag else strtitle
ylabel += r'$\>\>\>\> [\mu K]^2$'
ax.set_ylabel(ylabel, fontsize=12)
# title
title = captitle if geom == cap_flag else strtitle
title += r'$\,\,\,\, Vs \,\,\,\,$ {}'.format("Top Cap Size" if geom == cap_flag else "Stripe Center")
title += r', '
title += r'{} Map'.format("Masked" if is_masked else "Inpainted")
ax.set_title(title, fontsize = 12)


ax.plot(sampling_range, X2, '-k')

# maximum of sky
_max = X2.argmax()
ax.plot(sampling_range[_max], X2[_max], color = 'orange', marker = "o")

max_point_label = "maximum at {}".format(sampling_range[_max])
ax.legend(["CMB", max_point_label])

file_name = "./output/"
file_name += "{}".format(nside)
file_name += "_{}".format("masked" if is_masked else "inpainted")
file_name += "_{}".format("cap" if geom == cap_flag else "{}stripe".format(stripe_size))
file_name += "_{}".format("corr2" if measure_mode == correlation_flag else "sigma2")
file_name += ".pdf"
fig.savefig(file_name)

print("Total execution time: {} seconds".format(time.time() - t0))