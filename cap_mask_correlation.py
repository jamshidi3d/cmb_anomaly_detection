import numpy as np
import healpy as hp
# from astropy.io import fits
from numba import njit, prange 
import time
import matplotlib.pyplot as plt

from cmb_cpu.cap import *
from cmb_cpu.coords import *
from cmb_cpu.measure import *

D_CORR2_FLAG   = 'DELTA_CORRELATION_2'
D_STD2_FLAG    = 'DELTA_STD_2'
CORR_FLAG      = 'CORRELATION'
STD_FLAG       = 'STANDARD_DEVIATION'
CAP_FLAG       = 'CAP'
STRIPE_FLAG    = 'STRIPE'

t0 = time.time()

nside           = 1024
is_masked       = False
measure_mode    = STD_FLAG #D_STD2_FLAG #CORR_FLAG #D_CORR2_FLAG
geom            = STRIPE_FLAG #CAP_FLAG
n_samples       = 64*3
stripe_thickness = top_cap_size = 20
dtheta          = 1
sampling_range  = np.arange(15, 165, dtheta)
cacr = correlation_angle_cutoff_ratio = 2/3

print("- nSide: {} | Map: {} | Measure: {} | Geometry: {}".format(nside, "MASKED" if is_masked else "INPAINTED", measure_mode, geom))

print("- Reading CMB map")
cmb_file_name = "smica.fits"


cmb_map, _, _ = hp.read_map('smica.fits',field=(5, 1, 3), nest=True)
cmb_map = hp.ud_grade(cmb_map, nside_out=nside, order_in='NESTED')
cmb_map = hp.reorder(cmb_map, inp='NESTED', out='RING')


print("- Reading mask")
mask_file_name = 'mask.fits'
mask = hp.read_map(mask_file_name)
mask = hp.ud_grade(mask, nside_out=nside)

map_masked = hp.ma(cmb_map)
map_masked.mask = np.logical_not(mask)

# temperature
sky_temp = cmb_map #map_masked.data
sky_temp = sky_temp * 10**6

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

fullsky_int = 1
if measure_mode == CORR_FLAG or measure_mode == D_CORR2_FLAG:
    fullsky_corr = parallel_correlation_tt(sky_data, n_samples, 4)
    fullsky_int = np.sum(fullsky_corr ** 2)
else:
    fullsky_int = std_t(sky_data)

if geom == CAP_FLAG:
    cap_angles = sampling_range
    # measure
    for i in range(len(cap_angles)):
        ca = cap_angles[i]
        print("++ Cap of size {} degrees".format(ca))
        top, bottom = get_top_bottom_caps(sky_data, ca, _mask)
        if measure_mode == D_CORR2_FLAG:
            tctt = parallel_correlation_tt(top, n_samples, 4)
            bctt = parallel_correlation_tt(bottom, n_samples, 4)
            max_index = int(cacr * 2 * min(ca, 180-ca) / 180 * n_samples)
            X2[i] = np.sum((tctt[:max_index] - bctt[:max_index])**2)
        elif measure_mode == D_STD2_FLAG:
            X2[i] = (std_t(top) - std_t(bottom))**2
        elif measure_mode == STD_FLAG:
            X2[i] = std_t(top)
        elif measure_mode == CORR_FLAG:
            tctt = parallel_correlation_tt(top, n_samples, 4)
            X2[i] = np.sum(tctt ** 2) / fullsky_int - 1

elif geom == STRIPE_FLAG:
    # creating stripes
    height = 1 - np.cos(stripe_thickness * np.pi / 180)
    stripe_centers = sampling_range
    stripe_starts  = 180 / np.pi * np.arccos(np.cos(stripe_centers * np.pi / 180) + height/2)
    stripe_ends    = 180 / np.pi * np.arccos(np.cos(stripe_centers * np.pi / 180) - height/2)
    # stripe_starts  = np.arange(10, 171, top_cap_size)
    # stripe_ends    = np.arange(10 + top_cap_size, 180, top_cap_size)

    # measure
    for i in range(len(stripe_centers)):
        print("++ Stripe center {} degrees".format(stripe_centers[i]))
        start = stripe_starts[i]
        end = stripe_ends[i]
        stripe, rest_of_sky = get_stripe(sky_data, start, end, _mask)
        if measure_mode == D_CORR2_FLAG:
            sctt = parallel_correlation_tt(stripe, n_samples, 4)
            rctt = parallel_correlation_tt(rest_of_sky, n_samples, 4)
            max_index = int(cacr * 2 * stripe_thickness / 180 * n_samples)
            X2[i] = np.sum((sctt[:max_index] - rctt[:max_index])**2)
        elif measure_mode == D_STD2_FLAG:
            X2[i] = (std_t(stripe) - std_t(rest_of_sky))**2
        elif measure_mode == STD_FLAG:
            X2[i] = std_t(stripe)
        elif measure_mode == CORR_FLAG:
            tctt = parallel_correlation_tt(stripe, n_samples, 4)
            X2[i] = np.sum(tctt ** 2) / fullsky_int - 1

# plot
print("- Plotting")
fig, ax = plt.subplots(1,1)
fig.set_size_inches(8,5)

if measure_mode == D_CORR2_FLAG:
    captitle = r'$\int [C_{TT}^{top}(\gamma) - C_{TT}^{bottom}(\gamma)]^2 d\gamma$'
    strtitle = r'$\int [C_{TT}^{stripe}(\gamma) - C_{TT}^{rest\,of\,sky}(\gamma)]^2 d\gamma$'
elif measure_mode == CORR_FLAG:
    captitle = r'$\frac{\int [C_{TT}^{top}(\gamma)]^2 d\gamma}{\int [C_{TT}^{total}(\gamma)]^2 d\gamma} - 1$'
    strtitle = r'$\frac{\int [C_{TT}^{stripe}(\gamma)]^2 d\gamma}{\int [C_{TT}^{total}(\gamma)]^2 d\gamma} - 1$'
elif measure_mode == D_STD2_FLAG:
    captitle = r'$[\sigma_{top}(T) - \sigma_{bottom}(T)]^2$'
    strtitle = r'$[\sigma_{stripe}(T) - \sigma_{rest\,of\,sky}(T)]^2$'
elif measure_mode == STD_FLAG:
    captitle = r'$\sigma_{top}(T)$'
    strtitle = r'$\sigma_{stripe}(T)$'


# xlabel
capxlabel = r'Cap angle [$\degree$]'
strxlabel = r'Stripe Center [$\degree$]'
xlabel = capxlabel if geom == CAP_FLAG else strxlabel
ax.set_xlabel(xlabel, fontsize=11)
# ylabel
ylabel = captitle if geom == CAP_FLAG else strtitle
ylabel += r'$\>\>\>\>$'
if measure_mode == STD_FLAG:
    ylabel +=  r'$[\mu K]$'
elif measure_mode == CORR_FLAG:
    pass
else:
    ylabel +=  r'$[\mu K]^2$'
ax.set_ylabel(ylabel, fontsize=12)
# title
title = captitle if geom == CAP_FLAG else strtitle
title += r'$\,\,\,\, Vs \,\,\,\,$ {}'.format("Top Cap Size" if geom == CAP_FLAG else "Stripe Center")
title += r', '
title += r'{} Map'.format("Masked" if is_masked else "Inpainted")
ax.set_title(title, fontsize = 12)


ax.plot(sampling_range, X2, '-k')

# # maximum of sky
# _max = X2.argmax()
# ax.plot(sampling_range[_max], X2[_max], color = 'orange', marker = "o")
# print(X2)
# max_point_label = "maximum at {}".format(sampling_range[_max])
# ax.legend(["CMB", max_point_label])

file_name = "./output/"
file_name += "{}".format(nside)
file_name += "_{}".format("masked" if is_masked else "inpainted")
file_name += "_{}".format("cap" if geom == CAP_FLAG else "{}stripe".format(stripe_thickness))
file_name += "_{}".format(  "dcorr2" if measure_mode == D_CORR2_FLAG else \
                            "dstd2" if measure_mode == D_STD2_FLAG else \
                            "std" if measure_mode == STD_FLAG else "corr")
file_name += "_{}dtheta".format(dtheta)

# save data
fname = file_name + "_sampling_range" + ".txt"
with open(fname, "w") as file:
    np.savetxt(file, sampling_range)

fname = file_name + "_result" + ".txt"
with open(fname, "w") as file:
    np.savetxt(file, X2)

# save fig
fig.savefig(file_name + ".pdf")

print("Total execution time: {} seconds".format(time.time() - t0))