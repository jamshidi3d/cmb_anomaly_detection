import numpy as np
import healpy as hp
import os
import astropy.units as u
import matplotlib.pyplot as plt


# Read files
print(f"Reading maps")
map_fname = './input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits'
cmb_map = hp.read_map(map_fname, field=5)
mask_fname = './input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'
mask = hp.read_map(mask_fname)
map_masked = hp.ma(cmb_map)
map_masked.mask = np.logical_not(mask)
binned_fname = './input/COM_PowerSpect_CMB-TT-binned_R3.01.txt'
cmb_binned_spectrum = np.loadtxt(binned_fname)



lmax = 3000

print(f"fining Cl's")
# Always use use_pixel_weights=True in anafast
# to have a more precise spectrum estimation
test_cls_meas_frommap = hp.anafast(cmb_map, lmax=lmax, use_pixel_weights=True)
ll = np.arange(lmax+1)


print(f"fining sky fraction")
# If you compute the spectrum on the partial sky,
# first order correction is to divide by
# the sky fraction to retrieve the spectrum over the full sky
sky_fraction = 1#len(map_masked.compressed()) / len(map_masked)
print(f"The map covers {sky_fraction:.1%} of the sky")

# plt.style.use("seaborn-poster")

k2muK = 1e6

plt.plot(cmb_binned_spectrum[:,0], cmb_binned_spectrum[:,1], '--', alpha=1, label='Planck 2018 PS release')
plt.plot(ll, ll*(ll+1.)*test_cls_meas_frommap*k2muK**2/2./np.pi / sky_fraction, '--', alpha=0.6, label='Planck 2018 PS from Data Map')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell~[\mu K^2]$')
plt.grid()
plt.legend(loc='best')

plt.savefig('./output/without_beam.pdf', transparent = True)

# Reading the documentation of the Planck commander release,
# we see that the output has a resolution of 5 arcminutes.
# Therefore as a first order correction of the beam,
# we can divide the power spectrum by the square of the beam window function.

w_ell = hp.gauss_beam((5*u.arcmin).to_value(u.radian), lmax=lmax)
plt.plot(cmb_binned_spectrum[:,0], cmb_binned_spectrum[:,1], '--', alpha=1, label='Planck 2018 PS release')
plt.plot(ll, ll*(ll+1.)*test_cls_meas_frommap*k2muK**2/2./np.pi / sky_fraction / w_ell**2,
         alpha=0.6, label='Planck 2018 PS from Data Map (beam corrected)')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell~[\mu K^2]$')
plt.grid()
plt.legend(loc='best')
plt.savefig('./output/applied_beam.pdf', transparent = True)
