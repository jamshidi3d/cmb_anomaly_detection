import numpy as np
from astropy_healpix import HEALPix
import bmesh

nside = 16
hp = HEALPix(nside=nside, order='nested')
boundaries = hp.boundaries_lonlat(np.arange(12*nside**2), step=1)
print(boundaries.to_value())