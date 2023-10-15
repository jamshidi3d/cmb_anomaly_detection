# import pymaster
import healpy as hp
import numpy as np

from . import math_utils as mu
from . import file_reader
from .dtypes import PixMap

# ------ Remove Monopole/Dipole ------
def remove_monopole_dipole(pix_map:PixMap):
    result = hp.remove_dipole(pix_map.raw_data)
    pix_map.raw_data = result.data
    return result

# ------ Map Filling ------
def fill_map_with_cap(data_map, pole_lat, pole_lon, cap_size, fake_poles):
    nside = int(np.sqrt(len(data_map) / 12))
    filled_map = np.copy(data_map)
    mask = np.ones(len(data_map), dtype=bool) # all pixels masked
    # marking map pixels
    theta, phi = np.deg2rad(90 - pole_lat), np.deg2rad(pole_lon)
    _vec = hp.ang2vec(theta, phi)
    ipix_disc = hp.query_disc(nside= nside, vec=_vec, radius=np.radians(cap_size))
    mask[ipix_disc] = False
    # rotating the cap to north
    euler_rot_angles = np.array([phi, -theta, 0])
    r = hp.rotator.Rotator(rot = euler_rot_angles, deg=False, eulertype='ZYX')
    map_cap_north = r.rotate_map_pixel(data_map)
    # copying caps
    for _vec in fake_poles:
        # rotating the cap to its fake positions
        theta, phi = tuple((ang_arr[0] for ang_arr in hp.vec2ang(_vec)))
        euler_rot_angles = np.array([phi, -theta, 0])
        r = hp.rotator.Rotator(rot = euler_rot_angles, inv=True, deg=False, eulertype='ZYX')
        map_rotated = r.rotate_map_pixel(map_cap_north)
        # copying rotated pixels to the filled map and marking map pixels
        ipix_disc = hp.query_disc(nside=nside, vec=_vec, radius=np.radians(cap_size))
        filled_map[ipix_disc] = map_rotated[ipix_disc]
        mask[ipix_disc] = False
    return filled_map, mask