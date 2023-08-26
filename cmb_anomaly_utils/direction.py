'''
This module finds direction based on cap measures in different directions
'''
import numpy as np

from . import coords, stat_utils as su
from .dtypes import PixMap

# ------- Align map methods -------
def align_pole_to_average(pix_map:PixMap,
                          all_dir_cap_anom = None,
                          dir_cap_size = None,
                          geom_range = None,
                          top_ratio = 0.1,
                          all_dir_lat = None,
                          all_dir_lon = None):
    '''Aligns pix_map and returns pole's lat & lon'''
    if all_dir_cap_anom is None:
        return
    plat, plon = find_dir_by_average(all_dir_cap_anom,
                                     dir_cap_size,
                                     geom_range,
                                     top_ratio,
                                     all_dir_lat,
                                     all_dir_lon)
    pix_map.change_pole(plat, plon )
    return plat, plon

def align_pole_to_mac(pix_map:PixMap,
                      all_dir_cap_anom,
                      dir_cap_size:float = None,
                      geom_range = None,
                      all_dir_lat = None,
                      all_dir_lon = None):
    '''Aligns pix_map and returns pole's lat & lon'''
    if all_dir_cap_anom is None:
        return
    plat, plon = find_dir_by_mac(all_dir_cap_anom,
                                dir_cap_size,
                                geom_range,
                                all_dir_lat,
                                all_dir_lon)
    pix_map.change_pole(plat, plon )
    return plat, plon

# ------- Find direction methods -------
def find_dir_by_mac(all_dir_cap_anom,
                    dir_cap_size:float = None,
                    geom_range = None,
                    all_dir_lat = None,
                    all_dir_lon = None):
    dir_lat, dir_lon = all_dir_lat, all_dir_lon
    npix    = all_dir_cap_anom.shape[0]
    nside   = coords.get_nside(npix)
    if (all_dir_lat is None) or (all_dir_lon is None):
        dir_lat, dir_lon = coords.get_healpix_latlon(nside)
    if dir_cap_size is None:
        mac_i   = np.nanargmax(all_dir_cap_anom)
        _shape  = all_dir_cap_anom.shape
        mac_i   = np.unravel_index(mac_i, _shape)[0]
    else:
        cap_index   = su.find_nearest_index(geom_range, dir_cap_size)
        mac_i   = np.nanargmax(all_dir_cap_anom[:, cap_index])
    return dir_lat[mac_i], dir_lon[mac_i]

def find_dir_by_average(all_dir_cap_anom,
                        dir_cap_size:float = None,
                        geom_range = None,
                        top_ratio = 0.1,
                        all_dir_lat = None,
                        all_dir_lon = None):
    ''' lat, lon (in degrees)'''
    dir_lat, dir_lon = all_dir_lat, all_dir_lon
    npix    = all_dir_cap_anom.shape[0]
    nside   = coords.get_nside(npix)
    if (all_dir_lat is None) or (all_dir_lon is None):
        dir_lat, dir_lon = coords.get_healpix_latlon(nside)
    cap_index   = su.find_nearest_index(geom_range, dir_cap_size)
    dir_weights = all_dir_cap_anom[:, cap_index]
    _filter     = su.get_top_cut_filter(dir_weights, top_ratio)
    avg_lat, avg_lon    = coords.average_dir_by_xyz(all_dir_lat[_filter],
                                                    all_dir_lon[_filter],
                                                    dir_weights[_filter])
    avg_i   = coords.get_pix_by_ang(nside, avg_lat, avg_lon)
    return dir_lat[avg_i], dir_lon[avg_i]

def find_dir_accumulative(all_dir_cap_anom,
                          top_ratio = 0.1,
                          all_dir_lat = None,
                          all_dir_lon = None):
    ''' lat, lon (in degrees)'''
    dir_lat = all_dir_lat
    dir_lon = all_dir_lon
    ndir    = all_dir_cap_anom.shape[0]
    nside   = coords.get_nside(ndir)
    if (all_dir_lat is None) or (all_dir_lon is None):
        dir_lat, dir_lon = coords.get_healpix_latlon(nside)
    dir_weights = np.sum(all_dir_cap_anom, axis = 0)
    _filter     = su.get_top_cut_filter(dir_weights, top_ratio)
    return coords.average_dir_by_xyz(dir_lat[_filter], dir_lon[_filter])