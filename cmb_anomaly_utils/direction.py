'''
This module finds direction based on cap measures in different directions
'''
import numpy as np
import healpy as hp


from .stat_utils import find_nearest_index
from .coords import convert_polar_to_xyz


def get_healpix_latlon(ndir):
    dir_nside = int(np.sqrt(ndir / 12))
    dir_lon, dir_lat = hp.pix2ang(dir_nside, np.arange(ndir), lonlat = True)
    return dir_lat, dir_lon

def average_lon(lon_arr):
    x = np.cos(lon_arr)
    y = np.sin(lon_arr)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return np.arctan2(y_mean, x_mean) * 180 / np.pi

def average_directions_by_vec(vec_arr):
    pass

def average_dir_by_latlon(dir_lat : np.ndarray, dir_lon : np.ndarray):
    z_arr    = np.cos(dir_lat)
    z_mean   = np.mean(z_arr)
    lon_mean = average_lon(dir_lon)
    lat_mean = np.arccos(z_mean)
    return lat_mean, lon_mean


def find_dir_using_mac(all_dir_cap_anom,
                       special_cap_size:float = None,
                       geom_range = None,
                       all_dir_lat = None,
                       all_dir_lon = None):
    ''' lat, lon (in degrees)'''
    dir_lat = all_dir_lat
    dir_lon = all_dir_lon
    if (all_dir_lat is None) or (all_dir_lon is None):
        dir_lat, dir_lon = get_healpix_latlon(all_dir_cap_anom.shape[0])
    # MAC aligned
    if special_cap_size is None:
        index = np.unravel_index(np.nanargmax(all_dir_cap_anom),
                                 all_dir_cap_anom.shape)
        mac_i = index[0]
    else:
        # OR in specified cap size
        cap_index = find_nearest_index(geom_range, special_cap_size)
        mac_i =  np.nanargmax(all_dir_cap_anom[:, cap_index])
    return dir_lat[mac_i], dir_lon[mac_i]

def find_dir_accumulative(all_dir_cap_anom,
                          top_ratio = 0.1,
                          all_dir_lat = None,
                          all_dir_lon = None):
    ''' lat, lon (in degrees)'''
    dir_lat = all_dir_lat
    dir_lon = all_dir_lon
    if (all_dir_lat is None) or (all_dir_lon is None):
        dir_lat, dir_lon = get_healpix_latlon(all_dir_cap_anom.shape[0])
    dir_weights = np.sum(all_dir_cap_anom, axis = 0)
    _max, _min  = np.max(dir_weights), np.min(dir_weights)
    _min_weight = _max - top_ratio * (_max - _min)
    _screen     = dir_weights > _min_weight
    return average_dir_by_latlon(dir_lat[_screen], dir_lon[_screen])

