import numpy as np
import healpy as hp


from .stat_utils import find_nearest_index
from .coords import convert_polar_to_xyz


def get_healpix_lat_lon(ndir):
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
                       geom_range = None):
    ''' lat, lon (in degrees)'''
    print(all_dir_cap_anom.shape)
    # The first index determines direction
    ndir = all_dir_cap_anom.shape[0]
    dir_lat, dir_lon = get_healpix_lat_lon(ndir)
    # Select the cap size in which the anomaly is the most
    if special_cap_size is None:
        index = np.unravel_index(np.nanargmax(all_dir_cap_anom),
                                 all_dir_cap_anom.shape)
        mad_i = index[0]
    else:
        # OR mad in specified cap size
        cap_index = find_nearest_index(geom_range, special_cap_size)
        print(geom_range)
        print(cap_index)
        mad_i =  np.nanargmax(all_dir_cap_anom[:, cap_index])
    return dir_lat[mad_i], dir_lon[mad_i]

def find_dir_accumulative(all_dir_cap_anom, top_ratio = 0.1):
    dir_weights = np.sum(all_dir_cap_anom, axis = 0)
    _max, _min  = np.max(dir_weights), np.min(dir_weights)
    _min_weight = _max - top_ratio * (_max - _min)
    _screen     = dir_weights > _min_weight
    ndir        = len(dir_weights)
    dir_lat, dir_lon = get_healpix_lat_lon(ndir)
    return average_dir_by_latlon(dir_lat[_screen], dir_lon[_screen])

