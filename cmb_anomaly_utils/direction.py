'''
This module finds direction based on cap measures in different directions
'''
import numpy as np

from .stat_utils import find_nearest_index
from . import coords

def average_lon(lon_arr):
    lon_arr_rad = np.radians(lon_arr)
    x = np.cos(lon_arr_rad)
    y = np.sin(lon_arr_rad)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return np.degrees(np.arctan2(y_mean, x_mean))

def get_top_cut_filter(val_arr, top_ratio = 0.1):
    '''val_arr (e.g. weight, measure)'''
    _max, _min  = np.max(val_arr), np.min(val_arr)
    _min_weight = _max - top_ratio * (_max - _min)
    _filter     = val_arr > _min_weight
    return _filter

def average_dir_by_zphi(dir_lat : np.ndarray, dir_lon : np.ndarray):
    z_arr    = np.cos(dir_lat)
    z_mean   = np.mean(z_arr)
    lon_mean = average_lon(dir_lon)
    lat_mean = np.arccos(z_mean)
    return lat_mean, lon_mean

def average_dir_by_xyz(dir_lat : np.ndarray, dir_lon : np.ndarray):
    pos = coords.convert_polar_to_xyz(dir_lat, dir_lon)
    _x, _y, _z = np.mean(pos[:, 0]), np.mean(pos[:, 1]), np.mean(pos[:, 2])
    _r  = np.sqrt(_x**2 + _y**2 + _z**2)
    _x, _y, _z = _x/_r, _y/_r, _z/_r
    lat_arr, lon_arr = coords.convert_xyz_to_polar(np.array([_x]), np.array([_y]), np.array([_z]))
    return lat_arr[0], lon_arr[0]

def find_dir_by_mac(all_dir_cap_anom,
                    special_cap_size:float = None,
                    geom_range = None,
                    all_dir_lat = None,
                    all_dir_lon = None):
    ''' lat, lon (in degrees)'''
    dir_lat = all_dir_lat
    dir_lon = all_dir_lon
    npix    = all_dir_cap_anom.shape[0]
    if (all_dir_lat is None) or (all_dir_lon is None):
        dir_lat, dir_lon = coords.get_healpix_latlon(npix)
    cap_index   = find_nearest_index(geom_range, special_cap_size)
    dir_weights = all_dir_cap_anom[:, cap_index]
    _filter     = get_top_cut_filter(dir_weights, 0.1)
    avg_lat, avg_lon    = average_dir_by_xyz( all_dir_lat[_filter],
                                            all_dir_lon[_filter],
                                            dir_weights[_filter])
    mac_i       = coords.get_pix_by_ang(coords.get_nside(npix),
                                        avg_lat,
                                        avg_lon)
    return dir_lat[mac_i], dir_lon[mac_i]

def find_dir_accumulative(all_dir_cap_anom,
                          top_ratio = 0.1,
                          all_dir_lat = None,
                          all_dir_lon = None):
    ''' lat, lon (in degrees)'''
    dir_lat = all_dir_lat
    dir_lon = all_dir_lon
    if (all_dir_lat is None) or (all_dir_lon is None):
        dir_lat, dir_lon = coords.get_healpix_latlon(all_dir_cap_anom.shape[0])
    dir_weights = np.sum(all_dir_cap_anom, axis = 0)
    _filter     = get_top_cut_filter(dir_weights, top_ratio)
    return average_dir_by_xyz(dir_lat[_filter], dir_lon[_filter])