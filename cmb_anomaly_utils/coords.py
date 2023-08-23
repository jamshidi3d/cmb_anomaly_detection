import numpy as np
import healpy as hp

from . import const

# ------- Vector calculus methods -------
def angle_to_z(angle_deg):
    return np.cos(np.radians(angle_deg))

# ------- 2D Spherical methods -------
def convert_polar_to_spherical(lat, lon):
    theta, phi = np.deg2rad(90 - lat), np.deg2rad(lon)
    return theta, phi

def get_angle_dist_polar(lat1, lon1, lat2, lon2):
    '''returns in Degrees'''
    v1, v2 = convert_polar_to_xyz(  np.array([lat1, lat2]),
                                    np.array([lon1, lon2]))
    return get_angle_dist_xyz(np.array([v1]), np.array([v2]))
    
def average_lon(lon_arr):
    lon_arr_rad = np.radians(lon_arr)
    x = np.cos(lon_arr_rad)
    y = np.sin(lon_arr_rad)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return np.degrees(np.arctan2(y_mean, x_mean))

def average_dir_by_zphi(dir_lat : np.ndarray, dir_lon : np.ndarray):
    z_arr    = np.cos(90 - dir_lat)
    z_mean   = np.mean(z_arr)
    lon_mean = average_lon(dir_lon)
    lat_mean = np.arccos(z_mean)
    return lat_mean, lon_mean

def average_dir_by_xyz(dir_lat : np.ndarray, dir_lon : np.ndarray, dir_weights : np.ndarray = None):
    _weights = np.ones(len(dir_lat)) if dir_weights is None else dir_weights
    pos     = convert_polar_to_xyz(dir_lat, dir_lon)
    x, y, z = normalize_xyz(np.average(pos[:, 0], weights=_weights),
                            np.average(pos[:, 1], weights=_weights),
                            np.average(pos[:, 2], weights=_weights))
    lat_arr, lon_arr = convert_xyz_to_polar(combine_xyz(x, y, z))
    return lat_arr[0], lon_arr[0]

# ------- 3D methods -------
def combine_xyz(x, y, z) -> np.ndarray:
    return np.column_stack((x,y,z))

def separate_xyz(vec_ndarray):
    x, y, z = vec_ndarray[:, 0], vec_ndarray[:, 1], vec_ndarray[:, 2]
    return x, y, z

def normalize_xyz(x, y, z):
    r  = np.sqrt(x**2 + y**2 + z**2)
    return x/r, y/r, z/r

def normalize_vec(vec_ndarray):
    x, y, z     = separate_xyz(vec_ndarray)
    nx, ny, nz  = normalize_xyz(x, y, z)
    return combine_xyz(nx, ny, nz)

def dot_product(vec_nd1, vec_nd2):
    return np.dot(vec_nd1, np.transpose(vec_nd2))

def get_angle_dist_xyz(vec_nd1, vec_nd2):
    '''returns separation in Degrees, assuming vectors are normalized'''
    ang_arr = np.degrees(np.arccos(dot_product(vec_nd1, vec_nd2)))
    return ang_arr[0,0]

def convert_polar_to_xyz(lat_ndarray, lon_ndarray):
    theta, phi = np.radians(90 - lat_ndarray), np.radians(lon_ndarray)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return combine_xyz(nx,ny,nz)

def convert_xyz_to_polar(vec_ndarray):
    '''returns lat, lon '''
    x, y, z = separate_xyz(vec_ndarray)
    theta   = np.arccos(z)
    phi     = np.arctan2(y, x)
    lat, lon = 90 - np.degrees(theta), np.degrees(phi)
    return lat, lon

def rotate_angle_axis(vec_ndarray, angle, axis):
    x_arr, y_arr, z_arr = separate_xyz(axis)
    ux, uy, uz = x_arr[0], y_arr[0], z_arr[0]
    I3_mat = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    cross_mat = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0],
    ])
    dot_mat = np.array([
        [ux * ux, ux * uy, ux * uz],
        [uy * ux, uy * uy, uy * uz],
        [uz * ux, uz * uy, uz * uz],
    ])
    rot_mat = \
        I3_mat * np.cos(angle) + \
        cross_mat * np.sin(angle) + \
        dot_mat * (1 - np.cos(angle))
    return np.transpose(np.matmul(rot_mat , np.transpose(vec_ndarray)))

def rotate_pole_to_north(vec_ndarray, pole_lat, pole_lon):
    pole = convert_polar_to_xyz(
        np.array([pole_lat]),
        np.array([pole_lon])
    )
    north = combine_xyz(0.0, 0.0, 1.0)
    angle = np.arccos(np.dot(pole, north.transpose()))
    if angle < const.ANG_THRESHOLD:
        return vec_ndarray
    axis  = normalize_vec(np.cross(pole, north))
    return rotate_angle_axis(vec_ndarray, angle, axis)

# ------- Healpix methods -------
def get_nside(npix):
    return int(np.sqrt(npix / 12))

def get_npix(nside):
    return 12 * nside * nside

def get_healpix_xyz(nside = 64):
    npix     = np.arange(12 * nside **2)
    lon, lat = hp.pix2ang(nside, npix, lonlat = True)
    pos      = convert_polar_to_xyz(lat, lon)
    return pos

def get_healpix_latlon(nside):
    ndir = get_npix(nside)
    dir_lon, dir_lat = hp.pix2ang(nside, np.arange(ndir), lonlat = True)
    return dir_lat, dir_lon

def get_pix_by_ang(nside, lat, lon):
    theta, phi = convert_polar_to_spherical(lat, lon)
    pix_index = hp.pixelfunc.ang2pix(nside, theta, phi)
    return pix_index

def get_disc_indices(nside, disc_size, disc_lat, disc_lon):
    theta, phi = np.deg2rad(90 - disc_lat), np.deg2rad(disc_lon)
    _vec = hp.ang2vec(theta, phi)
    ipix_disc = hp.query_disc(nside= nside, vec=_vec, radius=np.radians(disc_size))
    return ipix_disc

# ------- Pixel Rotation -------
def rotate_pixels_pole_to_north(data_arr, pole_lat, pole_lon):
    theta, phi = convert_polar_to_spherical(pole_lat, pole_lon)
    euler_rot_angles = np.array([phi, -theta, 0])
    r = hp.rotator.Rotator(rot = euler_rot_angles, deg=False, eulertype='ZYX')
    data_rotated = r.rotate_map_pixel(data_arr)
    return data_rotated

def rotate_pixels_north_to_pole(data_arr, pole_lat, pole_lon):
    theta, phi = convert_polar_to_spherical(pole_lat, pole_lon)
    euler_rot_angles = np.array([phi, -theta, 0])
    r = hp.rotator.Rotator(rot = euler_rot_angles, inv=True, deg=False, eulertype='ZYX')
    data_rotated = r.rotate_map_pixel(data_arr)
    return data_rotated

