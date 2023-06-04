import numpy as np
from . import const

def angle_to_z(angle):
    return np.cos(angle * np.pi / 180)


def convert_polar_to_xyz(lat_ndarray, lon_ndarray):
    theta, phi = np.radians(90 - lat_ndarray), np.radians(lon_ndarray)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return np.column_stack((nx,ny,nz))


def rotate_angle_axis(vec_ndarray, angle, axis):
    ux, uy, uz = axis
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
    north = np.array([0.0, 0.0, 1.0])
    angle = np.arccos(np.dot(pole, north))
    if angle < const.ANG_THRESHOLD:
        return vec_ndarray
    axis  = np.cross(pole, north)[0]
    axis_length = np.sqrt(np.dot(axis, np.transpose(axis)))
    axis /= axis_length
    return rotate_angle_axis(vec_ndarray, angle, axis)