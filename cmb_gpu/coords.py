import cupy as cp

zero = cp.array(0)
one  = cp.array(1)


def convert_polar_to_xyz(lat_ndarray, lon_ndarray):
    theta, phi = cp.radians(90 - lat_ndarray), cp.radians(lon_ndarray)
    nx = cp.sin(theta) * cp.cos(phi)
    ny = cp.sin(theta) * cp.sin(phi)
    nz = cp.cos(theta)
    return cp.column_stack((nx,ny,nz))


def rotate_angle_axis(angle, axis, vec_ndarray):
    ux, uy, uz = axis
    I3_mat = cp.array([
        [one, zero, zero],
        [zero, one, zero],
        [zero, zero, one],
    ])
    cross_mat = cp.array([
        [zero, -uz, uy],
        [uz, zero, -ux],
        [-uy, ux, zero],
    ])
    dot_mat = cp.array([
        [ux * ux, ux * uy, ux * uz],
        [uy * ux, uy * uy, uy * uz],
        [uz * ux, uz * uy, uz * uz],
    ])
    rot_mat = \
        I3_mat * cp.cos(angle) + \
        cross_mat * cp.sin(angle) + \
        dot_mat * (1 - cp.cos(angle))
    return cp.transpose(cp.matmul(rot_mat , cp.transpose(vec_ndarray)))


def rotate_pole_to_north(pole_lat, pole_lon, vec_ndarray):
    pole = convert_polar_to_xyz(
        cp.array([pole_lat]),
        cp.array([pole_lon])
    )
    north = cp.array([0,0,1])
    angle = cp.arccos(cp.dot(pole, north))
    axis  = cp.cross(pole, north)[0]
    axis_length = cp.sqrt(cp.dot(axis, cp.transpose(axis)))
    axis /= axis_length
    return rotate_angle_axis(angle, axis, vec_ndarray)