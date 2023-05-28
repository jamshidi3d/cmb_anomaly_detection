import numpy as np
from astropy_healpix import HEALPix

# this script will create the 3d file of HEALPix_sphere in obj format
# one can open the file in Blender or any other 3d software

def convert_geo_to_xyz(lat_ndarray, lon_ndarray):
    theta, phi = np.radians(90 - lat_ndarray), np.radians(lon_ndarray)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return np.column_stack((nx,ny,nz))

def file3d_obj_txt(element_arr, str_key):
    txt = ""
    space, next_line = " ", "\n"
    for elem in element_arr:
        txt += str_key + "".join(space + str(component) for component in elem) + next_line
    return txt


nside = 16
hp = HEALPix(nside=nside, order='ring') # or 'nested'
boundaries = hp.boundaries_lonlat(np.arange(12*nside**2), step=1)

lon = np.array(boundaries[0].to_value())
lat = np.array(boundaries[1].to_value())

nfaces = len(lon)
lon = lon.reshape(4 * nfaces) * 180 / np.pi
lat = lat.reshape(4 * nfaces) * 180 / np.pi

v_coords = convert_geo_to_xyz(lat, lon)
v_indices = np.arange(len(v_coords)) + 1
faces = v_indices.reshape((nfaces, 4))

# optionally weld repetitious vertices.
# welding will drastically decrease the volume of the file
f3dname = "healpix_sphere.obj"
# mesh creation
with open(f3dname,'w') as f3d:
    # before saving, change coords from (right handed)xyz to (left handed)xzy
    v_coords[:, 1], v_coords[:, 2] = np.copy(v_coords[:, 2]), -np.copy(v_coords[:, 1])
    ftxt = file3d_obj_txt(v_coords, "v")
    ftxt += file3d_obj_txt(faces, "f")
    f3d.write(ftxt)