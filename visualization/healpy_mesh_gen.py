# this script will create the 3d file of HEALPix_sphere in obj format
# one can open the file in Blender or any other 3d software
import numpy as np
import healpy as hp

def file3d_obj_txt(element_arr, str_key):
    txt = ""
    space, next_line = " ", "\n"
    for elem in element_arr:
        txt += str_key + space + "".join(space + str(component) for component in elem) + next_line
    return txt

nside = 16
npix = 12 * nside**2
# 4 corners for each pixel and 3 coords for each corner
v_coords = np.zeros((npix * 4, 3))
for i in range(npix):
    pbounds = hp.boundaries(nside = nside, pix = i, step = 1, nest = False)
    v_coords[4 * i : 4 * (i + 1)] = np.transpose(pbounds)

v_indices = np.arange(len(v_coords)) + 1
faces = v_indices.reshape((npix, 4))

# optionally weld repetitious vertices here.
# welding will drastically decrease the volume of the file

# mesh creation
f3dname = "healpix_sphere.obj"
with open(f3dname,'w') as f3d:
    # before saving, change coords from (right handed)xyz to (left handed)xzy
    v_coords[:, 1], v_coords[:, 2] = np.copy(v_coords[:, 2]), -np.copy(v_coords[:, 1])
    ftxt = file3d_obj_txt(v_coords, "v")
    ftxt += file3d_obj_txt(faces, "f")
    f3d.write(ftxt)