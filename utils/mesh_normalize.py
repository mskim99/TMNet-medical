import numpy as np
import sys
import sklearn.preprocessing as sklp

sys.path.append('./auxiliary/')
import meshio_custom

mesh = meshio_custom.read_obj('./f_034_vrt_24.obj')
points_origin = mesh['vertices']
faces_origin = mesh['faces']

points_max = np.max(points_origin, axis=0)
points_min = np.min(points_origin, axis=0)
points_origin = (points_origin - points_min) / (points_max - points_min)

v10 = points_origin[faces_origin[:, 1]] - points_origin[faces_origin[:, 0]]
v20 = points_origin[faces_origin[:, 2]] - points_origin[faces_origin[:, 0]]
normals_origin = np.cross(v10, v20)
normals_origin = sklp.normalize(normals_origin, axis=1)

print(points_origin.shape)
print(faces_origin.shape)
print(normals_origin.shape)

mesh = meshio_custom.write_obj('./f_034_vrt_24_norm.obj', points_origin, faces_origin, None, normals_origin)