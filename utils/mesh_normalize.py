import numpy as np
import sys
import os

sys.path.append('./auxiliary/')
import meshio_custom

'''
mesh = meshio_custom.read_obj('./f_043_vrt_24.obj')
points_origin = mesh['vertices']
faces_origin = mesh['faces']

points_max = np.max(points_origin, axis=0)
points_min = np.min(points_origin, axis=0)
points_origin = (points_origin - points_min) / (points_max - points_min)

normals_origin = np.zeros(points_origin.shape)
v10 = points_origin[faces_origin[:, 1]] - points_origin[faces_origin[:, 0]]
v20 = points_origin[faces_origin[:, 2]] - points_origin[faces_origin[:, 0]]
normals_origin_value = np.cross(v10, v20)
normals_origin[faces_origin[:,0]] += normals_origin_value[:]
normals_origin[faces_origin[:,1]] += normals_origin_value[:]
normals_origin[faces_origin[:,2]] += normals_origin_value[:]
normals_origin_len = np.sqrt(normals_origin[:,0]*normals_origin[:,0]+normals_origin[:,1]*normals_origin[:,1]+normals_origin[:,2]*normals_origin[:,2])
normals_origin = normals_origin / normals_origin_len.reshape(-1, 1)

mesh = meshio_custom.write_obj('./f_043_vrt_24_norm2.obj', points_origin, faces_origin, None, normals_origin)
'''


for i in range (34, 58):
    mesh_str = './data/normalize/m_' + str(i).zfill(7) + '.obj'
    if os.path.isfile(mesh_str):
        mesh = meshio_custom.read_obj(mesh_str)
        points_origin = mesh['vertices']
        faces_origin = mesh['faces']

        points_max = np.max(points_origin, axis=0)
        points_min = np.min(points_origin, axis=0)
        points_origin = 2. * (points_origin - points_min) / (points_max - points_min) - 1.

        normals_origin = np.zeros(points_origin.shape)
        v10 = points_origin[faces_origin[:, 1]] - points_origin[faces_origin[:, 0]]
        v20 = points_origin[faces_origin[:, 2]] - points_origin[faces_origin[:, 0]]
        normals_origin_value = np.cross(v10, v20)
        normals_origin[faces_origin[:,0]] += normals_origin_value[:]
        normals_origin[faces_origin[:,1]] += normals_origin_value[:]
        normals_origin[faces_origin[:,2]] += normals_origin_value[:]
        normals_origin_len = np.sqrt(normals_origin[:,0]*normals_origin[:,0]+normals_origin[:,1]*normals_origin[:,1]+normals_origin[:,2]*normals_origin[:,2])
        normals_origin = normals_origin / normals_origin_len.reshape(-1, 1)

        mesh = meshio_custom.write_obj('./data/normalize/m_' + str(i).zfill(7) + '_norm.obj', points_origin, faces_origin, None, normals_origin)