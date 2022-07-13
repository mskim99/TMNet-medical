import meshio
import numpy as np
import sys

sys.path.append('./auxiliary/')
import meshio_custom

mesh = meshio_custom.read_obj('./data/customShapeNet_mat/00000024/f_0000001.obj')
points = mesh['vertices']
points_max = np.max(points, axis=0)
points_min = np.min(points, axis=0)
points = (points - points_min) / (points_max - points_min)
print(points.shape)

'''
indices = np.random.randint(points.shape[0], size=1000)
points = points[indices,:]
print(mesh.get_cells_type("triangle"))
normals = mesh.cells
normals = normals[indices,:]
'''

b_range = np.array([[0.0, 0.5, 1.0+1e-5], [0.0, 0.5, 1.0+1e-5], [0.0, 0.5, 1.0+1e-5]])
b_v_num = np.zeros([(b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)])
b_v_list = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),),
                    dtype=object)
points_comp = points.copy()

b_v_idx = 0
for x_i in range(0, (b_range[0].shape[0] - 1)):
    for y_i in range(0, (b_range[1].shape[0] - 1)):
        for z_i in range(0, (b_range[2].shape[0] - 1)):
            print('x_range : ' + str(b_range[0][x_i]) + ' ' + str(b_range[0][x_i + 1]))
            print('y_range : ' + str(b_range[1][y_i]) + ' ' + str(b_range[1][y_i + 1]))
            print('z_range : ' + str(b_range[2][z_i]) + ' ' + str(b_range[0][z_i + 1]))
            print(' ')

            b_v_list[b_v_idx] = []
            for e_i in range(0, points_comp.shape[0]):
                if b_range[0][x_i] <= points_comp[e_i][0] < b_range[0][x_i + 1] and \
                        b_range[1][y_i] <= points_comp[e_i][1] < b_range[0][y_i + 1] and \
                        b_range[2][z_i] <= points_comp[e_i][2] < b_range[2][z_i + 1]:
                    b_v_list[b_v_idx].append(points_comp[e_i][:])
                    # points_comp = np.delete(points_comp, points_comp[e_i][:])

            b_v_list[b_v_idx] = np.array(b_v_list[b_v_idx])
            b_v_idx = b_v_idx + 1

v_num = 0
for i in range(0, b_v_idx):
    print(b_v_list[i].shape)
    v_num = v_num + b_v_list[i].shape[0]

print(v_num)