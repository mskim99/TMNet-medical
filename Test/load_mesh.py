import meshio
import numpy as np

mesh = meshio.read('./data/customShapeNet_mat/00000024/f_0000001.obj')
points = mesh.points
points_max = np.max(points)
points = points / points_max
indices = np.random.randint(points.shape[0], size=1000)
points = points[indices,:]
print(mesh.get_cells_type("triangle"))
normals = mesh.cells
normals = normals[indices,:]