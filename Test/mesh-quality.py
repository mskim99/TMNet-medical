import numpy as np
import open3d as o3d
import copy

import sys
sys.path.append('./auxiliary/')
from utils import *


'''
knot_mesh = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
mesh.compute_vertex_normals()
mesh1 = copy.deepcopy(mesh)
mesh1.triangles = o3d.utility.Vector3iVector(
    np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
o3d.io.write_triangle_mesh("knot.obj", mesh)
o3d.io.write_triangle_mesh("knot_part.obj", mesh1)
'''

'''
mesh = o3d.io.read_triangle_mesh("cube.obj")
# mesh = o3d.geometry.TriangleMesh.create_box()
# mesh.compute_vertex_normals()
mesh_div_itr1 = mesh.subdivide_midpoint(number_of_iterations=1)
# o3d.io.write_triangle_mesh("cube_o3d.obj", mesh)
o3d.io.write_triangle_mesh("cube_div_itr1_o3d.obj", mesh_div_itr1)
'''
'''
knot_mesh = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
# print(np.asarray(mesh.vertices).shape)
mesh_div_itr4 = mesh.subdivide_midpoint(number_of_iterations=4)
mesh_simp = mesh.simplify_quadric_decimation(720)
o3d.io.write_triangle_mesh("knot_o3d.obj", mesh)
o3d.io.write_triangle_mesh("knot_div_itr4_o3d.obj", mesh_div_itr4)
o3d.io.write_triangle_mesh("knot_sim_720_o3d.obj", mesh_simp)
'''

mesh_gt = o3d.io.read_triangle_mesh("./Test/f_034_vrt_24_norm.obj")
# print(np.asarray(mesh_gt.vertices))
# print(np.asarray(mesh_gt.triangles))

vertices = np.asarray(mesh_gt.vertices)
tri_idx = np.asarray(mesh_gt.triangles)
edges = get_edges(np.asarray(mesh_gt.triangles)).cpu().numpy()

# Edge Ratio
edge_dists = vertices[edges[:, 0]] - vertices[edges[:, 1]]
edge_lengths = edge_dists[:, 0] * edge_dists[:, 0] + edge_dists[:, 1] * edge_dists[:, 1] + edge_dists[:, 2] * edge_dists[:, 2]

el_min = np.min(edge_lengths)
el_max = np.max(edge_lengths)

edge_ratio = el_min / el_max

print(edge_ratio)

# Edge Root Mean Square
edge_12 = vertices[tri_idx[:, 0]] - vertices[tri_idx[:, 1]]
edge_13 = vertices[tri_idx[:, 0]] - vertices[tri_idx[:, 2]]
edge_23 = vertices[tri_idx[:, 1]] - vertices[tri_idx[:, 2]]

ERMS = np.sqrt((edge_12 * edge_12 + edge_13 * edge_13 + edge_23 * edge_23) / 3)
ERMS = np.average(ERMS)
print(ERMS)

# Area
crs_tri = np.cross(edge_12, edge_13)
area_tri = np.linalg.norm(crs_tri, axis=1)
area_tri_avg = np.average(area_tri)

area = area_tri / area_tri_avg

print(np.min(area))
print(np.max(area))



