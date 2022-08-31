import numpy as np
import open3d as o3d
import copy
import math

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

edge_ratios = []
ERMSs = []
areas = []
ARs = []

num = 0
for i in range (34, 58):
    mesh_gt = o3d.io.read_triangle_mesh("./Test/eval/prop/m_00000" + str(i) + "_gen2.obj")

    if len(mesh_gt.vertices) <= 0:
        continue

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
    edge_ratios.append(edge_ratio)

    # Edge Root Mean Square
    edge_12 = vertices[tri_idx[:, 0]] - vertices[tri_idx[:, 1]]
    edge_13 = vertices[tri_idx[:, 0]] - vertices[tri_idx[:, 2]]
    edge_23 = vertices[tri_idx[:, 1]] - vertices[tri_idx[:, 2]]

    edge_lengths_12 = edge_12[:, 0] * edge_12[:, 0] + edge_12[:, 1] * edge_12[:, 1] + edge_12[:, 2] * edge_12[:, 2]
    edge_lengths_13 = edge_13[:, 0] * edge_13[:, 0] + edge_13[:, 1] * edge_13[:, 1] + edge_13[:, 2] * edge_13[:, 2]
    edge_lengths_23 = edge_23[:, 0] * edge_23[:, 0] + edge_23[:, 1] * edge_23[:, 1] + edge_23[:, 2] * edge_23[:, 2]

    ERMS = np.sqrt((edge_lengths_12 * edge_lengths_12 + edge_lengths_13 * edge_lengths_13 + edge_lengths_23 * edge_lengths_23) / 3)
    ERMSs.append(ERMS)

    # Area
    crs_tri = np.cross(edge_12, edge_13)
    area_tri = np.linalg.norm(crs_tri, axis=1)
    area_tri_avg = np.average(area_tri)

    area = area_tri - area_tri_avg
    areas.append(area)

    # Aspect Ratio
    AR = math.sqrt(3) * (edge_lengths_12 * edge_lengths_12 + edge_lengths_13 * edge_lengths_13 + edge_lengths_23 * edge_lengths_23)
    AR = AR / area_tri
    ARs.append(AR)

    num = num + 1

np_edge_ratios = np.array(edge_ratios)
np_ERMSs = np.concatenate(ERMSs, axis=0)
np_areas = np.concatenate(areas, axis=0)
np_ARs = np.concatenate(ARs, axis=0)

np_ERMSs = np_ERMSs.reshape(-1)
np_areas = np_areas.reshape(-1)
np_ARs = np_ARs.reshape(-1)

print(np_edge_ratios.shape)
print(np_ERMSs.shape)
print(np_areas.shape)
print(np_ARs.shape)

edge_ratio_min = np.min(np_edge_ratios)
edge_ratio_avg = np.average(np_edge_ratios)
edge_ratio_max = np.max(np_edge_ratios)

ERMS_min = np.min(np_ERMSs)
ERMS_avg = np.average(np_ERMSs)
ERMS_max = np.max(np_ERMSs)

area_min = np.min(np_areas)
area_avg = np.average(np_areas)
area_max = np.max(np_areas)

AR_min = np.min(np_ARs)
AR_avg = np.average(np_ARs)
AR_max = np.max(np_ARs)


print(num)

print("edge ratio min : " + str(edge_ratio_min))
print("edge ratio avg : " + str(edge_ratio_avg))
print("edge ratio max : " + str(edge_ratio_max))

print("ERMS min : " + str(ERMS_min))
print("ERMS avg : " + str(ERMS_avg))
print("ERMS max : " + str(ERMS_max))

print("Area min : " + str(area_min))
print("Area avg : " + str(area_avg))
print("Area max : " + str(area_max))

print("aspect ratio min : " + str(AR_min))
print("aspect ratio avg : " + str(AR_avg))
print("aspect ratio max : " + str(AR_max))
