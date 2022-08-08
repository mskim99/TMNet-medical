import open3d as o3d
import numpy as np
import copy

import pyvista

def split_mesh(orig_mesh, thres=np.array([0.5, 0.5, 0.5]), downer=np.array([True, True, True]), eps=0.):

    # the crop parts
    crop_triangle = []
    crop_triangle_normal = []

    vert = np.asarray(orig_mesh.vertices)
    tri = np.asarray(orig_mesh.triangles)
    norm = np.asarray(orig_mesh.triangle_normals)

    for i in range(0, tri.shape[0]):
        if downer[0]:
            if vert[tri[i, 0], 0] < thres[0] + eps and vert[tri[i, 1], 0] < thres[0] + eps and vert[tri[i, 2], 0] < thres[0] + eps:
                if downer[1]:
                    if vert[tri[i, 0], 1] < thres[1] + eps and vert[tri[i, 1], 1] < thres[1] + eps and vert[tri[i, 2], 1] < thres[1] + eps:
                        if downer[2]:
                            if vert[tri[i, 0], 2] < thres[2] + eps and vert[tri[i, 1], 2] < thres[2] + eps and vert[tri[i, 2], 2] < thres[2] + eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])
                        else:
                            if vert[tri[i, 0], 2] >= thres[2] - eps and vert[tri[i, 1], 2] >= thres[2] - eps and vert[tri[i, 2], 2] >= thres[2] - eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])
                else:
                    if vert[tri[i, 0], 1] >= thres[1] - eps and vert[tri[i, 1], 1] >= thres[1] - eps and vert[tri[i, 2], 1] >= thres[1] - eps:
                        if downer[2]:
                            if vert[tri[i, 0], 2] < thres[2] + eps and vert[tri[i, 1], 2] < thres[2] + eps and vert[tri[i, 2], 2] < thres[2] + eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])
                        else:
                            if vert[tri[i, 0], 2] >= thres[2] - eps and vert[tri[i, 1], 2] >= thres[2] - eps and vert[tri[i, 2], 2] >= thres[2] - eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])
        else:
            if vert[tri[i, 0], 0] >= thres[0] - eps and vert[tri[i, 1], 0] >= thres[0] - eps and vert[tri[i, 2], 0] >= thres[0] - eps:
                if downer[1]:
                    if vert[tri[i, 0], 1] < thres[1] + eps and vert[tri[i, 1], 1] < thres[1] + eps and vert[tri[i, 2], 1] < thres[1] + eps:
                        if downer[2]:
                            if vert[tri[i, 0], 2] < thres[2] + eps and vert[tri[i, 1], 2] < thres[2] + eps and vert[tri[i, 2], 2] < thres[2] + eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])
                        else:
                            if vert[tri[i, 0], 2] >= thres[2] - eps and vert[tri[i, 1], 2] >= thres[2] - eps and vert[tri[i, 2], 2] >= thres[2] - eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])
                else:
                    if vert[tri[i, 0], 1] >= thres[1] - eps and vert[tri[i, 1], 1] >= thres[1] - eps and vert[tri[i, 2], 1] >= thres[1] - eps:
                        if downer[2]:
                            if vert[tri[i, 0], 2] < thres[2] + eps and vert[tri[i, 1], 2] < thres[2] + eps and vert[tri[i, 2], 2] < thres[2] + eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])
                        else:
                            if vert[tri[i, 0], 2] >= thres[2] - eps and vert[tri[i, 1], 2] >= thres[2] - eps and vert[tri[i, 2], 2] >= thres[2] - eps:
                                crop_triangle.append(tri[i, :])
                                crop_triangle_normal.append(norm[i, :])

    crop_triangle = np.array(crop_triangle).astype(np.int32)
    crop_triangle_normal = np.array(crop_triangle_normal).astype(np.float)

    mesh1.vertices = mesh.vertices
    mesh1.triangles = o3d.utility.Vector3iVector(crop_triangle)
    # mesh1.triangle_normals = o3d.utility.Vector3iVector(crop_triangle_normal)
    mesh1.compute_vertex_normals()

    return mesh1


mesh = o3d.io.read_triangle_mesh("f_034_vrt_24_norm.obj")
mesh.compute_vertex_normals()

# Print related information of mesh
'''
print(np.asarray(mesh.vertices).shape) # Vertices
print(np.asarray(mesh.triangles).shape) # Faces
print(np.asarray(mesh.triangle_normals).shape) # Normals

print(np.asarray(mesh.vertices)) # Vertices
print(np.asarray(mesh.triangles)) # Faces
print(np.asarray(mesh.triangle_normals)) # Normals
'''

mesh1 = copy.deepcopy(mesh)
mesh2 = copy.deepcopy(mesh)
mesh3 = copy.deepcopy(mesh)
mesh4 = copy.deepcopy(mesh)
mesh5 = copy.deepcopy(mesh)
mesh6 = copy.deepcopy(mesh)
mesh7 = copy.deepcopy(mesh)
mesh8 = copy.deepcopy(mesh)

'''
mesh1.triangles = o3d.utility.Vector3iVector(
    np.asarray(mesh1.triangles)[0:len(mesh1.triangles) // 2, 0:len(mesh1.triangles) // 2])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[0:len(mesh1.triangle_normals) // 2, 0:len(mesh1.triangles) // 2])
    '''

threshold = np.array([0.5, 0.5, 0.5])  # z threshold
# the crop parts
crop_triangle = []
crop_triangle_normal = []

'''
mesh1 = o3d.geometry.TriangleMesh()

vert = np.asarray(mesh.vertices)
tri = np.asarray(mesh.triangles)
norm = np.asarray(mesh.triangle_normals)
for i in range (0, tri.shape[0]):
    if all(vert[tri[i, 0], :] < threshold) and all(vert[tri[i, 1], :] < threshold) and all(vert[tri[i, :], 2] < threshold):
        crop_triangle.append(tri[i, :])
        crop_triangle_normal.append(norm[i, :])
crop_triangle = np.array(crop_triangle).astype(np.int32)
crop_triangle_normal = np.array(crop_triangle_normal).astype(np.float)

mesh1.vertices = mesh.vertices
mesh1.triangles = o3d.utility.Vector3iVector(crop_triangle)
# mesh1.triangle_normals = o3d.utility.Vector3iVector(crop_triangle_normal)
mesh1.compute_vertex_normals()

print(np.asarray(mesh1.vertices).shape) # Vertices
print(np.asarray(mesh1.triangles).shape) # Faces
print(np.asarray(mesh1.triangle_normals).shape) # Normals
'''
mesh1 = split_mesh(mesh1, np.array([0.5, 0.5, 0.5]), np.array([True, True, True]), 0.025)
mesh1 = mesh1.simplify_quadric_decimation(int(np.asarray(mesh1.triangles).shape[0] / 2))
mesh2 = split_mesh(mesh2, np.array([0.5, 0.5, 0.5]), np.array([False, True, True]), 0.025)
mesh3 = split_mesh(mesh3, np.array([0.5, 0.5, 0.5]), np.array([True, False, True]), 0.025)
mesh4 = split_mesh(mesh4, np.array([0.5, 0.5, 0.5]), np.array([False, False, True]), 0.025)
mesh4 = mesh4.subdivide_midpoint(number_of_iterations=1)
mesh5 = split_mesh(mesh5, np.array([0.5, 0.5, 0.5]), np.array([True, True, False]), 0.025)
mesh6 = split_mesh(mesh6, np.array([0.5, 0.5, 0.5]), np.array([False, True, False]), 0.025)
mesh7 = split_mesh(mesh7, np.array([0.5, 0.5, 0.5]), np.array([True, False, False]), 0.025)
mesh8 = split_mesh(mesh8, np.array([0.5, 0.5, 0.5]), np.array([False, False, False]), 0.025)

mesh_recon = mesh1 + mesh2 + mesh3 + mesh4 + mesh5 + mesh6 + mesh7 + mesh8
mesh_recon = mesh_recon.remove_duplicated_triangles()
mesh_recon = mesh_recon.remove_non_manifold_edges()
mesh_recon = mesh_recon.remove_degenerate_triangles()

half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh1)
'''
half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh()
half_edge_mesh.vertices = mesh_recon.vertices
half_edge_mesh.triangles = mesh_recon.triangles
half_edge_mesh.triangle_normals = mesh_recon.triangle_normals
'''
'''
boundaries = half_edge_mesh.get_boundaries()
boundary_vertices = boundaries[0]

boundary_mesh = o3d.geometry.PointCloud()
vert_recon = np.asarray(half_edge_mesh.vertices)
boundary_mesh.points = o3d.utility.Vector3dVector(vert_recon[np.asarray(boundaries[0])])
o3d.io.write_point_cloud("f_034_vrt_24_norm_crop_recon_boundary.ply", boundary_mesh)
'''

o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_1.obj", mesh1)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_2.obj", mesh2)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_3.obj", mesh3)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_4.obj", mesh4)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_5.obj", mesh5)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_6.obj", mesh6)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_7.obj", mesh7)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_8.obj", mesh8)
o3d.io.write_triangle_mesh("f_034_vrt_24_norm_crop_recon.obj", mesh_recon)