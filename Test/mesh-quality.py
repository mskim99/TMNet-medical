import numpy as np
import open3d as o3d
import copy


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

