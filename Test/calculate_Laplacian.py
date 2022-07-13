import sys

sys.path.append('./auxiliary/')
from loss import *
import meshio_custom

mesh = meshio_custom.read_obj('./f_034_vrt_24.obj')
points_origin = mesh['vertices']
faces_origin = mesh['faces']