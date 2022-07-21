import numpy as np
import auxiliary.meshio_custom as mioc
import torch
import os

import extension.dist_chamfer as ext
distChamfer = ext.chamferDist()

for i in range (34, 58):
    gen_mesh_str = './data/evaluate/m_' + str(i).zfill(7) + '_gen.obj'
    gt_mesh_str = './data/evaluate/m_' + str(i).zfill(7) + '_GT.obj'

    if os.path.isfile(gen_mesh_str) and os.path.isfile(gt_mesh_str):
        gen_mesh = mioc.read_obj(gen_mesh_str)
        gt_mesh = mioc.read_obj(gt_mesh_str)
        points_gen = gen_mesh['vertices']
        points_gt = gt_mesh['vertices']
        points_gen = torch.from_numpy(points_gen)
        points_gt = torch.from_numpy(points_gt)
        points_gen = points_gen.cuda().float()
        points_gt = points_gt.cuda().float()

        dist1, dist2, _, _ = distChamfer(points_gen.unsqueeze(dim=0), points_gt.unsqueeze(dim=0))
        CD = torch.mean(dist1) + torch.mean(dist2)

        print(str(i) + 'th mesh CD : ' + str(CD.item()))

