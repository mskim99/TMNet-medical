import numpy as np
import auxiliary.meshio_custom as mioc
import torch
import os
import binvox_rw
from scipy import stats
import numpy as np

import extension.dist_chamfer as ext
distChamfer = ext.chamferDist()

CD_avg = 0.0
EMD_avg = 0.0
iou_avg = 0.0
f1_score_avg = 0.0
mesh_num = 0

for i in range (34, 58):
    gen_mesh_str = './data/evaluate/m_' + str(i).zfill(7) + '_gen.obj'
    gt_mesh_str = './data/evaluate/m_' + str(i).zfill(7) + '_norm.obj'
    gen_mesh_vox_str = './data/evaluate/m_' + str(i).zfill(7) + '_gen.binvox'
    gt_mesh_vox_str = './data/evaluate/m_' + str(i).zfill(7) + '_norm.binvox'

    if os.path.isfile(gen_mesh_str) and os.path.isfile(gt_mesh_str) and \
            os.path.isfile(gen_mesh_vox_str) and os.path.isfile(gt_mesh_vox_str):
        gen_mesh = mioc.read_obj(gen_mesh_str)
        gt_mesh = mioc.read_obj(gt_mesh_str)
        points_gen = gen_mesh['vertices']
        points_gt = gt_mesh['vertices']
        points_gen = torch.from_numpy(points_gen)
        points_gt = torch.from_numpy(points_gt)
        points_gen = points_gen.cuda().float()
        points_gt = points_gt.cuda().float()

        # Chamfer distance (CD)
        dist1, dist2, _, _ = distChamfer(points_gen.unsqueeze(dim=0), points_gt.unsqueeze(dim=0))
        CD = torch.mean(dist1) + torch.mean(dist2)

        # Earth-Mover distance (EMD)
        EMD = stats.wasserstein_distance(points_gen.cpu().view(-1), points_gt.cpu().view(-1))

        with open(gen_mesh_vox_str, 'rb') as f:
            gen_vox = binvox_rw.read_as_3d_array(f)

        with open(gt_mesh_vox_str, 'rb') as f:
            gt_vox = binvox_rw.read_as_3d_array(f)

        # Intersection over Union (IoU)
        gen_vox_data = torch.from_numpy(gen_vox.data).float()
        gt_vox_data = torch.from_numpy(gt_vox.data).float()

        volume_num = torch.sum(gen_vox_data)
        gt_volume_num = torch.sum(gt_vox_data)
        total_voxels = float(128 * 128 * 128)

        intersection = torch.sum(torch.ge(gen_vox_data.mul(gt_vox_data), 1e-4)).float()
        union = torch.sum(torch.ge(gen_vox_data.add(gt_vox_data), 1e-4)).float()

        iou = intersection / union

        TP = intersection / total_voxels
        TN = 1.0 - (union / total_voxels)
        FP = (volume_num - intersection) / total_voxels
        FN = (gt_volume_num - intersection) / total_voxels

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = (2. * precision * recall) / (precision + recall)

        print(str(i) + 'th mesh CD : ' + str(CD.item()))
        print(str(i) + 'th mesh EMD : ' + str(EMD.item()))
        print(str(i) + 'th mesh IoU : ' + str(iou.item()))
        print(str(i) + 'th mesh F1-score : ' + str(f1_score.item()))
        print()

        CD_avg = CD_avg + CD.item()
        EMD_avg = EMD_avg + EMD.item()
        iou_avg = iou_avg + iou.item()
        f1_score_avg = f1_score_avg + f1_score.item()

        mesh_num = mesh_num + 1

CD_avg = CD_avg / float(mesh_num)
EMD_avg = EMD_avg / float(mesh_num)
iou_avg = iou_avg / float(mesh_num)
f1_score_avg = f1_score_avg / float(mesh_num)

print()
print('average (CD) : ' + str(CD_avg))
print('average (EMD) : ' + str(EMD_avg))
print('average (IoU) : ' + str(iou_avg))
print('average (F1-score) : ' + str(f1_score_avg))
