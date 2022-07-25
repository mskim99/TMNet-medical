import torch
import numpy as np
import sys

import cv2

sys.path.append('./extension/')
import dist_chamfer as ext
distChamfer = ext.chamferDist()

def split_mesh(points_choice, vertices_input, level=0):

    # Split Regions
    if level == 0:
        b_range = np.array([[-1.0, 1.0 + 1e-5], [-1.0, 1.0 + 1e-5], [-1.0, 1.0 + 1e-5]])
    else:
        b_range = np.array([[-1.0, 0.0, 1.0 + 1e-5], [-1.0, 0.0, 1.0 + 1e-5], [-1.0, 0.0, 1.0 + 1e-5]])

    b_f_list_gt = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),), dtype=object)
    points_choice_parts = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),), dtype=object)
    b_f_list_gen = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),), dtype=object)
    vertices_input_parts = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),), dtype=object)
    range_part = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),), dtype=object)

    b_v_i = 0
    for x_i in range(0, (b_range[0].shape[0] - 1)):
        for y_i in range(0, (b_range[1].shape[0] - 1)):
            for z_i in range(0, (b_range[2].shape[0] - 1)):

                # Comparison between vertices of ground truth mesh
                b_f_list_gt[b_v_i] = []
                points_choice_parts[b_v_i] = []
                for e_i in range(0, points_choice.shape[1]):
                    if b_range[0][x_i] <= points_choice[0, e_i, 0] < b_range[0][x_i + 1] and \
                            b_range[1][y_i] <= points_choice[0, e_i, 1] < b_range[1][y_i + 1] and \
                            b_range[2][z_i] <= points_choice[0, e_i, 2] < b_range[2][z_i + 1]:
                        b_f_list_gt[b_v_i].append(e_i)
                        points_choice_parts[b_v_i].append(points_choice[0, e_i, :])
                b_f_list_gt[b_v_i] = torch.tensor(b_f_list_gt[b_v_i])
                points_choice_parts[b_v_i] = torch.stack(points_choice_parts[b_v_i], dim=1)

                # Comparison between vertices of generated mesh
                b_f_list_gen[b_v_i] = []
                vertices_input_parts[b_v_i] = []
                for e_i in range(0, vertices_input.shape[2]):
                    if b_range[0][x_i] <= vertices_input[0, 0, e_i] < b_range[0][x_i + 1] and \
                            b_range[1][y_i] <= vertices_input[0, 1, e_i] < b_range[0][y_i + 1] and \
                            b_range[2][z_i] <= vertices_input[0, 2, e_i] < b_range[2][z_i + 1]:
                        b_f_list_gen[b_v_i].append(e_i)
                        vertices_input_parts[b_v_i].append(vertices_input[0, :, e_i])
                b_f_list_gen[b_v_i] = torch.tensor(b_f_list_gen[b_v_i])
                vertices_input_parts[b_v_i] = torch.stack(vertices_input_parts[b_v_i], dim=1)

                range_part[b_v_i] = [b_range[0][x_i], b_range[0][x_i + 1], b_range[1][y_i], b_range[1][y_i + 1],
                                     b_range[2][z_i], b_range[0][z_i + 1], ]

                vertices_input_parts[b_v_i] = vertices_input_parts[b_v_i].reshape(1,
                                                                                  vertices_input_parts[b_v_i].size(0),
                                                                                  vertices_input_parts[b_v_i].size(
                                                                                      1)).contiguous()

                b_v_i = b_v_i + 1

    return b_f_list_gt, points_choice_parts, b_f_list_gen, vertices_input_parts, range_part


def split_volume(vol, level=0):

    vol_part = np.empty(((level + 1) * (level + 1) * (level + 1)), dtype=object)

    # Split Regions
    if level == 0:
        b_range = np.array([[0, 256], [0, 256], [0, 256]])
    else:
        b_range = np.array([[0, 128, 256], [0, 128, 256], [0, 128, 256]])

    b_v_i = 0
    for x_i in range(0, (b_range[0].shape[0] - 1)):
        for y_i in range(0, (b_range[1].shape[0] - 1)):
            for z_i in range(0, (b_range[2].shape[0] - 1)):
                vol = torch.squeeze(vol)
                vp = vol[b_range[0][x_i]:b_range[0][x_i+1], b_range[1][y_i]:b_range[1][y_i+1], b_range[2][z_i]:b_range[2][z_i+1]]
                vp_resize = torch.zeros([256, 256, 256])
                if level > 0:
                    for i in range(0, vp.shape[2]):
                        vp_resize_part = vp[:, :, i]
                        vp_resize_part = cv2.resize(vp_resize_part.cpu().numpy(), (256, 256))
                        vp_resize[:, :, i] = torch.tensor(vp_resize_part)
                    vol_part[b_v_i] = torch.tensor(vp_resize)
                else:
                    vol_part[b_v_i] = vp
                vol_part[b_v_i] = vol_part[b_v_i].unsqueeze(dim=0)
                vol_part[b_v_i] = vol_part[b_v_i].unsqueeze(dim=0)
                vol_part[b_v_i] = vol_part[b_v_i].cuda()
                b_v_i = b_v_i + 1

    return vol_part

def combine_meshes(pointsRec_parts, vertices_input_parts, points_choice_parts, range_part, b_f_list_gen, faces=None, norm=True, level=0, scale=1.0):
    CD_loss_part = 0.0
    pointsRec = torch.tensor([])
    points_orig_recon = torch.tensor([])
    v_idx_recon = torch.tensor([])
    pointsRec = pointsRec.cuda()

    for b_v_idx in range(0, vertices_input_parts.shape[0]):

        if norm is True:
            b_v_r_min = torch.tensor([range_part[b_v_idx][0], range_part[b_v_idx][2], range_part[b_v_idx][4]])
            b_v_r_max = torch.tensor([range_part[b_v_idx][1], range_part[b_v_idx][3], range_part[b_v_idx][5]])
            b_v_r_min = b_v_r_min.float()
            b_v_r_max = b_v_r_max.float()
            b_v_r_min = b_v_r_min.cuda()
            b_v_r_max = b_v_r_max.cuda()

            pointsRec_max = torch.max(pointsRec_parts[b_v_idx], axis=1).values
            pointsRec_min = torch.min(pointsRec_parts[b_v_idx], axis=1).values
            pointsRec_parts[b_v_idx] = (pointsRec_parts[b_v_idx] - pointsRec_min) / (
                    pointsRec_max - pointsRec_min + 1e-4)
            if level == 0:
                pointsRec_parts[b_v_idx] = b_v_r_min + (b_v_r_max - b_v_r_min) * pointsRec_parts[b_v_idx]
            elif level == 1:
                pointsRec_parts[b_v_idx] = -0.5 + pointsRec_parts[b_v_idx]
                pointsRec_parts[b_v_idx] = scale * pointsRec_parts[b_v_idx]
                vip = vertices_input_parts[b_v_idx].reshape(1, vertices_input_parts[b_v_idx].size(2),
                                                            vertices_input_parts[b_v_idx].size(1))
                pointsRec_parts[b_v_idx] = pointsRec_parts[b_v_idx] + vip

        points_choice_parts[b_v_idx] = points_choice_parts[b_v_idx].reshape(1, points_choice_parts[b_v_idx].size(1),
                                                        points_choice_parts[b_v_idx].size(0))
        dist1_p, dist2_p, _, _ = distChamfer(pointsRec_parts[b_v_idx], points_choice_parts[b_v_idx].detach())
        CD_loss_part_bv = torch.mean(dist1_p) + torch.mean(dist2_p)
        CD_loss_part = CD_loss_part + CD_loss_part_bv

        if b_v_idx == 0:
            pointsRec = pointsRec_parts[b_v_idx]
            v_idx_recon = b_f_list_gen[b_v_idx]
            points_orig_recon = points_choice_parts[b_v_idx]
        else:
            pointsRec = torch.concat([pointsRec, pointsRec_parts[b_v_idx]], dim=1)
            v_idx_recon = torch.concat([v_idx_recon, b_f_list_gen[b_v_idx]], dim=0)
            points_orig_recon = torch.concat([points_orig_recon, points_choice_parts[b_v_idx]], dim=1)

    CD_loss_part = CD_loss_part / float(vertices_input_parts.shape[0])

    v_idx_recon = v_idx_recon.cuda()
    faces_recon = None

    if vertices_input_parts.shape[0] > 1:
        pointsRec_recon = torch.empty_like(pointsRec)
        pointsRec_recon[:, v_idx_recon, :] = pointsRec
    else:
        pointsRec_recon = pointsRec

    return pointsRec_recon, points_orig_recon, CD_loss_part, v_idx_recon, faces_recon