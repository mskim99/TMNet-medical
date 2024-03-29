from __future__ import print_function
import argparse
import random
import numpy as np
import torch
from torch import autograd
import torch.optim as optim
import sys

sys.path.append('./auxiliary/')
from dataset_3D import *
from model_3D import *
from utils import *
from ply import *
import os
import json
import datetime
import visdom
import scipy.io as sio
from loss import *
import meshio_custom

sys.path.append('./utils/')
from split_mesh import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=420, help='number of epochs to train for') # 120
parser.add_argument('--epoch_decay', type=int, default=300, help='epoch to decay lr')
parser.add_argument('--epoch_decay2', type=int, default=400, help='epoch to decay lr for the second time')
parser.add_argument('--model', type=str, default='', help='model path from the pretrained model')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT point cloud') # 10000
parser.add_argument('--num_vertices', type=int, default=2562, help='number of vertices of the initial sphere')
parser.add_argument('--num_samples',type=int,default=5000, help='number of samples for error estimation') # 2500
parser.add_argument('--env', type=str, default="SVR_subnet1_1_split", help='visdom env')
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--tau', type=float, default=0.1, help='threshold to prune the faces')
parser.add_argument('--lambda_edge', type=float, default=1e-5, help='weight of edge loss') # 0.05
parser.add_argument('--lambda_smooth', type=float, default=5e-7, help='weight of smooth loss')
parser.add_argument('--lambda_normal', type=float, default=1e-3, help='weight of normal loss')
parser.add_argument('--lambda_uniform', type=float, default=1e-6, help='weight of uniform loss')
parser.add_argument('--pool', type=str, default='max', help='max or mean or sum')
parser.add_argument('--manualSeed', type=int, default=6185)
opt = parser.parse_args()
print(opt)

sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()

server = 'http://localhost/'
vis = visdom.Visdom(server=server, port=8886, env=opt.env, use_incoming_socket=False)
now = datetime.datetime.now()
save_path = opt.env
dir_name = os.path.join('./log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')
blue = lambda x: '\033[94m' + x + '\033[0m'
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.cuda.set_device(1)

dataset = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=True, class_choice='lumbar_vertebra_05')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=False, class_choice='lumbar_vertebra_05')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset)

name = 'sphere' + str(opt.num_vertices) + '.mat'
mesh = sio.loadmat('./data/' + name)
# name = 'sphere' + str(opt.num_vertices) + '.obj'
# mesh = meshio_custom.read_obj('./data/' + name)

faces = np.array(mesh['f'])
# faces = mesh['faces']
faces_cuda = torch.from_numpy(faces.astype(int)).type(torch.cuda.LongTensor)

vertices_sphere = np.array(mesh['v'])
# vertices_sphere = mesh['vertices']

vertices_sphere = (torch.cuda.FloatTensor(vertices_sphere)).transpose(0, 1).contiguous()
vertices_sphere = vertices_sphere.contiguous().unsqueeze(0)
edge_cuda = get_edges(faces)
# parameters = smoothness_loss_parameters(faces)

network = SVR_TMNet_Split()
network.apply(weights_init)

network.cuda()
if opt.model != '':
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in torch.load(opt.model).items() if (k in model_dict)}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    print(" Previous weight loaded ")
print(network)
network.cuda()

lrate = opt.lr
optimizer = optim.Adam([
    {'params': network.encoder.parameters()},
    {'params': network.estimate.parameters()},
    {'params': network.decoder.parameters()},
    # {'params': network.decoder_vol.parameters()}
], lr=lrate, weight_decay=1e-3)

train_CD_loss = AverageValueMeter()
val_CD_loss = AverageValueMeter()
train_l2_loss = AverageValueMeter()
val_l2_loss = AverageValueMeter()
train_CDs_loss = AverageValueMeter()
val_CDs_loss = AverageValueMeter()
train_vol_l1_loss = AverageValueMeter()
eval_vol_l1_loss = AverageValueMeter()


with open(logname, 'a') as f:
    f.write(str(network) + '\n')

train_CD_curve = []
val_CD_curve = []
train_l2_curve = []
val_l2_curve = []
train_CDs_curve = []
val_CDs_curve = []

CE_loss = torch.nn.CrossEntropyLoss()
l1_loss = torch.nn.L1Loss()

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_CD_loss.reset()
    train_CDs_loss.reset()
    train_l2_loss.reset()
    network.train()

    if epoch == opt.epoch_decay:
        optimizer = optim.Adam([
            {'params': network.encoder.parameters()},
            {'params': network.estimate.parameters()},
            {'params': network.decoder.parameters()},
            # {'params': network.decoder_vol.parameters()}
        ], lr=lrate / 10.0, weight_decay=1e-3)

    if epoch == opt.epoch_decay2:
        optimizer = optim.Adam([
            {'params': network.encoder.parameters()},
            {'params': network.estimate.parameters()},
            {'params': network.decoder.parameters()},
            # {'params': network.decoder_vol.parameters()}
        ], lr=lrate / 100.0, weight_decay=1e-3)

    torch.manual_seed(0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, normals, faces_gt, points_orig, name, cat = data
        img = img.cuda()
        img = img.unsqueeze(dim=0)
        img = img.float()

        choice = np.random.choice(points.size(1), opt.num_vertices, replace=False)
        points_choice = points[:, choice, :].contiguous()
        normals_choice = normals[:, choice, :].contiguous()

        points = points.cuda()
        normals = normals.cuda()
        points_choice = points_choice.cuda()
        normals_choice = normals_choice.cuda()
        faces_gt = torch.squeeze(faces_gt)
        faces_gt_cuda = faces_gt.cuda()
        edge_cuda_gt = get_edges(faces_gt.numpy())
        points = points.float()
        points_choice = points_choice.float()
        normals_choice = normals_choice.float()
        vertices_input = (vertices_sphere.reshape(img.size(0), vertices_sphere.size(1),
                                                 vertices_sphere.size(2)).contiguous())

        b_f_list_gt, points_choice_parts, b_f_list_gen, vertices_input_parts, range_part = split_mesh(points_choice, vertices_input, level=0)
        vol_part = split_volume(img, level=0)
        pointsRec_parts = network(vol_part, vertices_input_parts, mode='deform1')  # vertices_sphere 3*2562
        # pointsRec_parts, vol_recon = network(vol_part, vertices_input_parts, mode='deform1')  # vertices_sphere 3*2562
        pointsRec, _, CD_loss_part, _, _ = combine_meshes(pointsRec_parts, vertices_input_parts, points_choice_parts, range_part, b_f_list_gen, None, True, level=0)

        dist1, dist2, _, idx2 = distChamfer(points_choice, pointsRec)
        pointsRec_samples, _ = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
        dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples.detach())
        choice2 = np.random.choice(points.size(1), opt.num_samples, replace=False)
        error_GT = torch.sqrt(dist2_samples.detach()[:,choice2])
        error = network(vol_part, pointsRec_samples.detach()[:,choice2].transpose(1, 2), mode='estimate')

        pointsRec = torch.squeeze(pointsRec)
        normals_gen = torch.ones(pointsRec.shape).cuda()
        v10 = pointsRec[faces_cuda[:, 1]] - pointsRec[faces_cuda[:, 0]]
        v20 = pointsRec[faces_cuda[:, 2]] - pointsRec[faces_cuda[:, 0]]
        normals_gen_value = torch.cross(v10, v20)
        normals_gen[faces_cuda[:,0]] += normals_gen_value[:]
        normals_gen[faces_cuda[:,1]] += normals_gen_value[:]
        normals_gen[faces_cuda[:,2]] += normals_gen_value[:]
        normals_gen_len = torch.sqrt(normals_gen[:,0]*normals_gen[:,0]+normals_gen[:,1]*normals_gen[:,1]+normals_gen[:,2]*normals_gen[:,2])
        normals_gen = normals_gen / normals_gen_len.reshape(-1, 1)
        pointsRec = torch.unsqueeze(pointsRec, 0)

        CD_loss = torch.mean(dist1) + torch.mean(dist2)
        CDs_loss = torch.mean(dist1_samples) + torch.mean(dist2_samples)
        l2_loss = calculate_l2_loss(error, error_GT.detach())
        # edge_loss = get_edge_loss_stage1(pointsRec, edge_cuda.detach())
        edge_loss = get_edge_loss_stage1_whmr(pointsRec, points_orig, edge_cuda.detach(), edge_cuda_gt)
        # smoothness_loss = get_smoothness_loss_stage1(pointsRec, parameters)
        faces_cuda_bn = faces_cuda.unsqueeze(0).expand(pointsRec.size(0), faces_cuda.size(0),faces_cuda.size(1))
        # normal_loss = get_normal_loss(pointsRec, faces_cuda_bn, normals, idx2)
        normal_loss = get_normal_loss_mdf(normals_gen, normals_choice, idx2)

        # print(img)
        # print(vol_recon)

        # vol_l1_loss = l1_loss(img.view([256, 256, 256]), vol_recon)

        # uniform_loss_global, b_v_list_gen, b_v_list_gt = get_uniform_loss_global(pointsRec.squeeze().cpu().data.numpy(), points_orig.squeeze().cpu().data.numpy())
        # uniform_loss_local = get_uniform_loss_local(pointsRec.squeeze().cpu().data.numpy(),b_v_list_gen, b_v_list_gt)

        loss_net = CD_loss + opt.lambda_edge * edge_loss + l2_loss + opt.lambda_normal * normal_loss # + opt.lambda_uniform * uniform_loss_global * uniform_loss_local
        # loss_net = CD_loss + opt.lambda_edge * edge_loss + l2_loss + opt.lambda_normal * normal_loss

        loss_net.backward()
        train_CD_loss.update(CD_loss.item())
        train_CDs_loss.update(CDs_loss.item())
        train_l2_loss.update(l2_loss.item())
        # train_vol_l1_loss.update(vol_l1_loss.item())
        optimizer.step()

        # VIZUALIZE
        if i % 50 <= 0:
            vis.scatter(X=points[0].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            title="TRAIN_INPUT",
                            markersize=2,
                        ),
                        )
            vis.scatter(X=pointsRec[0].data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=2,
                        ),
                        )

        print(
            '[%d: %d/%d] train_cd_loss:  %f , CD_loss_part:  %f ,l2_loss: %f, edge_loss: %f, normal_loss: %f, vol_l1_loss: %f, loss_net: %f' %
            (epoch, i, len_dataset / opt.batchSize, CD_loss.item(), CD_loss_part, l2_loss.item(), edge_loss.item(), normal_loss.item(), 0.0, loss_net.item()))

        '''
        print(
            '[%d: %d/%d] train_cd_loss:  %f , CD_loss_part:  %f ,l2_loss: %f, edge_loss: %f, normal_loss: %f, loss_net: %f' %
            (epoch, i, len_dataset / opt.batchSize, CD_loss.item(), CD_loss_part, l2_loss.item(), edge_loss.item(),
             normal_loss.item(), loss_net.item()))
            '''

    train_CD_curve.append(train_CD_loss.avg)
    train_CDs_curve.append(train_CDs_loss.avg)
    train_l2_curve.append(train_l2_loss.avg)

    with torch.no_grad():
        val_CD_loss.reset()
        val_CDs_loss.reset()
        val_l2_loss.reset()
        for item in dataset_test.cat:
            dataset_test.perCatValueMeter[item].reset()

        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            img, points, normals, faces_gt, points_orig, name, cat = data
            img = img.cuda()
            img = img.unsqueeze(dim=0)
            img = img.float()

            points = points.cuda()
            normals = normals.cuda()
            faces_gt = torch.squeeze(faces_gt)
            faces_gt_cuda = faces_gt.cuda()
            edge_cuda_gt = get_edges(faces_gt.numpy())
            points = points.float()
            choice = np.random.choice(points.size(1), opt.num_vertices, replace=False)
            points_choice = points[:, choice, :].contiguous()
            points_choice = points_choice.float()
            vertices_input = (vertices_sphere.reshape(img.size(0), vertices_sphere.size(1),
                                                     vertices_sphere.size(2)).contiguous())
            b_f_list_gt, points_choice_parts, b_f_list_gen, vertices_input_parts, range_part = split_mesh(points_choice, vertices_input, level=0)
            vol_part = split_volume(img, level=0)
            # pointsRec_parts, vol_recon = network(vol_part, vertices_input_parts, mode='deform1')  # vertices_sphere 3*2562
            pointsRec_parts = network(vol_part, vertices_input_parts,mode='deform1')  # vertices_sphere 3*2562
            pointsRec, _, CD_loss_part, _, _ = combine_meshes(pointsRec_parts, vertices_input_parts, points_choice_parts, range_part, b_f_list_gen, None, True, level=0)

            # pointsRec = pointsRec.unsqueeze(dim=0)
            dist1, dist2, idx1, idx2 = distChamfer(points_choice, pointsRec)

            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples)
            error_GT = torch.sqrt(dist2_samples)
            error = network(vol_part, pointsRec_samples.detach().transpose(1, 2), mode='estimate')

            pointsRec = torch.squeeze(pointsRec)
            normals_gen = torch.ones(pointsRec.shape).cuda()
            v10 = pointsRec[faces_cuda[:, 1]] - pointsRec[faces_cuda[:, 0]]
            v20 = pointsRec[faces_cuda[:, 2]] - pointsRec[faces_cuda[:, 0]]
            normals_gen_value = torch.cross(v10, v20)
            normals_gen[faces_cuda[:, 0]] += normals_gen_value[:]
            normals_gen[faces_cuda[:, 1]] += normals_gen_value[:]
            normals_gen[faces_cuda[:, 2]] += normals_gen_value[:]
            normals_gen_len = torch.sqrt(normals_gen[:, 0] * normals_gen[:, 0] + normals_gen[:, 1] * normals_gen[:, 1] + normals_gen[:, 2] * normals_gen[:, 2])
            normals_gen = normals_gen / normals_gen_len.reshape(-1, 1)
            pointsRec = torch.unsqueeze(pointsRec, 0)

            CD_loss = torch.mean(dist1) + torch.mean(dist2)
            edge_loss = get_edge_loss_stage1_whmr(pointsRec, points_orig, edge_cuda, edge_cuda_gt)
            # smoothness_loss = get_smoothness_loss_stage1(pointsRec, parameters)
            l2_loss = calculate_l2_loss(error, error_GT.detach())
            CDs_loss = (torch.mean(dist1_samples)) + (torch.mean(dist2_samples))
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(error.size(0), faces_cuda.size(0), faces_cuda.size(1))
            # normal_loss = get_normal_loss(pointsRec, faces_cuda_bn, normals, idx2)
            normal_loss = get_normal_loss_mdf(normals_gen, normals, idx2)
            # vol_l1_loss = l1_loss(img.view([256, 256, 256]), vol_recon)

            # uniform_loss_global, b_v_list_gen, b_v_list_gt = get_uniform_loss_global(pointsRec.squeeze().cpu().data.numpy(), points_orig.squeeze().cpu().data.numpy())
            # uniform_loss_local = get_uniform_loss_local(b_v_list_gen.squeeze().cpu().data.numpy(), b_v_list_gt.squeeze().cpu().data.numpy())

            loss_net = CD_loss + opt.lambda_edge * edge_loss + l2_loss + opt.lambda_normal * normal_loss # + opt.lambda_uniform * uniform_loss_global * uniform_loss_local
            # loss_net = CD_loss + opt.lambda_edge * edge_loss + l2_loss + opt.lambda_normal * normal_loss

            val_CD_loss.update(CD_loss.item())
            dataset_test.perCatValueMeter[cat[0]].update(CDs_loss.item())
            val_l2_loss.update(l2_loss.item())
            val_CDs_loss.update(CDs_loss.item())
            # eval_vol_l1_loss.update(vol_l1_loss.item())

            if i % 25 == 0:
                vis.scatter(X=points[0].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title="VAL_INPUT",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsRec[0].data.cpu(),
                            win='VAL_INPUT_RECONSTRUCTED',
                            opts=dict(
                                title="VAL_INPUT_RECONSTRUCTED",
                                markersize=2,
                            ),
                            )

            print('[%d: %d/%d] val_cd_loss:  %f, CD_loss_part:  %f ,l2_loss: %f, edge_loss: %f, normal_loss: %f, vol_l1_loss: %f, loss_net: %f'
                  % (epoch, i, len(dataset_test) / opt.batchSize, CD_loss.item(), CD_loss_part, l2_loss.item(), edge_loss.item(), normal_loss.item(), 0.0, loss_net.item()))

            '''
            print(
                '[%d: %d/%d] val_cd_loss:  %f, CD_loss_part:  %f ,l2_loss: %f, edge_loss: %f, normal_loss: %f, loss_net: %f'
                % (epoch, i, len(dataset_test) / opt.batchSize, CD_loss.item(), CD_loss_part, l2_loss.item(), edge_loss.item(), normal_loss.item(), loss_net.item()))
                '''

        val_CD_curve.append(val_CD_loss.avg)
        val_l2_curve.append(val_l2_loss.avg)
        val_CDs_curve.append(val_CDs_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_CD_curve)), np.arange(len(val_CD_curve)))),
             Y=np.log(np.column_stack((np.array(train_CD_curve), np.array(val_CD_curve)))),
             win='CD_vertices',
             opts=dict(title="CD_vertices", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_l2_curve)), np.arange(len(val_l2_curve)))),
             Y=np.log(np.column_stack((np.array(train_l2_curve), np.array(val_l2_curve)))),
             win='L2_loss',
             opts=dict(title="L2_loss", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_CDs_curve)),np.arange(len(val_CDs_curve)))),
             Y=np.log(np.column_stack((np.array(train_CDs_curve),np.array(val_CDs_curve)))),
             win='CD_samples',
             opts=dict(title="CD_samples", legend=["train", "val"], markersize=2, ), )

    log_table = {
        "train_CD_loss": train_CD_loss.avg,
        "val_CD_loss": val_CD_loss.avg,
        "train_l2_loss": train_l2_loss.avg,
        "val_l2_loss": val_l2_loss.avg,
        "val_CDs_loss": val_CDs_loss.avg,
        # "train_vol_l1_loss": train_vol_l1_loss.avg,
        # "eval_vol_l1_loss": eval_vol_l1_loss.avg,
        "epoch": epoch,
        "lr": lrate,
    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
