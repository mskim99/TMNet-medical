from __future__ import print_function
import argparse
import random
import numpy as np
import torch
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
from data_transforms import *

sys.path.append('./utils/')
from split_mesh import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=420, help='number of epochs to train for')
parser.add_argument('--epoch_decay',type=int,default=100, help='epoch to decay lr ')
parser.add_argument('--model', type=str,default='./log/SVR_subnet2_usage2/network.pth',help='model path from the trained subnet1')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT point cloud')
parser.add_argument('--num_vertices', type=int, default=2562)
parser.add_argument('--num_samples',type=int,default=5000, help='number of samples for error estimation')
parser.add_argument('--env', type=str, default="SVR_subnet3_1_split", help='visdom env')
parser.add_argument('--lr', type=float, default=1e-9)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--tau_decay', type=float, default=2.0)
parser.add_argument('--lambda_normal', type=float, default=5e-3)
parser.add_argument('--lambda_edge', type=float, default=1e-5)
parser.add_argument('--lambda_smooth', type=float, default=2e-7)
parser.add_argument('--lambda_uniform_glob', type=float, default=1e-4, help='weight of global uniform loss')
parser.add_argument('--pool',type=str,default='max',help='max or mean or sum' )
parser.add_argument('--manualSeed',type=int,default=6185)
opt = parser.parse_args()
print(opt)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.cuda.set_device(0)

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

dataset = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=True,class_choice='lumbar_vertebra_05')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=False,class_choice='lumbar_vertebra_05')
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
network.estimate3 = network.estimate2
print(network)
network.cuda()

lrate = opt.lr
# optimizer = optim.Adam(network.refine.parameters(), lr=lrate / 10.0)
optimizer = optim.Adam([
    {'params':network.decoder3.parameters()},
    {'params':network.estimate3.parameters()}
    ], lr = 1e-7)

'''
optimizer2 = optim.Adam([
    {'params': network.decoder4.parameters()}
    ], lr = 1e-7, weight_decay=0.025)
    '''

train_l2_loss = AverageValueMeter()
val_l2_loss = AverageValueMeter()
train_CDs_stage3_loss = AverageValueMeter()
val_CDs_stage3_loss = AverageValueMeter()

with open(logname, 'a') as f:
    f.write(str(network) + '\n')

train_l2_curve = []
val_l2_curve = []
train_CDs_stage3_curve = []
val_CDs_stage3_curve = []

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_CDs_stage3_loss.reset()
    train_l2_loss.reset()
    network.eval()
    network.decoder3.train()
    # network.decoder4.train()
    network.estimate2.train()

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        # optimizer2.zero_grad()
        img, points, normals, faces_gt, points_orig, name, cat = data
        # img = train_transforms(img)
        img = img.cuda()
        img = img.unsqueeze(dim=0)
        img = img.float()

        points = points.cuda()
        normals = normals.cuda()
        faces_gt = torch.squeeze(faces_gt)
        faces_gt_cuda = faces_gt.cuda()
        edge_cuda_gt = get_edges(faces_gt.numpy())
        points = points.float()
        normals = normals.float()
        choice = np.random.choice(points.size(1), opt.num_vertices, replace=False)
        points_choice = points[:, choice, :].contiguous()
        normals_choice = normals[:, choice, :].contiguous()
        vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1), vertices_sphere.size(2)).contiguous())

        with torch.no_grad():
            b_f_list_gt, points_choice_parts, b_f_list_gen, vertices_input_parts, range_part = split_mesh(points_choice, vertices_input, level=0)
            vol_part = split_volume(img, level=0)
            pointsRec_parts = network(vol_part, vertices_input_parts, mode='deform1')
            pointsRec, _, _, _, _ = combine_meshes(pointsRec_parts, vertices_input_parts, points_choice_parts, range_part, b_f_list_gen, None, True, level=0)

            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            error_stage1 = network(vol_part, pointsRec_samples.detach().transpose(1, 2), mode='estimate')
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(img.size(0), faces_cuda.size(0), faces_cuda.size(1))

            b_f_list_gt2, points_choice_parts2, b_f_list_gen2, pointsRec_parts, range_part2 = split_mesh(points_choice, pointsRec.transpose(2, 1), level=0)
            vol_part = split_volume(img, level=0)
            pointsRec2_parts = network(vol_part, pointsRec_parts, mode='deform2')
            pointsRec2, _, _, _, _ = combine_meshes(pointsRec2_parts, pointsRec_parts, points_choice_parts2, range_part2,
                                                    b_f_list_gen2, faces_cuda_bn, False, level=0, scale=1.)

            pointsRec2_samples, index2 = samples_random(faces_cuda_bn.detach(), pointsRec2, opt.num_points)
            error_stage2 = network(vol_part, pointsRec2_samples.detach()[:, choice].transpose(1, 2), mode='estimate2')

        b_f_list_gt3, points_choice_parts3, b_f_list_gen3, pointsRec2_parts, range_part3 = split_mesh(points_choice, pointsRec2.transpose(2,1), level=1)
        vol_part = split_volume(img, level=1)
        pointsRec3_parts = network(vol_part, pointsRec2_parts, mode='deform3_mlt')

        pointsRec3, _, _, _, _ = combine_meshes(pointsRec3_parts, np.array(pointsRec2_parts), points_choice_parts3,
                                                            range_part3, b_f_list_gen3, faces_cuda_bn, False, level=1, scale=1.)

        pointsRec3, trianglesRec3, CD_loss_part3 = combine_meshes_simp_dec(pointsRec3_parts, pointsRec3, points_choice_parts3, b_f_list_gen3, faces_cuda_bn)
        pointsRec3 = torch.tensor(pointsRec3).unsqueeze(0).float().cuda()
        trianglesRec3 = torch.tensor(trianglesRec3).unsqueeze(0).float().cuda()

        edge_cuda_rec3 = get_edges(trianglesRec3.detach().cpu().squeeze(0).numpy().reshape(-1, 3))

        _, _, _, idx3 = distChamfer(points, pointsRec3)
        _, _, _, idx3_choice = distChamfer(points_choice, pointsRec3)

        # pointsRec3_samples, _ = samples_random(faces_cuda_bn.detach(), pointsRec3, opt.num_points)
        pointsRec3_samples, _ = samples_random(trianglesRec3.detach().int(), pointsRec3, opt.num_points)
        dist13_samples, dist23_samples, _, _ = distChamfer(points, pointsRec3_samples)
        choice3 = np.random.choice(points.size(1), opt.num_samples, replace=False)
        error_GT = torch.sqrt(dist23_samples.detach()[:,choice3])
        error = network(vol_part, pointsRec3_samples.detach()[:,choice3].transpose(1, 2), mode='estimate3')

        pointsRec3 = torch.squeeze(pointsRec3)
        trianglesRec3 = trianglesRec3.long()
        normals_gen = torch.zeros(pointsRec3.shape).cuda()
        v10 = pointsRec3[trianglesRec3[:, :, 1]] - pointsRec3[trianglesRec3[:, :, 0]]
        v20 = pointsRec3[trianglesRec3[:, :, 2]] - pointsRec3[trianglesRec3[:, :, 0]]
        normals_gen_value = torch.cross(v10, v20)
        normals_gen[trianglesRec3[:, :, 0]] += normals_gen_value[:]
        normals_gen[trianglesRec3[:, :, 1]] += normals_gen_value[:]
        normals_gen[trianglesRec3[:, :, 2]] += normals_gen_value[:]
        normals_gen_len = torch.sqrt(normals_gen[:,0]*normals_gen[:,0]+normals_gen[:,1]*normals_gen[:,1]+normals_gen[:,2]*normals_gen[:,2])
        normals_gen = normals_gen / normals_gen_len.reshape(-1, 1)
        pointsRec3 = torch.unsqueeze(pointsRec3, 0)

        cds_stage3 = (torch.mean(dist13_samples)) + (torch.mean(dist23_samples))
        l2_loss = calculate_l2_loss(error, error_GT.detach())
        # normal_loss = get_normal_loss(pointsRec2, faces_cuda_bn, normals, idx2)
        normal_loss = get_normal_loss_mdf(normals_gen, normals_choice, idx3_choice)
        # edge_loss = get_edge_loss(pointsRec2, faces_cuda_bn)
        # edge_loss = get_edge_loss_stage1_whmr(pointsRec3, points_orig, edge_cuda, edge_cuda_gt)
        edge_loss = get_edge_loss_stage1_whmr(pointsRec3, points_orig, edge_cuda_rec3, edge_cuda_gt)
        # smoothness_loss = get_smoothness_loss(pointsRec2, parameters, faces_cuda_bn)

        # uniform_loss_global, b_v_list_gen, b_v_list_gt = get_uniform_loss_global(pointsRec2.squeeze().cpu().data.numpy(), points_orig.squeeze().cpu().data.numpy())
        # uniform_loss_local = get_uniform_loss_local(pointsRec.squeeze().cpu().data.numpy(), b_v_list_gen, b_v_list_gt)

        loss_net = 100. * cds_stage3 + CD_loss_part3 + l2_loss # + opt.lambda_normal * normal_loss + opt.lambda_edge * edge_loss
                   # opt.lambda_smooth * smoothness_loss + opt.lambda_uniform_glob * uniform_loss_global * uniform_loss_local

        loss_net.requires_grad_(True)
        loss_net.backward()
        train_CDs_stage3_loss.update(cds_stage3.item())
        train_l2_loss.update(l2_loss.item())
        optimizer.step()
        # optimizer2.step()

        # VIZUALIZE
        if i % 50 <= 0:
            vis.image(img[0, :, 112, :].data.cpu().contiguous(), win='INPUT IMAGE TRAIN', opts=dict(title="INPUT IMAGE TRAIN"))
            vis.scatter(X=points_choice[0].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            title="TRAIN_INPUT",
                            markersize=2,
                        ),
                        )
            vis.scatter(X=pointsRec3_samples[0].data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=2,
                        ),
                        )
        print(
            '[%d: %d/%d] train_cd_loss:  %f, CD_loss_part:  %f, l2_loss: %f, edge_loss: %f, normal_loss: %f, loss_net: %f'
            % (epoch, i, len(dataset) / opt.batchSize, cds_stage3.item(), CD_loss_part3, l2_loss.item(), edge_loss.item(),
               normal_loss.item(), loss_net.item()))

    train_l2_curve.append(train_l2_loss.avg)
    train_CDs_stage3_curve.append(train_CDs_stage3_loss.avg)

    with torch.no_grad():
        # VALIDATION
        val_CDs_stage3_loss.reset()
        val_l2_loss.reset()
        for item in dataset_test.cat:
            dataset_test.perCatValueMeter[item].reset()

        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            img, points, normals, faces_gt, points_orig, name, cat = data
            # img = val_transforms(img)
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
            vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                     vertices_sphere.size(2)).contiguous())

            b_f_list_gt, points_choice_parts, b_f_list_gen, vertices_input_parts, range_part = split_mesh(points_choice, vertices_input, level=0)
            vol_part = split_volume(img, level=0)
            pointsRec_parts = network(vol_part, vertices_input_parts, mode='deform1')
            pointsRec, _, _, _, _ = combine_meshes(pointsRec_parts, vertices_input_parts, points_choice_parts, range_part, b_f_list_gen, None, True, level=0)

            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            error = network(vol_part, pointsRec_samples.detach().transpose(1, 2), mode='estimate')
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(img.size(0), faces_cuda.size(0), faces_cuda.size(1))

            b_f_list_gt2, points_choice_parts2, b_f_list_gen2, pointsRec_parts, range_part2 = split_mesh(points_choice, pointsRec.transpose(2, 1), level=0)
            vol_part = split_volume(img, level=0)
            pointsRec2_parts = network(vol_part, pointsRec_parts, mode='deform2')
            pointsRec2, _, _, _, _ = combine_meshes(pointsRec2_parts, pointsRec_parts, points_choice_parts2, range_part2, b_f_list_gen2, faces_cuda_bn, False, level=0, scale=1.)

            pointsRec2_samples, index2 = samples_random(faces_cuda_bn.detach(), pointsRec2, opt.num_points)
            error_stage2 = network(vol_part, pointsRec2_samples.detach()[:, choice].transpose(1, 2), mode='estimate2')

            b_f_list_gt3, points_choice_parts3, b_f_list_gen3, pointsRec_parts, range_part3 = split_mesh(points_choice,pointsRec2.transpose(2, 1),level=1)
            vol_part = split_volume(img, level=1)
            pointsRec3_parts = network(vol_part, pointsRec_parts, mode='deform3_mlt')
            pointsRec3, _, _, _, _ = combine_meshes(pointsRec3_parts, np.array(pointsRec2_parts), points_choice_parts3,
                                                    range_part3, b_f_list_gen3, faces_cuda_bn, False, level=1, scale=1.)

            pointsRec3, trianglesRec3, CD_loss_part3 = combine_meshes_simp_dec(pointsRec3_parts, pointsRec3, points_choice_parts3, b_f_list_gen3, faces_cuda_bn)
            pointsRec3 = torch.tensor(pointsRec3).unsqueeze(0).float().cuda()
            trianglesRec3 = torch.tensor(trianglesRec3).unsqueeze(0).float().cuda()

            edge_cuda_rec3 = get_edges(trianglesRec3.detach().cpu().squeeze(0).numpy())

            _, _, _, idx3 = distChamfer(points, pointsRec3)
            _, _, _, idx3_choice = distChamfer(points_choice, pointsRec3)

            # pointsRec3_samples, _ = samples_random(faces_cuda_bn.detach(), pointsRec3, opt.num_points)
            pointsRec3_samples, _ = samples_random(trianglesRec3.detach().int(), pointsRec3, opt.num_points)
            dist13_samples, dist23_samples, _, _ = distChamfer(points, pointsRec3_samples)
            choice3 = np.random.choice(points.size(1), opt.num_samples, replace=False)
            error_GT = torch.sqrt(dist23_samples.detach()[:, choice3])
            error = network(vol_part, pointsRec3_samples.detach()[:, choice3].transpose(1, 2), mode='estimate3')

            pointsRec3 = torch.squeeze(pointsRec3)
            trianglesRec3 = trianglesRec3.long()
            normals_gen = torch.zeros(pointsRec3.shape).cuda()
            v10 = pointsRec3[trianglesRec3[:, :, 1]] - pointsRec3[trianglesRec3[:, :, 0]]
            v20 = pointsRec3[trianglesRec3[:, :, 2]] - pointsRec3[trianglesRec3[:, :, 0]]
            normals_gen_value = torch.cross(v10, v20)
            normals_gen[trianglesRec3[:, :, 0]] += normals_gen_value[:]
            normals_gen[trianglesRec3[:, :, 1]] += normals_gen_value[:]
            normals_gen[trianglesRec3[:, :, 2]] += normals_gen_value[:]
            normals_gen_len = torch.sqrt(
                normals_gen[:, 0] * normals_gen[:, 0] + normals_gen[:, 1] * normals_gen[:, 1] + normals_gen[:,
                                                                                                2] * normals_gen[:, 2])
            normals_gen = normals_gen / normals_gen_len.reshape(-1, 1)
            pointsRec3 = torch.unsqueeze(pointsRec3, 0)

            cds_stage3 = (torch.mean(dist13_samples)) + (torch.mean(dist23_samples))
            l2_loss = calculate_l2_loss(error, error_GT.detach())
            # normal_loss = get_normal_loss(pointsRec2, faces_cuda_bn, normals, idx2)
            normal_loss = get_normal_loss_mdf(normals_gen, normals_choice, idx3_choice)
            # edge_loss = get_edge_loss(pointsRec2, faces_cuda_bn)
            # edge_loss = get_edge_loss_stage1_whmr(pointsRec3, points_orig, edge_cuda, edge_cuda_gt)
            edge_loss = get_edge_loss_stage1_whmr(pointsRec3, points_orig, edge_cuda_rec3, edge_cuda_gt)
            # smoothness_loss = get_smoothness_loss(pointsRec2, parameters, faces_cuda_bn)

            # uniform_loss_global, b_v_list_gen, b_v_list_gt = get_uniform_loss_global(pointsRec2.squeeze().cpu().data.numpy(), points_orig.squeeze().cpu().data.numpy())
            # uniform_loss_local = get_uniform_loss_local(pointsRec.squeeze().cpu().data.numpy(), b_v_list_gen, b_v_list_gt)

            loss_net = cds_stage3 + CD_loss_part3 + l2_loss # + opt.lambda_normal * normal_loss + opt.lambda_edge * edge_loss + l2_loss
            # opt.lambda_smooth * smoothness_loss + opt.lambda_uniform_glob * uniform_loss_global * uniform_loss_local

            val_CDs_stage3_loss.update(cds_stage3.item())
            val_l2_loss.update(l2_loss.item())
            dataset_test.perCatValueMeter[cat[0]].update(cds_stage3.item())

            if i % 25 == 0:
                vis.image(img[0, :, 112, :].data.cpu().contiguous(), win='INPUT IMAGE VAL', opts=dict(title="INPUT IMAGE TRAIN"))
                vis.scatter(X=points[0].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title="VAL_INPUT",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsRec3_samples[0].data.cpu(),
                            win='VAL_INPUT_RECONSTRUCTED',
                            opts=dict(
                                title="VAL_INPUT_RECONSTRUCTED",
                                markersize=2,
                            ),
                            )

            print(
                '[%d: %d/%d] val_cd_loss:  %f, CD_loss_part:  %f, l2_loss: %f, edge_loss: %f, normal_loss: %f, loss_net: %f'
                % (epoch, i, len(dataset_test) / opt.batchSize, cds_stage3.item(), CD_loss_part3, l2_loss.item(), edge_loss.item(),
                   normal_loss.item(), loss_net.item()))

        val_l2_curve.append(val_l2_loss.avg)
        val_CDs_stage3_curve.append(val_CDs_stage3_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_l2_curve)), np.arange(len(val_l2_curve)))),
             Y=np.log(np.column_stack((np.array(train_l2_curve), np.array(val_l2_curve)))),
             win='L2_loss',
             opts=dict(title="L2_loss", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_CDs_stage3_curve)), np.arange(len(val_CDs_stage3_curve)))),
             Y=np.log(np.column_stack((np.array(train_CDs_stage3_curve),np.array(val_CDs_stage3_curve)))),
             win='CDs_stage2',
             opts=dict(title="CDs_stage2", legend=["train", "val"], markersize=2, ), )

    log_table = {
        "train_l2_loss": train_l2_loss.avg,
        "train_cds_stage2": train_CDs_stage3_loss.avg,
        "val_l2_loss": val_l2_loss.avg,
        "val_cds_stage2": val_CDs_stage3_loss.avg,
        "epoch": epoch,
        "lr": lrate,
    }

    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
