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

sys.path.append('./utils/')
from losses import *
from hull import *

from sklearn.preprocessing import MinMaxScaler
import joblib

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=420, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--num_points', type=int, default=2500, help='number of points')
parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default=2500,
                    help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default="pretrain", help='visdom environment')
parser.add_argument('--lr',type=float,default=1e-6, help='initial learning rate')
parser.add_argument('--manualSeed', type=int, default=6185)
opt = parser.parse_args()
print(opt)

sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()

vis = visdom.Visdom(port=8887, env=opt.env)
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

dataset = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=True, class_choice='lumbar_vertebra_05')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(npoints=opt.num_points, SVR=True, normal=False, train=False, class_choice='lumbar_vertebra_05')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset)

network = Pretrain(num_points=opt.num_points)
network.cuda()  # put network on GPU
network.apply(weights_init)  # initialization of the weight
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

lrate = opt.lr  # learning rate
optimizer = optim.Adam([
    {'params': network.pc_encoder.parameters()},
    {'params': network.decoder.parameters()}
], lr=lrate)

# meters to record stats on learning
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')
# initialize learning curve on visdom, and color for each primitive in visdom display
train_curve = []
val_curve = []

volume_scaler = MinMaxScaler()
L1_loss = torch.nn.L1Loss()

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_loss.reset()
    network.train()
    # learning rate schedule
    optimizer = optim.Adam([{'params': network.encoder.parameters()}, {'params': network.decoder.parameters()}], lr=lrate)
    '''
    if epoch == 100:
        optimizer = optim.Adam([
            {'params': network.pc_encoder.parameters()},
            {'params': network.decoder.parameters()}
        ], lr=lrate/10.0)
    if epoch == 120:
        optimizer = optim.Adam(network.encoder.parameters(), lr=lrate)
    if epoch == 220:
        optimizer = optim.Adam(network.encoder.parameters(), lr=lrate / 10.0)
        '''

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, normals, name, cat = data

        img = volume_scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        img = torch.from_numpy(img).type(torch.FloatTensor)

        img = img.cuda()
        img = img.unsqueeze(dim=0)
        img = img.float()

        vol_noize = torch.randn([1, 1, 256, 256, 256])
        vol_noize = vol_noize.cuda()
        img = 0.95 * img + 0.05 * vol_noize

        points = points.transpose(2, 1).contiguous()
        points = points.cuda()
        # SUPER_RESOLUTION optionally reduce the size of the points fed to PointNet
        points = points[:, :, :opt.super_points].contiguous()
        points = points.float()

        # noises = torch.randn(points.size()) - 0.5
        # noises = noises.cuda()

        # Add noises (Gaussian distribution) to the points in training
        '''
        if epoch < 120:
            points = points + 0.1 * noises
        elif epoch >= 120 and epoch < 220:
            points = points + 0.25 * noises
        elif epoch >= 220 and epoch < 320:
            points = points + 0.0625 * noises
            '''

        # END SUPER RESOLUTION
        if epoch >= 0:
            pointsRec = network(img, mode='img')
        else:
            points = torch.concat([points, points], dim=0)
            pointsRec = network(points)  # forward pass

        pointsRec_max = torch.max(pointsRec, axis=1).values
        pointsRec_min = torch.min(pointsRec, axis=1).values
        pointsRec = (pointsRec - pointsRec_min) / (pointsRec_max - pointsRec_min + 1e-4)

        p = points.transpose(2, 1).contiguous().detach().cpu().numpy()
        pr = pointsRec.detach().cpu().numpy()

        dist1, dist2, idx1, idx2 = distChamfer(points.transpose(2, 1).contiguous(), pointsRec)  # loss function

        idx1_ln = idx1.long().unsqueeze(1).cpu().numpy()
        idx2_ln = idx2.long().unsqueeze(1).cpu().numpy()

        edge_len_points = p[:, idx1_ln, :] - p[:, idx2_ln, :]
        edge_len_pointsRec = pr[:, idx1_ln, :] - pr[:, idx2_ln, :]
        edge_len_points = torch.Tensor(edge_len_points).cuda()
        edge_len_pointsRec = torch.Tensor(edge_len_pointsRec).cuda()

        #  + 0.005 * L1_loss(edge_len_points, edge_len_pointsRec)
        loss_net = 100.0 * (((torch.mean(dist1)) + (torch.mean(dist2))) + 0.0025 * L1_loss(edge_len_points, edge_len_pointsRec))
        loss_net.backward()
        train_loss.update(loss_net.item())
        optimizer.step()  # gradient update
        # VIZUALIZE
        if i % 50 <= 0:
            vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
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
        print('[%d: %d/%d] train loss:  %f ' % (epoch, i, len_dataset / opt.batchSize, loss_net.item()))
    # UPDATE CURVES
    train_curve.append(train_loss.avg)

    # VALIDATION
    val_loss.reset()
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    network.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            img, points, normals, name, cat = data

            img = volume_scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
            img = torch.from_numpy(img).type(torch.FloatTensor)

            img = img.cuda()
            img = img.unsqueeze(dim=0)
            img = img.float()

            '''
            vol_noize = torch.randn([1, 1, 256, 256, 256])
            vol_noize = vol_noize.cuda()
            img = 0.95 * img + 0.05 * vol_noize
            '''

            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            # SUPER_RESOLUTION
            points = points[:, :, :opt.super_points].contiguous()
            points = points.float()

            # END SUPER RESOLUTION
            if epoch >= 0:
                pointsRec = network(img, mode='img')
            else:
                points = torch.concat([points, points], dim=0)
                pointsRec = network(points)  # forward pass

            pointsRec_max = torch.max(pointsRec, axis=1).values
            pointsRec_min = torch.min(pointsRec, axis=1).values
            pointsRec = (pointsRec - pointsRec_min) / (pointsRec_max - pointsRec_min + 1e-4)

            dist1, dist2, idx1, idx2 = distChamfer(points.transpose(2, 1).contiguous(), pointsRec)

            idx1_ln = idx1.long().unsqueeze(1).cpu().numpy()
            idx2_ln = idx2.long().unsqueeze(1).cpu().numpy()

            edge_len_points = p[:, idx1_ln, :] - p[:, idx2_ln, :]
            edge_len_pointsRec = pr[:, idx1_ln, :] - pr[:, idx2_ln, :]
            edge_len_points = torch.Tensor(edge_len_points).cuda()
            edge_len_pointsRec = torch.Tensor(edge_len_pointsRec).cuda()

            loss_net = 100.0 * (((torch.mean(dist1)) + (torch.mean(dist2))) + 0.0025 * L1_loss(edge_len_points, edge_len_pointsRec))
            val_loss.update(loss_net.item())
            dataset_test.perCatValueMeter[cat[0]].update(loss_net.item())
            if i % 200 == 0:
                vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
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
            print('[%d: %d/%d] val loss:  %f ' % (epoch, i, len(dataset_test)/opt.batchSize, loss_net.item()))
        # UPDATE CURVES
        val_curve.append(val_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
             Y=np.log(np.column_stack((np.array(train_curve), np.array(val_curve)))),
             win='loss',
             opts=dict(title="loss", legend=["train_curve" , "val_curve"], markersize=2, ), )

    # dump stats in log file
    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "super_points": opt.super_points,
    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
    joblib.dump(volume_scaler, '%s/volume_scaler.pkl' % (dir_name))

