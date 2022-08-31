from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import sys

sys.path.append('./auxiliary/')
from dataset_3D import *
from model_3D import *
from utils import *
from ply import *
import os
import scipy.io as sio
import pandas as pd
from loss import *
import meshio_custom
import sklearn.preprocessing as sklp

sys.path.append('./utils/')
from split_mesh import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default = './log/SVR_subnet2_usage2/network.pth',  help='your path to the trained model')
parser.add_argument('--num_points',type=int,default=10000)
parser.add_argument('--tau',type=float,default=0.1)
parser.add_argument('--tau_decay',type=float,default=2)
parser.add_argument('--pool',type=str,default='max',help='max or mean or sum' )
parser.add_argument('--num_vertices', type=int, default=2562) # 2562
parser.add_argument('--subnet',type=int,default=2)
parser.add_argument('--manualSeed', type=int, default=6185)
opt = parser.parse_args()
print (opt)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
torch.cuda.set_device(3)

sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()

blue = lambda x:'\033[94m' + x + '\033[0m'
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset_test = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=False,class_choice='lumbar_vertebra_05')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset_test)

# name = 'sphere' + str(opt.num_vertices) + '.mat'
# mesh = sio.loadmat('./data/' + name)
name = 'sphere' + str(opt.num_vertices) + '.obj'
mesh = meshio_custom.read_obj('./data/' + name)

# faces = np.array(mesh['f'])
faces = mesh['faces']
faces_cuda = torch.from_numpy(faces.astype(int)).type(torch.cuda.LongTensor)

# vertices_sphere = np.array(mesh['v'])
vertices_sphere = mesh['vertices']
vertices_sphere = (torch.cuda.FloatTensor(vertices_sphere)).transpose(0,1).contiguous()
vertices_sphere = vertices_sphere.contiguous().unsqueeze(0)

network = SVR_TMNet_Split()
network.apply(weights_init)
network.cuda()

if opt.model != '':
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in torch.load(opt.model).items() if (k in model_dict) }
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    print(" Previous weight loaded ")
print(network)
network.eval()

with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, points, normals, faces_gt, points_orig, name, cat = data
        cat = cat[0]
        fn = name[0]
        img = img.cuda()
        img = img.unsqueeze(dim=0)
        img = img.float()

        points = points.cuda()
        choice = np.random.choice(points.size(1), opt.num_vertices, replace=False)
        points_choice = points[:, choice, :].contiguous()
        normals_choice = normals[:, choice, :].contiguous()
        points = points.float()
        points_choice = points_choice.float()
        vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                 vertices_sphere.size(2)).contiguous())

        vol_part = split_volume(img, level=0)
        b_f_list_gt, points_choice_parts, b_f_list_gen, vertices_input_parts, range_part = split_mesh(points_choice, vertices_input, level=0)
        pointsRec_parts = network(vol_part, vertices_input_parts, mode='deform1')  # vertices_sphere 3*2562
        pointsRec, _, _, _, _ = combine_meshes(pointsRec_parts, vertices_input_parts, points_choice_parts, range_part, b_f_list_gen, True, level=0, scale=1.)

        _, _, _, idx2 = distChamfer(points.float() , pointsRec.float()) # PointsRec > Points

        pointsRec_samples, index = samples_random(faces_cuda, pointsRec, opt.num_points)
        error = network(vol_part, pointsRec_samples.detach().transpose(1, 2),mode='estimate')
        faces_cuda_bn = faces_cuda.unsqueeze(0)
        # faces_cuda_bn = prune(faces_cuda_bn, error, opt.tau, index, opt.pool)
        triangles_c1 = faces_cuda_bn[0].cpu().data.numpy()

        pointsRec = torch.squeeze(pointsRec)
        normals_gen = torch.zeros(pointsRec.shape).cuda()
        v10 = pointsRec[triangles_c1[:, 1]] - pointsRec[triangles_c1[:, 0]]
        v20 = pointsRec[triangles_c1[:, 2]] - pointsRec[triangles_c1[:, 0]]
        normals_gen_value = torch.cross(v10, v20)
        normals_gen[triangles_c1[:,0]] += normals_gen_value[:]
        normals_gen[triangles_c1[:,1]] += normals_gen_value[:]
        normals_gen[triangles_c1[:,2]] += normals_gen_value[:]
        normals_gen_len = torch.sqrt(normals_gen[:,0]*normals_gen[:,0]+normals_gen[:,1]*normals_gen[:,1]+normals_gen[:,2]*normals_gen[:,2])
        normals_gen = normals_gen / normals_gen_len.reshape(-1, 1)
        pointsRec = torch.unsqueeze(pointsRec, 0)

        '''
        for j in range(0, normals_gen.shape[0] - 1):
            if torch.dot(normals_gen[j, :].cpu().float(), normals[0, idx2[0, j], :].float()).item() < 0.0:
                normals_gen[j, :] = -normals_gen[j, :]
                '''

        ###################################################################################################
        if opt.subnet > 1:
            b_f_list_gt2, points_choice_parts2, b_f_list_gen2, pointsRec_parts, range_part2 = split_mesh(points_choice, pointsRec.transpose(2,1), level=1)
            vol_parts = split_volume(img, level=1)
            pointsRec2_parts = network(vol_parts, pointsRec_parts, mode='deform2')
            pointsRec2, _, _, _, _ = combine_meshes(pointsRec2_parts, pointsRec_parts, points_choice_parts2, range_part2, b_f_list_gen2, faces_cuda_bn, False, level=1, scale=1.)
            pointsRec2_sd, trianglesRec2, CD_loss_part2 = combine_meshes_simp_dec(pointsRec2_parts, pointsRec2,points_choice_parts2, b_f_list_gen2,faces_cuda_bn)
            pointsRec2_sd = torch.tensor(pointsRec2_sd).unsqueeze(0).float().cuda()
            trianglesRec2 = torch.tensor(trianglesRec2).unsqueeze(0).int().cuda()
            triangles_c2_sd = trianglesRec2[0].cpu().data.numpy()
            # _, _, _, idx2_2 = distChamfer(points.float(), pointsRec2.float())  # PointsRec > Points

            pointsRec2_samples, index = samples_random(faces_cuda_bn, pointsRec2, opt.num_points)
            error = network(vol_part, pointsRec2_samples.detach().transpose(1, 2),mode='estimate2')
            faces_cuda_bn = faces_cuda_bn.clone()
            # faces_cuda_bn = prune(faces_cuda_bn, error, opt.tau/opt.tau_decay, index, opt.pool)
            triangles_c2 = faces_cuda_bn[0].cpu().data.numpy()

            pointsRec2 = torch.squeeze(pointsRec2)
            normals_gen2 = torch.zeros(pointsRec2.shape).cuda()
            v10 = pointsRec2[triangles_c2[:, 1]] - pointsRec2[triangles_c2[:, 0]]
            v20 = pointsRec2[triangles_c2[:, 2]] - pointsRec2[triangles_c2[:, 0]]
            normals_gen2_value = torch.cross(v10, v20)
            normals_gen2[triangles_c2[:, 0]] += normals_gen2_value[:]
            normals_gen2[triangles_c2[:, 1]] += normals_gen2_value[:]
            normals_gen2[triangles_c2[:, 2]] += normals_gen2_value[:]
            normals_gen2_len = torch.sqrt(
                normals_gen2[:, 0] * normals_gen2[:, 0] + normals_gen2[:, 1] * normals_gen2[:, 1] + normals_gen2[:, 2] * normals_gen2[:, 2])
            normals_gen2 = normals_gen2 / normals_gen2_len.reshape(-1, 1)
            pointsRec2 = torch.unsqueeze(pointsRec2, 0)
            '''
            for j in range(0, normals_gen2.shape[0] - 1):
                if torch.dot(normals_gen2[j, :].cpu().float(), normals[0, idx2_2[0, j], :].float()).item() < 0.0:
                    normals_gen2[j, :] = -normals_gen2[j, :]
                    '''
        ###################################################################################################
        if opt.subnet > 2:
            b_f_list_gt3, points_choice_parts3, b_f_list_gen3, pointsRec2_parts, range_part3 = split_mesh(points_choice,
                                                                                                         pointsRec2.transpose(
                                                                                                             2, 1),
                                                                                                         level=1)
            pointsRec3_parts = network(vol_part, pointsRec2_parts, mode='deform3')

            # if i == 0:
            # combine_meshes_simp_dec(pointsRec2_parts, points_choice_parts2, b_f_list_gen2, faces_cuda_bn)

            pointsRec3, _, _, _, _ = combine_meshes(pointsRec3_parts, np.array(pointsRec2_parts),
                                                                        points_choice_parts3, range_part3,
                                                                        b_f_list_gen3, faces_cuda_bn, False, level=1,
                                                                        scale=1.)
            pointsRec3, trianglesRec3, CD_loss_part3 = combine_meshes_simp_dec(pointsRec3_parts, pointsRec3,
                                                                               points_choice_parts3, b_f_list_gen3,
                                                                               faces_cuda_bn)
            _, _, _, idx2_2 = distChamfer(points.float(), pointsRec2.float())  # PointsRec > Points
            pointsRec3 = torch.tensor(pointsRec3).unsqueeze(0).float().cuda()
            trianglesRec3 = torch.tensor(trianglesRec3).unsqueeze(0).float().cuda()

            pointsRec3_samples, index = samples_random(trianglesRec3.detach().int(), pointsRec3, opt.num_points)
            error = network(vol_part, pointsRec3_samples.detach().transpose(1, 2), mode='estimate3')
            faces_cuda_bn = faces_cuda_bn.clone()
            # faces_cuda_bn = prune(faces_cuda_bn, error, opt.tau/opt.tau_decay, index, opt.pool)
            # triangles_c2 = faces_cuda_bn[0].cpu().data.numpy()

            pointsRec3 = torch.squeeze(pointsRec3)
            trianglesRec3 = trianglesRec3.long()
            normals_gen3 = torch.zeros(pointsRec3.shape).cuda()
            v10 = pointsRec3[trianglesRec3[:, 1]] - pointsRec3[trianglesRec3[:, 0]]
            v20 = pointsRec3[trianglesRec3[:, 2]] - pointsRec3[trianglesRec3[:, 0]]
            normals_gen3_value = torch.cross(v10, v20)
            normals_gen3[trianglesRec3[:, 0]] += normals_gen3_value[:]
            normals_gen3[trianglesRec3[:, 1]] += normals_gen3_value[:]
            normals_gen3[trianglesRec3[:, 2]] += normals_gen3_value[:]
            normals_gen3_len = torch.sqrt(
                normals_gen3[:, 0] * normals_gen3[:, 0] + normals_gen3[:, 1] * normals_gen3[:, 1] + normals_gen3[:, 2] * normals_gen3[:, 2])
            normals_gen3 = normals_gen3 / normals_gen3_len.reshape(-1, 1)
            pointsRec3 = torch.unsqueeze(pointsRec3, 0)

        print(cat,fn)
        if not os.path.exists(opt.model[:-4]):
            os.mkdir(opt.model[:-4])
            print('created dir', opt.model[:-4])

        if not os.path.exists(opt.model[:-4] + "/" + str(cat)):
            os.mkdir(opt.model[:-4] + "/" + str(cat))
            print('created dir', opt.model[:-4] + "/" + str(cat))
        b = np.zeros((np.shape(faces)[0],4)) + 3
        b[:,1:] = faces

        '''
        triangles_c1_tosave = triangles_c1[triangles_c1.sum(1).nonzero()[0]]
        b_c1 = np.zeros((np.shape(triangles_c1_tosave)[0],4)) + 3
        b_c1[:,1:] = triangles_c1_tosave
        if opt.subnet>1:
            triangles_c2_tosave = triangles_c2[triangles_c2.sum(1).nonzero()[0]]
            b_c2 = np.zeros((np.shape(triangles_c2_tosave)[0],4)) + 3
            b_c2[:,1:] = triangles_c2_tosave
        if opt.subnet>2:
            triangles_c3_tosave = triangles_c3[triangles_c3.sum(1).nonzero()[0]]
            b_c3 = np.zeros((np.shape(triangles_c3_tosave)[0],4)) + 3
            b_c3[:,1:] = triangles_c3_tosave
            '''

        meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn+"_GT.obj",
                                points.cpu().data.squeeze().numpy(), triangles=faces_gt.cpu().data.squeeze().numpy().astype(int))
        meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen.obj",
                                pointsRec.cpu().data.squeeze().numpy(),
                                triangles=faces, normals=normals_gen.cpu().numpy())
        '''
        meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen_pruned.obj",
                                pointsRec.cpu().data.squeeze().numpy(),
                                triangles=triangles_c1, normals=normals_gen)
                                '''
        '''
        write_ply(filename=opt.model[:-4] + "/" + str(cat) + "/" + fn+"_GT",
                  points=pd.DataFrame(points.cpu().data.squeeze().numpy()), as_text=True)
        write_ply(filename=opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen",
                  points=pd.DataFrame(pointsRec.cpu().data.squeeze().numpy()), as_text=True,
                  faces = pd.DataFrame(b.astype(int)), normal = True)
        write_ply(filename=opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen_pruned",
                  points=pd.DataFrame(pointsRec.cpu().data.squeeze().numpy()), as_text=True,
                  faces = pd.DataFrame(b_c1.astype(int)), normal = True)
                    '''
        if opt.subnet>1:
            '''
            write_ply(filename=opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen2",
                      points=pd.DataFrame(pointsRec2.cpu().data.squeeze().numpy()), as_text=True,
                      faces = pd.DataFrame(b_c1.astype(int)))
            write_ply(filename=opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen2_pruned",
                      points=pd.DataFrame(pointsRec2.cpu().data.squeeze().numpy()), as_text=True,
                      faces = pd.DataFrame(b_c2.astype(int)))
                      '''
            '''
            meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn + "_gen2.obj",
                                    pointsRec2.cpu().data.squeeze().numpy(),
                                    triangles=triangles_c2)
                                    '''
            meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn + "_gen2.obj",
                                    pointsRec2_sd.cpu().data.squeeze().numpy(),
                                    triangles=triangles_c2_sd)
            '''
            meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn + "_gen2_pruned.obj",
                                    pointsRec2.cpu().data.squeeze().numpy(),
                                    triangles=triangles_c2, normals=normals_gen2)
                                    '''
        if opt.subnet>2:
            '''
            write_ply(filename=opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen3",
                      points=pd.DataFrame(pointsRec3.cpu().data.squeeze().numpy()), as_text=True,
                      faces = pd.DataFrame(b_c3.astype(int)))
                      '''
            meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn + "_gen3.obj",
                                    pointsRec3.cpu().data.squeeze().numpy(),
                                    triangles=trianglesRec3.cpu().data.squeeze().numpy(),
                                    normals=normals_gen3.cpu().data.squeeze().numpy())
