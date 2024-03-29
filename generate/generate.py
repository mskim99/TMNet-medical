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


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default = './log/SVR_subnet2_usage/network.pth',  help='your path to the trained model')
parser.add_argument('--num_points',type=int,default=20000)
parser.add_argument('--tau',type=float,default=0.1)
parser.add_argument('--tau_decay',type=float,default=2)
parser.add_argument('--pool',type=str,default='max',help='max or mean or sum' )
parser.add_argument('--num_vertices', type=int, default=10242) # 2562
parser.add_argument('--subnet',type=int,default=2)
parser.add_argument('--manualSeed', type=int, default=6185)
opt = parser.parse_args()
print (opt)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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

network = SVR_TMNet()
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
        vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                 vertices_sphere.size(2)).contiguous())
        pointsRec  = network(img, vertices_input,mode='deform1')

        dist1, dist2, _, idx2 = distChamfer(points.float() , pointsRec.float()) # PointsRec > Points

        pointsRec_samples, index = samples_random(faces_cuda, pointsRec, opt.num_points)
        error = network(img, pointsRec_samples.detach().transpose(1, 2),mode='estimate')
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

        # Validate normal flip
        '''
        for i in range (0, normals_gen.shape[0] - 1):
            if torch.dot(normals_gen[i, :].cpu().float(), normals[0, idx2[0, i], :].float()).item() < 0.0:
                normals_gen[i, :] = -normals_gen[i, :]
                '''

        ###################################################################################################
        if opt.subnet > 1:
            pointsRec2 = network(img, pointsRec,mode='deform2')
            pointsRec2_samples, index = samples_random(faces_cuda_bn, pointsRec2, opt.num_points)
            error = network(img, pointsRec2_samples.detach().transpose(1, 2),mode='estimate2')
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
                normals_gen2[:, 0] * normals_gen2[:, 0] + normals_gen2[:, 1] * normals_gen2[:, 1] + normals_gen2[:,2] * normals_gen2[:,2])
            normals_gen2 = normals_gen2 / normals_gen2_len.reshape(-1, 1)
            pointsRec2 = torch.unsqueeze(pointsRec2, 0)

        ###################################################################################################
        if opt.subnet > 2:
            indices = (torch.arange(0, faces_cuda_bn.size(0)) * (1 + faces_cuda_bn.size(0)))\
                .type(torch.cuda.LongTensor)
            pointsRec2_boundary, selected_pair_all, selected_pair_all_len = \
                get_boundary_points_bn(faces_cuda_bn, pointsRec2)
            vector1 = (pointsRec2_boundary[:, :, 1] - pointsRec2_boundary[:, :, 0])
            vector2 = (pointsRec2_boundary[:, :, 2] - pointsRec2_boundary[:, :, 0])
            vector1 = vector1 / (torch.norm((vector1 + 1e-6), dim=2)).unsqueeze(2)
            vector2 = vector2 / (torch.norm((vector2 + 1e-6), dim=2)).unsqueeze(2)
            vector1 = vector1.transpose(2,1).detach()
            vector2 = vector2.transpose(2,1).detach()

            if pointsRec2_boundary.shape[1] != 0:
                pointsRec3_boundary = network\
                    (img, pointsRec2_boundary[:, :, 0], vector1, vector2,mode='refine')
            else:
                pointsRec3_boundary = pointsRec2_boundary[:, :, 0]

            pointsRec3_set = []
            for ibatch in torch.arange(0, img.shape[0]):
                length = selected_pair_all_len[ibatch]
                if length != 0:
                    # index_bp = boundary_points_all[ibatch][:length]
                    index_bp = selected_pair_all[ibatch][:, 0][:length]
                    prb_final = pointsRec3_boundary[ibatch][:length]
                    pr = pointsRec2[ibatch]
                    index_bp = index_bp.view(index_bp.shape[0], -1).expand([index_bp.shape[0], 3])
                    pr_final = pr.scatter(dim=0, index=index_bp, source=prb_final)
                    pointsRec3_set.append(pr_final)
                else:
                    pr = pointsRec2[ibatch]
                    pr_final = pr
                    pointsRec3_set.append(pr_final)
            pointsRec3 = torch.stack(pointsRec3_set, 0)
            triangles_c3 = faces_cuda_bn[0].cpu().data.numpy()

        print(cat,fn)
        if not os.path.exists(opt.model[:-4]):
            os.mkdir(opt.model[:-4])
            print('created dir', opt.model[:-4])

        if not os.path.exists(opt.model[:-4] + "/" + str(cat)):
            os.mkdir(opt.model[:-4] + "/" + str(cat))
            print('created dir', opt.model[:-4] + "/" + str(cat))
        b = np.zeros((np.shape(faces)[0],4)) + 3
        b[:,1:] = faces

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


        meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn+"_GT.obj",
                                points.cpu().data.squeeze().numpy(), triangles=faces_gt.cpu().data.squeeze().numpy().astype(int))
        meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen.obj",
                                pointsRec.cpu().data.squeeze().numpy(),
                                triangles=faces, normals=normals_gen)
        meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen_pruned.obj",
                                pointsRec.cpu().data.squeeze().numpy(),
                                triangles=triangles_c1, normals=normals_gen)

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
            meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn + "_gen2.obj",
                                    pointsRec2.cpu().data.squeeze().numpy(),
                                    triangles=triangles_c1, normals=normals_gen2)
            meshio_custom.write_obj(opt.model[:-4] + "/" + str(cat) + "/" + fn + "_gen2_pruned.obj",
                                    pointsRec2.cpu().data.squeeze().numpy(),
                                    triangles=triangles_c2, normals=normals_gen2)
        if opt.subnet>2:
            write_ply(filename=opt.model[:-4] + "/" + str(cat) + "/" + fn+"_gen3",
                      points=pd.DataFrame(pointsRec3.cpu().data.squeeze().numpy()), as_text=True,
                      faces = pd.DataFrame(b_c3.astype(int)))
