from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import resnet_3D
import numpy as np


class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(1, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = (torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(1024)
        self.trans = trans
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        # x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


class DeformNet(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(DeformNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, (self.bottleneck_size // 2 + self.bottleneck_size // 4), 1)
        self.conv3 = torch.nn.Conv1d((self.bottleneck_size // 2 + self.bottleneck_size // 4), self.bottleneck_size // 2, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv5 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2 + self.bottleneck_size // 4)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn4 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

        # self.fc = nn.Linear(14508, 7500)
        self.th = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.maxpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, drp_idx):

        # Original Code
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print(x.shape)
        # x = self.Sigmoid(self.conv5(x))
        # x = self.conv5(x)
        x = self.th(self.conv5(x))
        # print(x.shape)

        return x


class DeformNet_layer8(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(DeformNet_layer8, self).__init__()

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, (self.bottleneck_size // 2 + self.bottleneck_size // 4 + self.bottleneck_size // 8), 1)
        self.conv3 = torch.nn.Conv1d((self.bottleneck_size // 2 + self.bottleneck_size // 4 + self.bottleneck_size // 8), (self.bottleneck_size // 2 + self.bottleneck_size // 4), 1)
        self.conv4 = torch.nn.Conv1d((self.bottleneck_size // 2 + self.bottleneck_size // 4), self.bottleneck_size // 2, 1)
        self.conv5 = torch.nn.Conv1d(self.bottleneck_size // 2, (self.bottleneck_size // 4 + self.bottleneck_size // 8), 1)
        self.conv6 = torch.nn.Conv1d((self.bottleneck_size // 4 + self.bottleneck_size // 8), self.bottleneck_size // 4, 1)
        self.conv7 = torch.nn.Conv1d(self.bottleneck_size // 4, self.bottleneck_size // 8, 1)
        self.conv8 = torch.nn.Conv1d(self.bottleneck_size // 8, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2 + self.bottleneck_size // 4 + self.bottleneck_size // 8)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 2 + self.bottleneck_size // 4)
        self.bn4 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn5 = torch.nn.BatchNorm1d(self.bottleneck_size // 4 + self.bottleneck_size // 8)
        self.bn6 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)
        self.bn7 = torch.nn.BatchNorm1d(self.bottleneck_size // 8)

        # self.fc = nn.Linear(14508, 7500)
        self.th = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.maxpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, drp_idx):

        # Original Code
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print(x.shape)
        x = F.relu(self.bn5(self.conv5(x)))
        # print(x.shape)
        x = F.relu(self.bn6(self.conv6(x)))
        # print(x.shape)
        x = F.relu(self.bn7(self.conv7(x)))
        # print(x.shape)
        x = self.th(self.conv8(x))
        # print(x.shape)

        return x


class DeformNet_scale(nn.Module):
    def __init__(self, bottleneck_size=1024):
        self.bottleneck_size = bottleneck_size
        super(DeformNet_scale, self).__init__()

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, (self.bottleneck_size // 2 + self.bottleneck_size // 4), 1)
        self.conv3 = torch.nn.Conv1d((self.bottleneck_size // 2 + self.bottleneck_size // 4), self.bottleneck_size // 2, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv5 = torch.nn.Conv1d(self.bottleneck_size // 4, 1, 1)

        '''
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2 + self.bottleneck_size // 4)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn4 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)
        '''

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, points):

        # Original Code
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x = torch.mean(self.conv5(x))
        # print(x.shape)

        return x


class DeformNet_Res(nn.Module):
    def __init__(self, bottleneck_size=1024):
        self.bottleneck_size = bottleneck_size
        super(DeformNet_Res, self).__init__()

        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.conv2_1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3_1 = torch.nn.Conv1d(1280, 640, 1)
        self.conv3_2 = torch.nn.Conv1d(640, 320, 1)
        self.conv4_1 = torch.nn.Conv1d(1600, 800, 1)
        self.conv4_2 = torch.nn.Conv1d(800, 400, 1)
        self.conv5_1 = torch.nn.Conv1d(1200, 600, 1)
        self.conv5_2 = torch.nn.Conv1d(600, 300, 1)
        self.conv6_1 = torch.nn.Conv1d(1500, 750, 1)
        self.conv6_2 = torch.nn.Conv1d(750, 375, 1)
        self.conv7_1 = torch.nn.Conv1d(1125, 562, 1)
        self.conv7_2 = torch.nn.Conv1d(562, 281, 1)
        self.conv8_1 = torch.nn.Conv1d(1406, 703, 1)
        self.conv8_2 = torch.nn.Conv1d(703, 351, 1)
        self.conv9_1 = torch.nn.Conv1d(1054, 527, 1)
        self.conv9_2 = torch.nn.Conv1d(527, 264, 1)
        self.conv10 = torch.nn.Conv1d(1318, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2_1 = torch.nn.BatchNorm1d(512)
        self.bn2_2 = torch.nn.BatchNorm1d(256)
        self.bn3_1 = torch.nn.BatchNorm1d(640)
        self.bn3_2 = torch.nn.BatchNorm1d(320)
        self.bn4_1 = torch.nn.BatchNorm1d(800)
        self.bn4_2 = torch.nn.BatchNorm1d(400)
        self.bn5_1 = torch.nn.BatchNorm1d(600)
        self.bn5_2 = torch.nn.BatchNorm1d(300)
        self.bn6_1 = torch.nn.BatchNorm1d(750)
        self.bn6_2 = torch.nn.BatchNorm1d(375)
        self.bn7_1 = torch.nn.BatchNorm1d(562)
        self.bn7_2 = torch.nn.BatchNorm1d(281)
        self.bn8_1 = torch.nn.BatchNorm1d(703)
        self.bn8_2 = torch.nn.BatchNorm1d(351)
        self.bn9_1 = torch.nn.BatchNorm1d(527)
        self.bn9_2 = torch.nn.BatchNorm1d(264)

        self.th = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        # Block 1
        residual = x
        # print(x.size())  # torch.Size([1, 1024, 2500])
        x = self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(x)))))
        # print(x.size())  # torch.Size([1, 256, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1280, 2500])

        # Block 1
        residual = x
        x = self.bn3_2(self.conv3_2(F.relu(self.bn3_1(self.conv3_1(x)))))
        # print(x.size())  # torch.Size([1, 320, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1600, 2500])

        # Block 2
        x = self.bn4_1(self.conv4_1(x))
        # print(x.size())  # torch.Size([1, 800, 2500])
        residual = x
        x = self.bn4_2(self.conv4_2(F.relu(x)))
        # print(x.size())  # torch.Size([1, 400, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1200, 2500])

        # Block 1
        residual = x
        x = self.bn5_2(self.conv5_2(F.relu(self.bn5_1(self.conv5_1(x)))))
        # print(x.size())  # torch.Size([1, 300, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1500, 2500])

        # Block 2
        x = self.bn6_1(self.conv6_1(x))
        # print(x.size())  # torch.Size([1, 750, 2500])
        residual = x
        x = self.bn6_2(self.conv6_2(F.relu(x)))
        # print(x.size())  # torch.Size([1, 375, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1125, 2500])

        # Block 1
        residual = x
        x = self.bn7_2(self.conv7_2(F.relu(self.bn7_1(self.conv7_1(x)))))
        # print(x.size())  # torch.Size([1, 281, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1406, 2500])

        # Block 2
        x = self.bn8_1(self.conv8_1(x))
        # print(x.size())  # torch.Size([1, 703, 2500])
        residual = x
        x = self.bn8_2(self.conv8_2(F.relu(x)))
        # print(x.size())  # torch.Size([1, 351, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1054, 2500])

        # Block 1
        residual = x
        x = self.bn9_2(self.conv9_2(F.relu(self.bn9_1(self.conv9_1(x)))))
        # print(x.size())  # torch.Size([1, 264, 2500])
        x = torch.concat([x, residual], dim=1)
        x = F.relu(x)
        # print(x.size())  # torch.Size([1, 1318, 2500])

        x = self.th(self.conv10(self.dropout(x)))
        # print(x.size())  # torch.Size([1, 3, 2500])

        return x


class Decoder_volume(torch.nn.Module):
    def __init__(self):
        super(Decoder_volume, self).__init__()

        self.d_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU(inplace=True),
        )
        self.d_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.d_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.d_layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.d_layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(inplace=True),
        )
        self.d_layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Sigmoid(),
        )
        self.d_layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Sigmoid(),
        )
        self.d_layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid(),
        )


    def forward(self, features):

        features = features.reshape([-1, 1024, 1, 1, 1])

        gen_volume = self.d_layer1(features)
        # print(gen_volume.size()) # torch.Size([1, 1024, 1, 1, 1])
        gen_volume = self.d_layer2(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 512, 2, 2, 2])
        gen_volume = self.d_layer3(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 256, 4, 4, 4])
        gen_volume = self.d_layer4(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 128, 8, 8, 8])
        gen_volume = self.d_layer5(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 64, 16, 16, 16])
        gen_volume = self.d_layer6(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 32, 32, 32, 32])
        gen_volume = self.d_layer7(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 16, 64, 64, 64])
        gen_volume = self.d_layer8(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 8, 128, 128, 128])
        gen_volume = torch.squeeze(gen_volume)

        return gen_volume


class Refiner(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Refiner, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 2, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.dropout3(x)
        x = self.th(self.conv4(x))
        return x


class Estimator(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Estimator, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 1, 1)

        self.sig = nn.Sigmoid()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.sig(self.conv4(x))
        return x


class SVR_TMNet(nn.Module):
    def __init__(self,  bottleneck_size = 1024):
        super(SVR_TMNet, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.encoder = resnet_3D.resnet50_3D(num_classes=self.bottleneck_size)
        self.decoder = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])
        self.decoder2 = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])
        self.estimate = Estimator(bottleneck_size=3 + self.bottleneck_size)
        self.estimate2 = Estimator(bottleneck_size=3+self.bottleneck_size)
        self.refine = Refiner(bottleneck_size=3 + self.bottleneck_size)

    def forward(self,x,points,vector1=0,vector2=0,mode='deform1'):
        x = x[:,:3,:,:].contiguous()
        x, _ = self.encoder(x)
        if points.size(1) != 3:
            points = points.transpose(2,1)
        y = x.unsqueeze(2).expand(x.size(0), x.size(1), points.size(2)).contiguous()
        y = torch.cat((points, y), 1).contiguous()
        if mode == 'deform1':
            outs = self.decoder[0](y, 0)
        elif mode == 'deform2':
            outs = self.decoder2[0](y, 0)
            outs = outs + points
        elif mode == 'estimate':
            outs = self.estimate(y)
        elif mode == 'estimate2':
            outs = self.estimate2(y)
        elif mode == 'refine':
            outs = self.refine(y)
            outs1 = outs[:, 0].unsqueeze(1)
            outs2 = outs[:, 1].unsqueeze(1)
            outs = outs1 * vector1 + outs2 * vector2 + points
        else:
            outs = None
        return outs.contiguous().transpose(2,1).contiguous().squeeze(2)


class SVR_TMNet_Split(nn.Module):
    def __init__(self,  bottleneck_size = 1024):
        super(SVR_TMNet_Split, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.encoder = resnet_3D.resnet18_3D(num_classes=self.bottleneck_size)
        # self.encoder2 = resnet_3D.resnet18_3D_part(num_classes=self.bottleneck_size)
        '''
        self.encoders2 = []
        for i in range(0, 8):
            self.encoders2.append(resnet_3D.resnet18_3D_part(num_classes=self.bottleneck_size))
            '''
        self.decoder = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])
        self.decoder2 = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])
        self.decoder3 = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])
        # self.decoder4 = nn.ModuleList([DeformNet_scale(bottleneck_size=3 + self.bottleneck_size)])
        self.decoder_vol = Decoder_volume()
        self.decoder_vol2 = Decoder_volume()
        self.estimate = Estimator(bottleneck_size=3 + self.bottleneck_size)
        self.estimate2 = Estimator(bottleneck_size=3+self.bottleneck_size)
        self.estimate3 = Estimator(bottleneck_size=3+self.bottleneck_size)
        self.refine = Refiner(bottleneck_size=3 + self.bottleneck_size)

    def forward(self,x,points,vector1=0,vector2=0,mode='deform1'):
        '''
        x = x[:,:3,:,:].contiguous()
        x, _ = self.encoder(x)
        '''
        outs = []
        outs_vol = []
        if mode == 'deform1':
            for i in range(0, points.shape[0]):
                x_part = x[i]
                # x_part = x_part[:,:3,:,:].contiguous()
                x_part, _ = self.encoder(x_part)
                if points[i].size(1) != 3:
                    points[i] = points[i].transpose(2, 1)
                y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points[i].size(2)).contiguous()
                y = torch.cat((points[i], y), 1).contiguous()
                res = self.decoder[0](y, points[i].shape[1])
                # res_vol = self.decoder_vol(x_part)
                # res = res + points[i].unsqueeze(dim=0)
                outs.append(res.contiguous().transpose(2,1).contiguous())
            return outs #, res_vol
        elif mode == 'deform2':
            x_part = x[0]
            x_part = x_part[:,:3,:,:].contiguous()
            x_part, _ = self.encoder(x_part)
            for i in range(0, points.shape[0]):
                if points[i].size(1) != 3:
                    points[i] = points[i].transpose(2, 1)
                y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points[i].size(2)).contiguous()
                y = torch.cat((points[i], y), 1).contiguous()
                res = self.decoder2[0](y, points[i].shape[1])
                # res_vol = self.decoder_vol2(x_part)
                # res = self.decoder2[0](y)
                res = res + points[i]
                outs.append(res.contiguous().transpose(2,1).contiguous())
                # outs_vol.append(res_vol)
            return outs
        elif mode == 'deform2_vol':
            for i in range(0, points.shape[0]):
                x_part = x[i]
                x_part = x_part[:, :3, :, :].contiguous()
                # x_part, _ = self.encoder2(x_part)
                x_part, _ = self.encoders2[i](x_part)

                if points[i].size(1) != 3:
                    points[i] = points[i].transpose(2, 1)
                y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points[i].size(2)).contiguous()
                y = torch.cat((points[i], y), 1).contiguous()
                res = self.decoder2[0](y, points[i].shape[1])
                # res_vol = self.decoder_vol2(x_part)
                # res = self.decoder2[0](y)
                res = res + points[i]
                outs.append(res.contiguous().transpose(2,1).contiguous())
                # outs_vol.append(res_vol)
            return outs# , outs_vol
        elif mode == 'deform2_mlt':
            for i in range(0, points.shape[0]):
                x_part = x[i]
                # x_part = x_part[:, :3, :, :].contiguous()
                x_part, _ = self.encoder(x_part)

                if points[i].size(1) != 3:
                    points[i] = points[i].transpose(2, 1)
                y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points[i].size(2)).contiguous()
                y = torch.cat((points[i], y), 1).contiguous()
                res = self.decoder2[0](y, points[i].shape[1])
                res_vol = self.decoder_vol2(x_part)
                # res = self.decoder2[0](y)
                res = res + points[i]
                outs.append(res.contiguous().transpose(2,1).contiguous().detach())
                outs_vol.append(res_vol.detach())
            return outs, outs_vol
        elif mode == 'deform3':
            x_part = x[0]
            x_part = x_part[:,:3,:,:].contiguous()
            x_part, _ = self.encoder(x_part)

            for i in range(0, points.shape[0]):
                if points[i].size(1) != 3:
                    points[i] = points[i].transpose(2, 1)
                y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points[i].size(2)).contiguous()
                y = torch.cat((points[i], y), 1).contiguous()

                vec = self.decoder3[0](y, 0)

                res = 1e-5 * vec + points[i]
                print(vec)
                outs.append(res.contiguous().transpose(2,1).contiguous())
            # np.savetxt('./reses.txt', np.array([t.detach().cpu().numpy() for t in outs]).reshape(-1))
            return outs
        elif mode == 'deform3_mlt':
            for i in range(0, points.shape[0]):
                x_part = x[i]
                x_part = x_part[:, :3, :, :].contiguous()
                x_part, _ = self.encoder(x_part)

                if points[i].size(1) != 3:
                    points[i] = points[i].transpose(2, 1)
                y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points[i].size(2)).contiguous()
                y = torch.cat((points[i], y), 1).contiguous()

                vec = self.decoder3[0](y, 0)

                res = 0.5 * vec + points[i]
                # print(vec)
                outs.append(res.contiguous().transpose(2, 1).contiguous().detach())
            # np.savetxt('./reses.txt', np.array([t.detach().cpu().numpy() for t in outs]).reshape(-1))
            return outs
        elif mode == 'estimate':
            x_part = x
            # x_part = x_part[:, :3, :, :].contiguous()
            x_part, _ = self.encoder(x_part)
            if points.size(1) != 3:
                points = points.transpose(2,1)
            y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points.size(2)).contiguous()
            y = torch.cat((points, y), 1).contiguous()
            outs = self.estimate(y)
            outs = outs.contiguous().transpose(2,1).contiguous().squeeze(2)
            return outs
        elif mode == 'estimate2':
            x_part = x
            # x_part = x_part[:, :3, :, :].contiguous()
            x_part, _ = self.encoder(x_part)
            if points.size(1) != 3:
                points = points.transpose(2,1)
            y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points.size(2)).contiguous()
            y = torch.cat((points, y), 1).contiguous()
            outs = self.estimate2(y)
            outs = outs.contiguous().transpose(2,1).contiguous().squeeze(2)
            return outs
        elif mode == 'estimate3':
            x_part = x[0]
            x_part = x_part[:, :3, :, :].contiguous()
            x_part, _ = self.encoder(x_part)
            if points.size(1) != 3:
                points = points.transpose(2,1)
            y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points.size(2)).contiguous()
            y = torch.cat((points, y), 1).contiguous()
            outs = self.estimate3(y)
            outs = outs.contiguous().transpose(2,1).contiguous().squeeze(2)
            return outs
        elif mode == 'refine':
            outs = self.refine(y)
            outs1 = outs[:, 0].unsqueeze(1)
            outs2 = outs[:, 1].unsqueeze(1)
            outs = outs1 * vector1 + outs2 * vector2 + points
            return outs
        elif mode == 'refine_split':
            x_part = x[0]
            x_part = x_part[:,:3,:,:].contiguous()
            x_part, _ = self.encoder(x_part)

            for i in range(0, points.shape[0]):
                if points[i].size(1) != 3:
                    points[i] = points[i].transpose(2, 1)
                y = x_part.unsqueeze(2).expand(x_part.size(0), x_part.size(1), points[i].size(2)).contiguous()
                y = torch.cat((points[i], y), 1).contiguous()

                outs = self.refine(y)
                outs1 = outs[:, 0].unsqueeze(1)
                outs2 = outs[:, 1].unsqueeze(1)
                outs = outs1 * vector1 + outs2 * vector2 + points
                outs = outs.contiguous().transpose(2,1).contiguous().squeeze(2)
            return outs
        else:
            outs = None
            return outs
        # return outs.contiguous().transpose(2,1).contiguous().squeeze(2)

class Pretrain(nn.Module):
    def __init__(self,  bottleneck_size = 1024,num_points=2500):
        super(Pretrain, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.num_points = num_points
        self.pc_encoder = nn.Sequential(
        PointNetfeat(self.num_points, global_feat = False, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.encoder = resnet_3D.resnet50_3D(num_classes=self.bottleneck_size)
        self.decoder = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])

    def forward(self, x, mode='point'):
        cat_features = []
        if mode == 'point':
            # print(x.size()) # (point) torch.Size([2, 3, 2500])
            x = self.pc_encoder(x)
            # print(x.size()) # (point) torch.Size([2, 3, 1024])
        else:
            # print(x.size())
            x, cat_features = self.encoder(x)
            # print(x.size())
        rand_grid = torch.cuda.FloatTensor(x.size(0),3,self.num_points)
        # print(rand_grid.size()) # (point) torch.Size([2, 3, 2500])
        rand_grid.data.normal_(0,1)
        # print(rand_grid.size()) # (point) torch.Size([2, 3, 2500])
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True))\
            .expand(x.size(0),3,self.num_points)
        # print(rand_grid.size())  # (point) torch.Size([2, 3, 2500])
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        # print(y.size()) # torch.Size([2, 1021, 2500])
        y = torch.cat( (rand_grid, y), 1).contiguous()
        # print(y.size()) # torch.Size([2, 1024, 2500])
        outs = self.decoder[0](y, 0)
        # print(outs.size()) # torch.Size([2, 3, 2500])
        # print(outs.contiguous().transpose(2,1).contiguous().size()) # torch.Size([2, 2500, 3])
        return outs.contiguous().transpose(2,1).contiguous()


