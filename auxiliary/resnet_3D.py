import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet_3D', 'resnet18_3D', 'resnet34_3D', 'resnet50_3D', 'resnet101_3D',
           'resnet152_3D']

def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock_3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.dropout = nn.Dropout3d(0.5)

    def forward(self, x):

        '''
        # 1st layer : # torch.Size([1, 64, 56, 56, 56])
        # 2nd layer : # torch.Size([1, 128, 28, 28, 28])
        # 3rd layer : # torch.Size([1, 256, 14, 14, 14])
        # 4th layer : # torch.Size([1, 512, 7, 7, 7])
        '''

        residual = x

        # print(x.size())
        out = self.conv1(x)
        # print(out.size())
        out = self.bn1(out)
        # print(out.size())
        out = self.relu(out)
        # print(out.size())
        # out = self.dropout(out)

        # print(out.size())
        out = self.conv2(out)
        # print(out.size())
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # out = self.dropout(out)

        return out


class Bottleneck_3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = torch.add(out, residual)
        out = self.relu(out)

        return out


class ResNet_3D(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout3d(0.625)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], kernel_size=1, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], kernel_size=1, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], kernel_size=1, stride=2)
        # self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.avgpool = nn.AvgPool3d(6)
        # self.avgpool = nn.AvgPool3d(3)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]  * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, kernel_size=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=kernel_size, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
                # nn.Dropout3d(0.625),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        cat_features = []

        # print(x.size())  # torch.Size([1, 1, 224, 224, 224])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.size())  # torch.Size([1, 64, 112, 112, 112])
        x = self.maxpool(x)
        # x = self.dropout(x)
        # print(x.size())  # torch.Size([1, 64, 56, 56, 56])

        x = self.layer1(x)
        # print(x.size())  # torch.Size([1, 64, 56, 56, 56])
        x = self.layer2(x)
        cat_features.append(x)
        # print(x.size())  # torch.Size([1, 128, 28, 28, 28])
        x = self.layer3(x)
        cat_features.append(x)
        # print(x.size())  # torch.Size([1, 256, 14, 14, 14])
        x = self.layer4(x)
        cat_features.append(x)
        # print(x.size())  # torch.Size([1, 512, 7, 7, 7])
        # x = self.layer5(x)
        # print(x.size())  # torch.Size([1, 1024, 3, 3, 3])

        x = self.avgpool(x)
        # print(x.size())  # torch.Size([1, 1024, 1, 1, 1])
        x = x.view(x.size(0), -1)
        # print(x.size())  # torch.Size([1, 512])
        x = self.fc(x)
        # print(x.size())  # torch.Size([1, 1024])

        return x, cat_features


def resnet18_3D(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet_3D(BasicBlock_3D, [2, 2, 2, 2], **kwargs)
    model = ResNet_3D(BasicBlock_3D, [2, 2, 2, 2, 2], **kwargs)
    return model


def resnet34_3D(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_3D(BasicBlock_3D, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_3D(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_3D(Bottleneck_3D, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_3D(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_3D(Bottleneck_3D, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_3D(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_3D(Bottleneck_3D, [3, 8, 36, 3], **kwargs)
    return model
