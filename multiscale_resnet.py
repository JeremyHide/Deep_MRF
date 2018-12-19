# Sheng Liu

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)



class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        

        out += residual
        out = self.relu(out)

        return out

class BasicBlock3x3_(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3_, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.prelu1 = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.prelu2 = nn.PReLU()
        self.downsample = downsample
        self.stride = stride
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.prelu1(x)
        out = self.conv1(x)
        out = self.prelu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)

        return out

class BasicBlock5x5_(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5_, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.prelu1 = nn.PReLU()
        self.conv2 = conv5x5(planes, planes)
        self.prelu2 = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.prelu1(x)
        out = self.conv1(x)
        out = self.prelu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out
        #out1 = self.relu(out1)
        # out += residual

        return out1
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
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

        out += residual
        out = self.relu(out)

        return out
class BasicBlock7x7_(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7_, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.prelu1 = nn.PReLU()
        self.conv2 = conv7x7(planes, planes)
        self.prelu2 = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.prelu1(x)
        out = self.conv1(x)
        out = self.prelu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        #out1 = self.relu(out1)
        # out += residual

        return out1


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1



class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1




class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        self.lin_in = nn.Linear(306,512)
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6 ---- 20 14 9
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        #self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(256*3, 500)
        self.bn_fc1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn_fc2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 500)
        self.bn_fc3 = nn.BatchNorm1d(500)
        self.fc4 = nn.Linear(500, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.lin_in(x0)
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        x0 = self.maxpool(x0)
        

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        #print(x.shape)
        
        #x = self.layer3x3_4(x)
        x = self.maxpool3(x)
        
        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        #print(y.shape)

        # y = self.layer5x5_4(y)
        y = self.maxpool5(y)
        

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        
        #print(z.shape)
        
        #z = self.layer7x7_4(z)
        z = self.maxpool7(z)
        

        out = torch.cat([x, y, z], dim=1)
        #out = torch.cat([x, y], dim=1)

        out = out.squeeze()
        #out = self.drop(out)
        out1 = self.relu(self.bn_fc1(self.fc1(out)))
        out1 = self.relu(self.bn_fc2(self.fc2(out1)))
        out1 = self.relu(self.bn_fc3(self.fc3(out1)))
        out1 = self.fc4(out1)

        return out1

class MSResNet2(nn.Module):
#This net removed  the first linear layer and the last few linear layers
    def __init__(self, input_channel, layers=[1, 1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet2, self).__init__()

        self.lin_in = nn.Linear(306,512*4)

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)
        self.layer3x3_5 = self._make_layer3(BasicBlock3x3, 512*2, layers[4], stride=2)

        # maxplooing kernel size: 16, 11, 6 ---- 20 14 9 -- 16 11 5
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.layer5x5_5 = self._make_layer5(BasicBlock5x5, 512*2, layers[4], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.layer7x7_5 = self._make_layer7(BasicBlock7x7, 512*2, layers[4], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=5, stride=1, padding=0)

        #self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(512*2*3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.lin_in(x0)
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        x0 = self.maxpool(x0)
        

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        
        
        x = self.layer3x3_4(x)
        x = self.layer3x3_5(x)
        #print(x.shape)
        x = self.maxpool3(x)


        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        y = self.layer5x5_4(y)
        y = self.layer5x5_5(y)
        #print(y.shape)
        y = self.maxpool5(y)
        

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)       
        #print(z.shape)
        z = self.layer7x7_4(z)
        z = self.layer7x7_5(z)
        #print(z.shape)
        z = self.maxpool7(z)
        

        out = torch.cat([x, y, z], dim=1)

        #out = torch.cat([x, y], dim=1)

        out = out.squeeze()
        

        #out = self.drop(out)
        #out1 = (self.fc1(out))
        out1 = self.relu(self.bn_fc1(self.fc1(out)))
        out1 = self.relu(self.bn_fc2(self.fc2(out1)))
        out1 = self.relu(self.bn_fc3(self.fc3(out1)))
        out1 = self.fc4(out1)
        
        return out1



class MSResNet3(nn.Module):
#This net removed  the first linear layer and the last few linear layers
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet3, self).__init__()

        self.lin_in = nn.Linear(306,512*2)

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6 ---- 20 14 9 -- 16 11 5 --2 extra layers 16 16
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=5, stride=1, padding=0)

        #self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(512*3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.lin_in(x0)
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        x0 = self.maxpool(x0)
        

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)       
        x = self.layer3x3_4(x)
        #print(x.shape)
        x = self.maxpool3(x)


        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        y = self.layer5x5_4(y)
        #print(y.shape)
        y = self.maxpool5(y)
        

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        #print(z.shape)
        z = self.layer7x7_4(z)
        z = self.maxpool7(z)
        

        out = torch.cat([x, y, z], dim=1)

        #out = torch.cat([x, y], dim=1)

        out = out.squeeze()
        

        #out = self.drop(out)
        #out1 = (self.fc1(out))
        out1 = self.relu(self.bn_fc1(self.fc1(out)))
        out1 = self.relu(self.bn_fc2(self.fc2(out1)))
        out1 = self.relu(self.bn_fc3(self.fc3(out1)))
        out1 = self.fc4(out1)
        
        return out1

class MSResNet4(nn.Module):
#This net removed  the first linear layer and the last few linear layers
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet4, self).__init__()

        self.lin_in = nn.Linear(306,512*2)

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3_, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3_, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3_, 256, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3_, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6 ---- 20 14 9 -- 16 11 5 --2 extra layers 16 16
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5_, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5_, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5_, 256, layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5_, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7_, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7_, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7_, 256, layers[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7_, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=5, stride=1, padding=0)

        #self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(512*3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.lin_in(x0)
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        x0 = self.maxpool(x0)
        

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)       
        x = self.layer3x3_4(x)
        #print(x.shape)
        x = self.maxpool3(x)


        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        y = self.layer5x5_4(y)
        #print(y.shape)
        y = self.maxpool5(y)
        

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        #print(z.shape)
        z = self.layer7x7_4(z)
        z = self.maxpool7(z)
        

        out = torch.cat([x, y, z], dim=1)

        #out = torch.cat([x, y], dim=1)

        out = out.squeeze()
        

        #out = self.drop(out)
        #out1 = (self.fc1(out))
        out1 = self.relu(self.bn_fc1(self.fc1(out)))
        out1 = self.relu(self.bn_fc2(self.fc2(out1)))
        out1 = self.relu(self.bn_fc3(self.fc3(out1)))
        out1 = self.fc4(out1)
        
        return out1

class ResNet(nn.Module):
#This net removed  the first linear layer and the last few linear layers
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(ResNet, self).__init__()

        self.lin_in = nn.Linear(512,512*2)
        self.bn_in = nn.BatchNorm1d(512*2)

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer5x5_1 = self._make_layer5(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock3x3, 256, layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock3x3, 512, layers[3], stride=2)
        #self.layer5x5_5 = self._make_layer5(Bottleneck, 512*2, layers[4], stride=2)
        #self.layer5x5_6 = self._make_layer5(BasicBlock5x5, 512*4, layers[4], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=8, stride=1, padding=0) #old dim: 11
        self.fc0 =nn.Linear(16,1)
        #self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256*2)
        self.bn_fc2 = nn.BatchNorm1d(256*2)
        self.fc3 = nn.Linear(256*2, 256*2)
        self.bn_fc3 = nn.BatchNorm1d(256*2)
        self.fc4 = nn.Linear(256*2, num_classes)

        #self.reset_params()

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x0):
        x0 = self.lin_in(x0)
        #x0 = self.bn_in(x0)
        #x0 = self.relu(x0)
        x0 = x0.unsqueeze(1)

        x0 = self.conv1(x0)

        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        x0 = self.maxpool(x0)
        
        
        y = self.layer5x5_1(x0)
        
        y = self.layer5x5_2(y)

        y = self.layer5x5_3(y)

        y = self.layer5x5_4(y)


        
        y = self.fc0(y)
        #print(y.shape)
        #y = self.layer5x5_5(y)
        #y = self.layer5x5_6(y)
        #y = self.maxpool5(y)
        


        out = y
        out = out.squeeze()

        

        #out = self.drop(out)
        out1 = self.relu(self.bn_fc1(self.fc1(out)))
        out1 = self.relu(self.bn_fc2(self.fc2(out1)))
        out1 = self.relu(self.bn_fc3(self.fc3(out1)))
        out1 = self.fc4(out1)
        return out1
        




if __name__ == '__main__':
    
    inp = torch.randn(3,512)
    model = ResNet(input_channel=1, num_classes=4)
    out= model(inp)
    print(out.shape)
    print(out)
    

