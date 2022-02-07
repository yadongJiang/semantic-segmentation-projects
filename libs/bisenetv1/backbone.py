import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

def conv3x3(in_chann, out_chann, stride=1):
    return nn.Conv2d(in_chann, out_chann, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_chann, out_chann, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chann, out_chann, stride)
        self.bn1 = nn.BatchNorm2d(out_chann)
        self.conv2 = conv3x3(out_chann, out_chann)
        self.bn2 = nn.BatchNorm2d(out_chann)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chann!=out_chann or stride!=1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chann, out_chann, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_chann)
            )

    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.downsample:
            residual = self.downsample(residual)
        
        out = residual + x
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layers(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layers(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layers(256, 512, num_blocks=2, stride=2)
        
        self._init_weights()

    def _init_weights(self):
        state_dict = model_zoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'fc' in k:
                continue
            new_state_dict[k] = v
        self_state_dict.update(new_state_dict)
        self.load_state_dict(self_state_dict)
        
    
    def _make_layers(self, in_chann, out_chann, num_blocks, stride=1):
        layers = [BasicBlock(in_chann, out_chann, stride=stride)]
        for i in range(1, num_blocks):
            layers.append(BasicBlock(out_chann, out_chann, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32