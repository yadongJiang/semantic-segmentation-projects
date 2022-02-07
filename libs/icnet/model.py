import torch
import torch.nn as nn
from .base_model import BaseModel
import torch.nn.functional as F
import warnings
import thop
warnings.filterwarnings('ignore')

class CBR(nn.Module):
    def __init__(self, in_chans, out_chans, k, s, p=None):
        super(CBR, self).__init__()
        p = k//2 if p is None else p
        self.conv = nn.Conv2d(in_chans, out_chans, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PyramidPoolingModule(nn.Module):
    def __init__(self, kernel=(1, 3, 5, 7), input_channels=2048):
        super(PyramidPoolingModule, self).__init__()
        out_channs = input_channels // len(kernel)
        self.pools = nn.ModuleList()
        for k in kernel:
            self.pools.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=k),
                    nn.Conv2d(input_channels, out_channs, 1, 1, bias=False)
                )
            )
        self.conv = nn.Conv2d(out_channs * len(kernel) + input_channels, input_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        features = []
        for i, pool in enumerate(self.pools):
            feat = pool(x)
            feat = F.interpolate(feat, size=x.size()[2:], mode="bilinear", align_corners=True)
            features.append(feat)
        outs = self.conv(torch.cat(features + [x], dim=1))

        return self.relu(self.bn(outs))

class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            # norm_layer(out_channels)
            nn.BatchNorm2d(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            # norm_layer(out_channels)
            nn.BatchNorm2d(out_channels)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        '''x = F.relu(x, inplace=True)''' # 原融合层
        x = self.fuse(x) # 新添加的卷积层，融合层
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls

class ICNetHead(nn.Module):
    def __init__(self, num_classes):
        super(ICNetHead, self).__init__()
        self.CFF24 = CascadeFeatureFusion(2048, 512, 128, num_classes)
        self.CFF12 = CascadeFeatureFusion(128, 64, 128, num_classes)
        # 原分类层
        '''self.conv_cls = nn.Conv2d(128, num_classes, 1, bias=False)'''
        # 新分类层
        self.conv_cls = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.1), 
            nn.Conv2d(64, num_classes, 1, 1, 0, bias=True)
        )
    
    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        out_24_feat, out_24_cls = self.CFF24(x_sub4, x_sub2) # 1/16
        outputs.append(out_24_cls)

        out_12_feat, out_12_cls = self.CFF12(out_24_feat, x_sub1) # 1/8
        outputs.append(out_12_cls)

        up_x2 = F.interpolate(out_12_feat, scale_factor=2, mode="bilinear", align_corners=True) # 1/4
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)

        up_x4 = F.interpolate(up_x2, scale_factor=4, mode="bilinear", align_corners=True) # 原图大小
        outputs.append(up_x4)
        
        outputs.reverse() # 翻转
        return outputs

class ICNet(BaseModel):
    def __init__(self, num_classes, backend="resnet50", pretrained=True):
        super(ICNet, self).__init__(backend, pretrained=pretrained)
        self.conv_sub1 = nn.Sequential(
            CBR(3, 32, 3, 2, 1),
            CBR(32, 32, 3, 2, 1),
            CBR(32, 64, 3, 2, 1),
        ) # 最大输入分支

        self.ppm = PyramidPoolingModule()

        self.head = ICNetHead(num_classes)

    def forward(self, x):
        sub1 = self.conv_sub1(x) # 最大输入分支, 输出, 1/8

        sub2_input = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True) # 中等输出分支， 1/2
        _, sub2, _, _ = self.base_forward(sub2_input) # 中等输入分支输出 1/2 * 1/8 = 1/16

        sub4_input = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=True) # 最小输入分支， 1/4
        _, _, _, sub4 = self.base_forward(sub4_input) # 最小输入分支 1/4 * 1/8 = 1/32
        sub4 = self.ppm(sub4)

        outputs = self.head(sub1, sub2, sub4)
        if self.training:
            return outputs
        return outputs[0]

from utils.utils import get_model_infos

@get_model_infos
def icnet(num_classes, backend="resnet50", pretrained=False):
    model = ICNet(num_classes, backend, pretrained)
    return model

if __name__ == "__main__":
    import time
    model = ICNet(4)
    inputs = torch.randn(1, 3, 640, 640)
    s = time.time()
    outputs = model(inputs)
    print("cost time: ", time.time() - s)
    for out in outputs:
        print(out.size())

    # icnet(4)