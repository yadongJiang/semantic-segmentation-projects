import torch
import torch.nn as nn
from .backbones import *
import torch.nn.functional as F

class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(in_channels, k) for k in kernel_size])
        self.conv = nn.Conv2d(in_channels * (len(kernel_size) + 1), out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_stage(self, in_channels, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
        return nn.Sequential(pool, conv)

    def forward(self, x):
        h, w = x.size()[2:]
        features = [F.interpolate(stage(x), size=[h, w], mode="bilinear", align_corners=True) for stage in self.stages] + [x]
        bottle = self.conv(torch.cat(features, dim=1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.PReLU(out_channels)
        )
    
    def forward(self, x):
        h, w = 2*x.size()[2], 2*x.size()[3]
        p = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return self.conv(p)

class PSPNet(nn.Module):
    def __init__(self,  
                 num_classes, 
                 psp_kernel=(1, 2, 3, 6), 
                 psp_in_channels=2048, 
                 deep_features_size=1024, 
                 backend='resnet34', 
                 pretrained=False):
        super(PSPNet, self).__init__()
        self.backbone = eval(backend)(pretrained=pretrained) # backbone
        self.psp = PSPModule(psp_in_channels, 1024, psp_kernel)
        self.drop1 = nn.Dropout(p=0.3)

        self.up1 = PSPUpsample(1024, 256)
        self.drop2 = nn.Dropout(p=0.15)

        self.up2 = PSPUpsample(256, 64)
        self.drop3 = nn.Dropout(p=0.15)

        self.up3 = PSPUpsample(64, 64)
        self.drop4 = nn.Dropout(p=0.15)

        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(64, num_classes, 1, 1, 0, bias=True)
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(deep_features_size, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        # feat: 1/8, [bs, 512, 80, 80] || aux_feat: 1/8, [bs, 256, 80, 80]
        feat, aux_feat = self.backbone(x)
        p = self.psp(feat)
        p = self.drop1(p)

        p = self.drop2(self.up1(p))
        p = self.drop3(self.up2(p))
        p = self.drop4(self.up3(p))

        out = self.head(p)
        auxiliary = F.interpolate(self.aux_head(aux_feat), size=(h, w), mode="bilinear", align_corners=True)
        
        if self.training:
            return out, auxiliary
        return out

if __name__ == "__main__":
    model = PSPNet(4, psp_in_channels=512, deep_features_size=256)
    inputs = torch.randn(1, 3, 640, 640)
    model(inputs)