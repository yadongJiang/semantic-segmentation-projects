import torch
import torch.nn as nn
from .backbone import ResNet18

def ConvBNReLU(in_chann, out_chann, ks=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_chann, out_chann, kernel_size=ks, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_chann), 
        nn.ReLU(inplace=True)
    )

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chann, out_chann):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chann, out_chann, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chann, out_chann, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chann)
        
        self._init_weight()

    def _init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if not layer.bias is None:
                    nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=[2, 3], keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.backbone = ResNet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)

        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.up16 = nn.Upsample(scale_factor=2)

        self._init_weight()
        
    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        # print("feat8 size: ", feat8.size(), "feat16 size: ", feat16.size(), "feat32 size: ", feat32.size())

        avg = torch.mean(feat32, dim=[2, 3], keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up) # 1/16

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up) # 1/8

        return feat16_up, feat32_up # 1/8 1/16
        
    def _init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if not layer.bias is None:
                    nn.init.constant_(layer.bias, 0)

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self._init_weight()

    def _init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)

        return feat

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chann, out_chann):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chann, out_chann, ks=1, stride=1, padding=0)
        self.conv = nn.Conv2d(out_chann, out_chann, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_chann)

    def forward(self, fsp, fcp):
        feat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(feat)

        atten = torch.mean(feat, dim=[2, 3], keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

class BiSeNetHead(nn.Module):
    def __init__(self, in_chann, mid_chann, num_classes, scale_factor=32):
        super(BiSeNetHead, self).__init__()
        self.conv = ConvBNReLU(in_chann, mid_chann, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chann, num_classes, kernel_size=1, stride=1, bias=True)
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

        self._init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def _init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)