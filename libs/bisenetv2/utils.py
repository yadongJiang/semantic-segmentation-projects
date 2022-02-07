import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBNReLU(in_chann, out_chann, ks=3, st=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_chann, out_chann, ks, st, p, bias=False), 
        nn.BatchNorm2d(out_chann), 
        nn.ReLU(inplace=True)
    )

class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, 2, 1), 
            ConvBNReLU(64, 64, 3, 1, 1)
        )

        self.s2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1), 
            ConvBNReLU(64, 64, 3, 1, 1)
        )

        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1), 
            ConvBNReLU(128, 128, 3, 1, 1), 
            ConvBNReLU(128, 128, 3, 1, 1)
        )

        self._init_weight()

    def forward(self, x):
        feat = self.s1(x)
        feat = self.s2(feat)
        feat = self.s3(feat)
        return feat

    def _init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

class Stem(nn.Module):
    def __init__(self, in_chann, out_chann):
        super(Stem, self).__init__()
        mid_chann = out_chann // 2
        self.conv = ConvBNReLU(in_chann, out_chann, 3, 2, 1)
        self.left = nn.Sequential(
            ConvBNReLU(out_chann, mid_chann, 1, 1, 0), 
            ConvBNReLU(mid_chann, out_chann, 3, 2, 1)
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.last_conv = ConvBNReLU(out_chann*2, out_chann, 3, 1, 1)

    def forward(self, x):
        feat = self.conv(x)
        left_feat = self.left(feat)
        right_feat = self.right(feat)
        out = torch.cat([left_feat, right_feat], dim=1)
        out = self.last_conv(out)
        
        return out

class GELayers1(nn.Module):
    def __init__(self, in_chann, out_chann):
        super(GELayers1, self).__init__()
        mid_chann = in_chann * 6
        self.conv1 = ConvBNReLU(in_chann, in_chann, 3, 1, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chann, mid_chann, kernel_size=3, stride=1, padding=1, groups=in_chann, bias=False), 
            nn.BatchNorm2d(mid_chann), 
            nn.ReLU(inplace=True), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chann, out_chann, kernel_size=1, stride=1,  bias=False), 
            nn.BatchNorm2d(out_chann)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        out = feat + x
        
        return self.relu(out)

class GELayers2(nn.Module):
    def __init__(self, in_chann, out_chann):
        super(GELayers2, self).__init__()
        mid_chann = in_chann * 6
        self.conv1 = ConvBNReLU(in_chann, in_chann, 3, 1, 1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_chann, mid_chann, kernel_size=3, stride=2, padding=1, groups=in_chann, bias=False), 
            nn.BatchNorm2d(mid_chann), 
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chann, mid_chann, kernel_size=3, stride=1, padding=1, groups=mid_chann, bias=False), 
            nn.BatchNorm2d(mid_chann), 
            nn.ReLU(inplace=True), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chann, out_chann, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_chann)
        )

        self.short = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, 3, 2, 1, groups=in_chann, bias=False), 
            nn.BatchNorm2d(in_chann), 
            nn.Conv2d(in_chann, out_chann, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_chann)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)

        out = feat + self.short(x)
        return self.relu(out)

class CEBlock(nn.Module):
    def __init__(self, in_chann):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_chann)
        self.conv1 = ConvBNReLU(in_chann, in_chann, 1, 1, 0)
        # self.conv2 = ConvBNReLU(in_chann, in_chann, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_chann, in_chann, 3, 1, 1, bias=True)
    
    def forward(self, x):
        # feat = self.bn(x)
        feat = torch.mean(x, dim=[2, 3], keepdim=True)
        feat = self.bn(feat)
        feat = self.conv1(feat)
        
        out = feat + x
        out = self.conv2(out)
        return out

class SegmentBranch(nn.Module):
    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.s1s2 = Stem(3, 16)
        self.s3 = nn.Sequential(
            GELayers2(16, 32), 
            GELayers1(32, 32), 
        )
        self.s4 = nn.Sequential(
            GELayers2(32, 64), 
            GELayers1(64, 64), 
        )

        self.s5_4 = nn.Sequential(
            GELayers2(64, 128), 
            GELayers1(128, 128), 
            GELayers1(128, 128), 
            GELayers1(128, 128), 
        )

        self.s5_5 = CEBlock(128)

    def forward(self, x):
        feat2 = self.s1s2(x) # 1/4
        feat3 = self.s3(feat2) # 1/8
        feat4 = self.s4(feat3) # 1/16
        feat5_4 = self.s5_4(feat4) # 1/32
        feat5_5 = self.s5_5(feat5_4) # 1/32
        return feat2, feat3, feat4, feat5_4, feat5_5

class ARMModule(nn.Module):
    def __init__(self, in_channs):
        super(ARMModule, self).__init__()
        self.conv = nn.Conv2d(in_channs, in_channs, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(in_channs)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        feature = torch.mean(x,  dim=[2, 3], keepdim=True)
        feature = self.bn(self.conv(feature))
        feature = self.sigmoid(feature)
        return feature * x

class FFMModule(nn.Module):
    def __init__(self, in_channs):
        super(FFMModule, self).__init__()
        self.conv1 = ConvBNReLU(in_channs * 2, in_channs, 3, 1, 1)

        self.conv2 = nn.Conv2d(in_channs, in_channs, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channs, in_channs, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seg_feat, detail_feat):
        feat = torch.cat([seg_feat, detail_feat], dim=1)
        feat = self.conv1(feat)

        f = torch.mean(feat, dim=[2, 3], keepdim=True)
        f = self.relu1(self.conv2(f))
        f = self.sigmoid(self.conv3(f))

        f = f*feat
        feat = f + feat
        return feat

class Aggreation(nn.Module):
    def __init__(self):
        super(Aggreation, self).__init__()
        self.conv1 = ConvBNReLU(128, 128, 1, 1, 0)
        self.arm1 = ARMModule(128)

        self.up32 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up32_conv = ConvBNReLU(128, 64, 3, 1, 1)

        self.arm2 = ARMModule(64)

        self.up16 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up16_conv = ConvBNReLU(64, 128, 3, 1, 1)

        self.ffm = FFMModule(128)

    def forward(self, seg32_1, seg32_2, seg16, detail):
        seg32_1 = torch.mean(seg32_1, dim=[2, 3], keepdim=True)
        seg32_1 = self.conv1(seg32_1)

        seg32_2 = self.arm1(seg32_2)

        feat_32 = seg32_1 + seg32_2
        feat_16 = self.up32_conv(self.up32(feat_32))

        seg_16 = self.arm2(seg16)
        seg_16 = seg_16 + feat_16
        feat_8 = self.up16_conv(self.up16(seg_16))

        out = self.ffm(feat_8, detail)

        return out, feat_8, feat_16

class BGABlock(nn.Module):
    def __init__(self, in_chann):
        super(BGABlock, self).__init__()
        self.detail_left = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, kernel_size=3, stride=1, padding=1, groups=in_chann, bias=False), 
            nn.BatchNorm2d(in_chann), 
            nn.Conv2d(in_chann, in_chann, kernel_size=1, stride=1, bias=True)
        )
        self.detail_right = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(in_chann), 
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.segment_left = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(in_chann), 
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )

        self.segment_right = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, kernel_size=3, stride=1, padding=1, groups=in_chann, bias=False), 
            nn.BatchNorm2d(in_chann), 
            nn.Conv2d(in_chann, in_chann, kernel_size=1, stride=1, bias=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(in_chann), 
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    def forward(self, detail, segment):
        detail_left = self.detail_left(detail) # 1/8
        detail_right = self.detail_right(detail) # 1/32

        segment_left = self.segment_left(segment) # 1/8
        segment_right = self.segment_right(segment) # 1/32
        
        left = segment_left.sigmoid() * detail_left # + detail_left
        right = detail_right * segment_right.sigmoid() # + segment_right
        out = left + self.up(right)
        
        return self.conv(out)

class SegmentHead(nn.Module):
    def __init__(self, in_chann, mid_chann, num_classes, up_factor=8):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chann, mid_chann, 3, 1, 1)
        self.drop = nn.Dropout(0.1)
        
        self.scale_factor = up_factor//2
        self.conv_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), 
            nn.Conv2d(mid_chann, num_classes, 3, 1, 1, bias=False), 
            nn.Upsample(scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.conv_out(x)
        return x

"""class SegmentHead(nn.Module):
    def __init__(self, in_chann, mid_chann, num_classes, up_factor=8):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chann, mid_chann, 3, 1, 1)
        self.drop = nn.Dropout(0.1)

        self.classifier = nn.Conv2d(mid_chann, num_classes, 1, 1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        ## x = self.conv_out(x)
        x = self.up(self.classifier(x))
        return x"""