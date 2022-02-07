import copy
import torch
import torch.nn as nn
from .backbone import *
import numpy as np
import torch.nn.functional as F
import thop

def ConvBNReLU(in_chann, out_chann, ks, st, p=1):
    return nn.Sequential(
        nn.Conv2d(in_chann, out_chann, kernel_size=ks, stride=st, padding=p, bias=False), 
        nn.BatchNorm2d(out_chann), 
        nn.ReLU(inplace=True)
    )

class Aggregation(nn.Module):
    def __init__(self, in_chann, out_chann, asy_ks=5):
        super(Aggregation, self).__init__()
        self.conv = ConvBNReLU(in_chann, out_chann, 3, 1, 1)
        self.left_asymmetric = nn.Sequential(
            nn.Conv2d(out_chann, out_chann, kernel_size=(1, asy_ks), stride=1, \
                        padding=(0, asy_ks//2), groups=out_chann, bias=True),
            nn.Conv2d(out_chann, out_chann, kernel_size=(asy_ks, 1), stride=1, \
                        padding=(asy_ks//2, 0), groups=out_chann, bias=True),
        )
        self.right_asymmetric = nn.Sequential(
            nn.Conv2d(out_chann, out_chann, kernel_size=(asy_ks, 1), stride=1, \
                        padding=(asy_ks//2, 0), groups=out_chann, bias=True),
            nn.Conv2d(out_chann, out_chann, kernel_size=(1, asy_ks), stride=1, \
                        padding=(0, asy_ks//2), groups=out_chann, bias=True),
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_chann), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        left = self.left_asymmetric(x)
        right = self.right_asymmetric(x)
        out = left + right
        out = self.bn_relu(out)
        return out

class DeepLabHead(nn.Module):
    def __init__(self, num_classes, last_channels, mid_channels, low_channels):
        super(DeepLabHead, self).__init__()
        self.low_process = ConvBNReLU(low_channels, 48, 1, 1, 0)
        self.mid_process = ConvBNReLU(mid_channels, 48, 1, 1, 0)

        self.mid_project = ConvBNReLU(304, 256, 3, 1, 1)
        self.classifier = nn.Sequential(
            ConvBNReLU(304, 256, 3, 1, 1), 
            nn.Dropout(0.1), 
            nn.Conv2d(256, num_classes, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=True)
        )
    
    def forward(self, last_feat, mid_feat, low_feat):
        low_feat = self.low_process(low_feat)
        mid_feat = self.mid_process(mid_feat)

        last_feat = F.interpolate(last_feat, size=mid_feat.size()[2:], mode="bilinear", align_corners=True)
        
        mid_feat = torch.cat([last_feat, mid_feat], dim=1)
        mid_feat = self.mid_project(mid_feat)
        mid_feat = F.interpolate(mid_feat, size=low_feat.size()[2:], mode="bilinear", align_corners=True)
        
        out_feat = torch.cat([mid_feat, low_feat], dim=1)
        out = self.classifier(out_feat)
        return out

class CPNet(nn.Module):
    def __init__(self, num_classes, input_channels=512, 
                 prior_channels=512, prior_size=(40, 40), backend="resnet34", pretrained=True):
        super(CPNet, self).__init__()
        self.prior_size = np.prod(prior_size)
        self.num_classes = num_classes
        self.prior_channels = prior_channels

        self.backbone = eval(backend)(pretrained=pretrained) # backbone
        self.aggregation = Aggregation(input_channels, prior_channels, 11) # 特征聚合，丰富特征的上下文信息
        
        self.prior_conv = nn.Sequential(
            nn.Conv2d(prior_channels, self.prior_size, kernel_size=1, stride=1, bias=True), 
            # nn.BatchNorm2d(self.prior_size)
        )

        self.intra_conv = ConvBNReLU(prior_channels, prior_channels, 1, 1, 0)
        self.inter_conv = ConvBNReLU(prior_channels, prior_channels, 1, 1, 0)

        self.post_process = nn.Sequential(
            ConvBNReLU(input_channels + prior_channels*2, 256, 1, 1, 0), 
            ConvBNReLU(256, 256, 3, 1, 1) # prior_channels
        )

        # without deeplab
        self.head = nn.Sequential(
            ConvBNReLU(256, 256, 3, 1, 1), # prior_channels
            nn.Dropout(0.1), 
            nn.Conv2d(256, num_classes, 1, 1, bias=True)
        )

        # with deeplab
        '''self.deeplab_head = DeepLabHead(num_classes, 256, 128, 64)'''
    
    def _reinit(self, input_size):
        input_size = input_size/16
        self.prior_size = int(np.prod(input_size))
        self.prior_conv = nn.Sequential(
            nn.Conv2d(self.prior_channels, self.prior_size, kernel_size=1, stride=1, bias=True), 
        )

    def forward(self, x):
        feat, feat_2, feat_1 = self.backbone(x)
        h, w = feat.size()[2:]

        value = self.aggregation(feat)
        context_proir_map = self.prior_conv(value)
        
        context_proir_map = context_proir_map.view(context_proir_map.size()[0], \
                                                    -1, self.prior_size).permute(0, 2, 1)
        intra_context_proir_map = torch.sigmoid(context_proir_map) # [bs, 40*40, 40*40], 类内
        inter_context_prior_map = 1 - context_proir_map # 类间

        value = value.view(value.size()[0], value.size()[1], -1).permute(0, 2, 1).contiguous() # [bs, 512, 40*40]==>[bs, 40*40, 512]
        intra_context_proir_map = F.softmax(intra_context_proir_map, dim=-1)
        intra_context = torch.matmul(intra_context_proir_map, value) # [bs, 40*40, 512] # 利用类内全局特征更新每一个特征
        # intra_context = intra_context.div(self.prior_size)
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(intra_context.size(0), self.prior_channels, h, w)
        intra_context = self.intra_conv(intra_context)

        inter_context_prior_map = F.softmax(inter_context_prior_map, dim=-1)
        inter_context = torch.matmul(inter_context_prior_map, value)
        # inter_context = inter_context.div(self.prior_size)
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(inter_context.size(0), self.prior_channels, h, w)
        inter_context = self.inter_conv(inter_context)

        out = torch.cat([feat, intra_context, inter_context], dim=1)
        out = self.post_process(out)
        
        # without deeplab
        seg_out = self.head(out)
        seg_out = F.interpolate(seg_out, size=(x.size()[2], x.size()[3]), mode="bilinear", align_corners=True)

        # with deeplab
        '''seg_out = self.deeplab_head(out, feat_2, feat_1)
        seg_out = F.interpolate(seg_out, size=x.size()[2:], mode="bilinear", align_corners=True)'''

        if self.training:
            return seg_out, intra_context_proir_map
        return seg_out

from utils.utils import get_model_infos
@get_model_infos
def cpnet(num_classes, backend="resnet34", pretrained=False):
    model = CPNet(num_classes, backend=backend, pretrained=pretrained)
    return model

if __name__ == "__main__":
    model = CPNet(20)
    inputs = torch.randn(1, 3, 640, 640)
    seg_out, context_map = model(inputs)
    print("segout: ", seg_out.size(), ' context_map siz: ', context_map.size())

    # labels = torch.randint(0, 20, (1, 640, 640)).long()
    # model._get_loss(context_map, labels, [80, 80])

    '''model = cpnet_resnet34(4, pretrained=False)
    feat = torch.randn(1, 3, 640, 640)
    out, context_proir_map = model(feat)
    print(out.size(), " context_proir_map size: ", context_proir_map.size())'''