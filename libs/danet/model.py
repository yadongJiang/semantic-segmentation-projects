import torch
import torch.nn as nn
from .backbone import *
import torch.nn.functional as F

def ConvBNReLU(in_channs, out_channs, ks=3, st=1, p=1, g=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channs, out_channs, kernel_size=ks, 
                  stride=st, padding=p, groups=g, bias=False), 
        nn.BatchNorm2d(out_channs), 
        nn.ReLU(inplace=True)
    )

class ChannelAttentionModule(nn.Module):
    def __init__(self, input_channels=2048, reduce_channels=512):
        super(ChannelAttentionModule, self).__init__()
        self.input_channels = input_channels
        self.reduce_channels = reduce_channels

        self.reduce_conv = ConvBNReLU(input_channels, reduce_channels, 1, 1, 0)
        self.up_conv = ConvBNReLU(reduce_channels, input_channels, 1, 1, 0)

    def forward(self, feature):
        bs, _, h, w = feature.size()
        feature = self.reduce_conv(feature)

        feat = feature.view((bs, self.reduce_channels, -1)) # [bs, c, hw]
        feat_t = feat.permute(0, 2, 1) # [bs, hw, c]
        smi_map = torch.matmul(feat, feat_t) # [bs, c, c]
        smi_map = F.softmax(smi_map, dim=2)

        feature_update = torch.matmul(smi_map, feat) # [bs, c, hw]
        feature_update = feature_update.view((bs, self.reduce_channels, h, w)) # [bs, c, h, w]
        feature = self.up_conv(feature+feature_update)
        return feature

class PositionAttentionModule(nn.Module):
    def __init__(self, input_channels=2048, reduce_channels=512):
        super(PositionAttentionModule, self).__init__()
        self.input_channels = input_channels
        self.reduce_channels = reduce_channels
        self.reduce_conv = ConvBNReLU(input_channels, reduce_channels, 1, 1, 0)
        
        self.q_conv = ConvBNReLU(reduce_channels, reduce_channels, 1, 1, 0)
        self.k_conv = ConvBNReLU(reduce_channels, reduce_channels, 1, 1, 0)
        self.v_conv = ConvBNReLU(reduce_channels, reduce_channels, 1, 1, 0)

        self.up_conv = ConvBNReLU(reduce_channels, input_channels, 1, 1, 0)

    def forward(self, feature):
        bs, _, h, w = feature.size()
        feature = self.reduce_conv(feature)

        query = self.q_conv(feature).view(bs, self.reduce_channels, -1) # [bs, c, hw]
        key = self.k_conv(feature).view(bs, self.reduce_channels, -1) # [bs, c, hw]
        value = self.v_conv(feature).view(bs, self.reduce_channels, -1) # [bs, c, hw]

        query = query.permute(0, 2, 1).contiguous() # [bs, hw, c]
        smi_map = torch.matmul(query, key) # [bs, hw, hw]
        smi_map = F.softmax(smi_map, dim=2)

        value = value.permute(0, 2, 1).contiguous() # [bs, hw, c]
        feature_update = torch.matmul(smi_map, value) # [bs, hw, c]
        feature_update = feature_update.view((bs, h, w, self.reduce_channels)).permute(0, 3, 1, 2) # [bs, c, h, w]
        feature = self.up_conv(feature+feature_update)
        return feature

class DANet(nn.Module):
    def __init__(self, num_classes, backend="resnet50", pretrained=False):
        super(DANet, self).__init__()
        # backbone
        self.backbone = eval(backend)(pretrained=pretrained)
        # positioin attentioin
        self.position_attention = PositionAttentionModule(2048, 512)
        # channels attention
        self.channel_attention = ChannelAttentionModule(2048, 512)

        self.aggregate = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, groups=2048, bias=False), 
            nn.BatchNorm2d(2048), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1), 
            nn.Dropout(0.1), 
            nn.Conv2d(256, num_classes, 1, 1, bias=True)
        )
        self.aux_head = nn.Sequential(
            ConvBNReLU(1024, 256, 3, 1, 1), 
            nn.Dropout(0.1), 
            nn.Conv2d(256, num_classes, 1, 1, bias=True)
        )

    def forward(self, x):
        bs, c, h, w = x.size()
        feat4, feat3, _, _ = self.backbone(x)
        
        position_feature = self.position_attention(feat4)
        channels_feature = self.channel_attention(feat4)

        features = position_feature + channels_feature
        features = self.aggregate(features)
        seg_out = self.head(features)
        seg_out = F.interpolate(seg_out, size=(h, w), mode="bilinear", align_corners=True)
        
        if not self.training:
            return seg_out
        else:
            aux_out = self.aux_head(feat3)
            aux_out = F.interpolate(aux_out, size=(h, w), mode="bilinear", align_corners=True)
            return seg_out, aux_out

from utils.utils import get_model_infos
@get_model_infos
def danet(num_classes, backend="resnet50", pretrained=False):
    model = DANet(num_classes, backend=backend, pretrained=pretrained)
    return model

if __name__ == "__main__":
    model = DANet(4)
    inputs = torch.randn(1, 3, 640, 640)
    seg_out, aux_out = model(inputs)
    print(seg_out.size(), ' ', aux_out.size())