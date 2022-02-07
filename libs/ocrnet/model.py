import torch
import torch.nn as nn
from .backbone import *
import torch.nn.functional as F

def ConvBNReLU(in_chans, out_chans, ks=3, st=1, p=1, g=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size=ks, stride=st, padding=p, groups=g, bias=bias), 
        nn.BatchNorm2d(out_chans), 
        nn.ReLU(inplace=True)
    )

class SpatialGatherModule(nn.Module):
    """
    获得每一个类别的特征表达
    """
    def __init__(self):
        super(SpatialGatherModule, self).__init__()
        
    def forward(self, feats, probs):
        bs, c, h, w = feats.size()
        feats = feats.view((bs, c, -1)).permute(0, 2, 1).contiguous() # [bs, hw, c]
        probs = probs.view((bs, probs.size()[1], h*w)).contiguous() # [bs, num_classes, hw]
        probs = F.softmax(probs, dim=2)
        context = torch.matmul(probs, feats) # [bs, num_classes, c]
        context = context.permute(0, 2, 1) # [bs, c, num_classes]
        return context.unsqueeze(3) # [bs, b, num_classes, 1]

class ObjectAttentionBlock2D(nn.Module):
    """
    更新每一个位置的特征
    """
    def __init__(self, input_channels, key_channels):
        super(ObjectAttentionBlock2D, self).__init__()
        self.key_channels = key_channels
        self.f_pixel = nn.Sequential(
            ConvBNReLU(input_channels, key_channels, 1, 1, 0), 
            ConvBNReLU(key_channels, key_channels, 1, 1, 0)
        )
        self.f_object = nn.Sequential(
            ConvBNReLU(input_channels, key_channels, 1, 1, 0), 
            ConvBNReLU(key_channels, key_channels, 1, 1, 0)
        )

        self.f_down = ConvBNReLU(input_channels, key_channels, 1, 1, 0)
        self.f_up = ConvBNReLU(key_channels, input_channels, 1, 1, 0)
    
    def forward(self, x, proxy):
        bs, c, h, w = x.size()
        query = self.f_pixel(x).view(bs, self.key_channels, -1) # Q
        query = query.permute(0, 2, 1) # [bs, hw, key_channels]

        key = self.f_object(proxy).view(bs, self.key_channels, -1) # K, [bs, key_channels, num_classes]

        value = self.f_down(proxy).view(bs, self.key_channels, -1).permute(0, 2, 1).contiguous() # V, [bs, num_classes, key_channels]

        # 计算像素与各类之间的特征相似度
        smi_map = torch.matmul(query, key) # [bs, hw, num_classes]
        smi_map = (self.key_channels ** -0.5) * smi_map
        smi_map = F.softmax(smi_map, dim=2)

        # 根据像素与各类之间的相似度，更新每一个像素的特征，使特征更具有全局性
        context = torch.matmul(smi_map, value) # [bs, hw, key_channels]
        context = context.view((bs, h, w, self.key_channels)).permute(0, 3, 1, 2).contiguous() # [bs, key_channels, h, w]
        context = self.f_up(context)
        return context

class SpatialOCRModule(nn.Module):
    """
    利用各个类的特征更新每一个位置的特征
    """
    def __init__(self, input_channels, key_channels, output_channels):
        super(SpatialOCRModule, self).__init__()
        self.object_attention_block = ObjectAttentionBlock2D(input_channels, key_channels)
        _in_channels = 2 * input_channels
        self.conv_bn_drop = nn.Sequential(
            ConvBNReLU(_in_channels, output_channels, 1, 1, 0), 
            nn.Dropout(0.05)
        )
        
    def forward(self, feats, proxy_feats):
        # 利用各类的特征更新每个像素点的特征，使特征更具全局性
        context = self.object_attention_block(feats, proxy_feats)
        output = self.conv_bn_drop(torch.cat([feats, context], dim=1))
        return output

class OCRModule(nn.Module):
    def __init__(self, num_classes, input_channels=None):
        super(OCRModule, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.dsn_head = nn.Sequential(
            ConvBNReLU(input_channels[0], 512, 3, 1, 1),
            nn.Dropout(0.05), 
            nn.Conv2d(512, num_classes, 1, 1, 0, bias=True)
        )
        self.conv3x3 = ConvBNReLU(input_channels[1], 512, 3, 1, 1)

        # 获得各个类的特征表达
        self.spatial_context_head = SpatialGatherModule()
        # 利用各个类的特征更新每一个位置的特征
        self.spatial_ocr_head = SpatialOCRModule(512, 256, 512)
        self.head = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1), 
            nn.Dropout(0.1), 
            nn.Conv2d(256, num_classes, 1, 1, 0, bias=True)
        )

    def forward(self, input_features):
        input_feat3 = input_features[0]
        input_feat4 = input_features[1]

        assert input_feat3.size()[1] == self.input_channels[0] and \
               input_feat4.size()[1] == self.input_channels[1], "please check out the input_chanenls!"
            
        x_dsn = self.dsn_head(input_feat3)
        x = self.conv3x3(input_feat4)

        # 获得每一个类别的物体区域的特征表达
        context = self.spatial_context_head(x, x_dsn)
        # 更新每一个位置的特征
        x = self.spatial_ocr_head(x, context)

        x = self.head(x)
        return x, x_dsn # , context

class OCRNet(nn.Module):
    def __init__(self, num_classes, input_channels=(1024, 2048), backend="resnet50", pretrained=False):
        super(OCRNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        # backbone
        self.backbone = eval(backend)(pretrained=pretrained)
        # ocr module
        self.ocr_module = OCRModule(num_classes, input_channels=input_channels)

    def forward(self, x):
        bs, _, h, w = x.size()
        out4, out3, _, _ = self.backbone(x)

        x, x_dsn = self.ocr_module((out3, out4))
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        x_dsn = F.interpolate(x_dsn, size=(h, w), mode="bilinear", align_corners=True)
        if self.training:
            return x, x_dsn
        return x

from utils.utils import get_model_infos
@get_model_infos
def ocrnet(num_classes, backend="resnet50", pretrained=True):
    model = OCRNet(num_classes, backend=backend, pretrained=pretrained)
    return model

if __name__ == "__main__":
    model = ocrnet(4)
    inputs = torch.randn(1, 3, 640, 640)
    x, x_dsn = model(inputs)
    print("x size: ", x.size(), " x_dsn size: ", x_dsn.size())