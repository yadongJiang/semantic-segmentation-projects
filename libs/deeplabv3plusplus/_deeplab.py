import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import _SimpleSegmentationModel
from .coordConv import CoordConv

class DeepLabV3(_SimpleSegmentationModel):
    pass

class DeepLabHeadV3PlusResNet50(nn.Module):
    def __init__(self, in_channels, low_level_channels, middle_level_planes, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3PlusResNet50, self).__init__()
        self.project = CoordConv(low_level_channels, 48, 1)

        self.middle_project = CoordConv(middle_level_planes, 48, 1)
        
        self.middle_process = CoordConv(304, 256, 3)

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            CoordConv(304, 256, 3),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] ) 

        middle_level_feature = self.middle_project(feature['middle_level'])

        output_feature = self.aspp(feature['out']) 

        output_feature = F.interpolate(output_feature, size=middle_level_feature.shape[2:], mode='bilinear', align_corners=True)
        output_feature = torch.cat([output_feature, middle_level_feature], dim=1)
        output_feature = self.middle_process(output_feature)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=True)

        output = self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) ) 
        return output
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[8, 16, 18]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=True)
        output_feature = torch.cat([output_feature, low_level_feature], dim=1)
        
        output = self.classifier(output_feature)
        return output


class ASPPConv(nn.Sequential):
    def __init__(self, inp, oup, dilation):
        modules = [
            nn.Conv2d(inp, oup, 3, 1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, inp, oup):
        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        ]
        super(ASPPPooling, self).__init__(*modules)

    def forward(self, x):
        size = x.size()[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

class ASPP(nn.Module):
    def __init__(self, inp, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(inp, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        ))

        for rate in atrous_rates:
            modules.append(ASPPConv(inp, out_channels, rate))
        modules.append(ASPPPooling(inp, out_channels))
        
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = None
        for module in self.convs:
            if out is None:
                out = module(x)
            else:
                out = torch.cat([out, module(x)], dim=1)
        out = self.project(out)
        return out
