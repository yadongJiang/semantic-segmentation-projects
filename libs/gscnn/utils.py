import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

class GateSpatialConv2d(nn.Module):
    def __init__(self, in_channs, out_channs):
        super(GateSpatialConv2d, self).__init__()
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channs+1), 
            nn.Conv2d(in_channs+1, in_channs+1, 1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channs+1, 1, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(1), 
            nn.Sigmoid()
        )
        self.process = nn.Conv2d(in_channs, out_channs, 1, 1, 0, bias=True)
    
    def forward(self, input_features, gating_features):
        feat = torch.cat([input_features, gating_features], dim=1)
        gate = self._gate_conv(feat)
        input_features = input_features + gate*input_features
        input_features = self.process(input_features)

        return input_features

class GSCNNASPP(nn.Module):
    def __init__(self, in_channs, out_channs, output_stride=16, rates=[6, 12, 18]):
        super(GSCNNASPP, self).__init__()
        if output_stride == 8:
            rates = [r*2 for r in rates]
        elif output_stride == 16:
            rates = rates
        else:
            raise "output stride of {} not supported".format(output_stride)
        
        self.features = nn.ModuleList()
        self.features.append(nn.Sequential(
            nn.Conv2d(in_channs, out_channs, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_channs), 
            nn.ReLU(inplace=True)
        ))
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_channs, out_channs, kernel_size=3, stride=1, padding=r, dilation=r, bias=False), 
                nn.BatchNorm2d(out_channs), 
                nn.ReLU(inplace=True)
            ))
        
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_channs, out_channs, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(out_channs), 
            nn.ReLU(inplace=True)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, out_channs, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(out_channs), 
            nn.ReLU(inplace=True)
        )
        self.post_process = nn.Sequential(
            nn.Conv2d(out_channs*6, out_channs, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_channs), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x, edge):
        x_size = x.size()[2:]

        pool_feat = self.img_pooling(x)
        pool_feat = self.img_conv(pool_feat)
        pool_feat = F.interpolate(pool_feat, size=x_size, mode="bilinear", align_corners=True)

        out = pool_feat
        # print("out size: ", out.size())
        edge =F.interpolate(edge, size=x_size, mode="bilinear", align_corners=True)
        edge_feat = self.edge_conv(edge)
        # print("edge feat size: ", edge_feat.size())
        out = torch.cat([out, edge_feat], dim=1)

        for f in self.features:
            y = f(x)
            out = torch.cat([out, y], dim=1)
        out = self.post_process(out)
        # print("out size: ", out.size())
        return out