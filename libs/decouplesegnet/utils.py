import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

class ASPPModule(nn.Sequential):
    def __init__(self, in_channs, out_channs, dilate):
        modules = [
            nn.Conv2d(in_channs, out_channs, 3, 1, padding=dilate, dilation=dilate, bias=False), 
            nn.BatchNorm2d(out_channs), 
            nn.ReLU(inplace=True)
        ]
        super(ASPPModule, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channs, out_channs):
        modules = [
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channs, out_channs, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_channs), 
            nn.ReLU(inplace=True)
        ]
        super(ASPPPooling, self).__init__(*modules)
    
    def forward(self, x):
        bs, _, h, w = x.size()
        x = super(ASPPPooling, self).forward(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channs):
        super(ASPP, self).__init__()
        self.aspp_dilate = [12, 24, 36]
        out_channel = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channs, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel), 
            nn.ReLU(inplace=True)
        ))
        
        for rate in self.aspp_dilate:
            modules.append(ASPPModule(in_channs, out_channel, rate))
        modules.append(ASPPPooling(in_channs, out_channel))

        self.convs = nn.ModuleList(modules)

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channel*5, out_channel, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channel), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outs=  []
        for module in self.convs:
            outs.append(module(x))
        outs = torch.cat(outs, dim=1)
        out = self.out_conv(outs)
        return out

class GenerateBodyEdge(nn.Module):
    def __init__(self, in_channs):
        super(GenerateBodyEdge, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channs, in_channs, kernel_size=3, stride=2, padding=1, groups=in_channs, bias=False), 
            nn.BatchNorm2d(in_channs), 
            nn.ReLU(inplace=True), 

            nn.Conv2d(in_channs, in_channs, kernel_size=3, stride=2, padding=1, groups=in_channs, bias=False), 
            nn.BatchNorm2d(in_channs), 
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channs, in_channs, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channs), 
            nn.ReLU(inplace=True)
        )

        self.flow_maker = nn.Conv2d(in_channs * 2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        size = x.size()[2:]
        x_down = self.downsample(x)
        x_up = F.interpolate(x_down, size=size, mode="bilinear", align_corners=True)
        
        flow = torch.cat([x, x_up], dim=1)
        flow = self.flow_maker(flow) # [bs, 2, h, w]
        body = self.flow_warp(x, flow) # [bs, 256, h, w]
        edge = x - body # [bs, 256, h, w]
        return body, edge

    def flow_warp(self, inputs, flow):
        bs, c, h, w = flow.size()
        norm = torch.tensor([w, h]).unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, 2]
        norm = norm.type_as(inputs).to(inputs.device)
        # new 
        h_grid = torch.linspace(-1., 1., h).view(-1, 1).repeat(1, w) # [h, w]
        w_grid = torch.linspace(-1., 1., w).view(1, -1).repeat(h, 1) # [h, w]
        grid = torch.stack([w_grid, h_grid], dim=2) # [h, w, 2]
        grid = grid.repeat(bs, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1)/norm # 将flow归一化到0~1

        output = F.grid_sample(inputs, grid) # [bs, c, h, w]
        return output