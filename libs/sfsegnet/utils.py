import torch
import torch.nn as nn
import torch.nn.functional as F

class FAM(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(FAM, self).__init__()
        pre_low_channels = 128
        mid_channels = 64
        self.pre_conv = nn.Sequential(
            nn.Conv2d(low_channels, pre_low_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(pre_low_channels), 
            nn.ReLU(inplace=True) 
        )
        self.convl = nn.Conv2d(pre_low_channels, mid_channels, 1, 1, 0)
        self.convh = nn.Conv2d(high_channels, mid_channels, 1, 1, 0)
        self.conv_flow = nn.Conv2d(mid_channels*2, 2, 3, 1, 1, bias=False)
        self.conv_out = nn.Sequential(
            nn.Conv2d(pre_low_channels, pre_low_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(pre_low_channels), 
            nn.ReLU(inplace=True)
        )
    
    def _flow_wrap(self, inputs, flow):
        bs, _, h, w = flow.size()
        norm = torch.tensor([w, h]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        norm = norm.type_as(inputs).to(inputs.device)

        h_grid = torch.linspace(-1., 1., h).view(h, 1).repeat(1, w)
        w_grid = torch.linspace(-1., 1., w).view(1, w).repeat(h, 1)
        grid = torch.stack([w_grid, h_grid], dim=2) # [h, w, 2], 出错了
        grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).type_as(inputs).to(inputs.device) # [bs, h, w, 2]
        grid = grid + flow.permute(0, 2, 3, 1)/norm
        return F.grid_sample(inputs, grid)

    def forward(self, low_feats, high_feats):
        low_feats = self.pre_conv(low_feats)
        featl = self.convl(low_feats)
        feath = self.convh(high_feats)
        feath = F.interpolate(feath, size=featl.size()[2:], mode="bilinear", align_corners=True)

        feat = torch.cat([featl, feath], dim=1)
        flow = self.conv_flow(feat) # [bs, 2, h, w]
        swap_feat = self._flow_wrap(high_feats, flow)
        out = low_feats + swap_feat
        return self.conv_out(out)

class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.ppm = nn.ModuleList([self._stage(in_channels, out_channels, s) for s in sizes])

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(sizes) + in_channels, out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def _stage(self, in_channels, out_channels, s):
        pool = nn.AdaptiveAvgPool2d(output_size=(s, s))
        conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        return nn.Sequential(pool, conv, bn)
    
    def forward(self, x):
        feats = []
        for stage in self.ppm:
            t = stage(x)
            t = F.interpolate(t, size=x.size()[2:], mode="bilinear", align_corners=True)
            feats.append(t)
        feats.append(x)
        feats = torch.cat(feats, dim=1)

        return self.project(feats)

class SegHead(nn.Module):
    def __init__(self, num_classes, in_channels, mid_channels):
        super(SegHead, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(mid_channels, num_classes, 1, 1, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.project(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(h*4, w*4), mode="bilinear", align_corners=True)
        return x

if __name__ == "__main__":
    ppm = PPM(10, 20)
    inputs = torch.randn(1, 10, 20, 20)
    outputs = ppm(inputs)
    print(outputs.size())

    fam = FAM(256, 128)
    low = torch.randn(1, 256, 40, 40)
    high = torch.randn(1, 128, 20, 20)
    out = fam(low, high)
    print(out.size())