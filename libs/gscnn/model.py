import torch
import torch.nn as nn
from .backbone import *
import torch.nn.functional as F
from .utils import GateSpatialConv2d, GSCNNASPP
import numpy as np
import cv2

class GSCNN(nn.Module):
    def __init__(self, num_classes, backend="resnet34", pretrained=True):
        super(GSCNN, self).__init__()
        self.backbone = eval(backend)(pretrained=pretrained)

        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)

        self.res1 = BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1, 1, 0, bias=True)

        self.res2 = BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1, 1, 0, bias=True)

        self.res3 = BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1, 1, 0, bias=True)

        self.gate1 = GateSpatialConv2d(32, 32)
        self.gate2 = GateSpatialConv2d(16, 16)
        self.gate3 = GateSpatialConv2d(8, 8)

        self.fuse = nn.Conv2d(8, 1, 1, 1, 0, bias=True)
        self.cw = nn.Conv2d(2, 1, 3, 1, 1, bias=True)

        self.aspp = GSCNNASPP(512, 256, output_stride=16)
        self.low_process = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256+64, 256, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.1), 
            nn.Conv2d(256, num_classes, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        x_size = x.size()[2:]
        # [bs, 64, 160, 160], [bs, 128, 80, 80], 
        # [bs, 256, 40, 40], [bs, 512, 40, 40]
        x_1, x_2, x_3, x_4 = self.backbone(x) 
        
        cs = self.res1(x_1) # 与x_1大小相同
        cs = F.interpolate(cs, size=x_size, mode="bilinear", align_corners=True)
        cs = self.d1(cs)
        s2 = self.dsn2(x_2)
        s2 = F.interpolate(s2, size=x_size, mode="bilinear", align_corners=True)
        cs = self.gate1(cs, s2) # [bs, 32, 640, 640]

        cs = self.res2(cs)
        cs = F.interpolate(cs, size=x_size, mode="bilinear", align_corners=True)
        cs = self.d2(cs)
        s3 = self.dsn3(x_3)
        s3 = F.interpolate(s3, size=x_size, mode="bilinear", align_corners=True)
        cs = self.gate2(cs, s3)

        cs = self.res3(cs)
        cs = F.interpolate(cs, size=x_size, mode="bilinear", align_corners=True)
        cs = self.d3(cs)
        s4 = self.dsn4(x_4)
        s4 = F.interpolate(s4, size=x_size, mode="bilinear", align_corners=True)
        cs = self.gate3(cs, s4)
        
        cs = self.fuse(cs)
        edge_out = torch.sigmoid(cs) # out of edge

        im_arr = x.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.uint8)
        canny = np.zeros((x.size()[0], 1, x_size[0], x_size[1])).astype(np.uint8)
        for i in range(x.size()[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).to(x.device)

        act = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(act)
        acts = torch.sigmoid(acts)

        x = self.aspp(x_4, acts)
        x = F.interpolate(x, size=x_1.size()[2:], mode="bilinear", align_corners=True)
        x_1 = self.low_process(x_1)
        x = torch.cat([x, x_1], dim=1)
        
        seg_out = self.classifier(x)
        seg_out = F.interpolate(seg_out, size=x_size, mode="bilinear", align_corners=True) # out of segmentation

        if self.training:
            return seg_out, cs # edge_out
        return seg_out  ## , edge_out

from utils.utils import get_model_infos
@get_model_infos
def gscnn(num_classes, backend="resnet34", pretrained=False):
    model = GSCNN(num_classes=num_classes, backend=backend, pretrained=pretrained)
    return model

if __name__ == "__main__":
    model = GSCNN(4)
    model.train()
    x = torch.randn(2, 3, 640, 640)
    seg_out, edge_out = model(x)
    print("seg_out size: ", seg_out.size(), " edge_out size: ", edge_out.size())