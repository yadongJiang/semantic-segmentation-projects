import torch
import torch.nn as nn
from .backbone import *
from .utils import *
import copy
import torch.nn.functional as F

class SFSegNet(nn.Module):
    def __init__(self, num_classes, backend="resnet18", pretrained=False):
        super(SFSegNet, self).__init__()
        self.backbone = eval(backend)(pretrained=pretrained)

        self.ppm = PPM(512, 128)
        
        self.fam3 = FAM(256, 128)
        self.fam2 = FAM(128, 128)
        self.fam1 = FAM(64, 128)

        self.seghead = SegHead(num_classes, 512, 256)
        self.auxs = nn.ModuleList()
        for i in range(4):
            self.auxs.append(SegHead(num_classes, 128, 256))

    def forward(self, x):
        s1, s2, s3, s4 = self.backbone(x)
        feat4 = self.ppm(s4)
        feat3 = self.fam3(s3, feat4)
        feat2 = self.fam2(s2, feat3)
        feat1 = self.fam1(s1, feat2)
        
        outs = [feat4, feat3, feat2, feat1]
        size = feat1.size()[2:]
        for i, out in enumerate(outs):
            outs[i] = F.interpolate(out, size=size, mode="bilinear", align_corners=True)
        feat = torch.cat(outs, dim=1)

        segout = self.seghead(feat)
        if self.training:
            auxes = []
            for i in range(len(outs)):
                auxes.append(self.auxs[i](outs[i]))
            return [segout] + auxes # segout, *aux

        return segout

from utils.utils import get_model_infos
@get_model_infos
def sfsegnet(num_classes, backend="resnet18", pretrained=False):
    model = SFSegNet(num_classes, backend=backend, pretrained=pretrained)
    return model

if __name__ == "__main__":
    model = sfsegnet(4) # SFSegNet(4)
    model.eval()
    inputs = torch.randn(1, 3, 640, 640)
    outputs = model(inputs)
    for out in outputs:
        print(out.size())