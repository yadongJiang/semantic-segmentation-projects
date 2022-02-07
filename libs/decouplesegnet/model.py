from webbrowser import get
from cv2 import pencilSketch
import torch
import torch.nn as nn
from .backbone import *
from .utils import *
import torch.nn.functional as F

def ConvBNReLU(in_chans, out_chans, ks=3, st=1, p=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_chans, out_chans, ks, st, p, bias=bias), 
        nn.BatchNorm2d(out_chans), 
        nn.ReLU(inplace=True)
    )

class DecoupleSegNet(nn.Module):
    def __init__(self, num_classes, backend="resnet34", pretrained=False):
        super(DecoupleSegNet, self).__init__()

        self.backbone = eval(backend)(pretrained=pretrained)
        # aspp
        self.aspp = ASPP(512)
        # generate body and edge
        self.gen_body_edge = GenerateBodyEdge(256)
        self.fine_process = ConvBNReLU(64, 64, 3, 1, 1) # nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.fine_edge = ConvBNReLU(256+64, 256, 1, 1, 0) # nn.Conv2d(256+64, 256, 1, 1, 0, bias=True)

        # edge out
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(48), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(48, 1, kernel_size=1, stride=1, bias=True)
        )
        # final seg
        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, num_classes, 1, 1, 0, bias=True)
        )

        self.final_body = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, num_classes, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        x4, _, x1 = self.backbone(x) # 1/8, 1/4
        # aspp, [bs, 256, 80, 80]
        aspp = self.aspp(x4)
        # fine, edge, both:[bs, 256, 80, 80]
        body, edge = self.gen_body_edge(aspp)
        # fine edge, [bs, 64, 160, 160]
        fine_feat = self.fine_process(x1)
        edge = F.interpolate(edge, size=fine_feat.size()[2:], mode="bilinear", align_corners=True)
        fine_edge = torch.cat([edge, fine_feat], dim=1)
        # [bs, 256, 160, 160]
        fine_edge = self.fine_edge(fine_edge)
        
        # final edge out, for training edge mask, [bs, 1, 160, 160]
        edge_out = self.edge_out(fine_edge)
        final_edge = F.interpolate(edge_out, size=x.size()[2:], mode="bilinear", align_corners=True)
        '''final_edge = torch.sigmoid(final_edge) # [bs, 1, 640, 640]'''

        # final seg out, [bs, num_classes, 640, 640]
        seg_out = fine_edge + F.interpolate(body, size=fine_edge.size()[2:], mode="bilinear", align_corners=True)
        aspp = F.interpolate(aspp, size=seg_out.size()[2:], mode="bilinear", align_corners=True)
        seg_out = torch.cat([aspp, seg_out], dim=1)
        final_segout = self.final_seg(seg_out)
        final_segout = F.interpolate(final_segout, size=x.size()[2:], mode="bilinear", align_corners=True)

        # final body out, [bs, num_classes, 640, 640]
        final_body = self.final_body(body)
        final_body = F.interpolate(final_body, size=x.size()[2:], mode="bilinear", align_corners=True)
        if self.training:
            return final_segout, final_body, final_edge
        return final_segout

from utils.utils import get_model_infos
@get_model_infos
def decouplesegnet(num_classes, backend="resnet34", pretrained=True):
    model = DecoupleSegNet(num_classes, backend=backend, pretrained=pretrained)
    return model

if __name__ == "__main__":
    model = DecoupleSegNet(4)
    model.train()
    x = torch.randn(2, 3, 640, 640)
    model(x)