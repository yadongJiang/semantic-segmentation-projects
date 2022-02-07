import torch
import torch.nn as nn
from .resnet import resnet50_v1b

class BaseModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(BaseModel, self).__init__()
        if backbone == "resnet50":
            self.backbone = resnet50_v1b(pretrained=pretrained)

    def base_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        return c1, c2, c3, c4

if __name__ == "__main__":
    model = BaseModel()
    inputs = torch.randn(1, 3, 640, 640)
    c1, c2, c3, c4 = model.base_forward(inputs)
    print('c1 size: ', c1.size())
    print('c2 size: ', c2.size())
    print('c3 size: ', c3.size())
    print('c4 size: ', c4.size())