import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import BR, C, CBR, DilatedParllelResidualBlockB, DownSamplerB, InputProjectionA

class ESPNetEncoder(nn.Module):
    def __init__(self, num_classes, p, q):
        super(ESPNetEncoder, self).__init__()
        self.level1 = CBR(3, 16, 3, 2)

        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.b1 = BR(16+3)

        self.level2_0 = DownSamplerB(16+3, 64)
        level2 = nn.ModuleList()
        for i in range(p):
            level2.append(DilatedParllelResidualBlockB(64, 64))
        self.level2 = nn.Sequential(*level2)
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128+3, 128)
        level3 = nn.ModuleList()
        for i in range(q):
            level3.append(DilatedParllelResidualBlockB(128, 128))
        self.level3 = nn.Sequential(*level3)
        self.b3 = BR(256)

        self.classifier = C(256, num_classes, 1, 1)

    def forward(self, input):
        output0 = self.level1(input) # 1/2
        inp1 = self.sample1(input) # 1/2
        inp2 = self.sample2(input) # 1/4

        output0_cat = self.b1(torch.cat([output0, inp1], dim=1)) # level1的输出与输入的1/2下采样融合下， 1/2
        
        output1_0 = self.level2_0(output0_cat) # 下采样+多路并行膨胀卷积， 1/4
        output1 = self.level2(output1_0)
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], dim=1))

        output2_0 = self.level3_0(output1_cat) # 下采样+多路并行膨胀卷积， 1/4
        output2 = self.level3(output2_0)
        output2_cat = self.b3(torch.cat([output2_0, output2], dim=1))

        classifier = self.classifier(output2_cat)
        return output0_cat, output1_cat, classifier


class ESPNet(nn.Module):
    def __init__(self, num_classes=20, p=8, q=2):
        super(ESPNet, self).__init__()
        self.encoder = ESPNetEncoder(num_classes, p, q)

        self.br = BR(num_classes)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.level3_c = C(128+3, num_classes, 1, 1)

        self.combine_l2_l2 = nn.Sequential(
            BR(2*num_classes),
            DilatedParllelResidualBlockB(2*num_classes, num_classes, add=False)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = CBR(19 + num_classes, num_classes, 1, 1)

        self.classifier = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.PReLU(num_classes),
            nn.Dropout(0.1),
            nn.Conv2d(num_classes, num_classes, 1, 1)
        )

    def forward(self, x):
        bs, _, h, w = x.size()
        output0_cat, output1_cat, classifier = self.encoder(x)
        output2_c = self.up1(self.br(classifier))

        output1_c = self.level3_c(output1_cat)
        comb_l2_l3 = self.up2(self.combine_l2_l2(torch.cat([output1_c, output2_c], dim=1)))

        concat_feature = self.conv(torch.cat([comb_l2_l3, output0_cat], dim=1))
        classifier = self.classifier(concat_feature)
        out = F.interpolate(classifier, size=(h, w), mode="bilinear", align_corners=True)
        return out

from utils.utils import get_model_infos

@get_model_infos
def espnet(num_classes, p, q):
    model = ESPNet(num_classes, p, q)
    return model

if __name__ == "__main__":
    model = ESPNet(4, 8, 2)
    inputs = torch.randn(1, 3, 640, 720)
    output = model(inputs)
    # print(model)
    print("output size: ", output.size())

    # espnet(4, 8, 2)