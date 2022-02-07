import torch
import torch.nn as nn
from .utils import DownsamplingBottleneck, InitialBlock, RegularBottleneck, UpsamplingBottleneck
import thop

class ENet(nn.Module):
    def __init__(self, num_classes):
        super(ENet, self).__init__()

        self.initial_stem = InitialBlock(3, 16)

        # Stage1 Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01)

        # Stage2 Encoder
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,)

        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1)
        
        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), 
            nn.Conv2d(16, 16, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.1),
            nn.Conv2d(16, num_classes, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        input_size = x.size()
        x = self.initial_stem(x) # 1/2

        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x) # 1/4
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x) # 1/8
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x) # 1/8
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size) # 1/4
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size) # 1/2
        x = self.regular5_1(x)
        
        x = self.head(x)
        return x

from utils.utils import get_model_infos
@get_model_infos
def enet(num_classes, **kwargs):
    model = ENet(num_classes)
    return model

if __name__ == "__main__":
    model =ENet(4)
    inputs = torch.randn(1, 3, 640, 640)
    x = model(inputs)
    print(x.size())