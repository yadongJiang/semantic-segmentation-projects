import torch
import torch.nn as nn
from .utils import *

class BiSeNetV1(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNetV1, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.head = BiSeNetHead(256, 256, num_classes, scale_factor=8)

        self.head_out8 = BiSeNetHead(128, 64, num_classes, scale_factor=8)
        self.head_out16 = BiSeNetHead(128, 64, num_classes, scale_factor=16)

        self._init_weight()

    def forward(self, x):
        h, w = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.head(feat_fuse) # 1/8
        if not self.training:
            return feat_out
        feat_out8 = self.head_out8(feat_cp8)
        feat_out16 = self.head_out16(feat_cp16)

        if self.training:
            return feat_out, feat_out8, feat_out16
        return feat_out

    def _init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

from utils.utils import get_model_infos
@get_model_infos
def bisenetv1(num_classes, **kwargs):
    model = BiSeNetV1(num_classes)
    return model

def model2onnx(model, device, onnx_snap_path):
    '''
    torch模型转onnx模型
    '''
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, onnx_snap_path + 'bisenetv1.onnx', 
                      verbose=True, input_names=input_names, do_constant_folding=True, 
                      output_names=output_names, opset_version=11, 
                      dynamic_axes={'input':{2:'height', 3:'width'}, 
                                    'output':{2:'height', 3:'width'}})

    return onnx_snap_path + 'bisenetv1.onnx'

if __name__ == "__main__":
    model = BiSeNetV1(4)
    model.eval()
    inputs = torch.randn(1, 3, 640, 640)
    outs = model(inputs)
    for out in outs:
        print(out.size())

    # model = bisenetv1(4)
    # print(model)