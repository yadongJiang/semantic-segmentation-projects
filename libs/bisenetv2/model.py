import torch
import torch.nn as nn
from .utils import *
import copy

class BiSeNetV2(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNetV2, self).__init__()
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGABlock(128)
        '''self.aggreate = Aggreation()'''
        self.head = SegmentHead(128, 512, num_classes, up_factor=8)
        '''self.head = SegmentHead(num_classes, in_channs=128, scale_factor=8)'''

        if self.training:
            self.aux2 = SegmentHead(16, 128, num_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, num_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 256, num_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 256, num_classes, up_factor=32)
        '''self.head8 = SegmentHead(num_classes, in_channs=128, scale_factor=8)
        self.head16 = SegmentHead(num_classes, in_channs=64, scale_factor=16)'''

    def forward(self, x):
        detail = self.detail(x) # [bs, 128, 80, 80]
        feat2, feat3, feat4, feat5_4, feat5_5 = self.segment(x) # feat5_5 : [bs, 128, 20, 20]
        feat_head = self.bga(detail, feat5_5) # [bs, 128, 80, 80]
        out = self.head(feat_head)
        '''out, feat_8, feat_16 = self.aggreate(feat5_5, feat5_4, feat4, detail)
        out = self.head(out)'''

        if self.training:
            out_aux2 = self.aux2(feat2)
            out_aux3 = self.aux3(feat3)
            out_aux4 = self.aux4(feat4)
            out_aux5_4 = self.aux5_4(feat5_4)
            return out, out_aux2, out_aux3, out_aux4, out_aux5_4
        '''if self.training:
            out_8 = self.head8(feat_8)
            out_16 = self.head16(feat_16)
            return out, out_8, out_16'''
        return out

from utils.utils import get_model_infos
@get_model_infos
def bisenetv2(num_classes, **kwargs):
    model = BiSeNetV2(num_classes)
    return model

def model2onnx(model, device, onnx_snap_path):
    '''
    torch模型转onnx模型
    '''
    dummy_inputs = torch.randn(1, 3, 640, 640).to(device)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_inputs, onnx_snap_path + 'bisenetv2.onnx', 
                      verbose=True, input_names=input_names, do_constant_folding=True, 
                      output_names=output_names, opset_version=11, 
                      dynamic_axes={'input':{2:'height', 3:'width'}, 
                                    'output':{2:'height', 3:'width'}})
    return onnx_snap_path + 'bisenetv2.onnx'

if __name__ == "__main__":
    model = BiSeNetV2(4)
    model.train()
    inputs = torch.randn(2, 3, 640, 640)
    out = model(inputs)
    for o in out:
        print(o.size())