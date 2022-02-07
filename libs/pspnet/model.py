import torch
import torch.nn as nn
from .pspnet import PSPNet

from utils.utils import get_model_infos

@get_model_infos
def pspnet_resnet34(num_classes):
    model = PSPNet(num_classes=num_classes,  
                   psp_kernel=(1, 2, 3, 6), 
                   psp_in_channels=512, 
                   deep_features_size=256, 
                   backend='resnet34')
    return model

@get_model_infos
def pspnet_resnet50(num_classes):
    model = PSPNet(num_classes=num_classes, 
                   psp_kernel=(1, 2, 3, 6), 
                   psp_in_channels=2048, 
                   deep_features_size=1024, 
                   backend='resnet50')
    return model

def pspnet(num_classes, backend="resnet34"):
    if backend=="resnet34":
        model = pspnet_resnet34(num_classes)
    elif backend=="resnet50":
        model = pspnet_resnet50(num_classes)
    else:
        raise Exception("backend type is Unknow!!!")

    return model

if __name__ == "__main__":
    model = pspnet(4)
    inputs = torch.randn(1, 3, 640, 640)
    model(inputs)