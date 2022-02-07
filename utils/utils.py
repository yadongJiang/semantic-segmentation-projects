import torch
import torch.nn as nn
import os
from functools import wraps
import copy
import thop
import logging

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_model_infos(func):
    @wraps(func)
    def wraper(*args, **kwargs):
        model = func(*args, **kwargs)

        model_tmp = copy.deepcopy(model)
        model_tmp.eval()
        inputs = torch.randn(1, 3, 640, 640)
        logging.info("\nCalculate model size...")
        flops, params = thop.profile(model_tmp, inputs=(inputs, ))
        
        s1 = "*************************************************"
        s2 = '*' + ' '*(len(s1)-2) + '*'
        s30 = '*' + ' '*8
        s31 = 'Flops= %.2fG     Params= %.2fM' % (flops/1e9, params/1e6)
        s32 = ' '*(len(s1) - len(s30) - len(s31)-1) + '*'

        print(s1)
        print(s2)
        print(s30+s31+s32)
        print(s2)
        print(s1)

        del model_tmp, inputs
        return model
    return wraper