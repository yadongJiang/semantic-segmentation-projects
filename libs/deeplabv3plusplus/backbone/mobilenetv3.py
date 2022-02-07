from numpy import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            Hsigmoid(inplace=True)
        )
    
    def forward(self, x):
        bs, c, h, w = x.size()
        y = self.avg_pool(x).view(bs, c)
        y = self.fc(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl="RE", dilation_rate=1):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel // 2) if dilation_rate==1 else dilation_rate
        self.use_res_connect = (inp==oup and stride==1)

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == "RE":
            nlin_layer = nn.ReLU
        elif nl == "HS":
            nlin_layer = Hswish
        else:
            raise NotImplementedError

        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # point-wise
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),

            # depth-wise
            conv_layer(exp, exp, kernel, stride, padding, dilation_rate, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),

            # point_wise
            conv_layer(exp, oup, 1, 1, 0,  bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, input_size=224, output_stride=16, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                #k, exp, c,   se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [3, 72,  40,  True,  'RE', 2], # 5
                [3, 120, 40,  True,  'RE', 1], # 5
                [3, 120, 40,  True,  'RE', 1], # 5
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [3, 672, 160, True,  'HS', 2], # 5
                [3, 960, 160, True,  'HS', 1], # 5
                [3, 960, 160, True,  'HS', 1], # 5
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [3, 96,  40,  True,  'HS', 2], # 5
                [3, 240, 40,  True,  'HS', 1],
                [3, 240, 40,  True,  'HS', 1],
                [3, 120, 48,  True,  'HS', 1],
                [3, 144, 48,  True,  'HS', 1],
                [3, 288, 96,  True,  'HS', 2],
                [3, 576, 96,  True,  'HS', 1],
                [3, 576, 96,  True,  'HS', 1], # 5
            ]
        else:
            raise NotImplementedError


        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel # 1280
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]

        current_stride *= 2
        dilation = 1
        # previous_dilation = 1

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            # previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s

            self.features.append(MobileBottleneck(input_channel, output_channel, k, stride, exp_channel, se, nl, dilation))
            input_channel = output_channel

        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def mobilenet_v3(**kwargs):
    model = MobileNetV3(**kwargs)
    return model

if __name__ == "__main__":
    backbone = mobilenet_v3(input_size=640, output_stride=32, mode='large')
    # print(backbone.features)
    x = torch.randn(1, 3, 640, 640)
    for i, module in enumerate(backbone.features):
        x = module(x)
        print(i, '==>', x.size())

    """backbone.low_level_features = backbone.features[0:5]
    backbone.high_level_features = backbone.features[5:]
    
    x = torch.randn(1, 3, 640, 640)

    low_feature = backbone.low_level_features(x)
    print("low_feature size: ", low_feature.size())

    high_feature = backbone.high_level_features(low_feature)
    print("high_feature size: ", high_feature.size())"""