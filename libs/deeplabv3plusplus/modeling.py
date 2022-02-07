from .backbone import resnet50, mobilenetv3, ghostnet
from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHeadV3Plus, DeepLabV3, DeepLabHeadV3PlusResNet50

def _segm_resnet50(num_classes, output_stride):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
        replace_stride_with_dilation=[False, True, True]
    elif output_stride == 16:
        aspp_dilate = [6, 12, 18]
        replace_stride_with_dilation=[False, False, True]

    backbone = resnet50.resnet50(replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256
    middle_level_planes = 512

    return_layers = {'layer4': 'out', 'layer1': 'low_level', 'layer2': 'middle_level'}
    classifier = DeepLabHeadV3PlusResNet50(inplanes, low_level_planes, middle_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_mobilenetv3(num_classes, output_stride):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    elif output_stride == 16:
        aspp_dilate = [6, 12, 18]
    
    backbone = mobilenetv3.mobilenet_v3(input_size=640, output_stride=output_stride, mode='large')
    backbone.low_level_features = backbone.features[0:7] # ocr 13 
    backbone.high_level_features = backbone.features[7:] # OCR 13
    backbone.features = None

    inplanes = 160
    low_level_planes = 40

    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_ghostnet(num_classes, output_stride):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    elif output_stride == 16:
        aspp_dilate = [6, 12, 18]
    
    backbone = ghostnet.ghostnet()
    backbone.low_level_features = backbone.blocks[:4]
    backbone.high_level_features = backbone.blocks[4:]
    backbone.blocks = None

    inplanes = 160
    low_level_planes = 24

    return_layers = {'high_level_features' : 'out', 'low_level_features' : 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(backbone, num_classes, output_stride):
    if backbone == "resnet50":
        model = _segm_resnet50(num_classes, output_stride)
    elif backbone == "mobilenetv3":
        model = _segm_mobilenetv3(num_classes, output_stride)
    elif backbone == "ghostnet":
        model = _segm_ghostnet(num_classes, output_stride)
    return model

from utils.utils import get_model_infos
@get_model_infos
def deeplabv3_mobilenetv3(num_classes=21, output_stride=16):
    return _load_model("mobilenetv3", num_classes, output_stride=output_stride)

@get_model_infos
def deeplabv3_ghostnet(num_classes=21, output_stride=16):
    return _load_model("ghostnet", num_classes, output_stride=output_stride)

@get_model_infos
def deeplabv3_resnet50(num_classes=21, output_stride=16):
    return _load_model("resnet50", num_classes, output_stride=output_stride)

def deeplabv3plusplus(num_classes=21, backend="resnet50", output_stride=16):
    if backend == "resnet50":
        return deeplabv3_resnet50(num_classes=num_classes, output_stride=output_stride)
    elif backend == "mobilenetv3":
        return deeplabv3_mobilenetv3(num_classes=num_classes, output_stride=output_stride)
    elif backend == "ghostnet":
        return deeplabv3_ghostnet(num_classes=num_classes, output_stride=output_stride)
    else:
        raise ValueError("Unkown backend type!")

if __name__ == "__main__":
    import torch
    # model = deeplabv3_mobilenetv3(num_classes=4)
    # model = deeplabv3_resnet50(num_classes=4)
    model = deeplabv3_ghostnet(num_classes=4)
    model.eval()
    inputs = torch.randn(1, 3, 640, 640)
    outputs = model(inputs)
    print("outputs size: ", outputs.size())