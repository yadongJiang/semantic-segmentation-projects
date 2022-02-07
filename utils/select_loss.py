from .loss import JointEdgeSegLoss, ICNetLoss, CPNetLoss, \
                       BiSeNetV2Loss, GSCNNLoss, \
                       DecoupleSegNetLoss, MainAuxLoss
NUM_CLASSES = 21

criterions = {
    'deeplabv3_mobilenetv3' : JointEdgeSegLoss, # (classes=NUM_CLASSES, upper_bound=1.0), 
    'deeplabv3_ghostnet' : JointEdgeSegLoss, #(classes=NUM_CLASSES, upper_bound=1.0), 
    'deeplabv3_resnet50' : JointEdgeSegLoss, #(classes=NUM_CLASSES, upper_bound=1.0), 
    'enet' : JointEdgeSegLoss, #(classes=NUM_CLASSES, upper_bound=1.0), 
    'espnet' : JointEdgeSegLoss, #(classes=NUM_CLASSES, upper_bound=1.0), 
    'icnet' : ICNetLoss, #(aux_weight=0.4), 
    'pspnet_res34' : MainAuxLoss, #(num_classes=NUM_CLASSES, aux_weight=0.4), 
    'pspnet_res50' : MainAuxLoss, #(num_classes=NUM_CLASSES, aux_weight=0.4), 
    'cpnet' : CPNetLoss, #(num_classes=NUM_CLASSES), 
    'bisenetv1' : MainAuxLoss, #(num_classes=NUM_CLASSES, aux_weight=[0.4, 0.4]), 
    'bisenetv2' : MainAuxLoss, #(num_classes=NUM_CLASSES, aux_weight=[0.1, 0.2, 0.3, 0.4]), 
    'ocrnet' : MainAuxLoss, #(num_classes=NUM_CLASSES, aux_weight=0.4), 
    'gscnn' : GSCNNLoss, #(num_classes=NUM_CLASSES), 
    'decoupsegnet' : DecoupleSegNetLoss, #(num_classes=NUM_CLASSES), 
    'sfsegnet' : MainAuxLoss, #(classes=NUM_CLASSES, upper_bound=1.0), 
    'danet' : MainAuxLoss, 
}

def get_criterion(arch):
    criterion = criterions[arch]
    if 'deeplabv3' in arch:
        return criterion(num_classes=NUM_CLASSES, upper_bound=1.0)
    if arch == 'enet':
        return criterion(num_classes=NUM_CLASSES, upper_bound=1.0)
    if arch == 'espnet':
        return criterion(num_classes=NUM_CLASSES, upper_bound=1.0)
    if arch == 'icnet':
        return criterion(aux_weight=0.4)
    if 'pspnet' in arch:
        return criterion(num_classes=NUM_CLASSES, aux_weight=0.4)
    if arch == 'cpnet':
        return criterion(num_classes=NUM_CLASSES)
    if arch == 'bisenetv1':
        return criterion(num_classes=NUM_CLASSES, aux_weight=[0.4, 0.4])
    if arch == 'bisenetv2':
        return criterion(num_classes=NUM_CLASSES, aux_weight=[0.1, 0.2, 0.3, 0.4], loss_type="entropy")
    if arch == 'ocrnet':
        return criterion(num_classes=NUM_CLASSES, aux_weight=0.4)
    if arch == 'gscnn':
        return criterion(num_classes=NUM_CLASSES)
    if arch == 'decoupsegnet':
        return criterion(num_classes=NUM_CLASSES)
    if arch == 'sfsegnet':
        return criterion(num_classes=NUM_CLASSES, aux_weight=[0.4, 0.4, 0.4, 0.4])
    if arch == "danet":
        return criterion(num_classes=NUM_CLASSES, aux_weight=0.4)