from utils import ext_transforms as et 
from datasets import VOCSegmentation
import numpy as np
import os
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    print(opts.DATA.DATASET)
    if opts.DATA.DATASET == 'voc' or opts.DATA.DATASET == "shelves":
        train_transform = et.ExtCompose([
            et.ExtResize(size=opts.DATA.INPUT_HEIGHT),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.DATA.INPUT_HEIGHT, opts.DATA.INPUT_WIDTH), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            et.ExtRandomHorizontalFlip(),
            et.PerspectiveTransform(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.TRAIN.CROP_VAL:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.DATA.INPUT_HEIGHT),
                et.ExtCenterCrop(opts.DATA.INPUT_HEIGHT),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.DATA.DATA_ROOT, year=opts.DATA.YEAR, num_classes=opts.MODEL.NUM_CLASS, 
                                    image_set='train', download=opts.DATA.DOWNLOAD, transform=train_transform, edge_radius=opts.TRAIN.EDGE_RADIUS)
        val_dst = VOCSegmentation(root=opts.DATA.DATA_ROOT, year=opts.DATA.YEAR, num_classes=opts.MODEL.NUM_CLASS, 
                                  image_set='val', download=False, transform=val_transform)

    return train_dst, val_dst

def get_params(model):
    backbone_no_ppsa_params = []
    backbone_ppsa_params = []
    for name, param in model.backbone.named_parameters():
        if "ppsa" in name:
            backbone_ppsa_params.append(param)
        else:
            backbone_no_ppsa_params.append(param)
    classifier_params = model.classifier.parameters()

    return classifier_params, backbone_no_ppsa_params, backbone_ppsa_params

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.TRAIN.SAVE_VAL_RESULTS:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    # 网络前向传播的时候不会保存梯度信息，节约显存
    with torch.no_grad():
        for i, (images, labels, _) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy() #[batch_size,513,513]
            targets = labels.cpu().numpy() #[batch_size,513,513]

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.TRAIN.SAVE_VAL_RESULTS:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples