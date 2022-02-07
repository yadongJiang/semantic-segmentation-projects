from tokenize import Number
from typing import Iterable
import torch.nn as nn
import torch.nn.functional as F
import torch 
import logging
import numpy as np
import numbers

__all__ = ['FocalLoss', 'HardMiningLoss', 'OhemCELoss', 'SSIMLoss', 
           'JointEdgeSegLoss', 'IOU', "BiSeNetV2Loss", "DecoupleSegNetLoss", "MainAuxLoss"]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


## IOU Loss
def _iou(pred, target, size_average = True):
    
    b = pred.shape[0] #batch_size
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


## SSIM Loss
from math import exp
from torch.autograd import Variable
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  #[1, 1, window_size, window_size]
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()) #[channel, 1, window_size, window_size]
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel) #[channel, 1, window_size, window_size]

    def forward(self, img1, img2):
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

criterion_ssim = SSIM(window_size=11, size_average=True)
def ssim_loss(preds, targets):
    pred_pre = preds.max(dim=1)[1]
    ssim_l = 1 - criterion_ssim(pred_pre, targets)
    return ssim_l

class HardMiningLoss(object):
    def __init__(self, alpha0=0.6, alpha1=2.0, last_iter=-1):
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.criterion1 = nn.CrossEntropyLoss(reduction="none")
        self.criterion2 = nn.CrossEntropyLoss(reduction="mean", ignore_index=255)

        self.last_iter = last_iter
    
    def set_step(self, cur_iter):
        self.last_iter = cur_iter

    def step(self):
        self.last_iter += 1
    
    def strategy1(self, preds, labels):
        # pred_log = F.softmax(preds, dim = 1)
        ax = preds.argmax(dim=1)  ## pred_log
        idx1 = (labels!=ax)
        idx0 = (labels==ax)

        loss = self.criterion1(preds, labels)
        loss[idx0] = self.alpha0 * loss[idx0]
        loss[idx1] = self.alpha1 * loss[idx1]
        loss = loss.mean()

        return loss
    
    def strategy2(self, preds, labels):
        ax = preds.argmax(dim=1)
        # idx = (labels == ax)
        labels_t = torch.ones_like(labels)* 255
        loss = self.criterion2(preds, torch.where(labels == ax, labels_t, labels))
        return loss
    
    def strategy3(self, preds, labels):
        pred_log = F.softmax(preds, dim=1)
        values, indices = pred_log.max(dim=1)
        # idx = values < 0.8
        labels_t = torch.ones_like(labels)*255
        loss = self.criterion2(preds, torch.where(values<0.8, labels, labels_t))
        
        return loss
    
    def __call__(self, preds, labels):
        return self.strategy1(preds, labels)
        
class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thresh = self.thresh.to(device)
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=self.ignore_lb, reduction="none")

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

class SSIMLoss(object):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    def __call__(self, outputs, label_gt):
        total_loss = 0.0
        total_loss += self.cross_ssim_iou_loss(outputs, label_gt)
        return total_loss
    
    def cross_ssim_iou_loss(self, pred, target):
        cross_loss = self.criterion(pred, target)
        pred_pre = pred.max(dim=1)[1]
        ssim_l = 1 - criterion_ssim(pred_pre, target)

BECLoss = nn.BCEWithLogitsLoss()
def NewLoss(outputs, label, bin_label):
    bin_pred = outputs[:, 0:1, :, :]
    muti_pred = outputs[:, 1:, :, :]

    bin_pred = bin_pred.permute((0, 2, 3, 1)).contiguous().view((-1, 1))
    bin_label = bin_label.unsqueeze(3).view((-1, 1)).float()
    L1 = BECLoss(bin_pred, bin_label)

    muti_pred = muti_pred.permute((0, 2, 3, 1)).contiguous().view((-1, 3))
    label_muti = torch.full( (label.size()[0], label.size()[1], label.size()[2], 3), 0, dtype=torch.long).view((-1, 3))
    label = label.unsqueeze(3).view((-1, 1))

    label_t = label_muti.scatter_(1, label, 1).float()
    L2 = BECLoss(muti_pred, label_t)
    return L1 + L2


class ImageBasedCrossEntropyLoss2d(nn.Module):
    
    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1 ## 每一类的权重，数量越多，则权重越大
        return hist

    def forward(self, inputs, targets):
        targets = targets.long()
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights) ## .cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]): # bs
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights)  ## .cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)), 
                                                targets[i].unsqueeze(0))
        return loss

class JointEdgeSegLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, upper_bound=1.0):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = num_classes
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=num_classes, ignore_index=ignore_index, upper_bound=upper_bound)  ## .cuda()

    def edge_attention(self, inputs, targets, edgemask):
        n, c, h, w = inputs.size()
        filter = torch.ones_like(targets) * 255
        return self.seg_loss(inputs, 
                             torch.where(edgemask.max(1)[0] > 0, targets, filter))

    def forward(self, inputs, targets):
        segmask, edgemask = targets

        losses = {}
        losses['seg_loss'] = self.seg_loss(inputs, segmask)
        losses['att_loss'] = self.edge_attention(inputs, segmask, edgemask)
        loss_ = 0.0
        loss_ += losses['seg_loss']
        loss_ += losses['att_loss']

        return loss_

class ICNetLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(ICNetLoss, self).__init__()
        self.aux_weight = aux_weight if isinstance(aux_weight, numbers.Number) \
                                        else aux_weight[0]
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, target):
        pred, pred_sub4, pred_sub8, pred_sub16 = inputs
        
        if isinstance(target, tuple):
            target = target[0]
        if len(target.size()) == 3:
            target = target.unsqueeze(1)
        
        target = target.float()

        target_sub4 = F.interpolate(target, size=pred_sub4.size()[2:], \
                                    mode="bilinear", align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, size=pred_sub8.size()[2:], \
                                    mode="bilinear", align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, size=pred_sub16.size()[2:], \
                                     mode="bilinear", align_corners=True).squeeze(1).long()

        loss_4 = self.criterion(pred_sub4, target_sub4)
        loss_8 = self.criterion(pred_sub8, target_sub8)
        loss_16 = self.criterion(pred_sub16, target_sub16)
        return loss_4 + loss_8 * self.aux_weight + loss_16 * self.aux_weight


class CPNetLoss(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(CPNetLoss, self).__init__()
        self.seg_criterion = nn.CrossEntropyLoss()
        # self.seg_criterion = JointEdgeSegLoss(classes=num_classes, upper_bound=1.0)
        self.criterion = nn.BCEWithLogitsLoss()
        self.prior_size = (40, 40)

    def _aux_loss(self, context_map, ideal_context_map):
        l1 = -torch.log((context_map * ideal_context_map).sum() / context_map.sum())
        l2 = -torch.log((context_map * ideal_context_map).sum() / ideal_context_map.sum())
        l3 = -torch.log(((1-context_map) * (1-ideal_context_map)).sum()/ (1-ideal_context_map).sum())
        
        return (l1 + l2 + l3).sum() / self.prior_size[0]

    def forward(self, inputs, targets):
        seg_preds, context_map = inputs
        if isinstance(targets, tuple):
            targets = targets[0]
        assert isinstance(self.prior_size, list) or isinstance(self.prior_size, tuple)
        if len(targets.size()) == 3:
            labels_c = targets.unsqueeze(1)

        # seg_loss = self.seg_criterion(seg_preds, targets) # with jointEdgeSegLoss
        seg_loss = self.seg_criterion(seg_preds, targets) # with regular crossentropy loss

        labels_c = F.interpolate(labels_c.float(), size=self.prior_size, \
                               mode="bilinear", align_corners=True).squeeze(1).long()
        labels_c = F.one_hot(labels_c, self.num_classes)
        labels_c = labels_c.view(labels_c.size()[0], -1, self.num_classes).float()
        ideal_context_map = torch.matmul(labels_c, labels_c.permute(0, 2, 1)).float()

        assert context_map.size() == ideal_context_map.size(), \
                "the pred contex_map's size must be equal to ideal_context_map's size"
        loss = self.criterion(context_map, ideal_context_map)
        # aux_loss = self._aux_loss(context_map, ideal_context_map)
        return seg_loss+loss #, aux_loss

class BiSeNetV2Loss(nn.Module):
    def __init__(self, criterion, mid_criterion, aux_alpha=0.4):
        super(BiSeNetV2Loss, self).__init__()
        self.criterion = criterion
        self.mid_criterion = mid_criterion
        self.aux_alpha = aux_alpha

    def forward(self, preds, label_edge):
        assert len(preds) >= 1, "number of preds must be greater 1"
        loss = self.criterion(preds[0], label_edge)
        aux_loss = 0
        aux_loss = 0.1*self.mid_criterion(preds[1], label_edge) + \
                   0.2*self.mid_criterion(preds[2], label_edge) + \
                   0.3*self.mid_criterion(preds[3], label_edge) + \
                   0.4*self.mid_criterion(preds[4], label_edge)
        
        return loss + aux_loss

class GSCNNLoss(JointEdgeSegLoss):
    def __init__(self, num_classes, ignore_index=255, upper_bound=1.0):
        super(GSCNNLoss, self).__init__(num_classes=num_classes, 
                                        ignore_index=ignore_index, 
                                        upper_bound=upper_bound)
        
    def bce2d(self, input, target):
        n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1) # [bs, 1, h, w]==>[bs, h, w, 1]==>[1, bs*h*w*1]
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1) # [bs, 1, h, w]==>[bs, h, w, 1]==>[1, bs*h*w*1]
        target_trans = target_t.clone()

        pos_index = (target_t == 1) # 边缘位置
        neg_index = (target_t == 0) # 非边缘位置
        ignore_index=(target_t > 1) # 忽略位置

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0) # [1, bs*h*w]
        weight = weight.numpy()
        pos_num = pos_index.sum() # 正样本的个数
        neg_num = neg_index.sum() # 负样本的个数
        sum_num = pos_num + neg_num # 有效样本的总数
        weight[pos_index] = neg_num*1.0 / sum_num # 正样本的权重
        weight[neg_index] = pos_num*1.0 / sum_num # 负样本的权重

        weight[ignore_index] = 0 # 将非法样本的权重置为0

        weight = torch.from_numpy(weight)
        weight = weight #.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, inputs, targets, edgemask):
        n, c, h, w = inputs.size()
        edgemask = F.sigmoid(edgemask)
        filter = torch.ones_like(targets) * 255
        return self.seg_loss(inputs, 
                             torch.where(edgemask.max(1)[0] > 0.5, targets, filter))

    def forward(self, inputs, targets):
        segin, edgein = inputs # 最终分割输出，边缘分割输出
        segmask, edgemask = targets # 分割标签，边缘标签

        losses = {}
        losses['seg_loss'] = self.seg_loss(segin, segmask) # 分割损失
        losses['edge_loss'] = 20 * self.bce2d(edgein, edgemask) # 边缘分割损失
        losses['att_loss'] = self.edge_attention(segin, segmask, edgein) # 在属于边缘的像素上计算分割损失
        loss_ = 0.0
        loss_ += losses['seg_loss']
        loss_ += losses['att_loss']
        loss_ += losses['edge_loss']

        return loss_

class DecoupleSegNetLoss(GSCNNLoss):
    def __init__(self, num_classes, ignore_index=255, upper_bound=1.0):
        super(DecoupleSegNetLoss, self).__init__(num_classes=num_classes, 
                                                 ignore_index=ignore_index, 
                                                 upper_bound=upper_bound)

    def body_seg(self, body_inputs, targets, edgein):
        filter = torch.ones_like(targets) * 255
        edgein = F.sigmoid(edgein)
        new_targets = torch.where(edgein.max(1)[0]>0.5, filter, targets)
        return self.seg_loss(body_inputs, new_targets)
    
    def forward(self, inputs, targets):
        #分割,  body,   edge
        segin, bodyin, edgein = inputs
        segmask, edgemask = targets
        
        losses = {}
        losses['seg_loss'] = self.seg_loss(segin, segmask) # 分割损失
        losses["edge_loss"] = 20 * self.bce2d(edgein, edgemask) # 边缘分割损失
        losses['att_loss'] = self.edge_attention(segin, segmask, edgein) # 在边缘像素部分计算分割损失
        
        # 计算body分割损失
        losses["body_loss"] = self.body_seg(bodyin, segmask, edgein)

        loss = losses['seg_loss'] + \
               losses["edge_loss"] + \
               losses['att_loss'] + \
               losses["body_loss"]
        return loss

class MainAuxLoss(nn.Module):
    def __init__(self, num_classes, aux_weight:list, loss_type="jointedge"):
        super(MainAuxLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "entropy":
            self.criterion = nn.CrossEntropyLoss(ignore_index = 255, reduction = 'mean') 
            self.aux_criterion = nn.CrossEntropyLoss(ignore_index = 255, reduction = 'mean') 
        elif loss_type == "jointedge":
            self.criterion = JointEdgeSegLoss(num_classes = num_classes, upper_bound = 1.0)
            self.aux_criterion = JointEdgeSegLoss(num_classes = num_classes, upper_bound = 1.0)
        else:
            raise ValueError("unkowned loss type: {}".format(loss_type))
        self.aux_weight = aux_weight if isinstance(aux_weight, list) else [aux_weight]

    def forward(self, inputs:tuple, targets):
        if not isinstance(inputs, Iterable):
            inputs = [inputs]
        
        loss = self.criterion(inputs[0], targets if 
                    self.loss_type=='jointedge' else targets[0])
        for i, aux in enumerate(inputs[1:]):
            loss += self.aux_weight[i] * self.aux_criterion(aux, targets if 
                    self.loss_type=='jointedge' else targets[0])
        
        return loss