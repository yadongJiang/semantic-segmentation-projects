import torch
import numpy as np
from PIL import Image
import collections
import torchvision.transforms.functional as F
import torch.nn.functional as Fc
import cv2

class ExtToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
    def __call__(self, pic):
        if self.normalize:
            return F.to_tensor(pic)
        else:
            return torch.from_numpy( np.array( pic, dtype=np.float32).transpose(2, 0, 1) )

class ExtNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ExtRateResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    
    def resize(self, img, new_size):
        img = np.array(img).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
        img = Fc.interpolate(img, size=new_size, mode = "bilinear", align_corners=True)
        img = img.squeeze(0).permute(1, 2, 0).data.numpy()
        img = Image.fromarray(np.uint8(img))
        return img
    
    def __call__(self, img):
        w, h = img.size
        if h<w:
            r = self.size / w
            new_h = int(h * r)-1 if int(h * r) % 2 != 0 else int(h * r)
            new_size = (new_h, self.size)
        else:
            r = self.size / h
            new_w = int(w * r)-1 if int(w * r) % 2 != 0 else int(w * r)
            new_size = (self.size, new_w)
        return F.resize(img, new_size, self.interpolation)

class ExtLetterResize(object):
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), stride=32):
        self.new_shape = new_shape # [h, w]
        self.color = color
        self.stride = stride

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.asarray(img)

        shape = img.shape[:2] # [h, w]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)
        
        r = min(self.new_shape[0] / shape[0], self.new_shape[1]/shape[1])
        new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r))) # [w, h]
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)

        # tar_w = new_unpad[0] + dw
        # tar_h = new_unpad[1] + dh
        # img = cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
        
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        return img