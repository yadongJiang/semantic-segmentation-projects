import argparse
import torch
import numpy as np
import os
import libs
from torchvision import transforms
from utils.common import *
import torch.nn.functional as Func
import copy
from functools import cmp_to_key
import time

class Inference(object):
    total_time = 0
    def __init__(self, args):
        self.args = args
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        print("Device: %s" % self.device)

        model_map = {
            'deeplabv3_mobilenetv3' : libs.deeplabv3_mobilenetv3,
            'deeplabv3_ghostnet' : libs.deeplabv3_ghostnet,
            "espnet" : libs.espnet,
            'bisenetv1' : libs.bisenetv1,
            'cpnet' : libs.cpnet_resnet34,
            'gscnn' : libs.gscnn_resnet34,
            'bisenetv2' : libs.bisenetv2, 
            'icnet' : libs.icnet, 
            'pspnet_res34' : libs.pspnet_resnet34
        }
        if "espnet" in args.arch:
            self.model = model_map[args.arch](num_classes=args.num_classes, p=args.p, q=args.q)
        elif "deeplabv3" in args.arch:
            self.model = model_map[args.arch](num_classes=args.num_classes, output_stride=args.output_stride)
        elif "bisenet" in args.arch:
            self.model = model_map[args.arch](num_classes=args.num_classes)
        elif 'cpnet' in args.arch:
            self.model = model_map[args.arch](num_classes=args.num_classes, pretrained=False)
        elif 'gscnn' in self.args.arch:
            self.model = model_map[args.arch](num_classes=args.num_classes, backend="resnet34")
        elif 'icnet' in self.args.arch:
            self.model = model_map[args.arch](num_classes=args.num_classes, pretrained=False)
        elif 'pspnet' in self.args.arch:
            self.model = model_map[args.arch](num_classes=args.num_classes)
        self._load_weight(args.weight)

        self.val_transform = transforms.Compose([
            # ExtRateResize(opts.crop_size),
            ExtLetterResize(opts.crop_size),
            ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        self.input_transform = transforms.Compose([
            # ExtRateResize(opts.crop_size),
            ExtLetterResize(opts.crop_size)
        ])

    def _load_weight(self, weight):
        checkpoints = torch.load(weight, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoints["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def _gap_areas(self, gap_output, input_tensor):
        def compare(contour1, contour2):
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            if area1 > area2:
                return -1
            elif area1 < area2:
                return 1
            else:
                return 0
        contours, _ = cv2.findContours(gap_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 650 * ( (input_tensor.size(2)* \
               input_tensor.size(3)) / (640 * 640)):
                continue
            areas.append(contour)
        
        if len(areas)>=2:
            areas=sorted(areas, key=cmp_to_key(compare))
        
        return areas

    def _getPoints(self, contour, gap):
        contour = contour.reshape(contour.shape[0], -1)
        output = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        k = output[1] / output[0]
        # print("k:", k)
        point1 = []
        point2 = []
        if abs(k) > 200:
            point1.append(output[2][0])
            point1.append(0)
            point2.append(output[2][0])
            point2.append(gap.shape[0])
        else:
            point1.append(0)
            point1.append(k*(0-output[2])+output[3])
            point2.append(gap.shape[1])
            point2.append(k*(gap.shape[1]-output[2])+output[3])
        return point1, point2, output
    
    def _gap_points(self, areas, gap_output):
        points = []
        mid_points= []
        for contour in areas:
            if len(points)>=2:
                break
            point1, point2, output = self._getPoints(contour, gap_output)
            if len(points) > 0:
                mid_point=mid_points[-1]
                # print("mid_point : ", mid_point)
                if abs(output[2]-mid_point[0]) < 100:
                    continue
            points.append([point1, point2])
            mid_points.append([output[2], output[3]])
        
        return points

    def __call__(self, image):
        image = image.convert("RGB")

        img_resize = self.input_transform(image)
        img_resize = np.asarray(img_resize)

        input_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        print("input_tensor size: ", input_tensor.size())

        start = time.time()
        
        if "cpnet" in self.args.arch:
            self.model._reinit(np.array(input_tensor.size()[2:]))

        outputs = self.model(input_tensor) # , context || 
        cost = time.time() - start
        self.total_time += cost
        print("cost time: ", cost)
        
        if "gscnn" in self.args.arch:
            edge_mask = outputs[1]
            outputs = outputs[0]
            edge_mask = (edge_mask.gt(0.9).long()) * 255
            edge_mask = edge_mask[0]
            edge_mask = edge_mask.permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
            cv2.imshow("edge_mask", edge_mask)

        ## debug
        deb_output = outputs.squeeze(dim=0)
        deb_output = Func.softmax(deb_output, dim=0)
        deb_output = deb_output.data.cpu().numpy()

        # layers
        layer_output = deb_output[2, :, :]
        indx = layer_output > 0.6 # 0.3
        layer_output[:, :] = 0
        layer_output[indx] = 255
        layer_output = layer_output.astype(np.uint8)
        layer_output = layer_output[:, :, None]

        # gaps
        gap_output = deb_output[1, :, :]
        indx0 = gap_output > 0.3
        gap_output[:, :] = 0
        gap_output[indx0] = 255
        gap_output = gap_output.astype(np.uint8)
        gap_output_tmp = copy.deepcopy(gap_output)
        cv2.imshow("gap_output_tmp", gap_output_tmp)
        gap_output = gap_output[:, :, None]

        # top
        top_output = deb_output[3, :, :]
        indx3 = top_output > 0.5
        top_output[:, :] = 0
        top_output[indx3] = 255
        top_output = top_output.astype(np.uint8)
        top_output = top_output[:, :, None]

        areas = self._gap_areas(gap_output, input_tensor)

        points = self._gap_points(areas, gap_output)

        ## 可视化
        B = np.zeros((layer_output.shape[0], layer_output.shape[1], 1), dtype=np.uint8)  # opts.crop_size, opts.crop_size
        G = np.zeros((layer_output.shape[0], layer_output.shape[1], 1), dtype=np.uint8)  # opts.crop_size, opts.crop_size
        mask = np.concatenate((B, top_output, layer_output), axis=2) # G || top_output

        ind0 = (layer_output==255)
        ind1 = layer_output!=255
        ind0 = np.tile(ind0, (1, 1, 3))
        ind1 = np.tile(ind1, (1, 1, 3))

        img = np.zeros(img_resize.shape, dtype=np.uint8)
        img[ind0] = img_resize[ind0] * 0.5 + mask[ind0] * 0.5
        img[ind1] = img_resize[ind1]

        img = img.astype(np.uint8)

        for point_list in points:
            cv2.line(img, tuple(point_list[0]), tuple(point_list[1]), (0,255, 0), 3)

        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        cv2.imshow("gap_output_tmp", gap_output_tmp)
        key = cv2.waitKey()
        if key == 113:
            exit()

def get_argparser():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--arch", type=str, default="espnet", \
                        choices=["deeplabv3_mobilenetv3", "deeplabv3_ghostnet", 
                                 "espnet" ,"bisenetv1", "cpnet", "gscnn", "bisenetv2", 
                                 "icnet", "pspnet_res34"], help="model type")
    parser.add_argument("--weight", type=str, default= \
                        "C:/Jyd/project2/segmentation/checkpoints/best_espnet_shelves_os16.pth", \
                        help="model's path")
    parser.add_argument("--num_classes", type=int, default=4, help="num classes")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--imgs_root", type=str, default="C:/Jyd/test/jyd_test/images2/detection/", help="infernces imgs root")
    return parser

if __name__=="__main__":
    opts=get_argparser().parse_args()

    inference = Inference(opts)

    for i, im in enumerate(os.listdir(opts.imgs_root)):
        img_path = opts.imgs_root + im
        image = Image.open(img_path)
        inference(image)

    print("total time: ", inference.total_time / (i+1))