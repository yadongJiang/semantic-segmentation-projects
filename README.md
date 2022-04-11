# Segmentation Projects
实现了一些常用的语义分割算法，可以支持VOC以及自定义的数据集训练

## 环境依赖
pytorch-1.6
PIL
Opencv
thop


## 支持的算法
1. [deeplabv3++](https://arxiv.org/pdf/1802.02611.pdf)
2. [enet](https://arxiv.org/pdf/1606.02147.pdf)
3. [espnet](https://arxiv.org/abs/1803.06815v2)
4. [icnet](https://arxiv.org/pdf/1704.08545.pdf)
5. [pspnet](https://arxiv.org/pdf/1612.01105.pdf)
6. [gscnn](https://arxiv.org/pdf/1907.05740.pdf)
7. [decouplesegnet](https://arxiv.org/pdf/2007.10035.pdf)
8. [danet](https://arxiv.org/pdf/1809.02983.pdf)
9. [cpnet](https://arxiv.org/pdf/2004.01547.pdf)
10. [ocrnet](https://arxiv.org/pdf/1909.11065.pdf)
11. [bisenetv1](https://arxiv.org/pdf/1808.00897.pdf)
12. [bisenetv2](https://arxiv.org/pdf/2004.02147.pdf)
13. [sfsegnet](https://arxiv.org/pdf/2002.10120v3.pdf)

## Training
参数可以在opts.py中配置，比如一些关键参数，如--model,--batch_size, --lr, --dataset, --data_root等等。
    python main.py --model bisenetv1 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 640 --batch_size 16