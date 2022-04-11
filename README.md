# Dmall_segment
    实现了一些常用的语义分割算法，可以支持VOC以及自定义的数据集训练

# 环境依赖
    pytorch-1.6
    PIL
    Opencv
    thop


## Supported algorithms
    [deeplabv3++](https://arxiv.org/pdf/1802.02611.pdf)
    [enet](https://arxiv.org/pdf/1606.02147.pdf)
    [espnet](https://arxiv.org/abs/1803.06815v2)
    [icnet](https://arxiv.org/pdf/1704.08545.pdf)
    [pspnet](https://arxiv.org/pdf/1612.01105.pdf)
    [gscnn](https://arxiv.org/pdf/1907.05740.pdf)
    [decouplesegnet](https://arxiv.org/pdf/2007.10035.pdf)
    [danet](https://arxiv.org/pdf/1809.02983.pdf)
    [cpnet](https://arxiv.org/pdf/2004.01547.pdf)
    [ocrnet](https://arxiv.org/pdf/1909.11065.pdf)
    [bisenetv1](https://arxiv.org/pdf/1808.00897.pdf)
    [bisenetv2](https://arxiv.org/pdf/2004.02147.pdf)
    [sfsegnet](https://arxiv.org/pdf/2002.10120v3.pdf)

## Training
    参数可以在opts.py中配置，比如一些关键参数，如--model,--batch_size, --lr, --dataset, --data_root等等。
    For example
        python main.py --model bisenetv1 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 640 --batch_size 16