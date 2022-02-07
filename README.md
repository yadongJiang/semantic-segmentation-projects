# Dmall_segment
    实现了一些常用的语义分割算法，可以支持VOC、Cityscapes以及自己的数据集训练， 使用的深度学习框架为pytorch

## Supported algorithms
    1. deeplabv3++ (mobilenetv3, ghostnet, resnet50)
    2. enet
    3. espnet
    4. icnet
    5. pspnet (resnet34, resnet50)
    6. gscnn
    7. decouplesegnet
    8. danet
    9. cpnet (resnet34)
    10.ocrnet
    11.bisenetv1
    12.bisenetv2
    13.sfsegnet

## Training
    参数可以在opts.py中配置，比如一些关键参数，如--model,--batch_size, --lr, --dataset, --data_root等等。
    For example
        python main.py --model bisenetv1 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 640 --batch_size 16