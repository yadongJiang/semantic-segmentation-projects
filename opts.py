import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--cfg", type=str, required=True, default="./configs/bisenetv1.yaml")
    parser.add_argument("--data_root", type=str, default='C:/Users/dmall/Downloads/VOCtrainval_11-May-2012',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', "shelves"], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=21,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3_mobilenetv3',
                        choices=[ 'deeplabv3_mobilenetv3', 'deeplabv3_ghostnet', 'enet', 
                                  'espnet', 'icnet', 'pspnet_res34', 'pspnet_res50', 'ocrnet',
                                  'bisenetv1', 'bisenetv2' 'cpnet', "gscnn", 'deeplabv3_resnet50', 
                                  'decoupsegnet', 'sfsegnet', 'danet'], help='model name')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    ## aux loss weight
    parser.add_argument("--aux-alpha", type=float, default=0.3, help="aux supervise's weights")
    ## espnet, p and q
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--edge_radius", type=int, default=3, help="training phase, the width of edge")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='JointEdgeSegLoss',
                        choices=['cross_entropy', 'focal_loss', "hard_mine_loss", "ssim_loss", "JointEdgeSegLoss"], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=3,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=3,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--save_val_results", type=bool, default=False, help="save the val result")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012_aug',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    parser.add_argument("--s", type=float, default=0.0001, help="scale sparse rate (default:0.0001)")
    return parser