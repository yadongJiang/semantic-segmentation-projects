import torch
import opts
from common import get_dataset, get_params, validate
import os
import numpy as np
import random
import torch.utils.data as data
from metrics import StreamSegMetrics
import utils
import torch.nn as nn
import libs
import utils.select_loss as sl
import torch.cuda.amp as amp

def main():
    args = opts.get_argparser().parse_args()
    if args.dataset.lower() == 'voc':
        args.num_classes = 21
    elif args.dataset.lower() == "shelves":
        args.num_classes = 4
    print("num_classes:", args.num_classes)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    train_dst, val_dst = get_dataset(args)
    train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (args.dataset, len(train_dst), len(val_dst)))
          
    model_map = {
        'deeplabv3_mobilenetv3' : libs.deeplabv3_mobilenetv3, 
        'deeplabv3_ghostnet' : libs.deeplabv3_ghostnet, 
        'deeplabv3_resnet50' : libs.deeplabv3_resnet50, 
        'enet' : libs.enet, 
        "espnet" : libs.espnet, 
        "icnet" : libs.icnet, 
        "pspnet_res34" : libs.pspnet_resnet34, 
        "pspnet_res50" : libs.pspnet_resnet50, 
        "cpnet" : libs.cpnet, 
        "bisenetv1" : libs.bisenetv1, 
        "bisenetv2" : libs.bisenetv2, 
        "ocrnet" : libs.ocrnet, 
        "gscnn" : libs.gscnn, 
        "decoupsegnet" : libs.decouplesegnet, 
        "sfsegnet" : libs.sfsegnet, 
        "danet" : libs.danet, 
    }

    if "deeplabv3" in args.model:
        model = model_map[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    elif "espnet" in args.model:
        model = model_map[args.model](num_classes=args.num_classes, p=args.p, q=args.q)
    else:
        model = model_map[args.model](num_classes=args.num_classes)

    sl.NUM_CLASSES = args.num_classes
    criterion = sl.get_criterion(args.model)
    print("criterion: ", criterion)

    # Set up metrics
    metrics = StreamSegMetrics(args.num_classes)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, args.total_itrs, power=0.9)
    elif args.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if args.ckpt is not None and os.path.isfile(args.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        
        state_dicts=checkpoint["model_state"]
        pretrained_dict={k:v for k, v in state_dicts.items() if k in model.state_dict() and ("classifier.classifier.3" not in k)}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)
        
        '''model.load_state_dict(checkpoint["model_state"])'''
        model = nn.DataParallel(model)
        model.to(device)
        if args.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % args.ckpt)
        print("Model restored from %s" % args.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    vis_sample_id = None

    # scaler = amp.GradScaler()
    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, edgemask) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)  #.half()
            labels = labels.to(device, dtype=torch.long)
            edgemask = edgemask.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            # with amp.autocast(enabled=True):
            outputs = model(images)

            if args.model == "icnet":
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, (labels, edgemask)) # 加权交叉熵函数与边缘注意力损失函数

            """scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()"""
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % args.print_interval == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, args.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % args.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (args.model, args.dataset, args.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=args, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (args.model, args.dataset,args.output_stride))

                model.train()
            scheduler.step()

            if cur_itrs >=  args.total_itrs:
                print("best_score:", best_score)
                return

if __name__ == '__main__':
    main()