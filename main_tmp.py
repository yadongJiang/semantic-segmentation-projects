from matplotlib.cbook import print_cycles
import opts
from config import get_config, build_model, build_criterion
import os
import torch
import numpy as np
import random
from common import get_dataset, validate
import torch.utils.data as data
from metrics import StreamSegMetrics
import utils
import torch.nn as nn
import torch.cuda.amp as amp

import logging
FORMAT = "%(asctime)s %(levelname)s [%(filename)s, %(lineno)d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main(cfgs):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.TRAIN.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    torch.manual_seed(cfgs.TRAIN.RANDOM_SEED)
    np.random.seed(cfgs.TRAIN.RANDOM_SEED)
    random.seed(cfgs.TRAIN.RANDOM_SEED)

    train_dst, val_dst = get_dataset(cfgs)
    train_loader = data.DataLoader(
        train_dst, batch_size=cfgs.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=cfgs.TRAIN.VAL_BATCH_SIZE, shuffle=True, num_workers=2)
    logging.info("Dataset: {}, Train set: {}, Val set: {}".format(cfgs.DATA.DATASET, len(train_dst), len(val_dst)))

    model = build_model(cfgs)
    criterion = build_criterion(cfgs)
    logging.info("Criterion Info :{}".format(criterion))

    # Set up metrics
    metrics = StreamSegMetrics(cfgs.MODEL.NUM_CLASS)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=cfgs.TRAIN.LR, momentum=0.9, weight_decay=cfgs.TRAIN.WEIGHT_DECAY)
    if cfgs.TRAIN.LR_POLICY == 'poly':
        scheduler = utils.PolyLR(optimizer, cfgs.TRAIN.TOTAL_ITRS, power=0.9)
    elif cfgs.TRAIN.LR_POLICY == 'step':
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
        logging.info("Model saved as {}".format(path))

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if cfgs.MODEL.RESUME is not None and os.path.isfile(cfgs.MODEL.RESUME):
        checkpoint = torch.load(cfgs.MODEL.RESUME, map_location=torch.device('cpu'))
        
        state_dicts=checkpoint["model_state"]
        pretrained_dict={k:v for k, v in state_dicts.items() if k in model.state_dict()} #  and ("classifier.classifier.3" not in k)
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)
        
        '''model.load_state_dict(checkpoint["model_state"])'''
        model = nn.DataParallel(model)
        model.to(device)
        if cfgs.TRAIN.CONTINUE_TRAINING:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            logging.info("Training state restored from {}".format(cfgs.MODEL.RESUME))
        logging.info("Model restored from {}".format(cfgs.MODEL.RESUME))
        del checkpoint  # free memory
    else:
        logging.warn("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    scaler = amp.GradScaler()
    interval_loss = 0
    while True:
        # =====  Train  =====
        model.train()
        cur_epochs += 1

        for (images, labels, edgemask) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)  #.half()
            labels = labels.to(device, dtype=torch.long)
            edgemask = edgemask.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, (labels, edgemask)) # 加权交叉熵函数与边缘注意力损失函数
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            '''loss.backward()
            optimizer.step()'''

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % cfgs.TRAIN.PRINT_INTERVAL == 0:
                interval_loss = interval_loss/10
                logging.info("Epoch {}, Itrs {}/{}, Loss={}".format(cur_epochs, cur_itrs, cfgs.TRAIN.TOTAL_ITRS, interval_loss))
                interval_loss = 0.0
            
            if (cur_itrs) % cfgs.TRAIN.VAL_INTERVAL == 0:
                save_ckpt('checkpoints/latest_%s_%s.pth' %
                          (cfgs.MODEL.TYPE, cfgs.DATA.DATASET))
                logging.info("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=cfgs, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=None)
                logging.warn(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s.pth' %
                              (cfgs.MODEL.TYPE, cfgs.DATA.DATASET))
                
                model.train()
            scheduler.step()

            if cur_itrs >=  cfgs.TRAIN.TOTAL_ITRS:
                logging.warn("best_score: {}".format(best_score))
                return

if __name__ == "__main__":
    args = opts.get_argparser().parse_args()
    cfgs = get_config(args)
    main(cfgs)