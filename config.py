from yacs.config import CfgNode as CN
import importlib
import logging

_C = CN()

_C.DATA = CN()
_C.DATA.DATA_ROOT = ''
_C.DATA.DATASET = 'shelves'
_C.DATA.INPUT_HEIGHT = 640
_C.DATA.INPUT_WIDTH = 640
_C.DATA.NUM_WORKERS = 4
_C.DATA.YEAR = ''
_C.DATA.DOWNLOAD = False

_C.MODEL = CN()
_C.MODEL.TYPE = ''
_C.MODEL.BACKBONE = None
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASS = 4
_C.MODEL.OUTPUT_STRIDE = 16
_C.MODEL.P = 8
_C.MODEL.Q = 2

_C.LOSS = CN()
_C.LOSS.TYPE = None
_C.LOSS.META_LOSS_TYPE = ''
_C.LOSS.AUX_WEIGHT = [0.4]

_C.TRAIN = CN()
_C.TRAIN.TOTAL_ITRS = 30e3
_C.TRAIN.LR = 0.1
_C.TRAIN.LR_POLICY = "poly"
_C.TRAIN.STEP_SIZE = 10000
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.VAL_BATCH_SIZE = 2
_C.TRAIN.CONTINUE_TRAINING = False
_C.TRAIN.GPU_ID = '0'
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.RANDOM_SEED = 1
_C.TRAIN.PRINT_INTERVAL = 10
_C.TRAIN.VAL_INTERVAL = 100
_C.TRAIN.CROP_VAL = True
_C.TRAIN.SAVE_VAL_RESULTS = False
_C.TRAIN.EDGE_RADIUS = 3

def _update_config_from_file(config, cfg_file):
    config.defrost()
    config.merge_from_file(cfg_file)
    config.freeze()

def _update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.val_batch_size:
        config.TRAIN.VAL_BATCH_SIZE = args.val_batch_size
    if args.total_itrs:
        config.TRAIN.TOTAL_ITRS = args.total_itrs
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.lr_policy:
        config.TRAIN.LR_POLICY = args.lr_policy
    if args.step_size:
        config.TRAIN.STEP_SIZE = args.step_size
    if args.continue_training:
        config.TRAIN.CONTINUE_TRAINING = args.continue_training
    if args.gpu_id:
        config.TRAIN.GPU_ID = args.gpu_id
    if args.random_seed:
        config.TRAIN.RANDOM_SEED = args.random_seed
    if args.print_interval:
        config.TRAIN.PRINT_INTERVAL = args.print_interval
    if args.val_interval:
        config.TRAIN.VAL_INTERVAL = args.val_interval
    if args.crop_val:
        config.TRAIN.CROP_VAL = args.crop_val
    if args.data_root:
        config.DATA.DATA_ROOT = args.data_root
    if args.year:
        config.DATA.YEAR = args.year
    if args.download:
        config.DATA.DOWNLOAD = args.download
    if args.save_val_results:
        config.TRAIN.SAVE_VAL_RESULTS = args.save_val_results
    if args.output_stride:
        config.TRAIN.OUTPUT_STRIDE = args.output_stride
    if args.edge_radius:
        config.TRAIN.EDGE_RADIUS = args.edge_radius
    if args.num_classes:
        config.MODEL.NUM_CLASS = args.num_classes
    config.freeze()

def get_config(args):
    config = _C.clone()
    _update_config(config, args)
    return config

def build_model(cfgs):
    mod = importlib.import_module("libs")
    net_func = getattr(mod, cfgs.MODEL.TYPE)
    logging.info("Build Backbone Type: {}".format(cfgs.MODEL.BACKBONE))
    if cfgs.MODEL.TYPE == "deeplabv3plusplus":
        print("=== cfgs.MODEL.NUM_CLASS: ", cfgs.MODEL.NUM_CLASS)
        net = net_func(num_classes=cfgs.MODEL.NUM_CLASS, backend=cfgs.MODEL.BACKBONE, 
                                        output_stride=cfgs.MODEL.OUTPUT_STRIDE)
    elif cfgs.MODEL.TYPE == "espnet":
        net = net_func(num_classes=cfgs.MODEL.NUM_CLASS, p=cfgs.MODEL.P, q=cfgs.MODEL.Q)
    else:
        net = net_func(num_classes=cfgs.MODEL.NUM_CLASS, backend=cfgs.MODEL.BACKBONE)
    return net

def build_criterion(cfgs):
    mod = importlib.import_module("utils.loss")
    loss_type = cfgs.LOSS.TYPE
    loss_module = getattr(mod, loss_type)
    logging.info("Build Loss Type: {}".format(loss_module))
    if loss_type == "MainAuxLoss":
        criterion = loss_module(num_classes=cfgs.MODEL.NUM_CLASS, aux_weight=cfgs.LOSS.AUX_WEIGHT, 
                                        loss_type=cfgs.LOSS.META_LOSS_TYPE)
    elif loss_type == "ICNetLoss":
        criterion = loss_module(aux_weight=cfgs.LOSS.AUX_WEIGHT)
    else:
        criterion = loss_module(num_classes=cfgs.MODEL.NUM_CLASS)
    return criterion