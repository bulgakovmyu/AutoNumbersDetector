from detectron2 import model_zoo
import os

def make_config_detector(cfg, prefix):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("AutoNumbers_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = os.path.join(prefix, 'checkpoints', 'init','model_final_68b088.pkl')
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # LR
    cfg.SOLVER.MAX_ITER = 2000    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg.OUTPUT_DIR = os.path.join(prefix,'checkpoints', 'current')
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    return cfg
    

def make_config_symbols(cfg, prefix):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("plates_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = os.path.join(prefix, 'checkpoints', 'symbols', 'init','model_final_68b088.pkl')
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 22
    cfg.OUTPUT_DIR = os.path.join(prefix,'checkpoints', 'symbols', 'current')
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    
    return cfg
