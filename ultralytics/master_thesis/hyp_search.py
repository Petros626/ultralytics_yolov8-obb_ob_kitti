from ultralytics import YOLO
import torch
import os

# Set expandable_segments to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check for GPU use
print('Use GPU:', torch.cuda.is_available())
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU cache cleared.\n")

# Build a new model from scratch and check model specific values
print('--Check YOLO intern values for custom purposes--')
model = YOLO("yolo11s-obb.yaml", task='obb', verbose=False) # Initialize the YOLO model
print(model.info(detailed=False, verbose=True))
print('reg_max: ', model.model.model[-1].reg_max)
dataset_cfg = '/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/kitti_bev.yaml'

# Define search space
search_space = {
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.005),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (0.02, 0.2),
    "cls": (0.2, 4.0),
    "dfl": (0.4, 6.0),
    # deactivate augmentations
    "hsv_h": (0.0, 0.0),
    "hsv_s": (0.0, 0.0),
    "hsv_v": (0.0, 0.0),
    "degrees": (0.0, 0.0),
    "translate": (0.0, 0.0),
    "scale": (0.0, 0.0),
    "shear": (0.0, 0.0),
    "perspective": (0.0, 0.0),
    "flipud": (0.0, 0.0),
    "fliplr": (0.0, 0.0),
    "bgr": (0.0, 0.0),
    "mosaic": (0.0, 0.0),
    "mixup": (0.0, 0.0),
    "cutmix": (0.0, 0.0),
    "copy_paste": (0.0, 0.0),
    "close_mosaic": (0, 0),
    "erasing": (0.0, 0.0),
}

# Tune hyperparameters on KITTI for 50 epochs
model.tune(data=dataset_cfg, epochs=50, iterations=15, optimizer="Adam", space=search_space, plots=True,
    save=True, val=True, time=None, patience=0, batch=6, imgsz=1024, save_period=25, cache=True, device=[0, 1], workers=8, project='mt_param_search',
    name='iterations', exist_ok=False, pretrained=False, seed=0, deterministic=False, verbose=True, single_cls=False, classes=None, rect=False, multi_scale=True, 
    cos_lr=False, resume=True, amp=False, fraction=1.0, profile=False, freeze=None, overlap_mask=False, mask_ratio=0,
    dropout=0.0, compile=False, auto_augment=None, augmentations=None)