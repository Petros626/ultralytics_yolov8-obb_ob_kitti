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
model = YOLO("yolo11s-obb.yaml", task='obb', verbose=False)  # build a new model from YAML
#model = YOLO("yolo11s-obb.yaml", task='obb', verbose=False).load("yolo11s.pt")  # build from YAML and transfer weights
#model= YOLO('/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/master_thesis/mt_kitti_bev/default3/weights/last.pt', task='obb', verbose=False) # Resume to training e.g. run_84 last.pt

# Print layers, parameters, gradients, GFLOPS (computation depends on https://github.com/ultralytics/ultralytics/issues/17547#issuecomment-2481925742)
# source: https://github.com/ultralytics/ultralytics/issues/14749
print(model.info(detailed=False, verbose=True))
print('reg_max: ', model.model.model[-1].reg_max)

# train configuration
dataset_cfg = '/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/zod_bev.yaml'

# https://community.ultralytics.com/t/about-yolo-configuration-file-yaml/300
results = model.train(data=dataset_cfg, epochs=50, time=None, patience=0, batch=6, imgsz=1024, save=True, save_period=25, cache=True,
                      device=[0, 1], workers=8, project='mt_zod_bev', name='default', exist_ok=False, 
                      pretrained=False, optimizer='Adam', seed=0, deterministic=False, verbose=True, single_cls=False,
                      classes=None, rect=False, multi_scale=True, cos_lr=True, close_mosaic=0, resume=False, amp=False,
                      fraction=1.0, profile=False, freeze=None, lr0=0.01, lrf=0.001,  momentum=0.937,
                      weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, 
                      box=7.0, cls=0.3, dfl=0.9, pose=0.0, kobj=0.0, nbs=64, overlap_mask=False, mask_ratio=0,
                      dropout=0.0, val=True, plots=True, compile=False,
                      
                      hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                      flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, copy_paste_mode='mixup', auto_augment=None,
                      erasing=0.0, augmentations=None) # https://github.com/ultralytics/ultralytics/issues/15721#issuecomment-2307978771