from ultralytics import YOLO
import torch

"""
BEVDetNet params:
epochs: 50, 60, 80, 120 (try 100)
batch_size: 8, 16, 24, 32, 48
imgsz: 640x640x3
optimizer: Adam
loss weights: 0.95, 0.98, 1
"""

# Check for GPU use
print('Use GPU:', torch.cuda.is_available())
print('\n')

# Build a new model from scratch and check model specific values
print('--Check YOLO intern values for custom purposes--')
model = YOLO('yolov8s-obb.yaml', 'obb', verbose=False) # n/s/m/l/x 
# Print layers, parameters, gradients, GFLOPS (computation depends on https://github.com/ultralytics/ultralytics/issues/17547#issuecomment-2481925742)
# https://github.com/ultralytics/ultralytics/issues/14749
print(model.info(detailed=False, verbose=True))

# https://community.ultralytics.com/t/about-yolo-configuration-file-yaml/300
# results = model.train(data='/home/rlab10/ultralytics/ultralytics/cfg/datasets/kitti_bev.yaml', epochs=50,
#                       time=None, patience=10, batch=8, imgsz=640, save=True, save_period=50, cache=False,
#                       device='cuda', workers=4, project='kitti_bev_yolo', name='run_1', exist_ok=False,
#                       pretrained=False, optimizer='Adam', seed=0, deterministic=True, single_cls=False,
#                       classes=None, rect=False, cos_lr=False, close_mosaic=0, resume=True, amp=False,
#                       fraction=1.0, profile=False, freeze=None, lr0=0.001, lrf=0.01,  momentum=0.937,
#                       weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, 
#                       box=7.5, cls=0.5, dfl=1.5, pose=0.0, kobj=0.0, nbs=64, overlap_mask=False, mask_ratio=0,
#                       dropout=0.0, val=False, plots=True,
                      
#                       hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
#                       flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0, copy_paste_mode='', auto_augment='',
#                       erasing=0.0, crop_fraction=0.0)


