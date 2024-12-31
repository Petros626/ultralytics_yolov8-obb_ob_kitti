from ultralytics import YOLO

model = YOLO("yolov8s.engine")
results = model.val(
    data="kitti_bev.yaml",
    batch=1,
    imgsz=640,
    verbose=False,
    device="cuda",
)