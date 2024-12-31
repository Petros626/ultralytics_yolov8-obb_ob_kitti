from ultralytics import YOLO

"""
BEVDetNet export params:
We also experiment with reduced precision quantized models for 
16-bit floating point (FP16) and 8-bit fixed point integer (INT8) 
as shown in Table IV. This is along the lines of our goal to improve 
efficiency for embedded deployment. FP16 precision model was created 
using the quantization option in TensorRT while the INT8 model was
 converted using a calibration dataset made from the training BEV images. 
 The latency improved significantly reaching 2 ms for INT8 resolution. 
 There is a noticeable degradation in accuracy as we are doing offline 
 quantization but this gap can be reduced using quantization aware training.
"""
# Load the YOLOv8 model
model = YOLO("xxxxxx.pt", task='obb')

# Export the model to TensorRT format with Precision: FP16 and INT8
# Nms: only needed for CoreML
model.export(format="engine", imgsz=640, keras=False, optimize=False, half=True, int8=False, 
             dynamic=True, simplify=False, opset=None, workspace=None, nms=False, 
             batch=1, device='0')  # creates 'yolov8s.engine FP16

# NOTE: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c
model.export(format="engine", imgsz=640, keras=False, optimize=False, half=True, int8=False, 
             dynamic=True, simplify=False, opset=None, workspace=None, nms=False, 
             batch=1, device='0', data='kitti_bev.yaml')  # creates 'yolov8s.engine INT8

# Load the exported TensorRT model
tensorrt_model = YOLO("yolov8n.engine")