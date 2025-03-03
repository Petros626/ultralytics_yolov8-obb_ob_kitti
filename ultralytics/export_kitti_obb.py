from ultralytics import YOLO
import argparse

"""
BEVDetNet export hints:
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

def export_model(model, export_args):
    
    model.export(**export_args)

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 OBB model to TensorRT with user-configurable parameters.')

    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained YOLOv8 model (.pt file).')
    parser.add_argument('--export_fp16', type=bool, default=False, help='Export to FP16 TensorRT model.')
    parser.add_argument('--export_int8', type=bool, default=False, help='Export to INT8 TensorRT model.')
    parser.add_argument('--format', type=str, default='engine', help="Format to export to (e.g., 'engine', 'onnx').")
    parser.add_argument('--imgsz', type=int, default=640, help='Desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions.')
    parser.add_argument('--keras', type=bool, default=False, help='Enables export to Keras format for TensorFlow SavedModel, providing compatibility with TensorFlow serving and APIs.')
    parser.add_argument('--optimize', type=bool, default=False, help='Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.')
    parser.add_argument('--half', type=bool, default=False, help='Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.')
    parser.add_argument('--int8', type=bool, default=False, help='Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.')
    parser.add_argument('--dynamic', type=bool, default=False, help='Allows dynamic input sizes for ONNX, TensorRT and OpenVINO exports, enhancing flexibility in handling varying image dimensions.')
    parser.add_argument('--simplify', type=bool, default=False, help='Simplifies the model graph for ONNX exports with onnxslim, potentially improving performance and compatibility.')
    parser.add_argument('--opset', type=int, default=None, help='Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.')
    parser.add_argument('--workspace', type=int, default=None, help='Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance; use None for auto-allocation by TensorRT up to device maximum.')
    parser.add_argument('--nms', type=bool, default=False, help='Adds Non-Maximum Suppression (NMS) to the exported model when supported (see Export Formats), improving detection post-processing efficiency.')
    parser.add_argument('--batch', type=int, default=8, help="Batch size for export. For INT8 it's recommended using a larger batch like batch=8 (calibrated as batch=16))")
    parser.add_argument('--device', type=str, default='0', help="Device to use for export (e.g., '0' for GPU 0).")
    parser.add_argument('--data', type=str, default=None, help="Path to the dataset configuration file for INT8 calibration.")

    args = parser.parse_args()

    # Load the final trained YOLOv8 model
    model = YOLO(args.model_path, task='obb')

    export_args = {
        'format': args.format,
        'imgsz': args.imgsz,
        'keras': args.keras,
        'optimize': args.optimize,
        'half': args.half,
        'int8': args.int8,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'workspace': args.workspace,
        'nms': args.nms,
        'batch': args.batch,
        'device': args.device,
        'data': args.data,
    }

    if args.export_fp16: # data argument isn't needed for FP16 exports since no calibration is required
        print('Exporting to FP16 TensorRT model...')
        fp16_args = export_args.copy()
        fp16_args['half'] = True
        fp16_args['int8'] = False
        export_model(model, fp16_args)
        print('FP16 export completed.')
    
    if args.export_int8: # NOTE: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c, for INT8 calibration, the kitti_bev.yaml val split with 3769 images is used.
        print('Exporting to INT8 TensorRT model...')
        int8_args = export_args.copy()
        int8_args['half'] = False
        int8_args['int8'] = True
        export_model(model, int8_args)
        print('INT8 export completed.')

    if not args.export_fp16 and not args.export_int8:
        print('No export option selected. Please specify --export_fp16 and/or --export_int8.')

if __name__ == '__main__':
    main()

# # FP16
# # data argument isn't needed for FP16 exports since no calibration is required
# model.export(format="engine", imgsz=640, keras=False, optimize=False, half=True, int8=False, 
#              dynamic=True, simplify=False, opset=None, workspace=None, nms=False, 
#              batch=1, device='0')  # creates 'yolov8s.engine' FP16 model

# # INT8 batch=8 * 2 recommended
# # NOTE: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c
# # for INT8 calibration, the kitti_bev.yaml val split with 3769 images
# model.export(format="engine", imgsz=640, keras=False, optimize=False, half=False, int8=True, 
#              dynamic=True, simplify=False, opset=None, workspace=None, nms=False, 
#              batch=1, device='0', data='kitti_bev.yaml')  # creates 'yolov8s.engine' INT8 model

# Load the exported TensorRT model, verify export worked
#tensorrt_model = YOLO("yolov8n.engine")