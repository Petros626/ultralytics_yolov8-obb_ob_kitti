import numpy as np
from ultralytics.utils import box_utils
import math, pickle, os, argparse
from pathlib import Path

def load_calib_from_pkl(dataset_path):
    """
    Load calibration data from validation data.

    Args:
        dataset_path (str or Path): Path to the pickle file

    Returns:
        dict: Dictionary mappind lidar_idx/frame_id to calibration data
    """
    print(f"Loading calibration data from {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Extract calibration data for each frame
    calib_data = {}
    for frame in dataset:
        if 'point_cloud' in frame and 'calib' in frame:
            lidar_idx = frame['point_cloud']['lidar_idx']
            calib = frame['calib']

            calib_data[lidar_idx] = {
                'P2': calib['P2'][:3],  # 3 x 4
                'R0': calib['R0_rect'][:3, :3],  # 3 x 3
                'Tr_velo2cam': calib['Tr_velo_to_cam'][:3],  # 3 x 4
                'image_shape': frame['image']['image_shape'] # [height, width]    
            }
    print(f"Loaded calibration data for {len(calib_data)} frames")
    return calib_data

def parse_bev_prediction(pred_line):
        """
        Parse a line from YOLO BEV prediction file
        
        Args:
            pred_line (str): Single line from prediction file containing space-separated values
            Format: class_id cx cy width height rotation confidence

        Returns:
            dict: Parsed prediction data with keys
        """
        parts = pred_line.strip().split()

        # YOLO11 OBB prediction format cls_id, cx, cy, w, h, rotation, confidence
        return {
            'class_id': int(parts[0]),
            'x_center': float(parts[1]),
            'y_center': float(parts[2]),
            'width': float(parts[3]), # OpenCV: width = KITTI: length
            'height': float(parts[4]), # OpenCV: height = KITTI: width
            'rotation': float(parts[5]), # in radians
            'confidence': float(parts[6]),
        }

def convert_to_lidar_label(pred, image_width=1024, image_height=1024, cell_size=0.06):
        """
        Convert YOLO BEV prediction to LiDAR frame
        
        Args:
            pred (dict): Parsed BEV prediction from parse_bev_prediction()
            image_width (int): BEV image width in pixels
            image_height (int): BEV image height in pixels
            cell_size (float): Meters per pixel
                           
        Returns:
            dict: LiDAR label in KITTI format with keys
        """
        
        # Convert image coordinates (px) to world coordinates (m)
        center, dimensions = box_utils.pixel_to_world_coordinates(
            (pred['x_center'], pred['y_center']),
            (pred['width'], pred['height']),
            image_width=image_width,
            image_height=image_height,
            cell_size=cell_size
        )
          
        # rotation is raw CW, angle range [-π/4, 3π/4]            
        # Image-space to LiDAR-space => -: Reflects the x-axis (due to y-axis difference) & -pi/2: Rotates coordinate system by -90°
        # NOTE: Both spaces are CW, but the axis orientation is different. The transformation heading = -pred['rotation'] - math.pi/2 corrects both the axis mirroring and the reference rotation.
        heading = -pred['rotation'] - math.pi/2
            
        # source: "BirdNet+: End-to-End 3D Object Detection in LiDAR Bird's Eye View"
        default_heights = {1: 1.53, 2: 1.76, 3: 1.74}  # Car: 1.53m, Ped: 1.76m, Cyc: 1.74m
        height = default_heights.get(pred['class_id'])
        # 3D Bounding Box Regression not supported by YOLO11 OBB Head. Approaches like BirdNet, BirdNet+ etc. can't be applied yet
        z = -1.55 # approx. value fitting for 3D Boxes vs 3D GT

        # Map class_id to type
        class_map = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
        obj_type = class_map.get(pred['class_id'], "DontCare")

        # Create label in LiDAR frame in KITTI format
        lidar_label = {
            "type": str(obj_type),
            "truncated": float(-1), # dummy value like BirdNet2 or OpenPCDet
            "occluded": int(-1), # dummy value
            "alpha": float(-10), # # calculated in convert_to_camera_label()
            "bbox": [-1, -1, -1, -1], #calculated in convert_to_camera_label()
            "dimensions": [dimensions[0], dimensions[1], height],  # l, w, h
            "location": [center[0], center[1], z], # x, y, z
            "rotation_z": heading, # rad
            "score": float(pred['confidence'])
        }

        return lidar_label

def convert_to_camera_label(lidar_label, calib, image_shape):
    """
    Convert LiDAR frame label to Camera frame label.
    
    Args:
        lidar_label (dict): Label data in LiDAR frame format from convert_to_lidar_label()
        calib (dict): Calibration data with keys 'P2', 'R0', 'Tr_velo2cam'
        image_shape (tuple): (height, width) of the camera image
        
    Returns:
        dict: Label data converted to Camera frame format with keys
    """
    # Extract box parameters
    x, y, z = lidar_label['location']
    l, w, h = lidar_label['dimensions']
    heading = lidar_label['rotation_z']

    # Create box array in OpenPCDet format [x, y, z, l, w, h, heading]
    box3d_lidar = np.array([[x, y, z, l, w, h, heading]])

    # Convert LiDAR 3D Boxes to Camera 3D Boxes using OpenPCDet method
    box3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(box3d_lidar, calib)
    x_rect, y_rect, z_rect, l_rect, h_rect, w_rect, rotation_y = box3d_camera[0]
    
    # Convert Camera 3D Boxes to Camera 2D Boxes using OpenPCDet method
    box2d_camera = box_utils.boxes3d_kitti_camera_to_imageboxes(box3d_camera, calib, image_shape=image_shape)
    xmin, ymin, xmax, ymax = box2d_camera[0]

    # Calculate observation angle
    alpha = rotation_y - np.arctan2(-y, x) # -y is necessary because you are in the LiDAR frame and are “rethinking” the camera frame
    #alpha = rotation_y - np.arctan2(x_rect, z_rect) # or use this in the camera frame
    
    # Create label in Camera frame in KITTI format
    return {
        'type': lidar_label['type'],
        'truncated': float(lidar_label['truncated']), # adopted dummy value
        'occluded': int(lidar_label['occluded']), # adopted dummy value
        'alpha': alpha,
        'bbox': [xmin, ymin, xmax, ymax], # use calculated 2D BBox values
        'dimensions': [h_rect, w_rect, l_rect],  # h, w, l 
        'location': [x_rect, y_rect, z_rect], # x, y, z
        'rotation_y': rotation_y,
        'score': float(lidar_label['score'])
    }

def generate_preds_txts(camera_labels, output_path=None, frame_id=None):

    if output_path is not None and frame_id is not None:
        os.makedirs(output_path, exist_ok=True)
        label_file = os.path.join(output_path, f"{frame_id}.txt")

        with open(label_file, 'w') as f:
            for label in camera_labels:
                # KITTI format: type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y score
                f.write('%s %.2f %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'
                      % (label['type'],
                        label['truncated'],
                        label['occluded'],
                        label['alpha'],
                        label['bbox'][0], label['bbox'][1], label['bbox'][2], label['bbox'][3],  # xmin, ymin, xmax, ymax
                        label['dimensions'][0], label['dimensions'][1], label['dimensions'][2],  # h, w, l
                        label['location'][0], label['location'][1], label['location'][2],  # x, y, z
                        label['rotation_y'],
                        label['score']))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_data", required=True, help="Path to kitti_val_dataset.pkl")
    ap.add_argument("--preds", required=True, help="Directory containing YOLO prediction txt files")
    ap.add_argument("--output", required=True, help="Output directory for KITTI format predictions")
    ap.add_argument("--image_width", type=int, default=1024, help="BEV image width in pixels")
    ap.add_argument("--image_height", type=int, default=1024, help="BEV image height in pixels")
    ap.add_argument("--cell_size", type=float, default=0.06, help="BEV cell resolution")
    args = ap.parse_args()

    calib_dict = load_calib_from_pkl(args.calib_data)
    pred_dir = Path(args.preds)
    pred_files = sorted(list(pred_dir.glob("*.txt")))

    print(f"Found {len(pred_files)} prediction files")

    for pred_file in pred_files:
        frame_id = pred_file.stem

        calib = calib_dict[frame_id]
        image_shape = calib['image_shape']

        with open(pred_file, 'r') as f:
            pred_lines = f.readlines()
        
        camera_labels = []

        for line in pred_lines:
            pred = parse_bev_prediction(line)
            lidar_label = convert_to_lidar_label(pred, image_width=args.image_width,
                                                image_height=args.image_height,
                                                cell_size=args.cell_size)
            
            camera_label = convert_to_camera_label(lidar_label, calib, image_shape)
            camera_labels.append(camera_label)

        generate_preds_txts(camera_labels, args.output, frame_id)    
    
    print(f"\nFinish label transformation, labels saved to {args.output}")

if __name__ == '__main__':
    main()