import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.models.yolo.obb.val import OBBValidatorCustom

# Contains all difficulty levels
bev_val_img = '/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/kitti/images/val/bev_val_000159.png'
bev_val_lbl = '/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/kitti/labels/val/bev_val_000159.txt'


def draw_obb(image_path, label_path, target_difficulty):
    """
    Draw oriented bounding boxes filtered by difficulty level.
    
    Args:
        image_path (str): Path to the image file
        label_path (str): Path to the label file (YOLO format required)
        target_difficulty (int): Difficulty level to display (1=Easy, 2=Moderate, 3=Hard)
    """
    image = cv2.imread(image_path)
    validator = OBBValidatorCustom()

    colors = {
        0: (255, 0, 0), # Unknown: Blue
        1: (0, 255, 0), # Easy: Green
        2: (0, 255, 255), # Moderate: Yellow
        3: (0, 0, 255) # Hard: Red
        
    }

    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            # Extract relevant information
            cls = int(parts[0])
            points = list(map(float, parts[1:9]))
            height = float(parts[9])
            truncation = float(parts[10])
            occlusion = float(parts[11])

            # Check difficulty level
            difficulty_info = np.array([[height, truncation, occlusion]])
            obj_difficulty = validator.get_kitti_obj_level(difficulty_info)[0]

            # Only draw if difficulty matches
            if obj_difficulty == target_difficulty:
                points = [(int(points[i] * image.shape[1]), int(points[i+1] * image.shape[0])) for i in range(0, len(points), 2)]
                points = np.array(points, np.int32).reshape((-1, 1, 2))
                color = colors.get(target_difficulty, (255, 255, 255)) # Default to white if not found
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=1)

    # less accrurate drawing
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()

    # more accurate drawing
    difficulty_name = {0: 'Unknown', 1: 'Easy', 2: 'Moderate', 3: 'Hard'}[target_difficulty]
    cv2.imshow(f'BEV Visualization - {difficulty_name}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Check one by one
draw_obb(bev_val_img, bev_val_lbl, 0) # Unknown, blue
#draw_obb(bev_val_img, bev_val_lbl, 1) # Easy, green
#draw_obb(bev_val_img, bev_val_lbl, 2) # Moderate, yellow
#draw_obb(bev_val_img, bev_val_lbl, 3) # Hard, red