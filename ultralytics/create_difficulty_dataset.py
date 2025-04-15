import os
import shutil
import numpy as np
from pathlib import Path

from ultralytics.models.yolo.obb.val import OBBValidatorCustom
from collections import defaultdict

def create_difficulty_dataset(source_image_dir, source_label_dir, target_base_dir):
    """
    Create cumulative datasets for each difficulty level (easy, moderate, hard).
    
    Easy: only easy objects
    Moderate: easy + moderate objects
    Hard: easy + moderate + hard objects
    """
    validator = OBBValidatorCustom()
    difficulties = ['easy', 'moderate', 'hard']

    val_base = Path(target_base_dir) / 'bevdata_obb_val'
    os.makedirs(val_base / 'images', exist_ok=True)

    for diff in difficulties:
        os.makedirs(val_base / f'labels_{diff}', exist_ok=True)

    valid_images = set()
    images_only_unknown = set()
    objects_by_difficulty = {diff: 0 for diff in difficulties}

    for label_file in Path(source_label_dir).glob('*.txt'):
        has_valid_object = False
        objects_by_level = defaultdict(list)

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                height = float(parts[9])
                truncation = float(parts[10])
                occlusion = int(float(parts[11]))
                
                difficulty_info = np.array([[height, truncation, occlusion]])
                difficulty = validator.get_kitti_obj_level(difficulty_info)[0]
                
                if difficulty > 0:  # Not unknown
                    has_valid_object = True
                    diff_name = difficulties[difficulty-1]
                    objects_by_level[diff_name].append(line)
                    objects_by_difficulty[diff_name] += 1
        
        if has_valid_object:
            valid_images.add(label_file.stem)

            if objects_by_level['easy']:
                with open(val_base / 'labels_easy' / label_file.name, 'w') as f:
                    f.writelines(objects_by_level['easy'])

            if objects_by_level['easy'] or objects_by_level['moderate']:
                with open(val_base / 'labels_moderate' / label_file.name, 'w') as f:
                    f.writelines(objects_by_level['easy'] + objects_by_level['moderate'])
          
            if objects_by_level['easy'] or objects_by_level['moderate'] or objects_by_level['hard']:
                with open(val_base / 'labels_hard' / label_file.name, 'w') as f:
                    f.writelines(objects_by_level['easy'] + objects_by_level['moderate'] + objects_by_level['hard'])
        else:
            images_only_unknown.add(label_file.stem)

    # Copy valid images
    for img_stem in valid_images:
        source_img = Path(source_image_dir) / f"{img_stem}.png"
        if source_img.exists():
            shutil.copy2(source_img, val_base / 'images' / f"{img_stem}.png")

    # Print statistics
    print("\n=== Dataset Creation Statistics ===")
    print(f"Total images processed: {len(valid_images) + len(images_only_unknown)}")
    print(f"Images with valid objects: {len(valid_images)}")
    print(f"Images with only unknown objects: {len(images_only_unknown)}")
    
    print("\nObjects per difficulty:")
    total_objects = 0
    cumulative_count = 0
    for diff in difficulties:
        cumulative_count += objects_by_difficulty[diff]
        print(f"{diff.capitalize()}: {objects_by_difficulty[diff]} (Cumulative: {cumulative_count})")
        total_objects += objects_by_difficulty[diff]
    print(f"Total objects: {total_objects}")

if __name__ == "__main__":
    source_img_dir = "/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/kitti/images/val"
    source_label_dir = "/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/kitti/labels/val"
    target_base_dir = "/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/kitti"

    create_difficulty_dataset(source_img_dir, source_label_dir, target_base_dir)
    
