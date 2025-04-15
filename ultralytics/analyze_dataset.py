import numpy as np
from ultralytics.models.yolo.obb.val import OBBValidatorCustom
from collections import defaultdict
from pathlib import Path

def analyze_label_difficulties(label_dir):
    """
    Analyze difficulty distribution in KITTI label files.

    Args:
        label_dir (str): Directory containing label files.
    """
    validator = OBBValidatorCustom()	
    difficulty_counts = defaultdict(int)
    class_difficulty_counts = {
        'Car': defaultdict(int),
        'Pedestrian': defaultdict(int),
        'Cyclist': defaultdict(int),
    }
    total_objects = 0
    labels_only_unknown = []

    # Process each label file
    for label_file in Path(label_dir).glob('*.txt'):
        file_difficulties = []

        with open(label_file, 'r') as file:
            objects = file.readlines()

        for obj in objects:
            total_objects += 1
            parts = obj.strip().split()

            class_id = int(parts[0])
            class_name = ['Car', 'Pedestrian', 'Cyclist'][class_id-1]

            # Extract difficulty information
            height = float(parts[9])
            truncation = float(parts[10])
            occlusion = float(parts[11])

            # Get difficulty level
            difficulty_info = np.array([[height, truncation, occlusion]], dtype=np.float32)
            difficulty = validator.get_kitti_obj_level(difficulty_info)[0]

            difficulty_counts[difficulty] += 1
            class_difficulty_counts[class_name][difficulty] += 1
            file_difficulties.append(difficulty)
        
        # Check if file contains only unknown objects
        if file_difficulties and all(d == 0 for d in file_difficulties):
            labels_only_unknown.append(label_file.name)

    # Print statistics
    print("\n=== KITTI Difficulty Distribution ===")
    print(f"Total objects analyzed: {total_objects}")

    # Validation checks
    expected = {
        'total': 16907,
        'easy': 4398,
        'moderate': 5929,
        'hard': 3599,
        'unknown': 2981
    }

    print("\n=== Per-Class Difficulty Distribution ===")
    diff_names = {0: 'Unknown', 1: 'Easy', 2: 'Moderate', 3: 'Hard'}
    
    for class_name, difficulties in class_difficulty_counts.items():
        print(f"\n{class_name}:")
        print("-" * 36)
        class_total = sum(difficulties.values())
        for diff_id in range(4):  # 0 to 3
            count = difficulties[diff_id]
            percentage = (count / class_total * 100) if class_total > 0 else 0
            print(f"{diff_names[diff_id]:<10}: {count:<8} ({percentage:.1f}%)")
        print(f"Total      : {class_total}")

    print("\nComparison with detect/val.py:")
    print(f"{'Level':<10} {'Found':<8} {'Expected':<8} {'Diff':<8}")
    print("-" * 36)
    print(f"Easy    : {difficulty_counts[1]:<8} {expected['easy']:<8} {difficulty_counts[1] - expected['easy']}")
    print(f"Moderate: {difficulty_counts[2]:<8} {expected['moderate']:<8} {difficulty_counts[2] - expected['moderate']}")
    print(f"Hard    : {difficulty_counts[3]:<8} {expected['hard']:<8} {difficulty_counts[3] - expected['hard']}")
    print(f"Unknown : {difficulty_counts[0]:<8} {expected['unknown']:<8} {difficulty_counts[0] - expected['unknown']}")
    print(f"Total   : {total_objects:<8} {expected['total']:<8} {total_objects - expected['total']}") 

if __name__ == "__main__":
    label_dir = '/home/heizung1/ultralytics_yolov8-obb_ob_kitti/ultralytics/cfg/datasets/kitti/labels/val'
    analyze_label_difficulties(label_dir)