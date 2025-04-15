from ultralytics.models.yolo.obb.val import OBBValidatorCustom
import numpy as np

def test_difficulty_levels():
    """Test difficulty level assignment for specific objects"""
    validator = OBBValidatorCustom()
    
    # Test objects from bev_val_000006.txt
    test_objects = [
        # Format: [height, truncation, occlusion]
        [24.09, 0.00, 2],  # Object 1
        [41.81, 0.00, 0],  # Object 2
        [62.31, 0.00, 0],  # Object 3
        [34.51, 0.00, 1],  # Object 4
    ]
    
    for i, obj in enumerate(test_objects, 1):
        difficulty_info = np.array([obj])
        level = validator.get_kitti_obj_level(difficulty_info)[0]
        
        print(f"Object {i}:")
        print(f"  Height: {obj[0]:.2f}")
        print(f"  Truncation: {obj[1]:.2f}")
        print(f"  Occlusion: {int(obj[2])}")
        print(f"  Difficulty Level: {level}")
        print(f"  Category: {['Unknown', 'Easy', 'Moderate', 'Hard'][level]}\n")

if __name__ == "__main__":
    test_difficulty_levels()