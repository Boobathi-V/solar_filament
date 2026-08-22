import os
import json
import shutil
import random
from collections import defaultdict

def prepare_yolo_dataset():
    # Paths (dynamically resolved to work on both local laptop and Colab/Kaggle)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coco_json_path = os.path.join(base_dir, "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    train_images_src = os.path.join(base_dir, "MAGFiLO_1.0_Kaggle_2026", "train", "train_images")
    yolo_dataset_dir = os.path.join(base_dir, "yolo_dataset")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Loading COCO annotations...")
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
        
    # Categories mapping (COCO class_id starts at 1, YOLO expects 0-indexed classes)
    categories = coco_data["categories"]
    print("Categories found in dataset:")
    for cat in categories:
        print(f"  ID: {cat['id']} - Name: {cat['name']}")
    
    # Organize annotations by image_id
    img_id_to_annots = defaultdict(list)
    for annot in coco_data["annotations"]:
        img_id_to_annots[annot["image_id"]].append(annot)
        
    # Map image ID to metadata
    images_metadata = coco_data["images"]
    print(f"Total images declared in JSON: {len(images_metadata)}")
    
    # Shuffle and split into train (85%) and validation (15%)
    random.shuffle(images_metadata)
    split_idx = int(len(images_metadata) * 0.85)
    train_metadata = images_metadata[:split_idx]
    val_metadata = images_metadata[split_idx:]
    
    print(f"Splitting: {len(train_metadata)} train images, {len(val_metadata)} validation images")
    
    # Create directories for YOLO
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset_dir, "labels", split), exist_ok=True)
        
    # Function to process and copy files for a split
    def process_split(metadata_list, split_name):
        copied_count = 0
        annotated_count = 0
        
        for img_info in metadata_list:
            img_id = img_info["id"]
            filename = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]
            
            src_img_path = os.path.join(train_images_src, filename)
            if not os.path.exists(src_img_path):
                # Print warning but continue (sometimes names differ slightly or some images are missing)
                print(f"Warning: Image file not found: {src_img_path}")
                continue
                
            # Target paths
            dest_img_path = os.path.join(yolo_dataset_dir, "images", split_name, filename)
            dest_label_path = os.path.join(yolo_dataset_dir, "labels", split_name, os.path.splitext(filename)[0] + ".txt")
            
            # Copy image
            shutil.copy(src_img_path, dest_img_path)
            copied_count += 1
            
            if copied_count % 50 == 0:
                print(f"[{split_name}] Processed {copied_count}/{len(metadata_list)} images...")
            
            # Process annotations
            annots = img_id_to_annots[img_id]
            label_lines = []
            
            for annot in annots:
                cat_id = annot["category_id"]
                yolo_class_idx = cat_id - 1  # Map 1-4 to 0-3
                
                # YOLO format for segmentation: class_id x1 y1 x2 y2 ... (all normalized)
                segmentations = annot.get("segmentation", [])
                for seg in segmentations:
                    if len(seg) < 6:
                        continue # A valid polygon needs at least 3 points (6 coords)
                        
                    normalized_coords = []
                    for i, coord in enumerate(seg):
                        if i % 2 == 0:
                            # X coordinate
                            normalized_coords.append(coord / width)
                        else:
                            # Y coordinate
                            normalized_coords.append(coord / height)
                            
                    coord_str = " ".join(f"{c:.6f}" for c in normalized_coords)
                    label_lines.append(f"{yolo_class_idx} {coord_str}")
            
            # Write labels (even if empty, creating empty files helps YOLO understand background samples)
            with open(dest_label_path, "w") as label_file:
                if label_lines:
                    label_file.write("\n".join(label_lines) + "\n")
                    annotated_count += 1
                    
        print(f"Split [{split_name}]: Copied {copied_count} images, wrote labels for {annotated_count} annotated images")

    # Run splits
    process_split(train_metadata, "train")
    process_split(val_metadata, "val")
    
    # Create dataset.yaml
    yaml_content = f"""# YOLOv8 Solar Filament Dataset Config
path: {yolo_dataset_dir.replace('\\\\', '/').replace('\\', '/')}
train: images/train
val: images/val

# Classes
names:
  0: Left
  1: Right
  2: Unidentifiable
  3: Ambiguous
"""
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"\nDataset configuration file written to: {yaml_path}")
    print("Dataset preparation complete!")

if __name__ == "__main__":
    prepare_yolo_dataset()
