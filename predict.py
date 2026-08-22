import os
import cv2
import numpy as np
import pandas as pd
import sys
import subprocess

def check_and_install_requirements():
    print("Checking dependencies...")
    # Check ultralytics
    try:
        import ultralytics
        print("Ultralytics library is already installed.")
    except ImportError:
        print("Ultralytics is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("Ultralytics installed successfully.")
    
    # Check pycocotools
    try:
        import pycocotools
        print("pycocotools library is already installed.")
    except ImportError:
        print("pycocotools is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools"])
        print("pycocotools installed successfully.")

def mask_to_rle(binary_mask):
    """
    Converts a binary mask array to COCO Run-Length Encoding (RLE) string.
    """
    from pycocotools import mask as mask_utils
    # Ensure mask is in Fortran-contiguous order as required by pycocotools
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    # Encode mask
    rle = mask_utils.encode(fortran_mask)
    # Decode bytes counts to utf-8 string
    return rle["counts"].decode("utf-8")

def run_prediction():
    from ultralytics import YOLO
    # Paths (dynamically resolved to work on both local laptop and Colab/Kaggle)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_images_dir = os.path.join(base_dir, "MAGFiLO_1.0_Kaggle_2026", "test", "test_images")
    submission_path = os.path.join(base_dir, "submission.csv")
    
    # Path to trained model weights (dynamically search for the latest version run)
    trained_model_path = os.path.join(base_dir, "output", "solar_filament_yolov8", "weights", "best.pt")
    
    output_parent = os.path.join(base_dir, "output")
    if os.path.exists(output_parent):
        versions = []
        for folder in os.listdir(output_parent):
            if folder.startswith("solar_filament_yolov8"):
                path_to_check = os.path.join(output_parent, folder, "weights", "best.pt")
                if os.path.exists(path_to_check):
                    parts = folder.split("-")
                    ver = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    versions.append((ver, path_to_check))
        if versions:
            versions.sort(reverse=True)
            trained_model_path = versions[0][1]
            
    # Check device (automatically use GPU on Colab/Kaggle, fallback to CPU)
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    if os.path.exists(trained_model_path):
        print(f"Loading trained model weights from: {trained_model_path}")
        model = YOLO(trained_model_path)
    else:
        print(f"Warning: Trained model weights not found at {trained_model_path}")
        print("Falling back to pre-trained yolov8n-seg.pt for testing/dry-run...")
        model = YOLO("yolov8n-seg.pt")
        
    # Get all test images
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    print(f"Found {len(test_images)} test images to predict.")
    
    submission_rows = []
    
    for idx, img_name in enumerate(test_images):
        img_path = os.path.join(test_images_dir, img_name)
        image_id = os.path.splitext(img_name)[0]
        
        # Load image to verify shape
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        orig_h, orig_w = img.shape[:2]  # Should be 2048 x 2048
        
        # Run inference
        results = model.predict(img, conf=0.25, imgsz=1024, device=device, verbose=False)
        
        instance_counter = 0
        for r in results:
            masks = r.masks
            
            if masks is not None and len(masks) > 0:
                # Initialize seen_mask for this image to track and prevent overlapping pixels
                seen_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for mask_tensor in masks.data:
                    mask_arr = mask_tensor.cpu().numpy()
                    
                    # Resize mask back to original resolution if shapes differ
                    if mask_arr.shape != (orig_h, orig_w):
                        mask_arr = cv2.resize(mask_arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        
                    # Binarize the mask (convert values to 0 or 1)
                    binary_mask = (mask_arr > 0.5).astype(np.uint8)
                    
                    # Remove overlapping pixels already claimed by higher-confidence masks
                    binary_mask[seen_mask > 0] = 0
                    
                    # Skip if the mask is empty after resolving overlaps
                    if np.sum(binary_mask) == 0:
                        continue
                        
                    # Update seen_mask with new claimed pixels
                    seen_mask[binary_mask > 0] = 1
                    
                    # Compute COCO RLE
                    rle = mask_to_rle(binary_mask)
                    
                    instance_counter += 1
                    filament_id = f"{image_id}_{instance_counter}"
                    
                    submission_rows.append({
                        "filament_id": filament_id,
                        "segmentation_rle": rle
                    })

                
        if (idx + 1) % 20 == 0 or (idx + 1) == len(test_images):
            print(f"Processed {idx + 1}/{len(test_images)} images...")
            
    # Create DataFrame and save to CSV
    df = pd.DataFrame(submission_rows)
    df.to_csv(submission_path, index=False)
    print(f"\nPredictions complete! Submission CSV saved to: {submission_path}")
    print(f"Total prediction records: {len(df)}")

if __name__ == "__main__":
    check_and_install_requirements()
    run_prediction()
