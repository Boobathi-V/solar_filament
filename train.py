import os
import sys
import subprocess

def check_and_install_requirements():
    print("Checking dependencies...")
    try:
        import ultralytics
        print("Ultralytics library is already installed.")
    except ImportError:
        print("Ultralytics is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("Ultralytics installed successfully.")

def run_training():
    from ultralytics import YOLO
    
    # Paths (dynamically resolved to work on both local laptop and Colab/Kaggle)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"Error: Dataset configuration file not found at {yaml_path}")
        print("Please run prepare_data.py first.")
        sys.exit(1)
        
    # Check if GPU is available (e.g. on Google Colab or Kaggle)
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading YOLOv8-segmentation model...")
    # Use yolov8n-seg.pt (nano) for CPU speed. If GPU is available later, 
    # you can switch to yolov8s-seg or yolov8m-seg for higher accuracy.
    model = YOLO("yolov8l-seg.pt")
    
    # Regularization & Augmentation configurations to prevent overfitting/underfitting:
    # 1. Flip horizontal & vertical (fliplr & flipud) because solar images are orientation-invariant
    # 2. Rotation, scaling, translation, mosaic, and mixup augmentation
    # 3. HSV color/brightness jitter to handle telescope camera exposure variance
    # 4. Early stopping validation patience of 20 epochs
    # 5. Weight decay (L2 regularization) of 0.0005
    # 6. AdamW optimizer for better convergence
    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=100,            # Increased to 100 epochs for better convergence
        imgsz=1024,            # Increased resolution from 640 to 1024 to preserve thin filament details
        batch=8,               # Batch size (reduce to 4 or 2 if GPU Out Of Memory occurs)
        workers=2,             # Avoid multithreading overhead on Windows CPUs
        device=device,         # Use GPU if available, fallback to CPU
        patience=20,           # Stop training if validation mAP doesn't improve for 20 epochs (prevent overfitting)
        project=os.path.join(base_dir, "output"),
        name="solar_filament_yolov8",
        optimizer="AdamW",     # Use AdamW optimizer for robust segmentation training
        # Augmentations & Regularization
        flipud=0.5,            # vertical flip
        fliplr=0.5,            # horizontal flip
        degrees=15.0,          # random rotation
        scale=0.5,             # random scaling
        translate=0.1,         # random translation
        mosaic=1.0,            # mosaic augmentation
        mixup=0.15,            # mixup augmentation to regularize masks
        hsv_h=0.015,           # color jitter (hue)
        hsv_s=0.7,             # color jitter (saturation)
        hsv_v=0.4,             # color jitter (value/brightness)
        weight_decay=0.0005,   # L2 weight regularization
    )
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    check_and_install_requirements()
    run_training()
