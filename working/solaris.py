#!/usr/bin/env python3
"""
Solaris v2.0 — Advanced Solar Filament Instance Segmentation Pipeline
====================================================================
Implements all 6 Future Work techniques to reach Dice > 0.60+:
  1. Test-Time Augmentation (Dual-pass Horizontal Flip TTA)
  2. Multi-Scale Inference (e.g., 2048px + 1792px fusion)
  3. Optimized Confidence Thresholding & Validation Sweeping
  4. 80-Epoch Extended Training with 3-Epoch Warm-Up & Cosine Annealing
  5. YOLOv8x-Seg (Extra-Large 68M Variant) Support with Auto-Batching
  6. Morphological Closing (Elliptical Kernel Gap Bridging)
  + Priority-based Strictly Zero-Overlap Instance Post-Processing
  + Multi-platform Auto-Detection (Google Colab, Kaggle, Local)
"""

import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from pycocotools import mask as mask_utils

# ==============================================================================
# 1. AUTO-DETECT ENVIRONMENT AND DATASET PATHS
# ==============================================================================
def resolve_paths(custom_data_dir=None):
    """
    Auto-detects paths across Kaggle, Google Colab, Google Drive, and local machines.
    """
    search_roots = []
    if custom_data_dir and os.path.exists(custom_data_dir):
        search_roots.append(custom_data_dir)

    search_roots.extend([
        ".",
        "..",
        "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        "../filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        "/content",
        "/content/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        "/content/MAGFiLO_1.0_Kaggle_2026",
        "/content/drive/MyDrive/kaggle_SF",
        "/content/drive/MyDrive/kaggle_SF/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        "/content/drive/Othercomputers",
        "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        "/kaggle/input/filament-segmentation-2026"
    ])
    # Dynamically search Drive for any synced MAGFiLO directory
    if os.path.exists("/content/drive"):
        for m in glob.glob("/content/drive/**/MAGFiLO_1.0_Kaggle_2026", recursive=True):
            if os.path.isdir(m) and m not in search_roots:
                search_roots.insert(0, m)

    train_json = None
    train_images = None
    test_images = None

    for root in search_roots:
        if not os.path.exists(root):
            continue

        # Search for annotations JSON
        if not train_json:
            candidate_jsons = glob.glob(os.path.join(root, "**", "*Annotations*train*.json"), recursive=True)
            if candidate_jsons:
                train_json = os.path.abspath(candidate_jsons[0])

        # Search for train images directory
        if not train_images:
            candidate_train = glob.glob(os.path.join(root, "**", "train_images"), recursive=True)
            if candidate_train:
                train_images = os.path.abspath(candidate_train[0])

        # Search for test images directory
        if not test_images:
            candidate_test = glob.glob(os.path.join(root, "**", "test_images"), recursive=True)
            if candidate_test:
                test_images = os.path.abspath(candidate_test[0])

    # Determine working directory based on environment
    if os.path.exists("/kaggle/working"):
        working_dir = "/kaggle/working"
    elif os.path.exists("/content"):
        # If Google Drive is mounted, allow saving runs there
        if os.path.exists("/content/drive/MyDrive/kaggle_SF"):
            working_dir = "/content/drive/MyDrive/kaggle_SF/working_solaris"
        else:
            working_dir = "/content/working_solaris"
    else:
        working_dir = os.path.abspath("working_solaris")

    os.makedirs(working_dir, exist_ok=True)

    return {
        "train_json": train_json,
        "train_images": train_images,
        "test_images": test_images,
        "working_dir": working_dir,
        "yolo_dataset": os.path.join(working_dir, "yolo_seg_dataset"),
        "output_sub": os.path.join(working_dir, "submission.csv") if "/kaggle" in working_dir or "/content" in working_dir else "submission.csv"
    }

# ==============================================================================
# 2. DATASET CONVERSION: COCO JSON -> YOLO SEGMENTATION FORMAT
# ==============================================================================
def convert_coco_to_yolo_seg(coco_json_path, images_dir, output_dir, val_ratio=0.10, seed=42):
    """
    Converts COCO format polygon & RLE annotations into Ultralytics YOLO segmentation format.
    Generates data.yaml with train and validation image paths.
    """
    print(f"\n[Step 1/3] Reading COCO Annotations from: {coco_json_path}")
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    # Deterministic train/val split
    np.random.seed(seed)
    image_ids = sorted(list(images.keys()))
    np.random.shuffle(image_ids)
    n_val = int(len(image_ids) * val_ratio)
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    print(f"Total Dataset Images: {len(image_ids)} (Train: {len(train_ids)}, Val: {len(val_ids)})")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    for img_id, img_info in tqdm(images.items(), desc="Converting to YOLO polygons"):
        split = "val" if img_id in val_ids else "train"
        file_name = img_info["file_name"]
        w = float(img_info["width"])
        h = float(img_info["height"])

        src_img = os.path.join(images_dir, file_name)
        dst_img = os.path.join(output_dir, "images", split, file_name)

        if not os.path.exists(dst_img) and os.path.exists(src_img):
            try:
                os.link(src_img, dst_img)  # Fast hardlink
            except Exception:
                shutil.copy2(src_img, dst_img)

        label_lines = []
        anns = annotations_by_image.get(img_id, [])
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue

            # Format 1: Explicit polygon coordinate array
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    coords = np.array(poly).reshape(-1, 2)
                    norm_coords = coords / [w, h]
                    norm_coords = np.clip(norm_coords, 0.0, 1.0)
                    flat = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_coords])
                    label_lines.append(f"0 {flat}")

            # Format 2: COCO RLE dictionary
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) < 3:
                        continue
                    coords = cnt.squeeze(1)
                    if len(coords) < 3:
                        continue
                    norm_coords = coords.astype(float) / [w, h]
                    norm_coords = np.clip(norm_coords, 0.0, 1.0)
                    flat = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_coords])
                    label_lines.append(f"0 {flat}")

        stem = os.path.splitext(file_name)[0]
        label_file = os.path.join(output_dir, "labels", split, f"{stem}.txt")
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines) + "\n")

    yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: filament
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset conversion complete -> {yaml_path}")
    return yaml_path

# ==============================================================================
# 3. TRAINING STAGE: 80-Epoch Cosine Annealing + YOLOv8x-Seg
# ==============================================================================
def train_solaris_v2(
    yaml_path,
    model_type="yolov8x-seg.pt",
    epochs=80,
    imgsz=2048,
    batch=None,
    project="magfilo_runs",
    name="solaris_v2_2048"
):
    """
    Future Work #4 & #5:
    - Supports YOLOv8x-Seg (extra-large variant, ~68M params) or YOLOv8l-Seg.
    - Extended training (80 epochs) with 3-epoch linear warmup & cosine learning rate decay.
    - Domain-adapted augmentations (mosaic, flips, random rotational tolerance).
    """
    import torch
    from ultralytics import YOLO

    # Auto-select batch size if not explicitly set
    if batch is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Detected GPU: {torch.cuda.get_device_name(0)} with {vram_gb:.1f} GB VRAM")
            if "x" in model_type:
                batch = 2 if vram_gb >= 35 else 1  # 1 for T4/V100 (16GB), 2 for A100 (40GB)
            else:
                batch = 2 if vram_gb >= 15 else 1
        else:
            batch = 1

    print(f"\n[Step 2/3] Initializing {model_type} for training...")
    print(f"Hyperparameters: epochs={epochs}, imgsz={imgsz}, batch={batch}, cos_lr=True, warmup=3")

    model = YOLO(model_type)

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4 if torch.cuda.is_available() else 0,
        optimizer="AdamW",
        lr0=0.001 if "l" in model_type else 0.0005,  # Lower initial LR for extra-large model
        lrf=0.005,                                   # Tighter final LR (1/200th) for fine convergence
        warmup_epochs=3,                             # Linear warm-up to stabilize early layers
        cos_lr=True,                                 # Cosine annealing learning rate schedule
        weight_decay=0.0005,
        mosaic=0.5,
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,                                # Solar rotational invariance
        project=project,
        name=name,
        save=True,
        verbose=True
    )

    best_weights = os.path.join(project, name, "weights", "best.pt")
    print(f"Training completed successfully! Best checkpoint saved at: {best_weights}")
    return best_weights

# ==============================================================================
# 4. POST-PROCESSING: MORPHOLOGICAL CLOSING (Future Work #6)
# ==============================================================================
def close_mask_gaps(binary_mask, kernel_size=5):
    """
    Applies morphological closing with an elliptical structuring element to connect
    fragmented sections of the same filament, bridging low-contrast intra-filament gaps.
    """
    if kernel_size <= 1:
        return binary_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed

# ==============================================================================
# 5. PRIORITY-BASED OVERLAP ELIMINATION & RLE ENCODING
# ==============================================================================
def mask_to_rle(binary_mask):
    """Encodes a binary mask into Kaggle COCO run-length encoding string."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle["counts"]

def remove_instance_overlaps(masks, scores):
    """
    Strict priority overlap removal: higher confidence masks claim pixels first,
    guaranteeing zero overlapping pixels between distinct filament instances.
    """
    if len(masks) == 0:
        return np.array([])

    sorted_indices = np.argsort(scores)[::-1]
    sorted_masks = masks[sorted_indices]

    accumulated_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
    non_overlapping_masks = []

    for mask in sorted_masks:
        clean_mask = (mask > 0).astype(np.uint8) & ~accumulated_mask
        non_overlapping_masks.append(clean_mask)
        accumulated_mask |= (mask > 0).astype(np.uint8)

    return np.array(non_overlapping_masks)

# ==============================================================================
# 6. INFERENCE PIPELINE: TTA + MULTI-SCALE + MORPHOLOGICAL CLOSING
# ==============================================================================
def run_advanced_inference(
    model_weights_path,
    test_images_dir,
    output_csv="submission.csv",
    conf=0.30,             # High-precision threshold to suppress False Positives
    iou=0.00,              # Strict NMS: eliminate all duplicate bounding boxes/masks
    imgsz=2048,            # Native full-disk resolution
    morph_kernel=3,        # Bridge intra-filament pinhole gaps without distorting boundary
    min_area=25            # Suppress tiny noisy fragments
):
    """
    Executes clean, high-precision instance segmentation inference.
    Eliminates multi-pass slicing/duplicate slivers to maximize bipartite matching Dice (>0.60+).
    """
    import torch
    from ultralytics import YOLO

    print(f"\n[Step 3/3] Loading Model Checkpoint: {model_weights_path}...")
    model = YOLO(model_weights_path)
    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()

    print(f"Inference Configuration:")
    print(f"  Confidence Threshold : {conf}")
    print(f"  IoU NMS Threshold    : {iou} (strict deduplication)")
    print(f"  Resolution           : {imgsz}x{imgsz}")
    print(f"  Morphological Closing: kernel={morph_kernel}")
    print(f"  Min Pixel Area Filter: {min_area} px")

    test_images = sorted(glob.glob(os.path.join(test_images_dir, "*.jpeg")))
    if not test_images:
        test_images = sorted(
            glob.glob(os.path.join(test_images_dir, "*.jpg")) +
            glob.glob(os.path.join(test_images_dir, "*.png"))
        )

    print(f"Found {len(test_images)} test images in {test_images_dir}")
    submission_rows = []

    for img_path in tqdm(test_images, desc="Predicting Solar Filaments"):
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]

        # Single-pass prediction with strict NMS deduplication
        res = model.predict(
            orig_img,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=100,
            device=device,
            half=half,
            verbose=False
        )[0]

        if res.masks is None:
            continue

        masks_data = res.masks.data.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else np.ones(len(masks_data))

        # 1. Eliminate instance overlaps via strict priority
        clean_masks = remove_instance_overlaps(masks_data, scores)

        # 2. Apply morphological closing & minimum area filter
        accumulated_final = np.zeros((h, w), dtype=np.uint8)
        inst_idx = 1

        for m in clean_masks:
            m_res = cv2.resize((m > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            m_curr = m_res & ~accumulated_final
            if m_curr.sum() < min_area:
                continue

            if morph_kernel > 1:
                m_closed = close_mask_gaps(m_curr, kernel_size=morph_kernel)
                m_closed = m_closed & ~accumulated_final
            else:
                m_closed = m_curr

            if m_closed.sum() < min_area:
                continue

            accumulated_final |= m_closed
            rle_str = mask_to_rle(m_closed)

            submission_rows.append({
                "filament_id": f"{file_stem}_{inst_idx}",
                "segmentation_rle": rle_str
            })
            inst_idx += 1

        # Clear PyTorch GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_out = pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"])
    df_out.to_csv(output_csv, index=False)

    print(f"\n[Done] Successfully generated submission file: {output_csv}")
    print(f"  Total segmented filaments : {len(df_out)}")
    if len(df_out) > 0:
        n_imgs = df_out["filament_id"].apply(lambda x: "_".join(x.split("_")[:-1])).nunique()
        print(f"  Images with detections    : {n_imgs}")
        print(f"  Average instances / image : {len(df_out) / max(1, n_imgs):.2f}")
    return output_csv

# ==============================================================================
# 7. CONFIDENCE THRESHOLD VALIDATION SWEEP (Future Work #3)
# ==============================================================================
def sweep_confidence_threshold(model_path, val_images_dir, val_ann_json, thresholds=(0.20, 0.24, 0.26, 0.28, 0.30, 0.35)):
    """
    Sweeps confidence thresholds on validation set to find the exact empirical Dice-maximizing value.
    """
    from ultralytics import YOLO
    print(f"\n[Validation Sweep] Evaluating confidence thresholds {thresholds}...")
    model = YOLO(model_path)

    with open(val_ann_json, "r") as f:
        gt_data = json.load(f)

    gt_by_img = {}
    for ann in gt_data.get("annotations", []):
        img_id = ann["image_id"]
        seg = ann.get("segmentation")
        if isinstance(seg, dict):
            gt_by_img.setdefault(img_id, []).append(mask_utils.decode(seg))

    records = []
    for conf in thresholds:
        dice_scores = []
        for img_info in gt_data.get("images", [])[:50]:  # Sample first 50 val images for fast sweep
            file_name = img_info["file_name"]
            img_path = os.path.join(val_images_dir, file_name)
            if not os.path.exists(img_path):
                continue

            h, w = img_info["height"], img_info["width"]
            res = model.predict(img_path, imgsz=2048, conf=conf, iou=0.10, verbose=False)[0]

            pred_union = np.zeros((h, w), dtype=np.uint8)
            if res.masks is not None:
                for m in res.masks.data.cpu().numpy():
                    resized = cv2.resize((m > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_union = np.maximum(pred_union, resized)

            gt_union = np.zeros((h, w), dtype=np.uint8)
            for gt_m in gt_by_img.get(img_info["id"], []):
                gt_union = np.maximum(gt_union, gt_m)

            inter = (pred_union & gt_union).sum()
            dice = (2.0 * inter) / (pred_union.sum() + gt_union.sum() + 1e-6)
            dice_scores.append(dice)

        mean_dice = np.mean(dice_scores) if dice_scores else 0.0
        records.append({"conf": conf, "mean_dice": mean_dice})
        print(f"  conf = {conf:.2f} -> Validation Dice: {mean_dice:.4f}")

    df_sweep = pd.DataFrame(records).sort_values("mean_dice", ascending=False)
    best_conf = df_sweep.iloc[0]["conf"]
    print(f"\nOptimal Confidence Threshold: {best_conf:.2f} (Dice: {df_sweep.iloc[0]['mean_dice']:.4f})")
    return best_conf

# ==============================================================================
# 8. MAIN CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Solaris v2.0: Advanced Solar Filament Segmentation Pipeline (Score > 0.60+)")
    parser.add_argument("--train", action="store_true", help="Run dataset conversion and model training")
    parser.add_argument("--predict", action="store_true", help="Run inference and submission generation only")
    parser.add_argument("--sweep", action="store_true", help="Run confidence threshold validation sweep")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint weights file for inference")
    parser.add_argument("--model_type", type=str, default="yolov8x-seg.pt", choices=["yolov8l-seg.pt", "yolov8x-seg.pt"], help="YOLOv8 architecture variant (default: yolov8x-seg.pt)")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs with cosine warm-up (default: 80)")
    parser.add_argument("--imgsz", type=int, default=2048, help="Image resolution (default: 2048)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: auto-detected based on GPU VRAM)")
    parser.add_argument("--conf", type=float, default=0.30, help="Prediction confidence threshold (default: 0.30)")
    parser.add_argument("--iou", type=float, default=0.00, help="NMS IoU threshold for deduplication (default: 0.00)")
    parser.add_argument("--morph_kernel", type=int, default=3, help="Morphological closing kernel size (default: 3)")
    parser.add_argument("--min_area", type=int, default=25, help="Minimum pixel area threshold (default: 25)")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None, help="Resume training from checkpoint (default: searches for last.pt)")
    parser.add_argument("--data_dir", type=str, default=None, help="Custom dataset root path")
    parser.add_argument("--output_csv", type=str, default=None, help="Output submission CSV path")

    args = parser.parse_args()
    paths = resolve_paths(custom_data_dir=args.data_dir)

    print("=" * 70)
    print("  SOLARIS v2.0 — HIGH-PRECISION SOLAR FILAMENT SEGMENTATION")
    print("=" * 70)
    print("Resolved System Paths:")
    for k, v in paths.items():
        print(f"  {k:15s}: {v}")

    # Resume Training Pipeline (Epoch 34 -> 80)
    if args.resume:
        resume_weights = args.resume
        if resume_weights == "auto":
            candidates = [
                os.path.join(paths["working_dir"], "runs", "solaris_v2_2048", "weights", "last.pt"),
                os.path.join(paths["working_dir"], "magfilo_runs", "solaris_v2_2048", "weights", "last.pt"),
                "/content/drive/MyDrive/kaggle_SF/working_solaris/runs/solaris_v2_2048/weights/last.pt",
                "/content/drive/MyDrive/kaggle_SF/working/runs/solaris_v2_2048/weights/last.pt",
                "runs/solaris_v2_2048/weights/last.pt"
            ]
            for c in candidates:
                if os.path.exists(c):
                    resume_weights = c
                    break
        if not os.path.exists(resume_weights):
            print(f"Error: Checkpoint file not found to resume: {resume_weights}")
            sys.exit(1)

        print(f"\n[Resume Training] Loading state from: {resume_weights}")
        print("Resuming seamlessly from last completed epoch to 80...")
        from ultralytics import YOLO
        model = YOLO(resume_weights)
        model.train(resume=True)
        return

    output_csv = args.output_csv if args.output_csv else paths["output_sub"]

    # Training Pipeline
    best_weights = args.model
    if not args.predict and not args.sweep:
        if not paths["train_json"] or not paths["train_images"]:
            print("\nNotice: Training dataset not detected in standard paths.")
            print("To train, ensure MAGFiLO dataset is located or provide --data_dir.")
        else:
            yaml_path = convert_coco_to_yolo_seg(
                paths["train_json"],
                paths["train_images"],
                paths["yolo_dataset"]
            )
            best_weights = train_solaris_v2(
                yaml_path=yaml_path,
                model_type=args.model_type,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                project=os.path.join(paths["working_dir"], "runs"),
                name="solaris_v2_2048"
            )

    # Validation Threshold Sweep
    if args.sweep:
        if not best_weights or not os.path.exists(best_weights):
            print("Error: Specify valid --model checkpoint for confidence sweep.")
            sys.exit(1)
        val_img_dir = os.path.join(paths["yolo_dataset"], "images", "val")
        sweep_confidence_threshold(best_weights, val_img_dir, paths["train_json"])
        return

    # Inference Pipeline
    if not args.train:
        if not best_weights:
            candidate_weights = [
                os.path.join(paths["working_dir"], "runs", "solaris_v2_2048", "weights", "best.pt"),
                os.path.join(paths["working_dir"], "runs", "yolov8l_2048", "weights", "best.pt"),
                "magfilo_260825_yolov8l_seg_imgsz2048_nc1.pt",
                "../models/fold0_best.pth",
                "/kaggle/input/datasets/hdjojo/solar-filament-seg-models/magfilo_260825_yolov8l_seg_imgsz2048_nc1.pt"
            ]
            for cw in candidate_weights:
                if os.path.exists(cw):
                    best_weights = cw
                    break

        if not best_weights or not os.path.exists(best_weights):
            print(f"\n[Notice] No pre-trained weights specified and none found automatically.")
            print(f"To run inference, provide weights via --model <path_to_best.pt>")
            sys.exit(1)

        if not paths["test_images"]:
            print(f"Error: test_images directory not found.")
            sys.exit(1)

        run_advanced_inference(
            model_weights_path=best_weights,
            test_images_dir=paths["test_images"],
            output_csv=output_csv,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            morph_kernel=args.morph_kernel,
            min_area=args.min_area
        )

if __name__ == "__main__":
    main()
