#!/usr/bin/env python3
"""
Clean High-Precision Inference for Solar Filament Segmentation (Kaggle 2026)
-----------------------------------------------------------------------------
Fixes the 0.16 score caused by multi-pass duplicate slivers and restores
high-precision single-pass instance segmentation (mAP50 = 0.6984).
Expected Score: > 0.60 - 0.65+
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools import mask as mask_utils
import torch
from ultralytics import YOLO

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def mask_to_rle(binary_mask):
    """Encodes binary mask into standard Kaggle COCO run-length encoding string."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle["counts"]

def close_mask_gaps(binary_mask, kernel_size=3):
    """Morphological closing with elliptical kernel to seal internal pinholes."""
    if kernel_size <= 1:
        return binary_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

def remove_instance_overlaps(masks, scores):
    """Priority overlap elimination: ensures strictly 0 overlapping pixels."""
    if len(masks) == 0:
        return np.array([])
    sorted_indices = np.argsort(scores)[::-1]
    sorted_masks = masks[sorted_indices]

    accumulated_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
    clean_masks = []
    for mask in sorted_masks:
        m = (mask > 0).astype(np.uint8) & ~accumulated_mask
        clean_masks.append(m)
        accumulated_mask |= (mask > 0).astype(np.uint8)
    return np.array(clean_masks)

# -------------------------------------------------------------------------
# Path Auto-Detection
# -------------------------------------------------------------------------
def find_test_dir():
    candidates = [
        "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        "../filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        "/content/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        "/content/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        "/content/drive/MyDrive/kaggle_SF/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    ]
    for c in candidates:
        if os.path.exists(c) and len(glob.glob(os.path.join(c, "*"))) > 0:
            return os.path.abspath(c)
    # Search recursively in current dir or /content
    for root in [".", "/content", "/content/drive/MyDrive/kaggle_SF"]:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", "test_images"), recursive=True)
            if matches:
                return os.path.abspath(matches[0])
    return None

def find_weights():
    candidates = [
        "working/runs/solaris_v2_2048/weights/best.pt",
        "working_solaris/runs/solaris_v2_2048/weights/best.pt",
        "/content/drive/MyDrive/kaggle_SF/working/runs/solaris_v2_2048/weights/best.pt",
        "/content/drive/MyDrive/kaggle_SF/working_solaris/runs/solaris_v2_2048/weights/best.pt",
        "/content/runs/solaris_v2_2048/weights/best.pt"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Recursive search
    for root in [".", "/content", "/content/drive/MyDrive/kaggle_SF"]:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", "solaris_v2_2048/**/best.pt"), recursive=True)
            if matches:
                return os.path.abspath(matches[0])
    return None

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission_clean_060.csv")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.00)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    weights_path = args.weights or find_weights()
    test_dir = args.test_dir or find_test_dir()

    print("=" * 60)
    print(" Solar Filament Clean Submission Generator")
    print("=" * 60)
    print(f" Model Weights : {weights_path}")
    print(f" Test Images   : {test_dir}")
    print(f" Confidence    : {args.conf}")
    print(f" IoU Deduplication : {args.iou}")
    print(f" Resolution    : {args.imgsz}x{args.imgsz}")
    print(f" Output File   : {args.output}")
    print("=" * 60)

    if not weights_path or not os.path.exists(weights_path):
        print(f"Error: Weights file not found! ({weights_path})")
        sys.exit(1)

    if not test_dir or not os.path.exists(test_dir):
        print(f"Error: Test directory not found! ({test_dir})")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()
    print(f"Loading YOLO model on {device} (half={half})...")
    model = YOLO(weights_path)

    test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")))
    if not test_images:
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png")))

    if args.limit:
        test_images = test_images[:args.limit]

    print(f"Processing {len(test_images)} test images...")
    submission_rows = []

    for img_path in tqdm(test_images, desc="Predicting Clean Filaments"):
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]

        # Single-pass prediction with strict NMS deduplication
        res = model.predict(
            orig_img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=100,
            device=device,
            verbose=False
        )[0]

        if res.masks is None:
            continue

        masks_data = res.masks.data.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else np.ones(len(masks_data))

        # Priority overlap removal
        clean_masks = remove_instance_overlaps(masks_data, scores)

        accumulated_final = np.zeros((h, w), dtype=np.uint8)
        inst_idx = 1

        for m in clean_masks:
            m_res = cv2.resize((m > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            m_curr = m_res & ~accumulated_final
            if m_curr.sum() < args.min_area:
                continue

            if args.morph_kernel > 1:
                m_closed = close_mask_gaps(m_curr, kernel_size=args.morph_kernel)
                m_closed = m_closed & ~accumulated_final
            else:
                m_closed = m_curr

            if m_closed.sum() < args.min_area:
                continue

            accumulated_final |= m_closed
            rle_str = mask_to_rle(m_closed)

            submission_rows.append({
                "filament_id": f"{file_stem}_{inst_idx}",
                "segmentation_rle": rle_str
            })
            inst_idx += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_out = pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"])
    df_out.to_csv(args.output, index=False)
    # Also save as submission.csv
    df_out.to_csv("submission.csv", index=False)

    print("\n" + "=" * 60)
    print(f" SUCCESS! Generated: {args.output} (and submission.csv)")
    print(f" Total Filament Instances: {len(df_out)}")
    images_with_filaments = df_out['filament_id'].apply(lambda x: '_'.join(x.split('_')[:-1])).nunique()
    print(f" Images with Filaments   : {images_with_filaments} / {len(test_images)}")
    avg_per_img = len(df_out) / max(images_with_filaments, 1)
    print(f" Average Instances/Image : {avg_per_img:.1f} (Ground Truth is ~5-10)")
    print("=" * 60)

if __name__ == "__main__":
    main()
