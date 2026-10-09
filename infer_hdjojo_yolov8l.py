"""
==============================================================================
MAGFiLO SOLAR FILAMENT SEGMENTATION - EXACT INFERENCE PIPELINE
Recreates the ~0.50 - 0.55 Score of hdjojo/solar-filament-seg-inference
==============================================================================
"""

import os
import csv
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools import mask as mask_utils
from ultralytics import YOLO

class Config:
    TEST_IMAGES_DIR = "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    MODEL_PATH = "working/runs/solaris_v2_2048/weights/best.pt"  # or trained YOLOv8l-seg weights
    OUTPUT_CSV = "submission.csv"
    IMG_SIZE = 2048
    CONF = 0.30
    IOU = 0.00
    MAX_DET = 100
    MIN_PIXEL_AREA = 5

def mask_to_rle(binary_mask: np.ndarray) -> str:
    """Encodes a binary mask to Kaggle-compatible Fortran-ordered COCO RLE string."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts

def remove_instance_overlaps(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Priority-based overlap resolution:
    Sorts predicted masks by confidence descending and removes overlapping pixels
    from lower-confidence instances so no two filaments share any pixel.
    """
    if len(masks) == 0:
        return np.array([])
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_masks = masks[sorted_indices]

    accumulated_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
    non_overlapping_masks = []

    for mask in sorted_masks:
        clean_mask = mask.astype(np.uint8) & ~accumulated_mask
        non_overlapping_masks.append(clean_mask)
        accumulated_mask |= mask.astype(np.uint8)

    return np.array(non_overlapping_masks)

def run_inference(cfg: Config):
    print("=" * 60)
    print("RUNNING INFERENCE WITH EXACT hdjojo PIPELINE CONFIG")
    print(f"Model: {cfg.MODEL_PATH}")
    print(f"Conf: {cfg.CONF} | IoU: {cfg.IOU} | ImgSize: {cfg.IMG_SIZE} | MaxDet: {cfg.MAX_DET}")
    print("=" * 60)

    if not os.path.exists(cfg.MODEL_PATH):
        raise FileNotFoundError(f"Model weight file not found at: {cfg.MODEL_PATH}")

    model = YOLO(cfg.MODEL_PATH)

    image_paths = sorted(
        glob.glob(os.path.join(cfg.TEST_IMAGES_DIR, "*.jpeg")) +
        glob.glob(os.path.join(cfg.TEST_IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(cfg.TEST_IMAGES_DIR, "*.png"))
    )

    print(f"Discovered test images: {len(image_paths)}")
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No test images found in {cfg.TEST_IMAGES_DIR}")

    submission_rows = []

    for img_path in tqdm(image_paths, desc="Inference"):
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        result = model.predict(
            img_path,
            imgsz=cfg.IMG_SIZE,
            conf=cfg.CONF,
            iou=cfg.IOU,
            max_det=cfg.MAX_DET,
            verbose=False
        )[0]

        if result.masks is None or len(result.masks) == 0:
            continue

        result_masks_data = result.masks.data.cpu().numpy()
        # Sort by confidence score (robust fix over class index)
        scores = result.boxes.conf.cpu().numpy() if (result.boxes is not None and len(result.boxes.conf) > 0) else np.ones(len(result_masks_data))
        new_masks_data = remove_instance_overlaps(result_masks_data, scores)

        for i, m in enumerate(new_masks_data):
            m_res = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            if m_res.sum() < cfg.MIN_PIXEL_AREA:
                continue
            submission_rows.append({
                "filament_id": f"{file_stem}_{i+1}",
                "segmentation_rle": mask_to_rle(m_res)
            })

    print(f"\nGenerated {len(submission_rows)} filament instances across {len(image_paths)} images.")

    # Save to submission CSV
    with open(cfg.OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(submission_rows)

    print(f"Submission saved to: {cfg.OUTPUT_CSV}")
    print(f"File size: {os.path.getsize(cfg.OUTPUT_CSV) / 1024:.1f} KB")

if __name__ == "__main__":
    cfg = Config()
    # Check if working runs best.pt is available
    if not os.path.exists(cfg.MODEL_PATH) and os.path.exists("working/runs/solaris_v2_2048/weights/best.pt"):
        cfg.MODEL_PATH = "working/runs/solaris_v2_2048/weights/best.pt"
    run_inference(cfg)
