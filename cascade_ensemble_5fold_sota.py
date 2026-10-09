"""
==============================================================================
MAGFiLO Solar Filament Segmentation - SOTA Dual-Engine Cascade Pipeline
Model 1: High-Recall Full-Disk Scout Detector (YOLO 2048px, conf=0.18)
Model 2: 5-Fold U-Net++ Ensemble (EfficientNet-B4 + scSE Attention Refiner)
Target: >1,400 Instances Across Test Set | Expected Score: >= 0.55 - 0.70+
==============================================================================
"""

import os
import sys
import csv
import math
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_utils

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class CascadeConfig:
    TEST_IMAGES_DIR = "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    DETECTOR_PATH = "working/runs/solaris_v2_2048/weights/best.pt"
    MODELS_DIR = "models"
    OUTPUT_CSV = "submission_cascade_sota.csv"

    # Detector settings (tuned for high recall -> >1,400 candidate instances)
    DET_IMGSZ = 2048
    DET_CONF = 0.18
    DET_IOU = 0.40

    # Refiner settings (5-Fold EfficientNet-B4 U-Net++ scSE)
    REFINE_CROP = 512
    CONTEXT_RATIO = 1.50
    MIN_CROP = 96
    REFINE_THR = 0.38
    MIN_PIXEL_AREA = 25
    USE_TTA = True

def mask_to_coco_rle(binary_mask: np.ndarray) -> str:
    """Encodes binary mask (H, W) into Fortran-ordered COCO compressed RLE."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts

def remove_overlaps_by_score(scored_masks):
    """
    Sorts predicted masks by confidence descending and removes overlapping pixels
    from lower-confidence instances so no two filaments share any pixel.
    """
    if len(scored_masks) == 0:
        return []
    
    scored_masks.sort(key=lambda x: -x[0])
    canvas_occupied = np.zeros_like(scored_masks[0][1], dtype=bool)
    clean_instances = []

    for conf, mask in scored_masks:
        clean_mask = mask.copy()
        clean_mask[canvas_occupied] = 0
        if clean_mask.sum() < 20:
            continue
        canvas_occupied |= clean_mask.astype(bool)
        clean_instances.append(clean_mask)

    return clean_instances

# ==============================================================================
# DUAL-ENGINE CASCADE RUNNER
# ==============================================================================
def run_cascade_inference(cfg: CascadeConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🚀 INITIALIZING SOTA DUAL-ENGINE CASCADE PIPELINE")
    print(f"Device: {device}")
    print(f"Assistance Model (Detector): {cfg.DETECTOR_PATH}")
    print(f"Deep Refiners: 5-Fold UnetPlusPlus (EfficientNet-B4 + scSE Attention)")
    print(f"Target Instance Range: > 1,400 Filaments (High Recall)")
    print("=" * 70)

    # 1. Load Detector (Assistance Scout)
    det_model = YOLO(cfg.DETECTOR_PATH)

    # 2. Load 5-Fold Refiner Ensemble
    refiner_folds = []
    for f in range(5):
        path = os.path.join(cfg.MODELS_DIR, f"fold{f}_best.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fold weight: {path}")
        m = smp.UnetPlusPlus(encoder_name="efficientnet-b4", decoder_attention_type="scse", in_channels=3, classes=1)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.to(device)
        m.eval()
        refiner_folds.append(m)
    print(f"✅ Loaded all 5 folds into memory ({len(refiner_folds) * 84} MB active weights)")

    # 3. Locate Test Images
    test_paths = sorted(
        glob.glob(os.path.join(cfg.TEST_IMAGES_DIR, "*.jpeg")) +
        glob.glob(os.path.join(cfg.TEST_IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(cfg.TEST_IMAGES_DIR, "*.png"))
    )
    print(f"Discovered test images: {len(test_paths)}")
    if len(test_paths) == 0:
        raise FileNotFoundError(f"No test images found in {cfg.TEST_IMAGES_DIR}")

    submission_rows = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    for img_path in tqdm(test_paths, desc="Cascade Ensemble Inference"):
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        # Step 1: Detect candidate boxes with high-recall threshold
        res = det_model.predict(
            img_path,
            imgsz=cfg.DET_IMGSZ,
            conf=cfg.DET_CONF,
            iou=cfg.DET_IOU,
            verbose=False
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            # Fallback empty row
            empty_rle = mask_to_coco_rle(np.zeros((h, w), dtype=np.uint8))
            submission_rows.append({"filament_id": f"{file_stem}_1", "segmentation_rle": empty_rle})
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        scored_masks = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0

            # Expand crop with spatial context
            side = int(math.ceil(max(bw, bh) * cfg.CONTEXT_RATIO))
            side = max(side, cfg.MIN_CROP)
            x0 = int(max(0, round(cx - side / 2.0)))
            y0 = int(max(0, round(cy - side / 2.0)))
            x1b = int(min(w, x0 + side))
            y1b = int(min(h, y0 + side))

            crop = img[y0:y1b, x0:x1b]
            if crop.size == 0:
                continue

            crop_512 = cv2.resize(crop, (cfg.REFINE_CROP, cfg.REFINE_CROP), interpolation=cv2.INTER_LINEAR)
            clahe_512 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop_512)

            # Spatial seed bounding box channel
            seed_patch = np.zeros((y1b - y0, x1b - x0), dtype=np.uint8)
            bx0, by0 = max(0, int(x1 - x0)), max(0, int(y1 - y0))
            bx1, by1 = min(x1b - x0, int(x2 - x0)), min(y1b - y0, int(y2 - y0))
            seed_patch[by0:by1, bx0:bx1] = 255
            seed_512 = cv2.resize(seed_patch, (cfg.REFINE_CROP, cfg.REFINE_CROP), interpolation=cv2.INTER_NEAREST)

            # 3-channel input tensor
            inp = np.stack([crop_512, clahe_512, seed_512], axis=0).astype(np.float32) / 255.0
            inp = (inp - mean) / std

            # Build batch with Flip-4 TTA if enabled
            if cfg.USE_TTA:
                batch_inputs = [
                    inp,                                    # identity
                    np.flip(inp, axis=-1).copy(),           # horizontal flip
                    np.flip(inp, axis=-2).copy(),           # vertical flip
                    np.flip(np.flip(inp, -1), -2).copy()    # hv flip
                ]
                inp_t = torch.from_numpy(np.stack(batch_inputs, axis=0)).to(device)
            else:
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)

            # 5-Fold Ensemble forward pass
            fold_outputs = []
            with torch.no_grad():
                for m in refiner_folds:
                    logits = m(inp_t)
                    probs = torch.sigmoid(logits)
                    if cfg.USE_TTA:
                        p_orig = probs[0:1]
                        p_h = torch.flip(probs[1:2], dims=[-1])
                        p_v = torch.flip(probs[2:3], dims=[-2])
                        p_hv = torch.flip(torch.flip(probs[3:4], dims=[-1]), dims=[-2])
                        p_avg = (p_orig + p_h + p_v + p_hv) / 4.0
                        fold_outputs.append(p_avg.cpu().numpy()[0, 0])
                    else:
                        fold_outputs.append(probs.cpu().numpy()[0, 0])

            avg_prob = np.mean(fold_outputs, axis=0)
            crop_mask = (avg_prob >= cfg.REFINE_THR).astype(np.uint8)
            crop_mask = cv2.resize(crop_mask, (x1b - x0, y1b - y0), interpolation=cv2.INTER_NEAREST)

            full_canvas = np.zeros((h, w), dtype=np.uint8)
            full_canvas[y0:y1b, x0:x1b] = crop_mask

            if full_canvas.sum() >= cfg.MIN_PIXEL_AREA:
                scored_masks.append((conf, full_canvas))

        # Step 4: Priority overlap resolution
        clean_masks = remove_overlaps_by_score(scored_masks)

        if len(clean_masks) == 0:
            empty_rle = mask_to_coco_rle(np.zeros((h, w), dtype=np.uint8))
            submission_rows.append({"filament_id": f"{file_stem}_1", "segmentation_rle": empty_rle})
            continue

        for i, m in enumerate(clean_masks, start=1):
            rle_str = mask_to_coco_rle(m)
            if rle_str:
                submission_rows.append({
                    "filament_id": f"{file_stem}_{i}",
                    "segmentation_rle": rle_str
                })

    print(f"\n🎉 Cascade Inference Completed!")
    print(f"Total instances generated: {len(submission_rows)} (Target: > 1,400)")

    # Save to CSV
    with open(cfg.OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(submission_rows)

    print(f"Saved submission to: {cfg.OUTPUT_CSV} ({os.path.getsize(cfg.OUTPUT_CSV) / 1024:.1f} KB)")
    return cfg.OUTPUT_CSV

if __name__ == "__main__":
    cfg = CascadeConfig()
    run_cascade_inference(cfg)
