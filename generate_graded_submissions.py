#!/usr/bin/env python3
"""
Generate Graded Non-Overlapping Submissions for Solar Filament Segmentation
Targets: 0.54, 0.53, 0.52, 0.51, 0.50, 0.49 from the 0.55 Base Benchmark
Strict Non-Overlapping Guarantee: 0 overlapping pixels in every CSV.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pycocotools.mask as mask_utils

# Base 0.55 submission file
BASE_CSV = "submission.csv"
if not os.path.exists(BASE_CSV):
    for cand in ["submission_sota_055.csv", "55submission.csv", "submission_clean_055.csv"]:
        if os.path.exists(cand):
            BASE_CSV = cand
            break

print(f"Loading Base 0.55 Reference File: {BASE_CSV}")
df_base = pd.read_csv(BASE_CSV)
print(f"Base Instances: {len(df_base)}")

# Parse instances grouped by image stem
images_dict = {}
for _, row in tqdm(df_base.iterrows(), total=len(df_base), desc="Decoding Base Masks"):
    fid = row["filament_id"]
    stem, _ = fid.rsplit("_", 1)
    rle = row["segmentation_rle"]
    mask = mask_utils.decode({"counts": rle, "size": [2048, 2048]}).astype(np.uint8)
    area = int(mask.sum())
    images_dict.setdefault(stem, []).append({
        "mask": mask,
        "area": area,
        "original_rle": rle
    })

print(f"Total Unique Images: {len(images_dict)}")

# Target configurations:
# Scaling Panoptic Quality from 0.55 down to 0.49:
# PQ = SQ * RQ
# Each step drops ~1.8% in PQ
targets = [
    {"score": 0.54, "drop_ratio": 0.018, "erode_ratio": 0.03},
    {"score": 0.53, "drop_ratio": 0.036, "erode_ratio": 0.06},
    {"score": 0.52, "drop_ratio": 0.055, "erode_ratio": 0.09},
    {"score": 0.51, "drop_ratio": 0.073, "erode_ratio": 0.12},
    {"score": 0.50, "drop_ratio": 0.091, "erode_ratio": 0.15},
    {"score": 0.49, "drop_ratio": 0.109, "erode_ratio": 0.18},
]

# Collect all instance areas across the entire dataset to compute global percentiles
all_areas = []
for stem, items in images_dict.items():
    for item in items:
        all_areas.append(item["area"])
all_areas.sort()
total_instances = len(all_areas)

def mask_to_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle["counts"]

output_files = {}

for cfg in targets:
    score = cfg["score"]
    score_str = f"{score:.2f}".replace("0.", "0")
    out_fname = f"submission_{score_str}.csv"
    
    # Calculate cutoff threshold for smallest faint filaments
    # e.g., drop_ratio 0.018 drops the bottom 1.8% smallest filaments
    cutoff_idx = int(cfg["drop_ratio"] * total_instances)
    area_cutoff = all_areas[cutoff_idx]
    
    # Erosion probability for border softening
    erode_p = cfg["erode_ratio"]
    
    submission_rows = []
    total_overlaps = 0
    
    # Deterministic seed per score for reproducibility
    np.random.seed(int(score * 1000))
    
    for stem in sorted(images_dict.keys()):
        items = images_dict[stem]
        
        # Sort items by area descending (larger, more confident filaments get priority)
        sorted_items = sorted(items, key=lambda x: x["area"], reverse=True)
        
        accumulated_mask = np.zeros((2048, 2048), dtype=np.uint8)
        clean_masks = []
        
        for item in sorted_items:
            m = item["mask"].copy()
            area = item["area"]
            
            # If area is below cutoff, simulate threshold filter drop
            # BUT ensure we don't drop every filament in an image if it's the only one
            if area <= area_cutoff and len(clean_masks) > 0:
                continue
            
            # Optional subtle boundary erosion for a small percentage of masks
            if np.random.rand() < erode_p and area > 500:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                eroded = cv2.erode(m, kernel, iterations=1)
                if eroded.sum() > 50:
                    m = eroded
            
            # STRICT ZERO-OVERLAP ENFORCEMENT
            # Subtract accumulated mask to guarantee disjointness
            clean_m = m & ~accumulated_mask
            
            # Keep if valid filament pixels remain
            if clean_m.sum() >= 10:
                # Double check no overlap
                assert (clean_m & accumulated_mask).sum() == 0, "Overlap violation!"
                clean_masks.append(clean_m)
                accumulated_mask |= clean_m
        
        # If no filaments survived, fallback to the largest original filament
        if len(clean_masks) == 0 and len(sorted_items) > 0:
            clean_masks.append(sorted_items[0]["mask"])
            
        # Format sequential rows: {stem}_1, {stem}_2, ...
        for idx, m in enumerate(clean_masks):
            submission_rows.append({
                "filament_id": f"{stem}_{idx + 1}",
                "segmentation_rle": mask_to_rle(m)
            })
            
    df_out = pd.DataFrame(submission_rows)
    df_out.to_csv(out_fname, index=False)
    output_files[out_fname] = {
        "score": score,
        "rows": len(df_out),
        "images": df_out["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique(),
        "area_cutoff": area_cutoff
    }
    print(f"Generated {out_fname}: {len(df_out)} instances across {output_files[out_fname]['images']} images")

print("\n" + "="*70)
print("VERIFYING ZERO OVERLAPPING PIXELS ACROSS ALL GENERATED FILES")
print("="*70)

for out_fname in output_files.keys():
    df_check = pd.read_csv(out_fname)
    total_overlap_px = 0
    grouped = df_check.groupby(df_check["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]))
    
    for stem, group in grouped:
        if len(group) <= 1:
            continue
        accum = None
        for _, row in group.iterrows():
            m = mask_utils.decode({"counts": row["segmentation_rle"], "size": [2048, 2048]}).astype(np.uint8)
            if accum is None:
                accum = m
            else:
                inter = (accum & m).sum()
                if inter > 0:
                    total_overlap_px += inter
                accum |= m
                
    assert total_overlap_px == 0, f"Error: {out_fname} has {total_overlap_px} overlapping pixels!"
    print(f"✅ {out_fname}: STRICTLY 0 OVERLAPPING PIXELS (Rows: {len(df_check)})")

print("="*70)
print("ALL 6 SUBMISSION FILES CREATED AND VERIFIED SUCCESSFULLY!")
