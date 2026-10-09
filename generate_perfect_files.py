#!/usr/bin/env python3
"""
Generate Clean, Strictly Non-Overlapping Graded Submissions in the `files/` folder
Base: submission_clean_055.csv (The verified 0.55 Kaggle-accepted file)
Generates: submission_49.csv, submission_50.csv, submission_51.csv,
           submission_52.csv, submission_53.csv, submission_54.csv
"""

import os
import shutil
import pandas as pd
import numpy as np
import pycocotools.mask as mask_utils

BASE_CSV = "submission_clean_055.csv"
print(f"Loading Base Kaggle-Accepted 0.55 Benchmark: {BASE_CSV}")
df_base = pd.read_csv(BASE_CSV)
print(f"Base rows: {len(df_base)}")

# Decode instance areas to rank by filament scale/confidence
print("Calculating instance areas for ranking...")
records = []
for _, row in df_base.iterrows():
    fid = row["filament_id"]
    stem, idx = fid.rsplit("_", 1)
    rle = row["segmentation_rle"]
    area = int(mask_utils.area({"counts": rle, "size": [2048, 2048]}))
    records.append({
        "stem": stem,
        "original_id": fid,
        "original_rle": rle,
        "area": area
    })

df_rec = pd.DataFrame(records)
print(f"Computed areas for {len(df_rec)} instances. Min area: {df_rec['area'].min()}, Max area: {df_rec['area'].max()}")

# Sort all instances by area ascending so we drop smallest/faintest filaments
df_sorted = df_rec.sort_values(by="area", ascending=True).reset_index(drop=True)

# Targets for 0.54, 0.53, 0.52, 0.51, 0.50, 0.49
targets = [
    {"score": 54, "n_drop": 22},
    {"score": 53, "n_drop": 45},
    {"score": 52, "n_drop": 67},
    {"score": 51, "n_drop": 89},
    {"score": 50, "n_drop": 112},
    {"score": 49, "n_drop": 135},
]

os.makedirs("files", exist_ok=True)
original_stems = df_base["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).unique()

for cfg in targets:
    score = cfg["score"]
    n_drop = cfg["n_drop"]
    
    # Identify IDs to drop (smallest instances)
    # Protection: never drop the only filament in an image
    stems_count = df_rec["stem"].value_counts().to_dict()
    
    dropped_indices = set()
    for idx, row in df_sorted.iterrows():
        if len(dropped_indices) >= n_drop:
            break
        # Never drop empty masks (used as fallback for images with 0 filaments)
        if row["area"] == 0:
            continue
        stem = row["stem"]
        if stems_count[stem] > 1:
            dropped_indices.add(idx)
            stems_count[stem] -= 1
            
    retained_records = [
        row for idx, row in df_sorted.iterrows()
        if idx not in dropped_indices
    ]
    
    by_stem = {}
    for r in retained_records:
        by_stem.setdefault(r["stem"], []).append(r)
        
    final_rows = []
    for stem in original_stems:
        if stem in by_stem:
            stem_instances = sorted(by_stem[stem], key=lambda x: x["area"], reverse=True)
            for new_idx, item in enumerate(stem_instances):
                final_rows.append({
                    "filament_id": f"{stem}_{new_idx + 1}",
                    "segmentation_rle": item["original_rle"]
                })
                
    df_out = pd.DataFrame(final_rows)
    
    # Save into files/submission_XX.csv AND root submission_0XX.csv
    files_path = f"files/submission_{score}.csv"
    root_path = f"submission_0{score}.csv"
    
    df_out.to_csv(files_path, index=False)
    df_out.to_csv(root_path, index=False)
    
    print(f"Generated {files_path:25s} | Rows: {len(df_out):4d} | Images: {len(original_stems)}")

print("\n" + "="*70)
print("AIRTIGHT OVERLAP VALIDATION ON GENERATED FILES IN `files/`")
print("="*70)

for cfg in targets:
    score = cfg["score"]
    fname = f"files/submission_{score}.csv"
    df_check = pd.read_csv(fname)
    
    overlap_count = 0
    grouped = df_check.groupby(df_check["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]))
    for stem, group in grouped:
        if len(group) <= 1:
            continue
        canvas = np.zeros((2048, 2048), dtype=np.uint16)
        for _, row in group.iterrows():
            m = mask_utils.decode({"counts": row["segmentation_rle"], "size": [2048, 2048]}).astype(np.uint16)
            canvas += m
        if canvas.max() > 1:
            overlap_count += 1
            
    assert overlap_count == 0, f"Error: {fname} has overlapping masks!"
    print(f"[PASS] {fname:25s} -> STRICTLY 0 OVERLAPS across all {len(grouped)} images! (Rows: {len(df_check)})")

print("="*70)
print("SUCCESS: All 6 files in `files/` are 100% verified non-overlapping and ready to submit!")
