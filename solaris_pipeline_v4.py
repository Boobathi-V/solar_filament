# ============================================================
# MAGFiLO SOLAR FILAMENT SEGMENTATION
# SINGLE-CELL COMPETITION PIPELINE
#
# Output:
#   /kaggle/working/submission.csv
#
# Pipeline:
#   COCO annotations
#        ↓
#   YOLO detector
#        ↓
#   Seed generation
#        ↓
#   U-Net refinement
#        ↓
#   RLE segmentation
#        ↓
#   submission.csv
#
# The code automatically searches /kaggle/input instead of
# assuming a fixed Kaggle dataset path.
# ============================================================

import sys
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import subprocess
import os
import json
import random
import shutil
import math
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================

print("=" * 70)
print("INSTALLING DEPENDENCIES")
print("=" * 70)

packages = [
    "segmentation-models-pytorch",
    "albumentations",
    "pycocotools",
    "ultralytics",
    "ensemble-boxes"
]

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    *packages
])

print("✅ Dependencies ready")


# ============================================================
# 2. IMPORTS
# ============================================================

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

from tqdm.auto import tqdm

from pycocotools import mask as mask_utils

try:
    from ultralytics import YOLO  # pyrefly: ignore [missing-import] # type: ignore
except ImportError:
    YOLO = None


# Safe display function for both Jupyter and standalone Python
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# ============================================================
# 3. GLOBAL CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_GPUS = (
    torch.cuda.device_count()
    if torch.cuda.is_available()
    else 0
)

WORKERS = 2

OUT_DIR = Path("/kaggle/working")

print()
print("=" * 70)
print("SYSTEM")
print("=" * 70)

print("Device :", DEVICE)
print("GPUs   :", N_GPUS)


# ============================================================
# 4. COMPETITION PARAMETERS
# ============================================================

YOLO_MODEL = "yolo11s.pt"

YOLO_IMGSZ = 1280

YOLO_EPOCHS = 40

YOLO_BATCH = 8

# Lowered from 0.28 to 0.15 to maximize recall on thin / faint solar filaments
YOLO_CONF = 0.15

YOLO_IOU = 0.50

# Detector Multi-Scale TTA + Weighted Boxes Fusion
YOLO_MULTI_SCALE = True
YOLO_SCALES = [1280, 1536]
USE_WBF = True
WBF_IOU_THR = 0.50

# Refiner Model Configuration
REFINER_ARCH = "UnetPlusPlus"        # SOTA nested skip connections for elongated thin filaments
REFINER_ENCODER = "timm-efficientnet-b3"  # Stronger encoder than standard resnet34
LOSS_TYPE = "focal_tversky"          # Beta=0.7 heavily penalizes false negatives on thin structures

REFINE_CROP = 512

REFINE_EPOCHS = 12

REFINE_BATCH = 8

REFINE_LR = 3e-4

# Initial threshold: will be automatically swept and tuned on val set!
REFINE_THR = 0.45

VAL_FRAC = 0.15

PAD_RATIO = 0.40

CONTEXT = 1.5

MIN_CROP = 96

MIN_AREA = 200

# SOTA Inference TTA enabled
USE_TTA = True

USE_BOX_JITTER_TTA = True

BOX_JITTER_PX = 8

# Aspect-preserving padding: pad to square with border constant instead of stretching
ASPECT_PRESERVE_PAD = True

MAX_TEST_IMAGES = None

# ============================================================
# IMPORTANT:
# Set these according to whether your trained models exist.
#
# True  = use existing trained checkpoint
# False = train model if checkpoint isn't found
# ============================================================

SKIP_YOLO_TRAIN = True

SKIP_REFINER_TRAIN = True


# ============================================================
# 5. SEARCH KAGGLE INPUT & COLAB PATHS AUTOMATICALLY
# ============================================================

SEARCH_PATHS = [
    Path("/kaggle/input"),
    Path("/content/drive/MyDrive"),
    Path("/content"),
    Path(".")
]

print()
print("=" * 70)
print("SEARCHING DATASET & CHECKPOINT PATHS")
print("=" * 70)

all_files = []
for sp in SEARCH_PATHS:
    if sp.exists():
        try:
            found = [p for p in sp.rglob("*") if not any(x.startswith(".") for x in p.parts)]
            print(f"Found {len(found)} entries in {sp}")
            all_files.extend(found)
        except Exception as e:
            print(f"Warning searching {sp}: {e}")

print("Files/directories discovered:", len(all_files))


# ============================================================
# 6. FIND COCO TRAIN ANNOTATION JSON
# ============================================================

json_candidates = [
    p for p in all_files
    if p.is_file() and p.suffix.lower() == ".json"
]

print()
print("JSON files found:")

for p in json_candidates[:30]:
    print(" ", p)


TRAIN_JSON = None


# strongest match first
priority_patterns = [
    lambda n: "magfilo" in n and "annotation" in n and "train" in n,
    lambda n: "magfilo" in n and "train" in n,
    lambda n: "annotation" in n and "train" in n,
    lambda n: "train" in n
]


for pattern in priority_patterns:

    for p in json_candidates:

        name = p.name.lower()

        if pattern(name):

            try:

                with open(p, "r") as f:
                    obj = json.load(f)

                if (
                    isinstance(obj, dict)
                    and "images" in obj
                    and "annotations" in obj
                ):

                    TRAIN_JSON = p
                    break

            except Exception:
                pass

    if TRAIN_JSON is not None:
        break


if TRAIN_JSON is None:

    raise FileNotFoundError(
        "\n❌ Could not find COCO training annotation JSON.\n"
        "Make sure the MAGFiLO training dataset is attached "
        "to this Kaggle notebook."
    )


print()
print("✅ TRAIN JSON")
print(TRAIN_JSON)


# ============================================================
# 7. FIND IMAGE DIRECTORIES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}


def count_images(directory):

    try:

        return sum(
            1
            for x in directory.iterdir()
            if x.is_file()
            and x.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png"
            }
        )

    except Exception:

        return 0


directories = [
    p for p in all_files
    if p.is_dir()
]


# ------------------------------------------------------------
# TRAIN IMAGES
# ------------------------------------------------------------

train_image_candidates = []

for p in directories:

    name = p.name.lower()

    score = 0

    if name == "train_images":
        score += 100

    if "train" in name:
        score += 30

    n = count_images(p)

    if n > 0:
        score += min(n, 100000) / 100000

        train_image_candidates.append(
            (score, n, p)
        )


train_image_candidates.sort(
    reverse=True,
    key=lambda x: x[0]
)


TRAIN_IMAGES = (
    train_image_candidates[0][2]
    if train_image_candidates
    else None
)


if TRAIN_IMAGES is None:

    raise FileNotFoundError(
        "❌ Could not find training image directory."
    )


print()
print("✅ TRAIN IMAGES")
print(TRAIN_IMAGES)

print(
    "Training images found:",
    train_image_candidates[0][1]
)


# ------------------------------------------------------------
# TEST IMAGES
# ------------------------------------------------------------

test_image_candidates = []

for p in directories:

    name = p.name.lower()

    score = 0

    if name == "test_images":
        score += 100

    if name == "test":
        score += 80

    if "test" in name:
        score += 30

    n = count_images(p)

    if n > 0:

        score += min(n, 100000) / 100000

        test_image_candidates.append(
            (score, n, p)
        )


test_image_candidates.sort(
    reverse=True,
    key=lambda x: x[0]
)


TEST_IMAGES = (
    test_image_candidates[0][2]
    if test_image_candidates
    else None
)


if TEST_IMAGES is None:

    raise FileNotFoundError(
        "❌ Could not find test image directory."
    )


print()
print("✅ TEST IMAGES")
print(TEST_IMAGES)

print(
    "Test images found:",
    test_image_candidates[0][1]
)


# ============================================================
# 8. FIND TRAINED YOLO CHECKPOINT
# ============================================================

pt_candidates = [
    p for p in all_files
    if p.is_file()
    and p.suffix.lower() == ".pt"
]


print()
print("=" * 70)
print("SEARCHING MODEL CHECKPOINTS")
print("=" * 70)

for p in pt_candidates:
    print(" ", p)


YOLO_WEIGHTS = None


# Strongest names first
for p in pt_candidates:

    name = p.name.lower()

    if (
        "yolo" in name
        and "detect" in name
        and "best" in name
    ):

        YOLO_WEIGHTS = p
        break


if YOLO_WEIGHTS is None:

    for p in pt_candidates:

        if p.name.lower() == "best.pt":

            YOLO_WEIGHTS = p

            break


# ============================================================
# 9. FIND REFINER CHECKPOINT
# ============================================================

REFINER_CKPT = None


for p in pt_candidates:

    name = p.name.lower()

    if (
        "seed" in name
        and "refiner" in name
        and "best" in name
    ):

        REFINER_CKPT = p

        break


if REFINER_CKPT is None:

    for p in pt_candidates:

        name = p.name.lower()

        if (
            "refiner" in name
            and "best" in name
        ):

            REFINER_CKPT = p

            break


print()
print("YOLO checkpoint   :", YOLO_WEIGHTS)

print("Refiner checkpoint:", REFINER_CKPT)


# ============================================================
# 10. OUTPUT FILES
# ============================================================

YOLO_DATA = OUT_DIR / "yolo_detect_data"

YOLO_RUNS = OUT_DIR / "ultralytics_runs"

SUBMISSION_CSV = OUT_DIR / "submission.csv"

PRED_VAL_CSV = OUT_DIR / "pred_val.csv"

GT_VAL_CSV = OUT_DIR / "gt_val.csv"


# ============================================================
# 11. LOAD COCO
# ============================================================

print()
print("=" * 70)
print("LOADING COCO DATASET")
print("=" * 70)


with open(TRAIN_JSON, "r") as f:

    coco = json.load(f)


images_by_id = {
    im["id"]: im
    for im in coco["images"]
}


anns_by_image = defaultdict(list)


for ann in coco["annotations"]:

    anns_by_image[
        ann["image_id"]
    ].append(ann)


print(
    "COCO images:",
    len(coco["images"])
)

print(
    "COCO annotations:",
    len(coco["annotations"])
)


# ============================================================
# 12. BUILD SAMPLES
# ============================================================

samples = []

missing = 0


# Build a filename lookup for robustness
train_files = {}

for p in TRAIN_IMAGES.iterdir():

    if (
        p.is_file()
        and p.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ):

        train_files[p.name] = p


for image_id, im in images_by_id.items():

    filename = im["file_name"]

    path = TRAIN_IMAGES / filename


    if not path.exists():

        # filename-only search
        if filename in train_files:

            path = train_files[filename]

        else:

            # stem-based fallback
            stem = Path(filename).stem

            matches = [
                p
                for name, p in train_files.items()
                if Path(name).stem == stem
            ]

            if matches:

                path = matches[0]

            else:

                missing += 1

                continue


    samples.append({

        "image_id": image_id,

        "file_name": filename,

        "path": path,

        "stem": path.stem,

        "height": int(
            im.get("height", 2048)
        ),

        "width": int(
            im.get("width", 2048)
        ),

        "annotations":
            anns_by_image[image_id]

    })


print()
print(
    "Usable training samples:",
    len(samples)
)

print(
    "Missing images:",
    missing
)


if len(samples) == 0:

    raise RuntimeError(
        "❌ Zero training images matched COCO annotations."
    )


# ============================================================
# 13. TRAIN / VALIDATION SPLIT
# ============================================================

stems = sorted({
    s["stem"]
    for s in samples
})


random.Random(SEED).shuffle(stems)


n_val = max(
    1,
    int(len(stems) * VAL_FRAC)
)


val_stem_list = stems[:n_val]

train_stem_list = stems[n_val:]


val_stems = set(val_stem_list)

train_stems = set(train_stem_list)


train_samples = [
    s
    for s in samples
    if s["stem"] in train_stems
]


val_samples = [
    s
    for s in samples
    if s["stem"] in val_stems
]


print()
print(
    "Train images:",
    len(train_samples)
)

print(
    "Validation images:",
    len(val_samples)
)


# ============================================================
# 14. ANNOTATION LIST
# ============================================================

def build_ann_list(sample_list):

    rows = []

    for s in sample_list:

        for ann in s["annotations"]:

            if ann.get("iscrowd", 0):
                continue

            if (
                not ann.get("segmentation")
                and not ann.get("bbox")
            ):
                continue

            rows.append(
                (s, ann)
            )

    return rows


train_ann_list = build_ann_list(
    train_samples
)

val_ann_list = build_ann_list(
    val_samples
)


print(
    "Train annotations:",
    len(train_ann_list)
)

print(
    "Val annotations:",
    len(val_ann_list)
)


# ============================================================
# 15. BUILD YOLO DATASET
# ============================================================

def build_yolo_split(
    sample_list,
    split_name
):

    image_dir = (
        YOLO_DATA
        / "images"
        / split_name
    )

    label_dir = (
        YOLO_DATA
        / "labels"
        / split_name
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    used = set()

    n_boxes = 0


    for s in tqdm(
        sample_list,
        desc=f"YOLO {split_name}"
    ):

        filename = s["path"].name

        if filename in used:
            continue

        used.add(filename)


        dst = image_dir / filename


        if not dst.exists():

            shutil.copy2(
                s["path"],
                dst
            )


        w = float(s["width"])

        h = float(s["height"])


        lines = []


        for ann in s["annotations"]:

            bbox = ann.get("bbox")

            if bbox is None:
                continue


            x, y, bw, bh = map(
                float,
                bbox
            )


            # clamp
            x = max(0, min(x, w - 1))

            y = max(0, min(y, h - 1))

            bw = max(
                1,
                min(bw, w - x)
            )

            bh = max(
                1,
                min(bh, h - y)
            )


            xc = (
                x + bw / 2
            ) / w

            yc = (
                y + bh / 2
            ) / h

            nw = bw / w

            nh = bh / h


            lines.append(
                f"0 {xc:.6f} {yc:.6f} "
                f"{nw:.6f} {nh:.6f}"
            )

            n_boxes += 1


        label_path = (
            label_dir
            / f"{Path(filename).stem}.txt"
        )


        label_path.write_text(
            "\n".join(lines)
            + ("\n" if lines else "")
        )


    return len(used), n_boxes


# ============================================================
# 16. TRAIN YOLO IF NECESSARY
# ============================================================

need_yolo_training = (
    YOLO_WEIGHTS is None
)


if (
    SKIP_YOLO_TRAIN
    and YOLO_WEIGHTS is None
):

    print()
    print(
        "⚠️ Trained YOLO checkpoint not found."
    )

    print(
        "Because SKIP_YOLO_TRAIN=True, "
        "the pretrained YOLO11s model will be used."
    )

    YOLO_WEIGHTS = None


if (
    not SKIP_YOLO_TRAIN
    or (
        SKIP_YOLO_TRAIN
        and YOLO_WEIGHTS is None
    )
):

    # If no trained checkpoint, train detector
    if YOLO_DATA.exists():

        shutil.rmtree(
            YOLO_DATA
        )


    ntr, btr = build_yolo_split(
        train_samples,
        "train"
    )

    nva, bva = build_yolo_split(
        val_samples,
        "val"
    )


    yaml_path = (
        YOLO_DATA
        / "data.yaml"
    )


    yaml_path.write_text(
        f"""
path: {YOLO_DATA}
train: images/train
val: images/val

names:
  0: filament
"""
    )


    print()
    print(
        f"YOLO train: {ntr} images / {btr} boxes"
    )

    print(
        f"YOLO val  : {nva} images / {bva} boxes"
    )


    detector = YOLO(
        YOLO_MODEL
    )


    device_arg = (
        ",".join(
            str(i)
            for i in range(N_GPUS)
        )
        if N_GPUS > 1
        else (
            0
            if DEVICE.type == "cuda"
            else "cpu"
        )
    )


    detector.train(

        data=str(yaml_path),

        epochs=YOLO_EPOCHS,

        imgsz=YOLO_IMGSZ,

        batch=YOLO_BATCH,

        workers=WORKERS,

        seed=SEED,

        project=str(
            YOLO_RUNS
        ),

        name="magfilo_detector",

        exist_ok=True,

        pretrained=True,

        patience=max(
            8,
            YOLO_EPOCHS // 4
        ),

        device=device_arg
    )


    best = (
        YOLO_RUNS
        / "magfilo_detector"
        / "weights"
        / "best.pt"
    )


    if not best.exists():

        raise FileNotFoundError(
            "❌ YOLO training finished but "
            "best.pt was not found."
        )


    YOLO_WEIGHTS = (
        OUT_DIR
        / "yolo_detect_best.pt"
    )


    shutil.copy2(
        best,
        YOLO_WEIGHTS
    )


    print(
        "✅ YOLO saved:",
        YOLO_WEIGHTS
    )


# ============================================================
# 17. LOAD DETECTOR
# ============================================================

if YOLO_WEIGHTS is not None:

    print()
    print(
        "Loading trained YOLO:",
        YOLO_WEIGHTS
    )

    det_model = YOLO(
        str(YOLO_WEIGHTS)
    )

else:

    print()
    print(
        "Loading pretrained YOLO11s"
    )

    det_model = YOLO(
        YOLO_MODEL
    )


print("✅ Detector ready")


# ============================================================
# 18. MASK UTILITIES
# ============================================================

def ann_to_mask(
    ann,
    h,
    w
):

    segmentation = ann.get(
        "segmentation"
    )


    if segmentation:

        try:

            rles = (
                mask_utils.frPyObjects(
                    segmentation,
                    h,
                    w
                )
            )


            rle = (
                mask_utils.merge(rles)
                if isinstance(rles, list)
                else rles
            )


            mask = mask_utils.decode(
                rle
            )


            if mask.ndim == 3:

                mask = np.any(
                    mask,
                    axis=2
                )


            return mask.astype(
                np.uint8
            )

        except Exception:
            pass


    # bbox fallback
    x, y, bw, bh = map(
        int,
        ann["bbox"]
    )


    x = max(
        0,
        min(x, w - 1)
    )

    y = max(
        0,
        min(y, h - 1)
    )

    x2 = min(
        w,
        x + max(1, bw)
    )

    y2 = min(
        h,
        y + max(1, bh)
    )


    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )


    mask[y:y2, x:x2] = 1


    return mask


# ============================================================
# 19. CROP UTILITIES & ASPECT-PRESERVING PADDING
# ============================================================

def get_padded_crop_bounds(
    bbox,
    h,
    w,
    pad_ratio=PAD_RATIO,
    minimum=MIN_CROP
):
    """Extract expanded bounding box crop coordinates clamped to image limits."""
    x, y, bw, bh = bbox
    cx = x + bw / 2.0
    cy = y + bh / 2.0

    exp_w = max(bw * (1.0 + 2.0 * pad_ratio), float(minimum))
    exp_h = max(bh * (1.0 + 2.0 * pad_ratio), float(minimum))

    x0 = int(max(0, math.floor(cx - exp_w / 2.0)))
    y0 = int(max(0, math.floor(cy - exp_h / 2.0)))
    x1 = int(min(w, math.ceil(cx + exp_w / 2.0)))
    y1 = int(min(h, math.ceil(cy + exp_h / 2.0)))

    return x0, y0, x1, y1


def pad_to_square(
    img,
    border_mode=cv2.BORDER_CONSTANT,
    value=0
):
    """
    Pads an HxW image symmetrically so height == width == max(H, W).
    Preserves true filament aspect ratio without stretching distortion.
    """
    h, w = img.shape[:2]
    max_side = max(h, w)

    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left

    padded = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        border_mode,
        value=value
    )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_mask(
    mask_square,
    pads,
    target_h,
    target_w
):
    """Crops out symmetric padding from a square mask back to target dimensions."""
    pad_top, pad_bottom, pad_left, pad_right = pads
    sq_h, sq_w = mask_square.shape[:2]
    end_y = sq_h - pad_bottom if pad_bottom > 0 else sq_h
    end_x = sq_w - pad_right if pad_right > 0 else sq_w
    unpadded = mask_square[pad_top:end_y, pad_left:end_x]
    if unpadded.shape[0] != target_h or unpadded.shape[1] != target_w:
        unpadded = cv2.resize(unpadded, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return unpadded


def square_bounds_from_bbox(
    bbox,
    h,
    w,
    context=CONTEXT,
    minimum=MIN_CROP
):

    x, y, bw, bh = bbox


    cx = x + bw / 2

    cy = y + bh / 2


    side = int(
        math.ceil(
            max(bw, bh)
            * context
        )
    )


    side = max(
        side,
        minimum
    )


    # do not exceed image dimensions
    side = min(
        side,
        h,
        w
    )


    x0 = int(
        round(
            cx - side / 2
        )
    )

    y0 = int(
        round(
            cy - side / 2
        )
    )


    x1 = x0 + side

    y1 = y0 + side


    if x0 < 0:

        x1 -= x0

        x0 = 0


    if y0 < 0:

        y1 -= y0

        y0 = 0


    if x1 > w:

        shift = x1 - w

        x0 -= shift

        x1 = w


    if y1 > h:

        shift = y1 - h

        y0 -= shift

        y1 = h


    x0 = max(
        0,
        x0
    )

    y0 = max(
        0,
        y0
    )

    x1 = min(
        w,
        x1
    )

    y1 = min(
        h,
        y1
    )


    return (
        x0,
        y0,
        x1,
        y1
    )


# ============================================================
# 20. SEED GENERATION
# ============================================================

def seed_from_box(
    h,
    w,
    bbox,
    pad_ratio=PAD_RATIO
):

    x, y, bw, bh = bbox


    px = bw * pad_ratio

    py = bh * pad_ratio


    x0 = max(
        0,
        int(x - px)
    )

    y0 = max(
        0,
        int(y - py)
    )


    x1 = min(
        w,
        int(x + bw + px)
    )

    y1 = min(
        h,
        int(y + bh + py)
    )


    seed = np.zeros(
        (h, w),
        dtype=np.uint8
    )


    seed[y0:y1, x0:x1] = 1


    return seed


# ============================================================
# 21. BOX JITTER
# ============================================================

def jitter_bbox(
    bbox,
    h,
    w,
    rng
):

    x, y, bw, bh = map(
        float,
        bbox
    )


    x += float(
        rng.uniform(
            -0.15,
            0.15
        ) * bw
    )


    y += float(
        rng.uniform(
            -0.15,
            0.15
        ) * bh
    )


    sx = float(
        rng.uniform(
            0.85,
            1.25
        )
    )


    sy = float(
        rng.uniform(
            0.85,
            1.25
        )
    )


    cx = x + bw / 2

    cy = y + bh / 2


    bw = max(
        8.0,
        bw * sx
    )

    bh = max(
        8.0,
        bh * sy
    )


    x = cx - bw / 2

    y = cy - bh / 2


    x = float(
        np.clip(
            x,
            0,
            max(0, w - 2)
        )
    )

    y = float(
        np.clip(
            y,
            0,
            max(0, h - 2)
        )
    )


    bw = float(
        min(
            bw,
            w - x
        )
    )

    bh = float(
        min(
            bh,
            h - y
        )
    )


    pad = float(
        rng.uniform(
            PAD_RATIO * 0.5,
            PAD_RATIO * 1.5
        )
    )


    return [
        x,
        y,
        bw,
        bh
    ], pad


# ============================================================
# 22. REFINER DATASET
# ============================================================

class SeedRefineDataset(
    Dataset
):

    def __init__(
        self,
        ann_list,
        crop_size,
        augment=False
    ):

        self.ann_list = ann_list

        self.crop_size = crop_size

        self.augment = augment

        self.cache = {}


    def __len__(self):

        return len(
            self.ann_list
        )


    def load_image(
        self,
        path
    ):

        key = str(path)


        if key not in self.cache:

            img = cv2.imread(
                key,
                cv2.IMREAD_GRAYSCALE
            )


            if img is None:

                raise FileNotFoundError(
                    key
                )


            self.cache[key] = img


        return self.cache[key]


    def __getitem__(
        self,
        idx
    ):

        sample, ann = (
            self.ann_list[idx]
        )


        img = self.load_image(
            sample["path"]
        )


        h, w = img.shape


        target = ann_to_mask(
            ann,
            h,
            w
        )


        gt_bbox = ann["bbox"]


        if self.augment:

            rng = np.random.default_rng(
                SEED
                + idx
                + random.randint(
                    0,
                    10000
                )
            )


            box_j, pad_j = jitter_bbox(
                gt_bbox,
                h,
                w,
                rng
            )


            seed_full = seed_from_box(
                h,
                w,
                box_j,
                pad_ratio=pad_j
            )

        else:

            seed_full = seed_from_box(
                h,
                w,
                gt_bbox
            )


        if ASPECT_PRESERVE_PAD:
            x0, y0, x1, y1 = get_padded_crop_bounds(gt_bbox, h, w, pad_ratio=PAD_RATIO)
            crop = img[y0:y1, x0:x1]
            target_crop = target[y0:y1, x0:x1]
            seed_crop = seed_full[y0:y1, x0:x1]

            crop_sq, _ = pad_to_square(crop, border_mode=cv2.BORDER_CONSTANT, value=0)
            tgt_sq, _ = pad_to_square(target_crop, border_mode=cv2.BORDER_CONSTANT, value=0)
            seed_sq, _ = pad_to_square(seed_crop, border_mode=cv2.BORDER_CONSTANT, value=0)

            img_c = cv2.resize(crop_sq, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            tgt_c = cv2.resize(tgt_sq, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
            seed_c = cv2.resize(seed_sq, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        else:
            x0, y0, x1, y1 = square_bounds_from_bbox(gt_bbox, h, w)
            crop = img[y0:y1, x0:x1]
            target_crop = target[y0:y1, x0:x1]
            seed_crop = seed_full[y0:y1, x0:x1]

            img_c = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            tgt_c = cv2.resize(target_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
            seed_c = cv2.resize(seed_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)


        # CLAHE improves faint filament contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(img_c)


        # augmentation
        if (
            self.augment
            and random.random() < 0.5
        ):

            img_c = np.fliplr(
                img_c
            ).copy()

            clahe = np.fliplr(
                clahe
            ).copy()

            tgt_c = np.fliplr(
                tgt_c
            ).copy()

            seed_c = np.fliplr(
                seed_c
            ).copy()


        if (
            self.augment
            and random.random() < 0.5
        ):

            img_c = np.flipud(
                img_c
            ).copy()

            clahe = np.flipud(
                clahe
            ).copy()

            tgt_c = np.flipud(
                tgt_c
            ).copy()

            seed_c = np.flipud(
                seed_c
            ).copy()


        x = np.stack(
            [
                img_c,
                clahe,
                seed_c * 255
            ],
            axis=0
        ).astype(
            np.float32
        ) / 255.0


        mean = np.array(
            [
                0.485,
                0.456,
                0.406
            ],
            dtype=np.float32
        )[:, None, None]


        std = np.array(
            [
                0.229,
                0.224,
                0.225
            ],
            dtype=np.float32
        )[:, None, None]


        x = (
            x - mean
        ) / std


        y = torch.from_numpy(
            tgt_c.astype(
                np.float32
            )
        )[None]


        return (
            torch.from_numpy(x),
            y
        )


# ============================================================
# 23. CREATE REFINER DATALOADERS
# ============================================================

train_ds = SeedRefineDataset(
    train_ann_list,
    REFINE_CROP,
    augment=True
)

val_ds = SeedRefineDataset(
    val_ann_list,
    REFINE_CROP,
    augment=False
)


train_loader = DataLoader(
    train_ds,
    batch_size=REFINE_BATCH,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
    drop_last=(
        len(train_ds) >= REFINE_BATCH
    )
)


val_loader = DataLoader(
    val_ds,
    batch_size=REFINE_BATCH * 2,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True
)


print()
print(
    "Refiner train samples:",
    len(train_ds)
)

print(
    "Refiner val samples:",
    len(val_ds)
)


# ============================================================
# 24. LOSS FUNCTIONS (FOCAL-TVERSKY FOR THIN FILAMENTS)
# ============================================================

from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss

# beta=0.7 > alpha=0.3 heavily penalizes False Negatives (essential for thin thread structures)
tversky_criterion = TverskyLoss(mode="binary", alpha=0.3, beta=0.7)
focal_criterion = FocalLoss(mode="binary")

def composite_loss(logits, targets):
    """Combined Focal-Tversky loss: balances boundary precision and FN penalty."""
    return 0.5 * tversky_criterion(logits, targets) + 0.5 * focal_criterion(logits, targets)

def soft_dice_loss(
    logits,
    targets,
    eps=1e-6
):

    probs = torch.sigmoid(
        logits
    )


    inter = (
        probs * targets
    ).sum(
        dim=(2, 3)
    )


    den = (
        probs.sum(
            dim=(2, 3)
        )
        +
        targets.sum(
            dim=(2, 3)
        )
    )


    dice = (
        2 * inter + eps
    ) / (
        den + eps
    )


    return 1.0 - dice.mean()


@torch.no_grad()
def batch_dice(
    logits,
    targets,
    threshold=0.5
):

    preds = (
        torch.sigmoid(
            logits
        )
        >
        threshold
    ).float()


    inter = (
        preds * targets
    ).sum(
        dim=(2, 3)
    )


    den = (
        preds.sum(
            dim=(2, 3)
        )
        +
        targets.sum(
            dim=(2, 3)
        )
    )


    return (
        (2 * inter + 1e-6)
        /
        (den + 1e-6)
    ).mean().item()


# ============================================================
# 25. CREATE U-NET REFINER
# ============================================================

print()
print("=" * 70)
print(f"CREATING REFINER: {REFINER_ARCH} ({REFINER_ENCODER})")
print("=" * 70)

arch_class = getattr(smp, REFINER_ARCH, smp.UnetPlusPlus)

try:
    refiner = arch_class(
        encoder_name=REFINER_ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)
    print(f"✅ Initialized {REFINER_ARCH} with encoder {REFINER_ENCODER}")
except Exception as e:
    print(f"⚠️ Notice: {e}. Falling back to smp.Unet('resnet34')...")
    refiner = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)


if N_GPUS > 1:

    refiner = nn.DataParallel(
        refiner
    )


def unwrap(model):

    return (
        model.module
        if isinstance(
            model,
            nn.DataParallel
        )
        else model
    )


# ============================================================
# 26. LOAD OR TRAIN REFINER
# ============================================================

optimizer = torch.optim.AdamW(
    refiner.parameters(),
    lr=REFINE_LR,
    weight_decay=1e-4
)


scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=REFINE_EPOCHS
    )
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        DEVICE.type == "cuda"
    )
)


if (
    SKIP_REFINER_TRAIN
    and REFINER_CKPT is not None
    and REFINER_CKPT.exists()
):

    print(
        "Loading trained refiner:"
    )

    print(
        REFINER_CKPT
    )


    state = torch.load(
        REFINER_CKPT,
        map_location=DEVICE,
        weights_only=False
    )


    if (
        isinstance(state, dict)
        and "model" in state
    ):

        unwrap(
            refiner
        ).load_state_dict(
            state["model"]
        )

        print(
            "Checkpoint val dice:",
            state.get(
                "val_dice",
                "unknown"
            )
        )

    else:

        unwrap(
            refiner
        ).load_state_dict(
            state
        )


else:

    print(
        "No usable refiner checkpoint."
    )

    print(
        "Training refiner..."
    )


    best_dice = -1.0


    for epoch in range(
        1,
        REFINE_EPOCHS + 1
    ):

        refiner.train()


        train_losses = []

        train_dices = []


        for x, y in tqdm(
            train_loader,
            desc=f"Refiner train {epoch}",
            leave=False
        ):

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.amp.autocast(
                "cuda",
                enabled=(
                    DEVICE.type == "cuda"
                )
            ):

                logits = refiner(x)


                loss = composite_loss(logits, y)


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            train_losses.append(
                loss.item()
            )


            train_dices.append(
                batch_dice(
                    logits.detach(),
                    y
                )
            )


        # validation
        refiner.eval()


        val_losses = []

        val_dices = []


        with torch.no_grad():

            for x, y in tqdm(
                val_loader,
                desc=f"Refiner val {epoch}",
                leave=False
            ):

                x = x.to(
                    DEVICE,
                    non_blocking=True
                )

                y = y.to(
                    DEVICE,
                    non_blocking=True
                )


                logits = refiner(x)


                loss = composite_loss(logits, y)


                val_losses.append(
                    loss.item()
                )


                val_dices.append(
                    batch_dice(
                        logits,
                        y
                    )
                )


        scheduler.step()


        tr_loss = float(
            np.mean(train_losses)
        )

        tr_dice = float(
            np.mean(train_dices)
        )

        va_loss = float(
            np.mean(val_losses)
        )

        va_dice = float(
            np.mean(val_dices)
        )


        print(
            f"Epoch {epoch:02d}/{REFINE_EPOCHS} | "
            f"train loss={tr_loss:.4f} "
            f"dice={tr_dice:.4f} | "
            f"val loss={va_loss:.4f} "
            f"dice={va_dice:.4f}"
        )


        if va_dice > best_dice:

            best_dice = va_dice


            REFINER_SAVE = (
                OUT_DIR
                / "seed_refiner_best.pt"
            )


            torch.save(
                {
                    "model":
                        unwrap(
                            refiner
                        ).state_dict(),

                    "epoch":
                        epoch,

                    "val_dice":
                        va_dice
                },
                REFINER_SAVE
            )


            print(
                "✅ Saved:",
                REFINER_SAVE
            )


    # Load best
    state = torch.load(
        REFINER_SAVE,
        map_location=DEVICE,
        weights_only=False
    )


    unwrap(
        refiner
    ).load_state_dict(
        state["model"]
    )


refiner.eval()


print(
    "✅ Refiner ready"
)

# Store refiner model(s) for ensemble inference
refiner_models = [refiner]

# ============================================================
# 26B. VALIDATION THRESHOLD TUNING SWEEP
# ============================================================

print()
print("=" * 70)
print("TUNING REFINER THRESHOLD ON VALIDATION SET")
print("=" * 70)

threshold_candidates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
best_thr = REFINE_THR
best_thr_dice = -1.0

if len(val_loader) > 0:
    refiner.eval()
    val_probs_list = []
    val_targets_list = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Collecting val predictions", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            p = torch.sigmoid(refiner(x))
            val_probs_list.append(p.cpu())
            val_targets_list.append(y.cpu())

    if val_probs_list:
        val_probs_all = torch.cat(val_probs_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)

        print("Refiner Threshold Sweep Results:")
        for thr in threshold_candidates:
            preds = (val_probs_all > thr).float()
            inter = (preds * val_targets_all).sum(dim=(2, 3))
            den = preds.sum(dim=(2, 3)) + val_targets_all.sum(dim=(2, 3))
            d_val = ((2.0 * inter + 1e-6) / (den + 1e-6)).mean().item()
            print(f"  Threshold {thr:.2f} -> Validation Dice: {d_val:.4f}")
            if d_val > best_thr_dice:
                best_thr_dice = d_val
                best_thr = thr

        print(f"\n🎯 Optimal REFINE_THR selected: {best_thr:.2f} (Val Dice: {best_thr_dice:.4f})")
        REFINE_THR = best_thr
else:
    print(f"No validation dataloader available; using default REFINE_THR={REFINE_THR}")


# ============================================================
# 27. RLE ENCODER
# ============================================================

def mask_to_rle(
    binary_mask
):

    binary_mask = (
        binary_mask
        .astype(np.uint8)
    )


    rle = mask_utils.encode(
        np.asfortranarray(
            binary_mask
        )
    )


    counts = rle["counts"]


    if isinstance(
        counts,
        bytes
    ):

        counts = counts.decode(
            "utf-8"
        )


    return counts


# ============================================================
# 28. REFINEMENT FUNCTION (ASPECT-PRESERVING + TTA + ENSEMBLE)
# ============================================================

@torch.no_grad()
def refine_box_on_image(
    gray,
    bbox_xyxy,
    conf
):
    h, w = gray.shape
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    bbox = [x1, y1, bw, bh]

    if ASPECT_PRESERVE_PAD:
        x0, y0, x1b, y1b = get_padded_crop_bounds(bbox, h, w, pad_ratio=PAD_RATIO)
        crop = gray[y0:y1b, x0:x1b]
        if crop.size == 0:
            return np.zeros((h, w), np.uint8), float(conf)
        crop_sq, pads = pad_to_square(crop, border_mode=cv2.BORDER_CONSTANT, value=0)
        img_c = cv2.resize(crop_sq, (REFINE_CROP, REFINE_CROP), interpolation=cv2.INTER_LINEAR)
    else:
        x0, y0, x1b, y1b = square_bounds_from_bbox(bbox, h, w)
        crop = gray[y0:y1b, x0:x1b]
        if crop.size == 0:
            return np.zeros((h, w), np.uint8), float(conf)
        img_c = cv2.resize(crop, (REFINE_CROP, REFINE_CROP), interpolation=cv2.INTER_LINEAR)
        pads = None

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(img_c)

    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

    if USE_BOX_JITTER_TTA:
        j = int(BOX_JITTER_PX)
        shifts = [(0, 0), (j, 0), (-j, 0), (0, j), (0, -j)]
    else:
        shifts = [(0, 0)]

    batch = []
    for dx, dy in shifts:
        bbox_j = [x1 + dx, y1 + dy, bw, bh]
        seed_full = seed_from_box(h, w, bbox_j)
        seed_crop = seed_full[y0:y1b, x0:x1b]

        if ASPECT_PRESERVE_PAD:
            seed_sq, _ = pad_to_square(seed_crop, border_mode=cv2.BORDER_CONSTANT, value=0)
            seed_c = cv2.resize(seed_sq, (REFINE_CROP, REFINE_CROP), interpolation=cv2.INTER_NEAREST)
        else:
            seed_c = cv2.resize(seed_crop, (REFINE_CROP, REFINE_CROP), interpolation=cv2.INTER_NEAREST)

        x = np.stack([img_c, clahe, seed_c * 255], axis=0).astype(np.float32) / 255.0
        x = (x - mean) / std
        batch.append(x)

    x_tensor = torch.from_numpy(np.stack(batch, axis=0)).to(DEVICE)

    # Multi-model ensemble + horizontal flip TTA
    models_to_run = refiner_models if ('refiner_models' in globals() and refiner_models) else [refiner]
    all_model_probs = []

    for m in models_to_run:
        m.eval()
        m_logits = m(x_tensor)
        if USE_TTA:
            flip_logits = m(x_tensor.flip(-1)).flip(-1)
            m_logits = (m_logits + flip_logits) / 2.0
        p_m = torch.sigmoid(m_logits).mean(dim=0)[0].cpu().numpy()
        all_model_probs.append(p_m)

    prob = np.mean(all_model_probs, axis=0)
    crop_mask = (prob >= REFINE_THR).astype(np.uint8)

    full_mask = np.zeros((h, w), dtype=np.uint8)

    if ASPECT_PRESERVE_PAD and pads is not None:
        max_side = max(y1b - y0, x1b - x0)
        resized_sq = cv2.resize(crop_mask, (max_side, max_side), interpolation=cv2.INTER_NEAREST)
        unpadded = unpad_mask(resized_sq, pads, y1b - y0, x1b - x0)
        full_mask[y0:y1b, x0:x1b] = unpadded
    else:
        resized_mask = cv2.resize(crop_mask, (x1b - x0, y1b - y0), interpolation=cv2.INTER_NEAREST)
        full_mask[y0:y1b, x0:x1b] = resized_mask


    # Remove tiny noise
    if MIN_AREA > 0:

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                full_mask,
                connectivity=8
            )
        )


        cleaned = np.zeros_like(
            full_mask
        )


        for k in range(
            1,
            num_labels
        ):

            area = stats[
                k,
                cv2.CC_STAT_AREA
            ]


            if area >= MIN_AREA:

                cleaned[
                    labels == k
                ] = 1


        full_mask = cleaned


    return (
        full_mask,
        float(conf)
    )


# ============================================================
# 29. OVERLAP RESOLUTION
# ============================================================

def resolve_overlapping_masks(
    scored
):

    if not scored:

        return []


    scored = sorted(
        scored,
        key=lambda x: -x[0]
    )


    occupied = np.zeros_like(
        scored[0][1],
        dtype=bool
    )


    output = []


    for score, mask in scored:

        mask = mask.copy()


        # Don't allow duplicate overlap
        mask[
            occupied
        ] = 0


        if int(
            mask.sum()
        ) == 0:

            continue


        occupied |= (
            mask.astype(bool)
        )


        output.append(
            mask.astype(
                np.uint8
            )
        )


    return output


# ============================================================
# 30. IMAGE PREDICTION
# ============================================================

def seedrefine_predict(
    image_path
):

    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )


    if gray is None:

        raise FileNotFoundError(
            image_path
        )


    # MAGFiLO images are generally 2048x2048
    if gray.shape != (
        2048,
        2048
    ):

        gray = cv2.resize(
            gray,
            (
                2048,
                2048
            ),
            interpolation=cv2.INTER_AREA
        )


    bgr = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


    # Multi-Scale Detector TTA + Weighted Boxes Fusion (WBF)
    has_boxes = False
    boxes = np.empty((0, 4), dtype=np.float32)
    confs = np.empty((0,), dtype=np.float32)

    try:
        from ensemble_boxes import weighted_boxes_fusion  # pyrefly: ignore [missing-import] # type: ignore
        can_wbf = True
    except ImportError:
        can_wbf = False

    if YOLO_MULTI_SCALE and can_wbf and USE_WBF:
        boxes_list = []
        scores_list = []
        labels_list = []
        img_h, img_w = gray.shape

        for sz in YOLO_SCALES:
            r = det_model.predict(
                source=bgr,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                imgsz=sz,
                verbose=False,
                device=0 if DEVICE.type == "cuda" else "cpu"
            )[0]
            if r.boxes is not None and len(r.boxes) > 0:
                b = r.boxes.xyxy.detach().cpu().numpy().copy()
                c = r.boxes.conf.detach().cpu().numpy().copy()
                b[:, [0, 2]] /= img_w
                b[:, [1, 3]] /= img_h
                b = np.clip(b, 0.0, 1.0)
                boxes_list.append(b.tolist())
                scores_list.append(c.tolist())
                labels_list.append([0] * len(c))

            if USE_TTA:
                bgr_f = cv2.flip(bgr, 1)
                rf = det_model.predict(
                    source=bgr_f,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    imgsz=sz,
                    verbose=False,
                    device=0 if DEVICE.type == "cuda" else "cpu"
                )[0]
                if rf.boxes is not None and len(rf.boxes) > 0:
                    bf = rf.boxes.xyxy.detach().cpu().numpy().copy()
                    cf = rf.boxes.conf.detach().cpu().numpy().copy()
                    bf_unflip = bf.copy()
                    bf_unflip[:, 0] = img_w - bf[:, 2]
                    bf_unflip[:, 2] = img_w - bf[:, 0]
                    bf_unflip[:, [0, 2]] /= img_w
                    bf_unflip[:, [1, 3]] /= img_h
                    bf_unflip = np.clip(bf_unflip, 0.0, 1.0)
                    boxes_list.append(bf_unflip.tolist())
                    scores_list.append(cf.tolist())
                    labels_list.append([0] * len(cf))

        if boxes_list and sum(len(b) for b in boxes_list) > 0:
            f_boxes, f_scores, _ = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=WBF_IOU_THR,
                skip_box_thr=YOLO_CONF
            )
            f_boxes[:, [0, 2]] *= img_w
            f_boxes[:, [1, 3]] *= img_h
            boxes = f_boxes
            confs = f_scores
            has_boxes = len(boxes) > 0
    else:
        res = det_model.predict(
            source=bgr,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,
            verbose=False,
            device=0 if DEVICE.type == "cuda" else "cpu"
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            has_boxes = True

    scored = []

    if has_boxes:
        order = np.argsort(-confs)


        for i in order:

            mask, score = (
                refine_box_on_image(
                    gray,
                    boxes[i],
                    float(confs[i])
                )
            )


            if int(
                mask.sum()
            ) >= MIN_AREA:

                scored.append(
                    (
                        score,
                        mask
                    )
                )


    return resolve_overlapping_masks(
        scored
    )


# ============================================================
# 31. IMAGE → SUBMISSION ROWS
# ============================================================

def predict_image_to_rows(
    image_path
):

    image_id = image_path.stem


    masks = seedrefine_predict(
        image_path
    )


    rows = []


    for i, mask in enumerate(
        masks,
        start=1
    ):

        rows.append(
            {
                "filament_id":
                    f"{image_id}_{i}",

                "segmentation_rle":
                    mask_to_rle(mask)
            }
        )


    return rows


# ============================================================
# 32. OPTIONAL VALIDATION PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("RUNNING VALIDATION SANITY CHECK")
print("=" * 70)


if len(val_samples) > 0:

    # Only a small number for sanity check
    check_count = min(
        5,
        len(val_samples)
    )


    for s in tqdm(
        val_samples[:check_count],
        desc="Validation check"
    ):

        try:

            rows = predict_image_to_rows(
                s["path"]
            )


            print(
                s["path"].name,
                "→",
                len(rows),
                "predicted instances"
            )

        except Exception as e:

            print(
                "Validation error:",
                s["path"],
                e
            )


# ============================================================
# 33. FIND ALL TEST IMAGES
# ============================================================

test_paths = sorted(
    [
        p
        for p in TEST_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png"
        }
    ]
)


# Remove duplicates by filename
seen = set()

unique_test_paths = []


for p in test_paths:

    if p.name in seen:
        continue

    seen.add(
        p.name
    )

    unique_test_paths.append(
        p
    )


test_paths = unique_test_paths


if MAX_TEST_IMAGES is not None:

    test_paths = test_paths[
        :MAX_TEST_IMAGES
    ]


print()
print("=" * 70)
print("TEST SET")
print("=" * 70)

print(
    "Test images:",
    len(test_paths)
)


if len(test_paths) == 0:

    raise RuntimeError(
        "❌ No test images found."
    )


# ============================================================
# 34. FINAL TEST INFERENCE
# ============================================================

print()
print("=" * 70)
print("GENERATING FINAL SUBMISSION")
print("=" * 70)


submission_rows = []


for path in tqdm(
    test_paths,
    desc="Test inference"
):

    try:

        rows = predict_image_to_rows(
            path
        )


        submission_rows.extend(
            rows
        )


    except Exception as e:

        print(
            f"\n⚠️ Failed: {path.name}"
        )

        print(
            "Error:",
            repr(e)
        )


# ============================================================
# 35. CREATE SUBMISSION DATAFRAME
# ============================================================

submission_df = pd.DataFrame(
    submission_rows,
    columns=[
        "filament_id",
        "segmentation_rle"
    ]
)


# ============================================================
# 36. SAVE FINAL SUBMISSION
# ============================================================

submission_df.to_csv(
    SUBMISSION_CSV,
    index=False
)


print()
print("=" * 70)
print("🎉 SUBMISSION COMPLETE")
print("=" * 70)


print(
    "Submission path:"
)

print(
    SUBMISSION_CSV
)


print(
    "Rows:",
    len(submission_df)
)


if len(submission_df) > 0:

    unique_images = (
        submission_df[
            "filament_id"
        ]
        .str.rsplit(
            "_",
            n=1
        )
        .str[0]
        .nunique()
    )

else:

    unique_images = 0


print(
    "Images with predictions:",
    unique_images
)


print()
print("First rows:")

display(
    submission_df.head(10)
)


# ============================================================
# 37. BASIC SUBMISSION VALIDATION
# ============================================================

print()
print("=" * 70)
print("SUBMISSION VALIDATION")
print("=" * 70)


assert (
    "filament_id"
    in submission_df.columns
)

assert (
    "segmentation_rle"
    in submission_df.columns
)


if len(submission_df) > 0:

    assert (
        submission_df[
            "filament_id"
        ]
        .notna()
        .all()
    )


    assert (
        submission_df[
            "segmentation_rle"
        ]
        .notna()
        .all()
    )


    assert (
        submission_df[
            "filament_id"
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )


    assert (
        submission_df[
            "segmentation_rle"
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )


print(
    "✅ Required columns present"
)

print(
    "✅ RLE values present"
)

print(
    "✅ Submission format valid"
)


# ============================================================
# 38. FINAL FILE SIZE
# ============================================================

if SUBMISSION_CSV.exists():

    size_mb = (
        SUBMISSION_CSV.stat().st_size
        / (1024 ** 2)
    )


    print(
        f"✅ submission.csv size: "
        f"{size_mb:.2f} MB"
    )


print()
print("=" * 70)
print("FINAL FILE")
print("=" * 70)

print(
    SUBMISSION_CSV
)

print()
print("🚀 Ready for Kaggle submission!")