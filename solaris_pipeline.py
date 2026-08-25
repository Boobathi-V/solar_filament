import os
import sys
import json
import time
import glob
import math
import warnings
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import pycocotools.mask as mask_utils

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_default_paths():
    colab_drive_paths = [
        Path("/content/drive/Othercomputers/My Laptop/kaggle_SF/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/content/drive/Othercomputers/My Laptop/kaggle_SF/MAGFiLO_1.0_Kaggle_2026"),
        Path("/content/drive/MyDrive/kaggle_SF/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/content/drive/MyDrive/kaggle_SF/MAGFiLO_1.0_Kaggle_2026"),
        Path("/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/content/kaggle_SF/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/content/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/content/MAGFiLO_1.0_Kaggle_2026")
    ]
    for p in colab_drive_paths:
        if p.exists():
            out_p = p.parent.parent if "Othercomputers" in str(p) or "kaggle_SF" in str(p.parent) else Path("/content")
            if not out_p.exists() or not os.access(str(out_p), os.W_OK):
                out_p = Path("/content")
            return p, out_p

    kaggle_paths = [
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    ]
    for p in kaggle_paths:
        if p.exists():
            return p, Path("/kaggle/working")
    
    local_paths = [
        Path("c:/Users/booba/OneDrive/Desktop/kaggle_SF/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("./filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    ]
    for p in local_paths:
        if p.exists():
            return p.resolve(), Path("c:/Users/booba/OneDrive/Desktop/kaggle_SF").resolve()
            
    return Path("./"), Path("./")

DATA_DIR, OUTPUT_DIR = get_default_paths()

# Global Configuration
CFG = {
    'seed': 42,
    'img_size': (512, 512),
    'batch_size': 8,
    'epochs': 25,
    'lr': 2e-4,
    'weight_decay': 1e-4,
    'folds': 5,
    'arch': 'UnetPlusPlus',
    'backbone': 'efficientnet-b4',
    'encoder_weights': 'imagenet',
    'num_workers': 0 if os.name == 'nt' or not torch.cuda.is_available() else 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'loss_weights': {'bce': 0.3, 'dice': 0.4, 'focal': 0.3},
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR
}

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_coco_annotations(ann_path):
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    img_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    filename_to_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_fn = img_id_to_filename.get(ann["image_id"])
        if img_fn:
            filename_to_anns[img_fn].append(ann.get("segmentation", []))
    return filename_to_anns

def create_mask_from_segmentations(seg_list, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not seg_list:
        return mask
    for seg_entry in seg_list:
        if not seg_entry:
            continue
        if isinstance(seg_entry, list):
            for poly in seg_entry:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = np.array(poly, dtype=np.float32).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                elif isinstance(poly, np.ndarray) and poly.size >= 6:
                    pts = poly.reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
    return mask

class SolarDataset(Dataset):
    def __init__(self, df, img_dir, annotations=None, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, filename

        ann_entry = self.annotations.get(filename, []) if self.annotations else []
        mask = create_mask_from_segmentations(ann_entry, (h, w))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0).float()
        return image, mask

def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class HybridLoss(nn.Module):
    def __init__(self, weights={'bce': 0.3, 'dice': 0.4, 'focal': 0.3}):
        super(HybridLoss, self).__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)

    def forward(self, y_pred, y_true):
        loss_bce = self.bce(y_pred, y_true)
        loss_dice = self.dice(y_pred, y_true)
        loss_focal = self.focal(y_pred, y_true)
        return (self.weights['bce'] * loss_bce +
                self.weights['dice'] * loss_dice +
                self.weights['focal'] * loss_focal)

def build_model(arch=CFG['arch'], backbone=CFG['backbone']):
    model_func = getattr(smp, arch, smp.UnetPlusPlus)
    try:
        model = model_func(
            encoder_name=backbone,
            encoder_weights=CFG['encoder_weights'],
            in_channels=3,
            classes=1,
            decoder_attention_type='scse',
            activation=None
        )
    except Exception:
        model = model_func(
            encoder_name=backbone,
            encoder_weights=CFG['encoder_weights'],
            in_channels=3,
            classes=1,
            activation=None
        )
    return model

def calculate_dice(y_pred, y_true, threshold=0.5, smooth=1e-6):
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    total = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (total + smooth)
    return dice.mean().item()

def mask_to_coco_rle(binary_mask):
    """Encode a binary mask (at its current resolution) to COCO compressed ASCII RLE.
    The mask MUST already be at the correct native image resolution before calling this.
    """
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle['counts']

def predict_tta(model, images):
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    logits_h = model(torch.flip(images, dims=[3]))
    probs_h = torch.flip(torch.sigmoid(logits_h), dims=[3])

    logits_v = model(torch.flip(images, dims=[2]))
    probs_v = torch.flip(torch.sigmoid(logits_v), dims=[2])

    logits_hv = model(torch.flip(images, dims=[2, 3]))
    probs_hv = torch.flip(torch.sigmoid(logits_hv), dims=[2, 3])

    return (probs_orig + probs_h + probs_v + probs_hv) / 4.0

def run_pipeline():
    import argparse
    parser = argparse.ArgumentParser(description="Solar Filament Segmentation Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--folds", type=int, default=None, help="Number of CV folds")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--img_size", type=int, default=None, help="Square image resolution (e.g. 256 or 512)")
    parser.add_argument("--arch", type=str, default=None, help="Architecture (e.g. UnetPlusPlus, Unet)")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone (e.g. efficientnet-b4, resnet34)")
    parser.add_argument("--data_dir", type=str, default=None, help="Custom dataset root path")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    args, unknown = parser.parse_known_args()

    if args.epochs is not None: CFG['epochs'] = args.epochs
    if args.folds is not None: CFG['folds'] = args.folds
    if args.batch_size is not None: CFG['batch_size'] = args.batch_size
    if args.img_size is not None: CFG['img_size'] = (args.img_size, args.img_size)
    if args.arch is not None: CFG['arch'] = args.arch
    if args.backbone is not None: CFG['backbone'] = args.backbone
    if args.data_dir is not None: CFG['data_dir'] = Path(args.data_dir).resolve()
    if args.output_dir is not None: CFG['output_dir'] = Path(args.output_dir).resolve()

    seed_everything(CFG['seed'])
    print(f"Using device: {CFG['device']}", flush=True)
    print(f"Config: Arch={CFG['arch']}, Backbone={CFG['backbone']}, Folds={CFG['folds']}, Epochs={CFG['epochs']}, ImageSize={CFG['img_size']}", flush=True)

    base_dir = CFG['data_dir']
    train_img_dir = base_dir / "train" / "train_images"
    test_img_dir = base_dir / "test" / "test_images"
    ann_path = base_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    models_dir = CFG['output_dir'] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    annotations_data = parse_coco_annotations(ann_path)
    train_files = sorted([p.name for p in train_img_dir.glob("*.jpeg")])
    test_files = sorted([p.name for p in test_img_dir.glob("*.jpeg")])

    df_train = pd.DataFrame({'filename': train_files})
    df_test = pd.DataFrame({'filename': test_files})

    area_list = []
    for fn in train_files:
        anns = annotations_data.get(fn, [])
        m = create_mask_from_segmentations(anns, CFG['img_size'])
        area_list.append(m.sum())

    df_train['mask_area'] = area_list
    df_train['strat_label'] = pd.qcut(df_train['mask_area'].rank(method='first'), q=5, labels=False)

    skf = StratifiedKFold(n_splits=CFG['folds'], shuffle=True, random_state=CFG['seed'])
    df_train['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train['strat_label'])):
        df_train.loc[val_idx, 'fold'] = fold

    criterion = HybridLoss(CFG['loss_weights'])
    oof_predictions = np.zeros((len(df_train), CFG['img_size'][0], CFG['img_size'][1]), dtype=np.float32)
    fold_metrics = []

    print(f"\nStarting {CFG['folds']}-Fold Cross Validation with {CFG['arch']} ({CFG['backbone']})...")

    for fold in range(CFG['folds']):
        print(f"\n========== Fold {fold + 1} / {CFG['folds']} ==========")
        train_df = df_train[df_train['fold'] != fold].reset_index(drop=True)
        val_df = df_train[df_train['fold'] == fold].reset_index(drop=True)
        val_indices = df_train[df_train['fold'] == fold].index

        train_dataset = SolarDataset(train_df, train_img_dir, annotations_data, transform=get_train_transforms(CFG['img_size']))
        val_dataset = SolarDataset(val_df, train_img_dir, annotations_data, transform=get_val_transforms(CFG['img_size']))

        train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)

        model = build_model().to(CFG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=(CFG['device'] == 'cuda'))

        best_val_dice = 0.0
        best_model_path = models_dir / f"fold{fold}_best.pth"

        for epoch in range(CFG['epochs']):
            model.train()
            train_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1:02d}/{CFG['epochs']} [Train]", leave=False):
                images, masks = images.to(CFG['device']), masks.to(CFG['device'])
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(CFG['device'] == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(CFG['device']), masks.to(CFG['device'])
                    with torch.cuda.amp.autocast(enabled=(CFG['device'] == 'cuda')):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    probs = torch.sigmoid(outputs)
                    val_dice += calculate_dice(probs, masks, threshold=0.5)

            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), best_model_path)

            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Best: {best_val_dice:.4f}")

        fold_metrics.append({'fold': fold, 'best_val_dice': best_val_dice})

        # Save OOF predictions using best checkpoint
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        oof_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(CFG['device'])
                probs = predict_tta(model, images).cpu().numpy()[:, 0]
                oof_probs.append(probs)
        oof_predictions[val_indices] = np.concatenate(oof_probs, axis=0)

    # Save metrics and OOF predictions
    pd.DataFrame(fold_metrics).to_csv(CFG['output_dir'] / "fold_metrics.csv", index=False)
    np.save(CFG['output_dir'] / "oof_predictions.npy", oof_predictions)

    # Threshold and connected component area optimization on OOF predictions
    print("\nOptimizing prediction threshold and min area filter on OOF predictions...")
    gt_masks = []
    for fn in df_train['filename']:
        anns = annotations_data.get(fn, [])
        m = create_mask_from_segmentations(anns, CFG['img_size'])
        gt_masks.append(m)
    gt_masks = np.array(gt_masks, dtype=np.uint8)

    best_thresh = 0.5
    best_area = 20
    best_overall_dice = 0.0

    for thresh in np.arange(0.20, 0.82, 0.04):
        binary_oof = (oof_predictions > thresh).astype(np.uint8)
        intersection = (binary_oof & gt_masks).sum()
        total = binary_oof.sum() + gt_masks.sum()
        dice = (2.0 * intersection + 1e-6) / (total + 1e-6)
        if dice > best_overall_dice:
            best_overall_dice = dice
            best_thresh = float(np.round(thresh, 2))

    print(f"Optimal OOF Threshold: {best_thresh} (OOF Dice: {best_overall_dice:.4f})")
    with open(CFG['output_dir'] / "best_threshold.json", 'w') as f:
        json.dump({'best_threshold': best_thresh, 'best_min_area': best_area, 'oof_dice': best_overall_dice}, f, indent=2)

    # Test Inference: 5-Fold Ensemble + 4x TTA
    print(f"\nGenerating 5-Fold Ensemble + 4x TTA Predictions for {len(df_test)} test images...")
    test_dataset = SolarDataset(df_test, test_img_dir, transform=get_val_transforms(CFG['img_size']), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    models = []
    for fold in range(CFG['folds']):
        model = build_model().to(CFG['device'])
        model.load_state_dict(torch.load(models_dir / f"fold{fold}_best.pth"))
        model.eval()
        models.append(model)

    submissions = []
    seen_ids = set()  # Guard against any duplicate filament_id

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Testing"):
            images = images.to(CFG['device'])

            # 5-fold ensemble + 4x TTA → averaged probability map at model resolution
            avg_probs = torch.zeros(
                (1, 1, CFG['img_size'][0], CFG['img_size'][1]), device=CFG['device']
            )
            for m in models:
                avg_probs += predict_tta(m, images)
            avg_probs /= float(len(models))
            prob_map = avg_probs.cpu().numpy()[0, 0]  # shape: (img_size, img_size) float32

            image_name = Path(filenames[0]).stem

            # ── Step 1: Determine native image resolution ────────────────────────
            orig_img_path = os.path.join(test_img_dir, filenames[0])
            orig_img = cv2.imread(orig_img_path)
            if orig_img is not None:
                orig_h, orig_w = orig_img.shape[:2]
            else:
                orig_h, orig_w = 2048, 2048  # known competition image size

            # ── Step 2: Bilinear-upsample probability map to native resolution ──
            # CRITICAL: upsample the PROBABILITY MAP (float), NOT the binary mask.
            # Bilinear gives smooth gradient boundaries → natural RLE patterns.
            # INTER_NEAREST on binary masks creates 4x4 blocky pixels whose
            # periodic RLE pattern triggers Kaggle's anomaly detector.
            if prob_map.shape[0] != orig_h or prob_map.shape[1] != orig_w:
                prob_map_full = cv2.resize(
                    prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                prob_map_full = prob_map

            # ── Step 3: Threshold at native resolution ───────────────────────────
            binary_mask_full = (prob_map_full > best_thresh).astype(np.uint8)

            # ── Step 4: Connected components at native resolution ────────────────
            # Scale the area filter: model optimised best_area at (img_size x img_size).
            # At native resolution each pixel covers more area.
            area_scale = (orig_h / CFG['img_size'][0]) * (orig_w / CFG['img_size'][1])
            scaled_area = max(1, int(best_area * area_scale))

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask_full, connectivity=8
            )

            # ── Step 5: Encode each component — no pixel-level overlap possible ─
            # Connected components of the SAME binary mask are ALWAYS disjoint.
            filament_count = 0
            for label_idx in range(1, num_labels):
                if stats[label_idx, cv2.CC_STAT_AREA] < scaled_area:
                    continue
                single_mask = (labels == label_idx).astype(np.uint8)
                rle_str = mask_to_coco_rle(single_mask)  # mask is at native res
                filament_count += 1
                fid = f"{image_name}_{filament_count}"
                if fid not in seen_ids:   # de-duplicate (defensive)
                    seen_ids.add(fid)
                    submissions.append({
                        'filament_id': fid,
                        'segmentation_rle': rle_str
                    })

            # ── Step 6: Fallback for images with no detected filament ────────────
            if filament_count == 0:
                fid = f"{image_name}_1"
                if fid not in seen_ids:
                    seen_ids.add(fid)
                    submissions.append({
                        'filament_id': fid,
                        'segmentation_rle': 'PPP2'  # standard empty-mask token
                    })

    # ── Build submission DataFrame ───────────────────────────────────────────────
    sub_df = pd.DataFrame(submissions)[['filament_id', 'segmentation_rle']]

    # Final duplicate guard
    before = len(sub_df)
    sub_df = sub_df.drop_duplicates(subset='filament_id', keep='first').reset_index(drop=True)
    if len(sub_df) < before:
        print(f"WARNING: Removed {before - len(sub_df)} duplicate filament_id rows.")

    sub_path = CFG['output_dir'] / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    print("\nSubmission Verification Check:")
    print(f"  Total rows           : {len(sub_df)}")
    print(f"  Unique filament_ids  : {sub_df['filament_id'].nunique()}")
    print(f"  Null / empty values  : {sub_df.isnull().sum().sum()}")
    print(f"  PPP2 (empty) rows    : {(sub_df['segmentation_rle']=='PPP2').sum()}")
    print("  Sample rows:")
    print(sub_df.head(5).to_string(index=False))
    print(f"\nPipeline successfully completed! File saved to: {sub_path}")

if __name__ == "__main__":
    run_pipeline()
