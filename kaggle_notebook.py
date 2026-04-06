#!/usr/bin/env python3
"""
Smart City Planning — Kaggle Notebook
======================================

Run this notebook on Kaggle with GPU enabled.

Prerequisites (already done by Krish):
  - Dataset uploaded at /kaggle/working/EarthVQA
  - Pretrained weights (sfpnr50.pth, soba.pth) uploaded

This notebook executes 4 steps:
  1. Setup & Install dependencies
  2. Segmentation → Feature Extraction (produces HDF5 files for VQA)
  3. VQA Evaluation using extracted features
  4. Smart City Spatial Analysis + Threshold Calibration

The outputs to download:
  - calibrated thresholds.yaml
  - sample analysis JSON files
  - VQA prediction results
"""

# ============================================================================
# CELL 1: Setup & Configuration
# ============================================================================

import os
import sys
import subprocess

# Install dependencies
print("=" * 60)
print("CELL 1: Installing dependencies...")
print("=" * 60)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'ever-beta',
    'segmentation_models_pytorch',
    'albumentations==1.4.3',
    'h5py',
    'scikit-image',
    'scipy',
    'pyyaml',
    'tqdm',
    'prettytable',
], check=True)

# === CONFIGURE PATHS ===
# Adjust these to match your Kaggle setup
DATASET_DIR = '/kaggle/working/EarthVQA'       # Root of the EarthVQA dataset
WEIGHTS_DIR = '/kaggle/working'                 # Where .pth files are stored

# Auto-detect weight file locations
SEG_WEIGHTS = None
VQA_WEIGHTS = None
for root, dirs, files in os.walk(WEIGHTS_DIR):
    for f in files:
        if f == 'sfpnr50.pth':
            SEG_WEIGHTS = os.path.join(root, f)
        elif f == 'soba.pth':
            VQA_WEIGHTS = os.path.join(root, f)

# Verify paths
print(f"\n📁 Dataset dir:   {DATASET_DIR}")
print(f"   Train images:  {os.path.exists(os.path.join(DATASET_DIR, 'Train/images_png'))}")
print(f"   Val images:    {os.path.exists(os.path.join(DATASET_DIR, 'Val/images_png'))}")
print(f"   Test images:   {os.path.exists(os.path.join(DATASET_DIR, 'Test/images_png'))}")
print(f"   Train_QA.json: {os.path.exists(os.path.join(DATASET_DIR, 'Train_QA.json'))}")
print(f"   Test_QA.json:  {os.path.exists(os.path.join(DATASET_DIR, 'Test_QA.json'))}")
print(f"\n🏋 Seg weights:   {SEG_WEIGHTS}")
print(f"🏋 VQA weights:   {VQA_WEIGHTS}")

# Check GPU
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥 Device: {device}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

assert os.path.exists(DATASET_DIR), f"Dataset not found at {DATASET_DIR}"
assert SEG_WEIGHTS is not None, "sfpnr50.pth not found!"
assert VQA_WEIGHTS is not None, "soba.pth not found!"
print("\n✅ Setup complete!")

# ============================================================================
# CELL 2: Step 1 — Segmentation Feature Extraction (GPU)
# ============================================================================

print("\n" + "=" * 60)
print("CELL 2: Segmentation Feature Extraction")
print("=" * 60)

import ever as er
from ever.core.builder import make_model, make_dataloader
from ever.core.config import import_config
import numpy as np
from tqdm import tqdm
import h5py
import importlib.util
import time

# Setup paths - the EarthVQA code expects to run from its parent dir
# We need to create a working directory structure that matches the config expectations
WORK_DIR = '/kaggle/working'
os.makedirs(os.path.join(WORK_DIR, 'log', 'sfpnr50'), exist_ok=True)

# Clone the EarthVQA code structure (configs, data, module) into working dir
# If your notebook has the full repo, adjust accordingly
EARTHVQA_CODE_DIR = WORK_DIR  # Where configs/, data/, module/ dirs exist

# Add to path
sys.path.insert(0, EARTHVQA_CODE_DIR)
os.chdir(EARTHVQA_CODE_DIR)

er.registry.register_all()

# Import and register the SemanticFPN model (has hyphenated filename)
sfpn_path = os.path.join(EARTHVQA_CODE_DIR, 'module', 'semantic-fpn.py')
if os.path.exists(sfpn_path):
    spec = importlib.util.spec_from_file_location("semantic_fpn", sfpn_path)
    sfpn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfpn_module)
    print("✅ SemanticFPN model registered")

# Load segmentation model
cfg = import_config('sfpnr50')
model_state_dict = torch.load(SEG_WEIGHTS, map_location=device)
model = make_model(cfg['model'])
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
print(f"✅ Segmentation model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")


import shutil

def check_disk():
    """Print available disk space."""
    usage = shutil.disk_usage('/kaggle/working')
    print(f"   💾 Disk: {usage.used/1e9:.1f}GB used / {usage.total/1e9:.1f}GB total / {usage.free/1e9:.1f}GB free")


def extract_features(model, split='test', batch_size=4):
    """
    Run segmentation on a split and save features + masks as HDF5.
    
    SPACE-OPTIMIZED:
    - Features stored as float16 (halves size: 2MB → 1MB per image)
    - HDF5 uses gzip compression (additional ~30% reduction)
    - pred_mask stored as uint8 (minimal)
    
    This produces the output format that the SOBA VQA model expects:
    - feature: (2048, H/32, W/32) array
    - pred_mask: (H, W) uint8 array (1-indexed, matching EarthVQA convention)
    """
    from albumentations import Compose, Normalize
    import ever as er
    from data.lovedav2 import LoveDALoaderV2
    
    split_name = split.capitalize()
    image_dir = os.path.join(DATASET_DIR, f'{split_name}/images_png')
    save_dir = os.path.join(WORK_DIR, 'log', 'sfpnr50', f'{split}_features')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Processing {split_name} split ---")
    print(f"   Images: {image_dir}")
    print(f"   Output: {save_dir}")
    check_disk()
    
    if not os.path.exists(image_dir):
        print(f"   ⚠ Image directory not found, skipping")
        return 0
    
    # Count images
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"   Found {len(image_files)} images")
    
    # Create dataloader using the LoveDA config (with test transforms)
    transforms = Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ])
    
    test_cfg = dict(
        type='LoveDALoaderV2',
        params=dict(
            image_dir=image_dir,
            mask_dir=None,
            transforms=transforms,
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=batch_size,
            num_workers=2,
        )
    )
    dataloader = make_dataloader(test_cfg)
    
    count = 0
    t0 = time.time()
    
    with torch.no_grad():
        for img, gt in tqdm(dataloader, desc=f'{split_name} features'):
            pred, img_feat = model(img.to(device))
            pred = pred.argmax(dim=1).cpu()
            
            for clsmap, feat_i, imname in zip(pred, img_feat, gt['imagen']):
                clsmap = clsmap.cpu().numpy().astype(np.uint8)
                feat_np = feat_i.cpu().numpy().astype(np.float16)  # float16 to save space
                hdf_path = os.path.join(save_dir, imname.replace('.png', '.hdf5'))
                with h5py.File(hdf_path, 'w') as f:
                    f.create_dataset('feature', data=feat_np,
                                     compression='gzip', compression_opts=1)  # fast compression
                    f.create_dataset('pred_mask', data=clsmap + 1)  # 1-indexed
                count += 1
            
            torch.cuda.empty_cache()
    
    elapsed = time.time() - t0
    print(f"   ✅ Processed {count} images in {elapsed:.1f}s ({count/elapsed:.1f} img/s)")
    check_disk()
    return count


# ┌──────────────────────────────────────────────────────────────────────┐
# │ IMPORTANT: Only extract TEST features for inference.                │
# │ Training features are NOT needed (we're using pretrained weights).  │
# │ Calibration (Cell 4) uses raw mask PNGs, not HDF5 features.         │
# │                                                                      │
# │ This saves ~5-6 GB of disk space.                                   │
# └──────────────────────────────────────────────────────────────────────┘

test_count = extract_features(model, 'test', batch_size=4)

print(f"\n✅ Feature extraction complete!")
print(f"   Test:  {test_count} images → log/sfpnr50/test_features/")
print(f"   (Train features skipped — not needed for inference)")
check_disk()

# ============================================================================
# CELL 3: Step 2 — VQA Evaluation (GPU)
# ============================================================================

print("\n" + "=" * 60)
print("CELL 3: VQA Evaluation")
print("=" * 60)

from data.earthvqa import EarthVQADataset
import json

def convert2str(indexes, map_dict=EarthVQADataset.QUESTION_VOC):
    if isinstance(indexes, np.int64):
        return map_dict[indexes]
    return ' '.join([map_dict[idx] for idx in indexes if map_dict[idx] != ' ']) + '?'


def run_vqa_evaluation():
    """Run VQA model on test set using extracted features."""
    from module.soba import SOBA  # noqa — registers via decorator
    
    cfg = import_config('soba')
    model_state_dict = torch.load(VQA_WEIGHTS, map_location=device)
    vqa_model = make_model(cfg['model'])
    vqa_model.load_state_dict(model_state_dict)
    vqa_model.to(device)
    vqa_model.eval()
    print(f"✅ VQA model loaded ({sum(p.numel() for p in vqa_model.parameters()) / 1e6:.1f}M params)")
    
    # Create test dataloader (uses HDF5 features, not raw images)
    test_dataloader = make_dataloader(cfg['data']['test'])
    
    pred_dict = dict()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for img, ret in tqdm(test_dataloader, desc='VQA evaluation'):
            ques, questypes, imagen = ret['question'], ret['questype'], ret['imagen']
            
            # Move data to GPU
            preds = vqa_model(img, ret)
            
            if isinstance(ques[0], str):
                ques = [q_i + '?' for q_i in ques]
            else:
                ques = [convert2str(q_i, EarthVQADataset.QUESTION_VOC) for q_i in ques]
            
            ans_idx = preds.argmax(dim=1).cpu().numpy()
            
            # Collect predictions
            for q_str, qt, ans_i, img_name in zip(ques, questypes, ans_idx, imagen):
                qa_list = pred_dict.get(img_name, [])
                ans_str = convert2str(ans_i, EarthVQADataset.ANSWER_VOC)
                qa_list.append({
                    'Type': qt,
                    'Question': q_str,
                    'Answer': ans_str,
                })
                pred_dict[img_name] = qa_list
                total += 1
    
    # Save predictions
    pred_path = os.path.join(WORK_DIR, 'vqa_predictions.json')
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, ensure_ascii=False, indent=1)
    
    print(f"\n✅ VQA evaluation complete!")
    print(f"   Total QA predictions: {total}")
    print(f"   Predictions saved to: {pred_path}")
    
    # Print sample predictions
    print(f"\n   Sample predictions:")
    for img_name, qas in list(pred_dict.items())[:3]:
        print(f"\n   Image: {img_name}")
        for qa in qas[:3]:
            print(f"     Q: {qa['Question']}")
            print(f"     A: {qa['Answer']}")
    
    return pred_dict


vqa_predictions = run_vqa_evaluation()

# ============================================================================
# CELL 4: Smart City Analysis + Threshold Calibration
# ============================================================================

print("\n" + "=" * 60)
print("CELL 4: Smart City Spatial Analysis & Calibration")
print("=" * 60)

# Install smart city dependencies (scipy, scikit-image already installed)
import yaml
from glob import glob

# --- We inline the smart city modules here for Kaggle (no need to upload the full repo) ---

# 4a. Inline SpatialFeatureEngine and DecisionEngine
# (Copy-paste from smart_city/ modules, or mount the repo)

# For now, we'll use a simplified inline version:

from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label as ndimage_label

CLASS_NAMES = {
    0: 'background', 1: 'building', 2: 'road', 3: 'water',
    4: 'barren', 5: 'forest', 6: 'agriculture', 7: 'playground',
}


def extract_spatial_features(seg_mask):
    """Quick spatial feature extraction for calibration."""
    seg_mask = seg_mask.astype(np.int32)
    h, w = seg_mask.shape
    total = h * w
    
    features = {}
    for cid, cname in CLASS_NAMES.items():
        binary = (seg_mask == cid)
        pix_count = int(binary.sum())
        area_pct = pix_count / total if total > 0 else 0.0
        
        if pix_count > 0:
            labeled, n_comp = ndimage_label(binary)
            features[cname] = {'area_pct': area_pct, 'components': n_comp}
        else:
            features[cname] = {'area_pct': 0.0, 'components': 0}
    
    # Building-water distance
    bw_dist = float('inf')
    if features['building']['area_pct'] > 0 and features['water']['area_pct'] > 0:
        mask_b = (seg_mask == 1)
        mask_w = (seg_mask == 3)
        dist_to_w = distance_transform_edt(~mask_w)
        bw_dist = float(dist_to_w[mask_b].min())
    
    return {
        'building_count': features['building']['components'],
        'building_area_pct': features['building']['area_pct'],
        'road_area_pct': features['road']['area_pct'],
        'water_area_pct': features['water']['area_pct'],
        'vegetation_pct': features['forest']['area_pct'] + features['agriculture']['area_pct'] + features['playground']['area_pct'],
        'building_water_dist': bw_dist,
    }


def calibrate_from_masks(mask_dir, sample_limit=0):
    """Calibrate thresholds from training masks."""
    from skimage.io import imread
    
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
    if sample_limit > 0:
        mask_paths = mask_paths[:sample_limit]
    
    print(f"Calibrating from {len(mask_paths)} masks...")
    
    stats = {
        'building_count': [], 'building_area_pct': [], 'road_area_pct': [],
        'water_area_pct': [], 'vegetation_pct': [], 'building_water_dist': [],
    }
    
    for mp in tqdm(mask_paths, desc="Processing masks"):
        try:
            mask = imread(mp)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.astype(np.int32) - 1  # EarthVQA masks are 1-indexed
            mask = np.clip(mask, 0, 7)
            
            feat = extract_spatial_features(mask)
            for k in stats:
                v = feat[k]
                if v != float('inf'):
                    stats[k].append(v)
        except Exception as e:
            continue
    
    # Compute percentile statistics
    calibrated_stats = {}
    for k, vals in stats.items():
        if vals:
            arr = np.array(vals)
            calibrated_stats[k] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'p90': float(np.percentile(arr, 90)),
                'max': float(np.max(arr)),
            }
    
    return calibrated_stats


# 4b. Run calibration on training masks
train_mask_dir = os.path.join(DATASET_DIR, 'Train/masks_png')
if os.path.exists(train_mask_dir):
    cal_stats = calibrate_from_masks(train_mask_dir)
    
    print("\n=== Calibration Statistics ===")
    for key, s in cal_stats.items():
        print(f"  {key:>25s}: mean={s['mean']:.4f}  p50={s['p50']:.4f}  p75={s['p75']:.4f}  max={s['max']:.4f}")
    
    # Generate calibrated thresholds.yaml
    bc = cal_stats.get('building_count', {})
    ba = cal_stats.get('building_area_pct', {})
    ra = cal_stats.get('road_area_pct', {})
    vp = cal_stats.get('vegetation_pct', {})
    
    calibrated_config = {
        'density': {
            'weights': {'building_count': 0.4, 'built_area_pct': 0.6},
            'thresholds': {
                'low': round(ba.get('p25', 0.1) * 2, 2),
                'moderate': round(ba.get('p50', 0.2) * 2, 2),
                'high': round(ba.get('p75', 0.35) * 2, 2),
            },
            'recommendations': {
                'high': 'Area is densely populated. Consider limiting further construction.',
                'moderate': 'Moderate building density. Suitable for controlled expansion.',
                'low': 'Low building density. Potential for development.',
            },
        },
        'green_coverage': {
            'weights': {'vegetation_pct': 1.0, 'playground_pct': 0.5},
            'thresholds': {
                'insufficient': 0.10,
                'low': round(vp.get('p25', 0.15), 2),
                'adequate': round(vp.get('p50', 0.25), 2),
                'good': round(vp.get('p75', 0.40), 2),
            },
            'target_pct': round(vp.get('p50', 0.20), 2),
            'recommendations': {
                'insufficient': 'Critical shortage of green spaces.',
                'low': 'Below recommended green coverage.',
                'adequate': 'Green coverage meets minimum standards.',
                'good': 'Excellent green coverage.',
            },
        },
        'flood_risk': {
            'weights': {'water_proximity': 0.5, 'water_area_pct': 0.3, 'building_water_ratio': 0.2},
            'thresholds': {'low': 0.25, 'moderate': 0.50, 'high': 0.70},
            'min_safe_distance_px': 50,
            'recommendations': {
                'high': 'High flood risk. Recommend buffer zones and flood barriers.',
                'moderate': 'Moderate flood risk. Consider drainage improvements.',
                'low': 'Low flood risk.',
            },
        },
        'infrastructure': {
            'weights': {'road_coverage_pct': 0.5, 'intersection_count': 0.3, 'road_connectivity': 0.2},
            'thresholds': {
                'poor': round(ra.get('p25', 0.05) * 4, 2),
                'moderate': round(ra.get('p50', 0.1) * 4, 2),
                'good': round(ra.get('p75', 0.15) * 4, 2),
            },
            'recommendations': {
                'poor': 'Poor road connectivity. Recommend expanding road network.',
                'moderate': 'Basic road infrastructure. Consider secondary roads.',
                'good': 'Good road connectivity.',
            },
        },
        'planning': {
            'suitability_weights': {
                'density': 0.25, 'green_coverage': 0.25,
                'flood_risk': 0.25, 'infrastructure': 0.25,
            },
            'suitability_labels': [
                {'label': 'Not Suitable', 'range': [0.0, 0.3]},
                {'label': 'Needs Improvement', 'range': [0.3, 0.5]},
                {'label': 'Moderately Suitable', 'range': [0.5, 0.7]},
                {'label': 'Suitable', 'range': [0.7, 0.85]},
                {'label': 'Highly Suitable', 'range': [0.85, 1.0]},
            ],
        },
        'normalization': {
            'building_count_max': int(bc.get('max', 200)),
            'intersection_count_max': 10,
            'image_area_px': 262144,
        },
    }
    
    # Save calibrated config
    cal_output = os.path.join(WORK_DIR, 'calibrated_thresholds.yaml')
    with open(cal_output, 'w') as f:
        yaml.dump(calibrated_config, f, default_flow_style=False, sort_keys=False)
    print(f"\n✅ Calibrated thresholds saved to: {cal_output}")
else:
    print(f"⚠ Training masks not found at {train_mask_dir}")


# 4c. Run spatial analysis on sample test images
print("\n--- Sample Analysis on Test Masks ---")
test_feat_dir = os.path.join(WORK_DIR, 'log', 'sfpnr50', 'test_features')
sample_results = {}

if os.path.exists(test_feat_dir):
    hdf_files = sorted(glob(os.path.join(test_feat_dir, '*.hdf5')))[:10]  # 10 samples
    
    for hdf_path in hdf_files:
        img_name = os.path.basename(hdf_path).replace('.hdf5', '.png')
        with h5py.File(hdf_path, 'r') as f:
            pred_mask = np.array(f['pred_mask']).astype(np.int32) - 1  # Convert to 0-indexed
        
        feat = extract_spatial_features(pred_mask)
        sample_results[img_name] = feat
        
        print(f"  {img_name}: buildings={feat['building_count']}, "
              f"built={feat['building_area_pct']*100:.1f}%, "
              f"veg={feat['vegetation_pct']*100:.1f}%, "
              f"water={feat['water_area_pct']*100:.1f}%")
    
    # Save sample results
    with open(os.path.join(WORK_DIR, 'sample_analysis.json'), 'w') as f:
        json.dump(sample_results, f, indent=2)
    print(f"\n✅ Sample analysis saved to: {WORK_DIR}/sample_analysis.json")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 ALL STEPS COMPLETE!")
print("=" * 60)
print(f"""
Files to download from Kaggle:
  1. calibrated_thresholds.yaml → Replace smart_city/config/thresholds.yaml
  2. vqa_predictions.json       → VQA model predictions for all test images
  3. sample_analysis.json       → Spatial analysis for sample images
  
Next steps (on your laptop):
  1. Copy calibrated_thresholds.yaml to smart_city/config/thresholds.yaml
  2. Run the Flask backend: python backend/app.py
  3. Build the React frontend
""")
