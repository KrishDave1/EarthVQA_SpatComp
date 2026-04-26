"""
Training Script for Change Detection Classifier (Module 8b)

Trains an SVM and Random Forest classifier to categorize urban sprawl
from delta (Δ) spatial features. Designed to work with:

    1. LEVIR-CD dataset (from Kaggle) — real satellite image pairs
    2. Synthetic data generation — fallback when no dataset available

Usage:
    # With LEVIR-CD dataset:
    python -m smart_city.train_change_classifier --dataset_dir /path/to/LEVIR-CD/train

    # With synthetic data (no external dataset needed):
    python -m smart_city.train_change_classifier --synthetic --num_samples 3000

    # On Kaggle (auto-detects LEVIR-CD path):
    python -m smart_city.train_change_classifier --kaggle

Output:
    smart_city/config/change_classifier.joblib
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from smart_city.change_detector import (
    DeltaFeatures, SPRAWL_CATEGORIES, ChangeDetector
)
from smart_city.spatial_engine import SpatialFeatureEngine
from smart_city.models import SpatialFeatures

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# ─── Synthetic Data Generation ────────────────────────────────────────

def generate_synthetic_mask(
    h: int = 512, w: int = 512,
    building_pct: float = 0.10,
    road_pct: float = 0.05,
    water_pct: float = 0.05,
    forest_pct: float = 0.20,
    agriculture_pct: float = 0.10,
    barren_pct: float = 0.05,
    playground_pct: float = 0.02,
) -> np.ndarray:
    """
    Generate a synthetic segmentation mask with approximate class percentages.

    Classes: 0=bg, 1=building, 2=road, 3=water, 4=barren, 5=forest, 6=agriculture, 7=playground
    """
    total = h * w
    mask = np.zeros((h, w), dtype=np.int32)  # background by default

    # Assign pixels for each class in order of priority
    classes = [
        (1, building_pct),
        (2, road_pct),
        (3, water_pct),
        (4, barren_pct),
        (5, forest_pct),
        (6, agriculture_pct),
        (7, playground_pct),
    ]

    available = np.ones((h, w), dtype=bool)
    rng = np.random.default_rng()

    for class_id, pct in classes:
        n_pixels = int(pct * total)
        if n_pixels == 0:
            continue

        # Get available pixel indices
        avail_indices = np.where(available.ravel())[0]
        if len(avail_indices) < n_pixels:
            n_pixels = len(avail_indices)

        chosen = rng.choice(avail_indices, size=n_pixels, replace=False)
        rows, cols = np.unravel_index(chosen, (h, w))
        mask[rows, cols] = class_id
        available[rows, cols] = False

    return mask


def generate_synthetic_dataset(
    num_samples: int = 3000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic training data: delta features + sprawl labels.

    Creates random "before" percentage distributions, applies category-specific
    perturbations to create "after" distributions, and directly constructs
    DeltaFeatures from the differences (no mask generation needed — fast).

    Returns:
        X: (num_samples, n_features) feature matrix
        y: (num_samples,) integer labels
        label_names: list of category names
    """
    rng = np.random.default_rng(seed)

    samples_per_category = num_samples // len(SPRAWL_CATEGORIES)
    X_list = []
    y_list = []

    print(f"[Training] Generating {num_samples} synthetic samples...")
    print(f"[Training] {samples_per_category} samples per category")

    for cat_idx, category in enumerate(SPRAWL_CATEGORIES):
        print(f"  Generating: {category} ({samples_per_category} samples)")

        for i in range(samples_per_category):
            # Random base percentages
            base = {
                'building': rng.uniform(0.0, 0.30),
                'road': rng.uniform(0.0, 0.10),
                'water': rng.uniform(0.0, 0.10),
                'forest': rng.uniform(0.0, 0.40),
                'agriculture': rng.uniform(0.0, 0.20),
                'barren': rng.uniform(0.0, 0.15),
                'playground': rng.uniform(0.0, 0.05),
            }

            # Apply category-specific perturbation to create "after"
            after = dict(base)

            if category == 'Aggressive Urbanization':
                after['building'] = base['building'] + rng.uniform(0.08, 0.25)
                after['forest'] = max(0, base['forest'] - rng.uniform(0.08, 0.20))
                after['agriculture'] = max(0, base['agriculture'] - rng.uniform(0.02, 0.08))

            elif category == 'Deforestation':
                after['forest'] = max(0, base['forest'] - rng.uniform(0.10, 0.30))
                after['barren'] = base['barren'] + rng.uniform(0.05, 0.15)
                after['agriculture'] = base['agriculture'] + rng.uniform(0.02, 0.10)

            elif category == 'Water Encroachment':
                after['water'] = base['water'] + rng.uniform(0.05, 0.15)
                after['building'] = base['building'] + rng.uniform(0.0, 0.05)
                after['forest'] = max(0, base['forest'] - rng.uniform(0.0, 0.05))

            elif category == 'Sustainable Expansion':
                after['building'] = base['building'] + rng.uniform(0.02, 0.08)
                after['road'] = base['road'] + rng.uniform(0.01, 0.04)
                after['forest'] = base['forest'] + rng.uniform(-0.02, 0.02)
                after['playground'] = base['playground'] + rng.uniform(0, 0.02)

            elif category == 'Infrastructure Development':
                after['road'] = base['road'] + rng.uniform(0.05, 0.15)
                after['building'] = base['building'] + rng.uniform(0.0, 0.03)
                after['barren'] = max(0, base['barren'] - rng.uniform(0.0, 0.05))

            elif category == 'Stable / No Change':
                for key in after:
                    after[key] = max(0, base[key] + rng.uniform(-0.015, 0.015))

            # Clamp all values
            for key in after:
                after[key] = max(0.0, min(after[key], 0.95))

            # Directly construct DeltaFeatures from percentages (fast path)
            base_veg = base['forest'] + base['agriculture'] + base['playground']
            after_veg = after['forest'] + after['agriculture'] + after['playground']

            # Estimate derived metrics from percentages
            base_bcount = max(1, int(base['building'] * 122))
            after_bcount = max(1, int(after['building'] * 122))
            base_density = base_bcount / 122
            after_density = after_bcount / 122
            base_connectivity = min(base['road'] / 0.15, 1.0) * 0.6
            after_connectivity = min(after['road'] / 0.15, 1.0) * 0.6
            base_intersections = int(base['road'] * 100)
            after_intersections = int(after['road'] * 100)

            # Water distance heuristic: closer if more water
            bw_dist_before = 200 * (1 - base['water'] * 5) + rng.uniform(-20, 20)
            bw_dist_after = 200 * (1 - after['water'] * 5) + rng.uniform(-20, 20)

            delta = DeltaFeatures(
                delta_building_area_pct=after['building'] - base['building'],
                delta_road_area_pct=after['road'] - base['road'],
                delta_water_area_pct=after['water'] - base['water'],
                delta_vegetation_area_pct=after_veg - base_veg,
                delta_barren_area_pct=after['barren'] - base['barren'],
                delta_forest_area_pct=after['forest'] - base['forest'],
                delta_agriculture_area_pct=after['agriculture'] - base['agriculture'],
                delta_playground_area_pct=after['playground'] - base['playground'],
                delta_building_count=after_bcount - base_bcount,
                delta_water_body_count=int((after['water'] - base['water']) * 10),
                delta_road_segment_count=int((after['road'] - base['road']) * 20),
                delta_building_density=after_density - base_density,
                delta_road_connectivity=after_connectivity - base_connectivity,
                delta_intersection_count=after_intersections - base_intersections,
                delta_building_water_distance=bw_dist_after - bw_dist_before,
                delta_building_road_distance=rng.uniform(-10, 10),
            )

            X_list.append(delta.to_feature_vector())
            y_list.append(cat_idx)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"[Training] Generated {len(X)} samples with {X.shape[1]} features each.")
    return X, y, SPRAWL_CATEGORIES


# ─── LEVIR-CD Dataset Loading ─────────────────────────────────────────

def extract_rgb_features(image: np.ndarray) -> dict:
    """
    Extract land-use proxy features from an RGB satellite image.
    Uses color-based heuristics to estimate land cover percentages.

    Args:
        image: H x W x 3 RGB numpy array (uint8)

    Returns:
        Dict with estimated area percentages for each land-use class.
    """
    h, w = image.shape[:2]
    total = h * w

    # Convert to float for computation
    r = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    b = image[:, :, 2].astype(float)

    # Brightness and saturation
    brightness = (r + g + b) / 3.0
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)

    # Vegetation index (excess green)
    excess_green = 2 * g - r - b
    veg_mask = (excess_green > 30) & (g > 60) & (g > r)

    # Water (blue dominant, dark)
    water_mask = (b > r + 20) & (b > g + 20) & (brightness < 140)

    # Building-like (gray/brown, moderate-high brightness)
    building_mask = (
        (saturation < 0.2) &
        (brightness > 80) &
        (brightness < 220) &
        ~veg_mask &
        ~water_mask
    )

    # Road-like (gray, elongated — approximate as gray with specific brightness)
    road_mask = (
        (saturation < 0.15) &
        (brightness > 60) &
        (brightness < 160) &
        (np.abs(r - g) < 20) &
        (np.abs(g - b) < 20) &
        ~building_mask &
        ~veg_mask &
        ~water_mask
    )

    # Barren (brown/reddish)
    barren_mask = (
        (r > g) & (r > b) &
        (saturation > 0.1) &
        (brightness > 80) &
        ~veg_mask &
        ~water_mask &
        ~building_mask
    )

    return {
        'building': building_mask.sum() / total,
        'road': road_mask.sum() / total,
        'water': water_mask.sum() / total,
        'forest': (veg_mask & (brightness < 120)).sum() / total,
        'agriculture': (veg_mask & (brightness >= 120)).sum() / total,
        'barren': barren_mask.sum() / total,
        'playground': 0.0,  # Hard to detect from RGB alone
    }


def features_from_rgb_stats(stats: dict) -> SpatialFeatures:
    """Create a SpatialFeatures object from RGB-estimated statistics."""
    features = SpatialFeatures(
        image_width=256,
        image_height=256,
        total_pixels=65536,
        building_area_pct=stats.get('building', 0.0),
        road_area_pct=stats.get('road', 0.0),
        water_area_pct=stats.get('water', 0.0),
        forest_area_pct=stats.get('forest', 0.0),
        agriculture_area_pct=stats.get('agriculture', 0.0),
        barren_area_pct=stats.get('barren', 0.0),
        playground_area_pct=stats.get('playground', 0.0),
    )
    features.vegetation_area_pct = (
        features.forest_area_pct +
        features.agriculture_area_pct +
        features.playground_area_pct
    )
    # Estimate counts from area
    features.building_count = max(1, int(features.building_area_pct * 50))
    features.building_density = features.building_area_pct
    features.road_connectivity_score = min(features.road_area_pct * 5, 1.0)
    features.intersection_count = max(0, int(features.road_area_pct * 20))
    return features


def label_from_change_mask_and_deltas(
    change_mask: np.ndarray, delta: DeltaFeatures
) -> int:
    """
    Auto-label a sprawl category from a LEVIR-CD binary change mask
    and computed delta features.

    Logic:
        - If very little change → Stable
        - If building area increased a lot with vegetation loss → Aggressive Urbanization
        - If forest decreased significantly → Deforestation
        - If water increased → Water Encroachment
        - If road increased a lot → Infrastructure Development
        - If moderate balanced change → Sustainable Expansion
    """
    change_pct = change_mask.mean() if change_mask is not None else 0.0

    # Very little change
    if change_pct < 0.02:
        return 5  # Stable

    d = delta

    # Scores for each category
    scores = {}

    # Aggressive Urbanization
    scores[0] = (
        max(d.delta_building_area_pct, 0) * 3 +
        max(-d.delta_vegetation_area_pct, 0) * 2
    )

    # Deforestation
    scores[1] = (
        max(-d.delta_forest_area_pct, 0) * 3 +
        max(d.delta_barren_area_pct, 0) * 1.5
    )

    # Water Encroachment
    scores[2] = max(d.delta_water_area_pct, 0) * 4

    # Sustainable Expansion
    if d.delta_building_area_pct > 0 and abs(d.delta_vegetation_area_pct) < 0.05:
        scores[3] = d.delta_building_area_pct * 2 + max(d.delta_road_area_pct, 0)
    else:
        scores[3] = 0.0

    # Infrastructure Development
    scores[4] = max(d.delta_road_area_pct, 0) * 4

    # Stable
    scores[5] = max(0, 0.1 - change_pct) * 5

    best = max(scores, key=scores.get)
    return best


def load_levir_cd_dataset(
    dataset_dir: str,
    max_pairs: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load image pairs from the LEVIR-CD dataset and extract delta features.

    Expected directory structure:
        dataset_dir/
        ├── A/       (before images)
        ├── B/       (after images)
        └── label/   (binary change masks)

    Args:
        dataset_dir: Path to LEVIR-CD train directory
        max_pairs: Maximum number of pairs to process

    Returns:
        X: feature matrix, y: labels, label_names
    """
    from PIL import Image

    dir_a = os.path.join(dataset_dir, 'A')
    dir_b = os.path.join(dataset_dir, 'B')
    dir_label = os.path.join(dataset_dir, 'label')

    if not all(os.path.isdir(d) for d in [dir_a, dir_b, dir_label]):
        raise FileNotFoundError(
            f"LEVIR-CD directory structure not found at {dataset_dir}. "
            f"Expected subdirectories: A/, B/, label/"
        )

    # Get paired filenames
    files_a = sorted(os.listdir(dir_a))
    files_a = [f for f in files_a if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    if len(files_a) == 0:
        raise FileNotFoundError(f"No image files found in {dir_a}")

    print(f"[Training] Found {len(files_a)} image pairs in LEVIR-CD")

    detector = ChangeDetector.__new__(ChangeDetector)
    detector.model = None
    detector.classifier_type = 'rule-based'

    X_list = []
    y_list = []
    processed = 0

    for filename in files_a[:max_pairs]:
        path_a = os.path.join(dir_a, filename)
        path_b = os.path.join(dir_b, filename)
        path_label = os.path.join(dir_label, filename)

        # Check all files exist
        if not os.path.exists(path_b):
            continue
        if not os.path.exists(path_label):
            # Try common label filename variations
            label_name = filename.replace('.jpg', '.png').replace('.jpeg', '.png')
            path_label = os.path.join(dir_label, label_name)
            if not os.path.exists(path_label):
                continue

        try:
            # Load images
            import torch
            from smart_city.pipeline import SmartCityPipeline
            
            # Lazy initialize pipeline
            if not hasattr(detector, '_pipeline'):
                detector._pipeline = SmartCityPipeline()
                # Required for proper Kaggle analysis matching inference
                if not getattr(detector._pipeline, '_seg_loaded', False):
                    detector._pipeline._load_seg_model()
            
            # If segmentation model is not available, we have to abort or mock
            if detector._pipeline.seg_model is None:
                raise RuntimeError("SemanticFPN segmentation model MUST be available to train accurately.")
            
            # Extract features EXACTLY as the inference pipeline does
            # We pass the paths directly so the pipeline resizes them properly to 512x512
            res_after = detector._pipeline.analyze_image(path_b)
            res_before = detector._pipeline.analyze_image(path_a)
            
            feat_b = res_after.spatial_features
            feat_a = res_before.spatial_features
            
            # Extract true change mask from Dataset Label
            change_mask = np.array(Image.open(path_label).convert('L'))
            change_mask = (change_mask > 127).astype(np.float32)  # binarize

            # Compute delta
            delta = detector.compute_delta(feat_a, feat_b)

            # Auto-label
            label = label_from_change_mask_and_deltas(change_mask, delta)

            X_list.append(delta.to_feature_vector())
            y_list.append(label)
            processed += 1

            if processed % 100 == 0:
                print(f"  Processed {processed}/{min(len(files_a), max_pairs)} pairs...")

        except Exception as e:
            print(f"  [WARNING] Failed to process {filename}: {e}")
            continue

    if len(X_list) == 0:
        raise RuntimeError("No valid image pairs processed from LEVIR-CD")

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"[Training] Processed {len(X)} pairs from LEVIR-CD")

    # Print label distribution
    for idx, name in enumerate(SPRAWL_CATEGORIES):
        count = (y == idx).sum()
        if count > 0:
            print(f"  {name}: {count} samples ({count/len(y)*100:.1f}%)")

    return X, y, SPRAWL_CATEGORIES


# ─── Model Training ───────────────────────────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    output_path: str,
    n_folds: int = 5,
) -> None:
    """
    Train SVM + Random Forest, cross-validate, and save the best model.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label vector (n_samples,)
        label_names: List of category names
        output_path: Path to save the best model (.joblib)
        n_folds: Number of cross-validation folds
    """
    print(f"\n{'='*60}")
    print(f"  Training Change Detection Classifiers")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Categories: {len(label_names)}")
    print(f"{'='*60}\n")

    # Define models
    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            ))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
            ))
        ]),
    }

    best_model = None
    best_score = 0.0
    best_name = ''

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for name, pipeline in models.items():
        print(f"\n--- {name} ---")

        # Cross-validate
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()

        print(f"  Cross-validation accuracy: {mean_score:.4f} ± {std_score:.4f}")
        print(f"  Per-fold scores: {[f'{s:.3f}' for s in scores]}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline
            best_name = name

    print(f"\n{'='*60}")
    print(f"  Best Model: {best_name} (accuracy: {best_score:.4f})")
    print(f"{'='*60}\n")

    # Train the best model on full data
    print(f"Training final {best_name} on all {len(X)} samples...")
    best_model.fit(X, y)

    # Store metadata
    best_model._classifier_type = best_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    best_model._feature_names = DeltaFeatures.feature_names()
    best_model._label_names = label_names

    # Full-data classification report
    y_pred = best_model.predict(X)
    present_labels = sorted(set(y) | set(y_pred))
    present_names = [label_names[i] for i in present_labels if i < len(label_names)]

    print(f"\nClassification Report (full training set):")
    print(classification_report(
        y, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    ))

    print(f"Confusion Matrix:")
    cm = confusion_matrix(y, y_pred, labels=present_labels)
    print(cm)

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"\n✅ Model saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train Change Detection Classifier for Urban Sprawl Classification'
    )
    parser.add_argument(
        '--dataset_dir', type=str, default=None,
        help='Path to LEVIR-CD train directory (containing A/, B/, label/ subdirs)'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Generate synthetic training data instead of using a real dataset'
    )
    parser.add_argument(
        '--kaggle', action='store_true',
        help='Auto-detect LEVIR-CD dataset on Kaggle'
    )
    parser.add_argument(
        '--num_samples', type=int, default=3000,
        help='Number of synthetic samples to generate (only with --synthetic)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for the trained model (.joblib)'
    )
    parser.add_argument(
        '--folds', type=int, default=5,
        help='Number of cross-validation folds'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config', 'change_classifier.joblib'
        )

    # Determine data source
    if args.kaggle:
        # Kaggle auto-detect paths
        kaggle_paths = [
            '/kaggle/input/levir-cd/train',
            '/kaggle/input/levircd/train',
            '/kaggle/input/levir-cd-dataset/train',
            '/kaggle/input/levir-change-detection/train',
        ]
        dataset_dir = None
        for p in kaggle_paths:
            if os.path.exists(p):
                dataset_dir = p
                break
        if dataset_dir is None:
            print("[WARNING] Could not auto-detect LEVIR-CD on Kaggle.")
            print("Falling back to synthetic data generation.")
            args.synthetic = True
        else:
            args.dataset_dir = dataset_dir

    if args.synthetic or (args.dataset_dir is None and not args.kaggle):
        print("=" * 60)
        print("  Using SYNTHETIC data generation")
        print("=" * 60)
        X, y, label_names = generate_synthetic_dataset(
            num_samples=args.num_samples,
            seed=42,
        )
    else:
        print("=" * 60)
        print(f"  Loading LEVIR-CD dataset from: {args.dataset_dir}")
        print("=" * 60)
        X, y, label_names = load_levir_cd_dataset(args.dataset_dir)

    # Train and save
    train_and_evaluate(X, y, label_names, output_path, n_folds=args.folds)


if __name__ == '__main__':
    main()
