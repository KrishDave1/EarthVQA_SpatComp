"""
Threshold Calibration Script

Runs the spatial engine on training set masks to compute distribution
statistics (mean, std, percentiles) for each metric.
Sets thresholds at percentile boundaries and writes to thresholds.yaml.

Usage:
    python -m smart_city.calibrate_thresholds \
        --mask_dir ./EarthVQA/Train/masks_png \
        --output ./smart_city/config/thresholds.yaml
"""

import os
import sys
import argparse
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_city.spatial_engine import SpatialFeatureEngine


def calibrate(mask_dir: str, output_path: str, sample_limit: int = 0):
    """
    Calibrate thresholds from training masks.
    
    Args:
        mask_dir: Directory containing mask PNG files (1-indexed, 0-7 classes).
        output_path: Where to write the calibrated thresholds.yaml.
        sample_limit: Max number of masks to process (0 = all).
    """
    engine = SpatialFeatureEngine()
    
    # Collect mask paths
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
    if sample_limit > 0:
        mask_paths = mask_paths[:sample_limit]
    
    print(f"Calibrating from {len(mask_paths)} masks in: {mask_dir}")
    
    # Accumulators
    building_counts = []
    building_area_pcts = []
    road_area_pcts = []
    water_area_pcts = []
    vegetation_pcts = []
    intersection_counts = []
    building_water_dists = []
    building_densities = []
    
    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        try:
            from skimage.io import imread
            mask = imread(mask_path)
            if mask.ndim == 3:
                mask = mask[:, :, 0]  # take first channel if RGB
            mask = mask.astype(np.int32)
        except Exception as e:
            print(f"  Skipping {mask_path}: {e}")
            continue
        
        features = engine.extract(mask)
        
        building_counts.append(features.building_count)
        building_area_pcts.append(features.building_area_pct)
        road_area_pcts.append(features.road_area_pct)
        water_area_pcts.append(features.water_area_pct)
        vegetation_pcts.append(features.vegetation_area_pct)
        intersection_counts.append(features.intersection_count)
        building_densities.append(features.building_density)
        
        if features.building_water_min_distance != float('inf'):
            building_water_dists.append(features.building_water_min_distance)
    
    if not building_counts:
        print("No masks processed! Check the mask directory.")
        return
    
    # Compute statistics
    stats = {
        'building_count': _percentile_stats(building_counts),
        'building_area_pct': _percentile_stats(building_area_pcts),
        'road_area_pct': _percentile_stats(road_area_pcts),
        'water_area_pct': _percentile_stats(water_area_pcts),
        'vegetation_pct': _percentile_stats(vegetation_pcts),
        'intersection_count': _percentile_stats(intersection_counts),
        'building_water_dist': _percentile_stats(building_water_dists) if building_water_dists else None,
        'building_density': _percentile_stats(building_densities),
    }
    
    print("\n=== Calibration Statistics ===")
    for key, s in stats.items():
        if s:
            print(f"{key:>25s}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"p25={s['p25']:.4f}  p50={s['p50']:.4f}  p75={s['p75']:.4f}  max={s['max']:.4f}")
    
    # Generate calibrated thresholds
    calibrated = _generate_thresholds(stats)
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(calibrated, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCalibrated thresholds written to: {output_path}")


def _percentile_stats(values: list) -> dict:
    """Compute percentile statistics for a list of values."""
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
    }


def _generate_thresholds(stats: dict) -> dict:
    """Generate threshold config from statistics."""
    bc = stats['building_count']
    ba = stats['building_area_pct']
    ra = stats['road_area_pct']
    vp = stats['vegetation_pct']
    ic = stats['intersection_count']
    
    config = {
        'density': {
            'weights': {'building_count': 0.4, 'built_area_pct': 0.6},
            'thresholds': {
                'low': round(ba['p25'] * 2, 2) if ba else 0.25,
                'moderate': round(ba['p50'] * 2, 2) if ba else 0.50,
                'high': round(ba['p75'] * 2, 2) if ba else 0.70,
            },
            'recommendations': {
                'high': 'Area is densely populated. Consider limiting further construction and introducing open spaces.',
                'moderate': 'Moderate building density. Area is suitable for controlled expansion.',
                'low': 'Low building density. Area has potential for residential or commercial development.',
            },
        },
        'green_coverage': {
            'weights': {'vegetation_pct': 1.0, 'playground_pct': 0.5},
            'thresholds': {
                'insufficient': 0.10,
                'low': round(vp['p25'], 2) if vp else 0.15,
                'adequate': round(vp['p50'], 2) if vp else 0.25,
                'good': round(vp['p75'], 2) if vp else 0.40,
            },
            'target_pct': round(vp['p50'], 2) if vp else 0.20,
            'recommendations': {
                'insufficient': 'Critical shortage of green spaces. Urgently recommend parks and tree planting programs.',
                'low': 'Below recommended green coverage (15-25%). Suggest introducing parks or community gardens.',
                'adequate': 'Green coverage meets minimum urban planning standards.',
                'good': 'Excellent green coverage. Area has strong environmental quality.',
            },
        },
        'flood_risk': {
            'weights': {'water_proximity': 0.5, 'water_area_pct': 0.3, 'building_water_ratio': 0.2},
            'thresholds': {'low': 0.25, 'moderate': 0.50, 'high': 0.70},
            'min_safe_distance_px': 50,
            'recommendations': {
                'high': 'High flood risk detected. Buildings are dangerously close to water bodies. Recommend buffer zones and flood barriers.',
                'moderate': 'Moderate flood risk. Some structures are near water. Consider drainage improvements.',
                'low': 'Low flood risk. Adequate distance between buildings and water bodies.',
            },
        },
        'infrastructure': {
            'weights': {'road_coverage_pct': 0.5, 'intersection_count': 0.3, 'road_connectivity': 0.2},
            'thresholds': {
                'poor': round(ra['p25'] * 4, 2) if ra else 0.25,
                'moderate': round(ra['p50'] * 4, 2) if ra else 0.50,
                'good': round(ra['p75'] * 4, 2) if ra else 0.70,
            },
            'recommendations': {
                'poor': 'Poor road connectivity. Recommend expanding road network and adding intersections for better access.',
                'moderate': 'Basic road infrastructure present. Consider adding secondary roads for improved connectivity.',
                'good': 'Good road connectivity and infrastructure coverage.',
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
            'building_count_max': int(bc['max']) if bc else 200,
            'intersection_count_max': int(ic['max']) if ic else 10,
            'image_area_px': 262144,
        },
    }
    
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate decision thresholds from training masks')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory with training mask PNGs')
    parser.add_argument('--output', type=str, default='./smart_city/config/thresholds.yaml',
                        help='Output path for calibrated thresholds.yaml')
    parser.add_argument('--sample_limit', type=int, default=0,
                        help='Max masks to process (0=all)')
    args = parser.parse_args()
    
    calibrate(args.mask_dir, args.output, args.sample_limit)
