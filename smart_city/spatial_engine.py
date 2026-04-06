"""
Spatial Feature Extraction Engine (Module 3)

Transforms a segmentation mask into quantitative urban planning metrics.
Uses scipy for connected component analysis and distance transforms,
and skimage for road network skeletonization.

Input:  segmentation mask (H x W numpy array, values 0-7)
Output: SpatialFeatures dataclass with all computed metrics
"""

import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label as ndimage_label
from collections import defaultdict

from smart_city.models import SpatialFeatures, ClassStats, DistanceMetric

# Try to import skimage for road analysis — graceful fallback if not installed
try:
    from skimage.morphology import skeletonize, remove_small_objects
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARNING] scikit-image not installed. Road connectivity analysis will be limited.")


class SpatialFeatureEngine:
    """
    Extracts spatial features from a segmentation mask.
    
    The segmentation mask uses EarthVQA class labels:
        0=background, 1=building, 2=road, 3=water, 
        4=barren, 5=forest, 6=agriculture, 7=playground
    """

    # Class ID to name mapping (from EarthVQA)
    CLASS_NAMES = {
        0: 'background',
        1: 'building',
        2: 'road',
        3: 'water',
        4: 'barren',
        5: 'forest',
        6: 'agriculture',
        7: 'playground',
    }

    # Which classes are "green space"
    GREEN_CLASSES = {5, 6, 7}  # forest, agriculture, playground

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to thresholds.yaml (for normalization params).
                         If None, uses defaults.
        """
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        # Default config
        return {
            'normalization': {
                'building_count_max': 200,
                'intersection_count_max': 10,
                'image_area_px': 262144,
            }
        }

    def extract(self, seg_mask: np.ndarray, one_indexed: bool = False) -> SpatialFeatures:
        """
        Extract all spatial features from a segmentation mask.
        
        Args:
            seg_mask: H x W numpy array with class IDs.
                      If one_indexed=False (default): values 0-7 (0=background, ..., 7=playground)
                      If one_indexed=True: values 1-8 (EarthVQA mask file convention)
            one_indexed: Set True if the mask uses 1-indexed labels (1-8).
                         Default False (0-indexed, 0-7).
        
        Returns:
            SpatialFeatures dataclass with all computed metrics.
        """
        # Ensure mask is the right type
        seg_mask = seg_mask.astype(np.int32)
        
        # Handle 1-indexed masks (EarthVQA convention: labels are 1-8 in files)
        if one_indexed:
            seg_mask = seg_mask - 1
            seg_mask = np.clip(seg_mask, 0, 7)  # safety clamp

        h, w = seg_mask.shape
        total_pixels = h * w
        
        features = SpatialFeatures(
            image_width=w,
            image_height=h,
            total_pixels=total_pixels,
        )

        # Step 1: Per-class statistics
        features.class_stats = self._compute_class_stats(seg_mask, total_pixels)

        # Step 2: Area coverages
        self._compute_area_coverages(features)

        # Step 3: Object counts (already computed in class_stats)
        self._extract_counts(features)

        # Step 4: Distance metrics
        features.distances = self._compute_distances(seg_mask, features.class_stats)
        self._extract_key_distances(features)

        # Step 5: Density metrics
        self._compute_density(features, total_pixels)

        # Step 6: Road connectivity
        self._compute_road_connectivity(seg_mask, features)

        return features

    def _compute_class_stats(self, seg_mask: np.ndarray, total_pixels: int) -> Dict[str, ClassStats]:
        """Compute per-class pixel counts, areas, and connected components."""
        stats = {}
        
        for class_id, class_name in self.CLASS_NAMES.items():
            # Binary mask for this class
            binary_mask = (seg_mask == class_id)
            pixel_count = int(binary_mask.sum())
            area_pct = pixel_count / total_pixels if total_pixels > 0 else 0.0
            
            # Connected component analysis
            if pixel_count > 0:
                # Label connected components
                labeled_array, num_components = ndimage_label(binary_mask)
                
                # Minimum component size filter (remove noise: components < 25 pixels)
                min_size = 25
                centroids = []
                valid_components = 0
                
                for comp_id in range(1, num_components + 1):
                    component_mask = (labeled_array == comp_id)
                    comp_size = component_mask.sum()
                    if comp_size >= min_size:
                        valid_components += 1
                        # Compute centroid
                        cy, cx = ndimage.center_of_mass(component_mask)
                        centroids.append((float(cy), float(cx)))
                
                stats[class_name] = ClassStats(
                    class_id=class_id,
                    class_name=class_name,
                    pixel_count=pixel_count,
                    area_percentage=area_pct,
                    component_count=valid_components,
                    centroids=centroids,
                )
            else:
                stats[class_name] = ClassStats(
                    class_id=class_id,
                    class_name=class_name,
                    pixel_count=0,
                    area_percentage=0.0,
                    component_count=0,
                    centroids=[],
                )
        
        return stats

    def _compute_area_coverages(self, features: SpatialFeatures):
        """Extract area coverage percentages from class stats."""
        stats = features.class_stats
        
        features.building_area_pct = stats.get('building', ClassStats(0, '', 0, 0.0, 0)).area_percentage
        features.road_area_pct = stats.get('road', ClassStats(0, '', 0, 0.0, 0)).area_percentage
        features.water_area_pct = stats.get('water', ClassStats(0, '', 0, 0.0, 0)).area_percentage
        features.barren_area_pct = stats.get('barren', ClassStats(0, '', 0, 0.0, 0)).area_percentage
        features.forest_area_pct = stats.get('forest', ClassStats(0, '', 0, 0.0, 0)).area_percentage
        features.agriculture_area_pct = stats.get('agriculture', ClassStats(0, '', 0, 0.0, 0)).area_percentage
        features.playground_area_pct = stats.get('playground', ClassStats(0, '', 0, 0.0, 0)).area_percentage
        
        # Vegetation = forest + agriculture + playground
        features.vegetation_area_pct = (
            features.forest_area_pct +
            features.agriculture_area_pct +
            features.playground_area_pct
        )

    def _extract_counts(self, features: SpatialFeatures):
        """Extract object counts from class stats."""
        stats = features.class_stats
        features.building_count = stats.get('building', ClassStats(0, '', 0, 0.0, 0)).component_count
        features.water_body_count = stats.get('water', ClassStats(0, '', 0, 0.0, 0)).component_count
        features.road_segment_count = stats.get('road', ClassStats(0, '', 0, 0.0, 0)).component_count

    def _compute_distances(self, seg_mask: np.ndarray, class_stats: Dict[str, ClassStats]) -> List[DistanceMetric]:
        """
        Compute pairwise distance metrics between key class pairs.
        Uses Euclidean distance transform for efficient boundary distance computation.
        """
        distances = []
        
        # Key pairs to compute distances for
        pairs = [
            ('building', 'water'),
            ('building', 'road'),
            ('building', 'forest'),
            ('water', 'agriculture'),
            ('road', 'water'),
        ]
        
        for class_a_name, class_b_name in pairs:
            stat_a = class_stats.get(class_a_name)
            stat_b = class_stats.get(class_b_name)
            
            if stat_a is None or stat_b is None:
                continue
            if stat_a.pixel_count == 0 or stat_b.pixel_count == 0:
                # One class is absent — infinite distance
                distances.append(DistanceMetric(
                    class_a=class_a_name,
                    class_b=class_b_name,
                    min_distance_px=float('inf'),
                    mean_centroid_distance_px=float('inf'),
                    proximity_score=0.0,
                ))
                continue
            
            class_a_id = stat_a.class_id
            class_b_id = stat_b.class_id
            
            # Minimum boundary distance via distance transform
            mask_a = (seg_mask == class_a_id)
            mask_b = (seg_mask == class_b_id)
            
            # Distance transform: distance of each pixel to nearest pixel of class B
            dist_to_b = distance_transform_edt(~mask_b)
            # Minimum distance from any pixel of class A to class B
            min_dist = float(dist_to_b[mask_a].min()) if mask_a.any() else float('inf')
            
            # Mean centroid distance
            centroids_a = stat_a.centroids
            centroids_b = stat_b.centroids
            mean_centroid_dist = float('inf')
            if centroids_a and centroids_b:
                centroid_dists = []
                for ca in centroids_a:
                    for cb in centroids_b:
                        d = np.sqrt((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2)
                        centroid_dists.append(d)
                mean_centroid_dist = float(np.mean(centroid_dists))
            
            # Proximity score: 1 when very close, 0 when far
            # Using sigmoid-like decay: score = exp(-min_dist / scale)
            scale = 100.0  # pixels — controls how fast proximity drops off
            proximity = float(np.exp(-min_dist / scale))
            
            distances.append(DistanceMetric(
                class_a=class_a_name,
                class_b=class_b_name,
                min_distance_px=min_dist,
                mean_centroid_distance_px=mean_centroid_dist,
                proximity_score=proximity,
            ))
        
        return distances

    def _extract_key_distances(self, features: SpatialFeatures):
        """Extract key distance values for quick access."""
        for d in features.distances:
            if d.class_a == 'building' and d.class_b == 'water':
                features.building_water_min_distance = d.min_distance_px
            elif d.class_a == 'building' and d.class_b == 'road':
                features.building_road_min_distance = d.min_distance_px

    def _compute_density(self, features: SpatialFeatures, total_pixels: int):
        """Compute building density metric."""
        norm_max = self.config.get('normalization', {}).get('building_count_max', 200)
        # Building density = building_count normalized by image area
        features.building_density = features.building_count / max(norm_max, 1)

    def _compute_road_connectivity(self, seg_mask: np.ndarray, features: SpatialFeatures):
        """
        Analyze road network connectivity using skeletonization.
        Detects intersections as junction points in the road skeleton.
        """
        road_mask = (seg_mask == 2)  # road class
        
        if road_mask.sum() < 100:
            # Not enough road pixels for meaningful analysis
            features.intersection_count = 0
            features.road_connectivity_score = 0.0
            return
        
        if not SKIMAGE_AVAILABLE:
            # Fallback: estimate from connected components
            features.intersection_count = 0
            features.road_connectivity_score = features.road_area_pct * 5  # rough estimate
            return
        
        try:
            # Clean the road mask: remove small isolated road pixels
            clean_road = remove_small_objects(road_mask, min_size=50)
            
            # Skeletonize the road network
            skeleton = skeletonize(clean_road)
            
            # Find intersection points (pixels with 3+ skeleton neighbors)
            intersection_count = self._count_intersections(skeleton)
            features.intersection_count = intersection_count
            
            # Road connectivity score
            # Based on: skeleton length relative to image, plus intersection density
            skeleton_length = skeleton.sum()
            norm_max_intersections = self.config.get('normalization', {}).get('intersection_count_max', 10)
            
            # Connectivity = weighted combination of road coverage and intersection density
            road_factor = min(features.road_area_pct / 0.15, 1.0)  # saturates at 15% road coverage
            intersection_factor = min(intersection_count / max(norm_max_intersections, 1), 1.0)
            features.road_connectivity_score = 0.6 * road_factor + 0.4 * intersection_factor
            
        except Exception as e:
            print(f"[WARNING] Road connectivity analysis failed: {e}")
            features.intersection_count = 0
            features.road_connectivity_score = 0.0

    def _count_intersections(self, skeleton: np.ndarray) -> int:
        """
        Count intersection points in a skeleton image.
        An intersection is a skeleton pixel with 3 or more skeleton neighbors.
        """
        if skeleton.sum() == 0:
            return 0
        
        # 3x3 structuring element for 8-connectivity
        kernel = np.ones((3, 3), dtype=np.int32)
        kernel[1, 1] = 0  # don't count the center pixel itself
        
        # Count neighbors for each skeleton pixel
        neighbor_count = ndimage.convolve(
            skeleton.astype(np.int32),
            kernel,
            mode='constant',
            cval=0
        )
        
        # Intersection pixels: skeleton pixels with 3+ neighbors
        intersections = skeleton & (neighbor_count >= 3)
        
        # Cluster nearby intersection pixels (they form groups at real intersections)
        if intersections.sum() == 0:
            return 0
        
        labeled_intersections, num_intersection_clusters = ndimage_label(
            ndimage.binary_dilation(intersections, iterations=3)
        )
        
        return num_intersection_clusters

    def colorize_mask(self, seg_mask: np.ndarray, one_indexed: bool = False) -> np.ndarray:
        """
        Convert segmentation mask to RGB visualization.
        
        Args:
            seg_mask: H x W array with class IDs (0-7 or 1-8)
            one_indexed: Set True if the mask uses 1-indexed labels (1-8).
        
        Returns:
            H x W x 3 uint8 RGB image
        """
        # Color map matching EarthVQA conventions
        color_map = {
            0: (255, 255, 255),   # background - white
            1: (255, 0, 0),       # building - red
            2: (255, 255, 0),     # road - yellow
            3: (0, 0, 255),       # water - blue
            4: (159, 129, 183),   # barren - purple
            5: (0, 255, 0),       # forest - green
            6: (255, 195, 128),   # agriculture - orange
            7: (165, 0, 165),     # playground - magenta
        }
        
        # Handle 1-indexed masks
        mask = seg_mask.copy().astype(np.int32)
        if one_indexed:
            mask = mask - 1
            mask = np.clip(mask, 0, 7)
        
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in color_map.items():
            rgb[mask == class_id] = color
        
        return rgb
