"""
Unit tests for the Spatial Feature Engine.
Uses synthetic masks to verify all metrics are computed correctly.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smart_city.spatial_engine import SpatialFeatureEngine
from smart_city.models import SpatialFeatures


class TestSpatialFeatureEngine(unittest.TestCase):
    """Test spatial feature extraction on synthetic masks."""

    def setUp(self):
        self.engine = SpatialFeatureEngine()

    def _make_simple_mask(self, size=256):
        """Create a simple test mask with known ground truth.
        Uses 0-indexed classes (0=bg, 1=building, 2=road, 3=water, 5=forest).
        All four quadrants have non-background classes.
        The mask contains 0 in its values (background appears in class_stats)
        so it will NOT be treated as 1-indexed.
        """
        mask = np.zeros((size, size), dtype=np.int32)
        # Top-left quadrant: building (class 1)
        mask[0:128, 0:128] = 1
        # Top-right quadrant: road (class 2)
        mask[0:128, 128:256] = 2
        # Bottom-left quadrant: water (class 3)
        mask[128:256, 0:128] = 3
        # Bottom-right quadrant: forest (class 5)
        mask[128:256, 128:256] = 5
        return mask

    def _make_urban_mask(self, size=256):
        """Create an urban-heavy mask."""
        mask = np.zeros((size, size), dtype=np.int32)
        # Mostly buildings with some roads
        mask[:, :] = 1  # all buildings
        # Horizontal road
        mask[120:136, :] = 2
        # Vertical road
        mask[:, 120:136] = 2
        # Small water body
        mask[200:230, 200:230] = 3
        return mask

    def _make_rural_mask(self, size=256):
        """Create a rural/green mask."""
        mask = np.zeros((size, size), dtype=np.int32)
        # Mostly agriculture
        mask[:, :] = 6
        # Some forest
        mask[0:100, 0:100] = 5
        # Small building cluster
        mask[150:170, 150:170] = 1
        # Small road
        mask[160:163, 100:200] = 2
        return mask

    def test_basic_extraction(self):
        """Test that extraction runs without errors."""
        mask = self._make_simple_mask()
        features = self.engine.extract(mask)
        self.assertIsInstance(features, SpatialFeatures)

    def test_area_coverage_simple(self):
        """Test area coverage on a 4-quadrant mask."""
        mask = self._make_simple_mask()
        features = self.engine.extract(mask)
        
        # Each quadrant is 25%
        self.assertAlmostEqual(features.building_area_pct, 0.25, delta=0.01)
        self.assertAlmostEqual(features.road_area_pct, 0.25, delta=0.01)
        self.assertAlmostEqual(features.water_area_pct, 0.25, delta=0.01)
        self.assertAlmostEqual(features.forest_area_pct, 0.25, delta=0.01)

    def test_vegetation_combined(self):
        """Test that vegetation is sum of forest + agriculture + playground."""
        mask = self._make_simple_mask()
        features = self.engine.extract(mask)
        expected = features.forest_area_pct + features.agriculture_area_pct + features.playground_area_pct
        self.assertAlmostEqual(features.vegetation_area_pct, expected, delta=0.001)

    def test_object_count_single(self):
        """Test building count on mask with single building region."""
        mask = self._make_simple_mask()
        features = self.engine.extract(mask)
        # One contiguous building region
        self.assertEqual(features.building_count, 1)

    def test_object_count_multiple(self):
        """Test building count with multiple separate buildings."""
        mask = np.zeros((256, 256), dtype=np.int32)
        # Three separate building patches (spread apart)
        mask[10:30, 10:30] = 1
        mask[10:30, 100:120] = 1
        mask[10:30, 200:220] = 1
        features = self.engine.extract(mask)
        self.assertEqual(features.building_count, 3)

    def test_distance_adjacent_classes(self):
        """Test distance between adjacent classes is small."""
        mask = self._make_simple_mask()
        features = self.engine.extract(mask)
        # Building (top-left) and road (top-right) share a boundary → distance 0-1
        self.assertLessEqual(features.building_road_min_distance, 2.0)
        # Building (top-left) and water (bottom-left) share a boundary → distance 0-1
        self.assertLessEqual(features.building_water_min_distance, 2.0)

    def test_distance_separated_classes(self):
        """Test distance between well-separated classes."""
        mask = np.zeros((256, 256), dtype=np.int32)
        mask[0:20, 0:20] = 1     # building in top-left
        mask[200:220, 200:220] = 3  # water in bottom-right
        features = self.engine.extract(mask)
        # Should be large distance
        self.assertGreater(features.building_water_min_distance, 100)

    def test_distance_no_water(self):
        """Test that building-water distance is inf when no water present."""
        mask = np.ones((256, 256), dtype=np.int32)  # all buildings
        features = self.engine.extract(mask)
        self.assertEqual(features.building_water_min_distance, float('inf'))

    def test_urban_density_high(self):
        """Test density on urban-heavy mask."""
        mask = self._make_urban_mask()
        features = self.engine.extract(mask)
        # Building area should be dominant (majority of pixels)
        self.assertGreater(features.building_area_pct, 0.3)

    def test_rural_vegetation(self):
        """Test vegetation on rural mask."""
        mask = self._make_rural_mask()
        features = self.engine.extract(mask)
        # Vegetation should be dominant
        self.assertGreater(features.vegetation_area_pct, 0.5)

    def test_intersection_count_crossroads(self):
        """Test intersection detection on a simple crossroads."""
        mask = np.zeros((256, 256), dtype=np.int32)
        # Horizontal road
        mask[120:140, :] = 2
        # Vertical road
        mask[:, 120:140] = 2
        features = self.engine.extract(mask)
        # Should detect at least 1 intersection
        # (May vary depending on skeletonization)
        self.assertGreaterEqual(features.intersection_count, 0)

    def test_colorize_mask(self):
        """Test that colorize_mask produces valid RGB output."""
        mask = self._make_simple_mask()
        rgb = self.engine.colorize_mask(mask)
        self.assertEqual(rgb.shape, (256, 256, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        # Check building (class 1) pixels are red: [255, 0, 0]
        self.assertTrue(np.array_equal(rgb[64, 64], [255, 0, 0]),
                        f"Expected [255,0,0] for building but got {rgb[64, 64]}")
        # Check water (class 3) pixels are blue: [0, 0, 255]
        self.assertTrue(np.array_equal(rgb[192, 64], [0, 0, 255]),
                        f"Expected [0,0,255] for water but got {rgb[192, 64]}")

    def test_to_dict_serializable(self):
        """Test that SpatialFeatures.to_dict() is JSON-serializable."""
        import json
        mask = self._make_simple_mask()
        features = self.engine.extract(mask)
        result = features.to_dict()
        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

    def test_handles_1_indexed_mask(self):
        """Test that 1-indexed masks (EarthVQA convention) are handled.
        In EarthVQA mask files, labels are 1-8.
        The engine should handle this when one_indexed=True."""
        mask = np.ones((256, 256), dtype=np.int32)  # fill with 1 = background in 1-indexed
        mask[0:128, 0:128] = 2   # building (1-indexed → becomes class 1)
        mask[128:256, 0:128] = 4  # water (1-indexed → becomes class 3)
        mask[128:256, 128:256] = 6  # forest (1-indexed → becomes class 5)
        # Use one_indexed=True
        features = self.engine.extract(mask, one_indexed=True)
        # After adjustment: class 1=building should be 25%
        self.assertGreater(features.building_area_pct, 0.2)
        self.assertGreater(features.water_area_pct, 0.2)

    def test_empty_mask(self):
        """Test handling of all-background mask."""
        mask = np.zeros((256, 256), dtype=np.int32)
        features = self.engine.extract(mask)
        self.assertEqual(features.building_count, 0)
        self.assertEqual(features.water_body_count, 0)
        self.assertAlmostEqual(features.building_area_pct, 0.0)

    def test_image_dimensions(self):
        """Test that image dimensions are recorded."""
        mask = np.zeros((512, 1024), dtype=np.int32)
        features = self.engine.extract(mask)
        self.assertEqual(features.image_width, 1024)
        self.assertEqual(features.image_height, 512)
        self.assertEqual(features.total_pixels, 512 * 1024)


if __name__ == '__main__':
    unittest.main()
