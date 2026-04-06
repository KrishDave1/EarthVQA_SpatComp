"""
Unit tests for the Decision Engine.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smart_city.decision_engine import DecisionEngine
from smart_city.models import SpatialFeatures, ClassStats, DistanceMetric


class TestDecisionEngine(unittest.TestCase):
    """Test decision intelligence layer."""

    def setUp(self):
        self.engine = DecisionEngine()

    def _make_urban_features(self) -> SpatialFeatures:
        """Create features for a dense urban scene."""
        f = SpatialFeatures()
        f.building_count = 150
        f.building_area_pct = 0.65
        f.road_area_pct = 0.15
        f.water_area_pct = 0.02
        f.forest_area_pct = 0.03
        f.agriculture_area_pct = 0.0
        f.playground_area_pct = 0.01
        f.vegetation_area_pct = 0.04
        f.barren_area_pct = 0.14
        f.building_density = 0.75
        f.intersection_count = 5
        f.road_connectivity_score = 0.6
        f.building_water_min_distance = 30.0
        f.building_road_min_distance = 5.0
        f.water_body_count = 1
        f.road_segment_count = 3
        f.image_width = 512
        f.image_height = 512
        f.total_pixels = 262144
        return f

    def _make_rural_features(self) -> SpatialFeatures:
        """Create features for a rural scene."""
        f = SpatialFeatures()
        f.building_count = 5
        f.building_area_pct = 0.03
        f.road_area_pct = 0.02
        f.water_area_pct = 0.10
        f.forest_area_pct = 0.40
        f.agriculture_area_pct = 0.35
        f.playground_area_pct = 0.0
        f.vegetation_area_pct = 0.75
        f.barren_area_pct = 0.10
        f.building_density = 0.025
        f.intersection_count = 0
        f.road_connectivity_score = 0.1
        f.building_water_min_distance = 200.0
        f.building_road_min_distance = 50.0
        f.water_body_count = 2
        f.road_segment_count = 1
        f.image_width = 512
        f.image_height = 512
        f.total_pixels = 262144
        return f

    def _make_flood_risk_features(self) -> SpatialFeatures:
        """Create features for a flood-risk scenario."""
        f = SpatialFeatures()
        f.building_count = 80
        f.building_area_pct = 0.30
        f.road_area_pct = 0.10
        f.water_area_pct = 0.20
        f.forest_area_pct = 0.05
        f.agriculture_area_pct = 0.10
        f.playground_area_pct = 0.0
        f.vegetation_area_pct = 0.15
        f.barren_area_pct = 0.25
        f.building_density = 0.4
        f.intersection_count = 2
        f.road_connectivity_score = 0.4
        f.building_water_min_distance = 10.0  # very close!
        f.building_road_min_distance = 5.0
        f.water_body_count = 3
        f.road_segment_count = 2
        f.image_width = 512
        f.image_height = 512
        f.total_pixels = 262144
        return f

    def test_evaluate_returns_report(self):
        """Test that evaluate returns a PlanningReport."""
        from smart_city.models import PlanningReport
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        self.assertIsInstance(report, PlanningReport)

    def test_all_four_decisions(self):
        """Test that all 4 decision categories are present."""
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        categories = [d.category for d in report.decisions]
        self.assertIn('density', categories)
        self.assertIn('green_coverage', categories)
        self.assertIn('flood_risk', categories)
        self.assertIn('infrastructure', categories)

    def test_urban_high_density(self):
        """Dense urban scene should have high density score."""
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        density_dec = next(d for d in report.decisions if d.category == 'density')
        self.assertGreater(density_dec.score, 0.5)
        self.assertEqual(density_dec.severity, 'high')

    def test_urban_low_green(self):
        """Dense urban scene should have low green coverage."""
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        green_dec = next(d for d in report.decisions if d.category == 'green_coverage')
        # severity should be 'high' (high concern about greenery)
        self.assertEqual(green_dec.severity, 'high')

    def test_rural_good_green(self):
        """Rural scene should have good green coverage."""
        features = self._make_rural_features()
        report = self.engine.evaluate(features)
        green_dec = next(d for d in report.decisions if d.category == 'green_coverage')
        self.assertEqual(green_dec.severity, 'low')  # low concern

    def test_rural_low_density(self):
        """Rural scene should have low density."""
        features = self._make_rural_features()
        report = self.engine.evaluate(features)
        density_dec = next(d for d in report.decisions if d.category == 'density')
        self.assertEqual(density_dec.severity, 'low')

    def test_flood_risk_high(self):
        """Close water-building proximity should trigger high flood risk."""
        features = self._make_flood_risk_features()
        report = self.engine.evaluate(features)
        flood_dec = next(d for d in report.decisions if d.category == 'flood_risk')
        self.assertGreater(flood_dec.score, 0.3)

    def test_no_water_no_flood(self):
        """No water should result in zero flood risk."""
        features = self._make_urban_features()
        features.water_area_pct = 0.0
        features.water_body_count = 0
        features.building_water_min_distance = float('inf')
        report = self.engine.evaluate(features)
        flood_dec = next(d for d in report.decisions if d.category == 'flood_risk')
        self.assertAlmostEqual(flood_dec.score, 0.0)

    def test_scene_classification_urban(self):
        """Dense built area should be classified as urban."""
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        self.assertEqual(report.scene_type, 'urban')

    def test_scene_classification_rural(self):
        """Green-dominant area should be classified as rural."""
        features = self._make_rural_features()
        report = self.engine.evaluate(features)
        self.assertEqual(report.scene_type, 'rural')

    def test_overall_score_range(self):
        """Overall suitability score should be between 0 and 1."""
        for features in [self._make_urban_features(), self._make_rural_features()]:
            report = self.engine.evaluate(features)
            self.assertGreaterEqual(report.overall_score, 0.0)
            self.assertLessEqual(report.overall_score, 1.0)

    def test_suitability_label(self):
        """Overall suitability should have a valid label."""
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        valid_labels = ['Not Suitable', 'Needs Improvement', 'Moderately Suitable', 'Suitable', 'Highly Suitable']
        self.assertIn(report.overall_suitability, valid_labels)

    def test_summary_not_empty(self):
        """Summary should contain meaningful text."""
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        self.assertGreater(len(report.summary), 50)

    def test_to_dict_serializable(self):
        """Test that PlanningReport.to_dict() is JSON-serializable."""
        import json
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        result = report.to_dict()
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

    def test_recommendations_not_empty(self):
        """All decisions should have non-empty recommendations."""
        for features in [self._make_urban_features(), self._make_rural_features()]:
            report = self.engine.evaluate(features)
            for decision in report.decisions:
                self.assertGreater(len(decision.recommendation), 0)
                self.assertGreater(len(decision.description), 0)
                self.assertGreater(len(decision.title), 0)

    def test_enrich_answer(self):
        """Test that enrich_answer adds context."""
        from smart_city.models import QuestionIntent, AnalysisResult
        features = self._make_urban_features()
        report = self.engine.evaluate(features)
        analysis = AnalysisResult(spatial_features=features, planning_report=report)
        
        intent = QuestionIntent(
            intent_type='density',
            target_objects=['building'],
            raw_question='Is this area overcrowded?',
        )
        
        enriched = self.engine.enrich_answer("Yes", analysis, intent)
        self.assertIn("Planning context", enriched)


if __name__ == '__main__':
    unittest.main()
