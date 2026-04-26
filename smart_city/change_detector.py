"""
Time-Series Change Detection Engine (Module 8)

Computes delta (Δ) spatial features between two temporal satellite images
and classifies the type of urban sprawl occurring.

Supports:
    - ML-based classification via a pre-trained SVM/Random Forest (.joblib)
    - Rule-based fallback classification using heuristic thresholds
    - Delta feature extraction from any two SpatialFeatures objects

Sprawl Categories:
    1. Aggressive Urbanization — Buildings ↑↑, Vegetation ↓↓
    2. Deforestation           — Forest ↓↓, Barren/Agriculture ↑
    3. Water Encroachment      — Water ↑↑, Buildings near water ↑
    4. Sustainable Expansion   — Buildings ↑, Green ≈ stable, Infra ↑
    5. Infrastructure Development — Roads ↑↑, Intersections ↑↑
    6. Stable / No Change      — All Δ values ≈ 0
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from smart_city.models import SpatialFeatures

# Conditional joblib import
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ─── Sprawl Category Constants ───────────────────────────────────────

SPRAWL_CATEGORIES = [
    'Aggressive Urbanization',
    'Deforestation',
    'Water Encroachment',
    'Sustainable Expansion',
    'Infrastructure Development',
    'Stable / No Change',
]

SPRAWL_DESCRIPTIONS = {
    'Aggressive Urbanization': (
        "Rapid, unplanned urban growth detected. Building area has significantly increased "
        "while vegetation has sharply declined. This pattern suggests aggressive construction "
        "activity replacing natural land cover."
    ),
    'Deforestation': (
        "Significant loss of forest cover detected. Tree coverage has declined substantially, "
        "often replaced by barren land or agricultural expansion. This indicates environmental "
        "degradation requiring immediate attention."
    ),
    'Water Encroachment': (
        "Water bodies have expanded significantly, with increased proximity to built structures. "
        "This may indicate flooding risk, rising water levels, or encroachment of development "
        "into flood-prone zones."
    ),
    'Sustainable Expansion': (
        "Balanced urban growth detected. Building area has increased moderately while green "
        "spaces have been maintained. Road infrastructure has kept pace with development, "
        "suggesting well-planned expansion."
    ),
    'Infrastructure Development': (
        "Major infrastructure expansion detected. Road network coverage and intersection "
        "density have increased significantly, indicating investment in transportation "
        "connectivity and urban infrastructure."
    ),
    'Stable / No Change': (
        "No significant land-use changes detected between the two time periods. "
        "The urban landscape has remained largely stable."
    ),
}

SPRAWL_RECOMMENDATIONS = {
    'Aggressive Urbanization': [
        "Implement strict zoning laws to control further unplanned expansion.",
        "Mandate green buffer zones between new construction and existing vegetation.",
        "Develop urban parks and green corridors to compensate for lost vegetation.",
        "Monitor building density to prevent overcrowding and ensure livability.",
        "Consider a moratorium on new construction in critically affected zones.",
    ],
    'Deforestation': [
        "Launch reforestation programs in affected areas immediately.",
        "Enforce protected forest zones with no-build regulations.",
        "Promote agroforestry as an alternative to pure agricultural expansion.",
        "Install satellite monitoring for early detection of further deforestation.",
        "Engage community stakeholders in forest conservation initiatives.",
    ],
    'Water Encroachment': [
        "Establish mandatory flood buffer zones around all water bodies.",
        "Relocate vulnerable structures away from expanding water boundaries.",
        "Invest in flood defense infrastructure (levees, drainage systems).",
        "Conduct hydrological risk assessments for all nearby developments.",
        "Implement early-warning systems for flood-prone areas.",
    ],
    'Sustainable Expansion': [
        "Continue current balanced development policies.",
        "Monitor green coverage to ensure it remains above 15% threshold.",
        "Expand public transit to support growing population sustainably.",
        "Promote energy-efficient building standards for new construction.",
        "Maintain and upgrade road infrastructure to match growth rate.",
    ],
    'Infrastructure Development': [
        "Ensure road expansion benefits all neighborhoods equitably.",
        "Add pedestrian and cycling infrastructure alongside road development.",
        "Plan green medians and tree-lined corridors along new roads.",
        "Monitor air quality impact of increased road infrastructure.",
        "Coordinate infrastructure growth with public transportation planning.",
    ],
    'Stable / No Change': [
        "Continue monitoring for future changes.",
        "Evaluate whether stability reflects successful planning or stagnation.",
        "Consider investment opportunities if the area has growth potential.",
        "Maintain existing infrastructure and green spaces.",
    ],
}

SPRAWL_ICONS = {
    'Aggressive Urbanization': '🔴',
    'Deforestation': '🟤',
    'Water Encroachment': '🔵',
    'Sustainable Expansion': '🟢',
    'Infrastructure Development': '🟡',
    'Stable / No Change': '⚪',
}


# ─── Data Classes ─────────────────────────────────────────────────────

@dataclass
class DeltaFeatures:
    """
    Delta (Δ) spatial features computed between two temporal snapshots.
    Positive values indicate increase from T₁ → T₂, negative = decrease.
    """
    # Absolute deltas (T₂ - T₁)
    delta_building_area_pct: float = 0.0
    delta_road_area_pct: float = 0.0
    delta_water_area_pct: float = 0.0
    delta_vegetation_area_pct: float = 0.0
    delta_barren_area_pct: float = 0.0
    delta_forest_area_pct: float = 0.0
    delta_agriculture_area_pct: float = 0.0
    delta_playground_area_pct: float = 0.0

    # Object count deltas
    delta_building_count: int = 0
    delta_water_body_count: int = 0
    delta_road_segment_count: int = 0

    # Metric deltas
    delta_building_density: float = 0.0
    delta_road_connectivity: float = 0.0
    delta_intersection_count: int = 0

    # Distance deltas (negative = getting closer)
    delta_building_water_distance: float = 0.0
    delta_building_road_distance: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to a numpy array for ML classification input."""
        return np.array([
            self.delta_building_area_pct,
            self.delta_road_area_pct,
            self.delta_water_area_pct,
            self.delta_vegetation_area_pct,
            self.delta_barren_area_pct,
            self.delta_forest_area_pct,
            self.delta_agriculture_area_pct,
            self.delta_playground_area_pct,
            self.delta_building_count,
            self.delta_water_body_count,
            self.delta_road_segment_count,
            self.delta_building_density,
            self.delta_road_connectivity,
            self.delta_intersection_count,
            self.delta_building_water_distance,
            self.delta_building_road_distance,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        """Return the names of features in the feature vector (same order)."""
        return [
            'Δ Building Area %',
            'Δ Road Area %',
            'Δ Water Area %',
            'Δ Vegetation Area %',
            'Δ Barren Area %',
            'Δ Forest Area %',
            'Δ Agriculture Area %',
            'Δ Playground Area %',
            'Δ Building Count',
            'Δ Water Body Count',
            'Δ Road Segment Count',
            'Δ Building Density',
            'Δ Road Connectivity',
            'Δ Intersection Count',
            'Δ Building-Water Distance',
            'Δ Building-Road Distance',
        ]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        names = self.feature_names()
        vec = self.to_feature_vector()
        return {
            'absolute_deltas': {
                'building_area_pct': round(self.delta_building_area_pct, 4),
                'road_area_pct': round(self.delta_road_area_pct, 4),
                'water_area_pct': round(self.delta_water_area_pct, 4),
                'vegetation_area_pct': round(self.delta_vegetation_area_pct, 4),
                'barren_area_pct': round(self.delta_barren_area_pct, 4),
                'forest_area_pct': round(self.delta_forest_area_pct, 4),
                'agriculture_area_pct': round(self.delta_agriculture_area_pct, 4),
                'playground_area_pct': round(self.delta_playground_area_pct, 4),
            },
            'count_deltas': {
                'building_count': self.delta_building_count,
                'water_body_count': self.delta_water_body_count,
                'road_segment_count': self.delta_road_segment_count,
                'intersection_count': self.delta_intersection_count,
            },
            'metric_deltas': {
                'building_density': round(self.delta_building_density, 4),
                'road_connectivity': round(self.delta_road_connectivity, 4),
            },
            'distance_deltas': {
                'building_water_distance': (
                    round(self.delta_building_water_distance, 2)
                    if self.delta_building_water_distance != float('inf')
                    and self.delta_building_water_distance != float('-inf')
                    else None
                ),
                'building_road_distance': (
                    round(self.delta_building_road_distance, 2)
                    if self.delta_building_road_distance != float('inf')
                    and self.delta_building_road_distance != float('-inf')
                    else None
                ),
            },
            'feature_vector': [
                {'name': n, 'value': round(float(v), 4)}
                for n, v in zip(names, vec)
                if abs(v) != float('inf')
            ],
        }


@dataclass
class ChangeDetectionResult:
    """Complete result of a temporal change detection analysis."""
    sprawl_type: str                              # Classification label
    confidence: float                             # 0-1 classifier confidence
    description: str                              # Human-readable explanation
    delta_features: DeltaFeatures                 # Computed Δ features
    features_before: Optional[SpatialFeatures] = None
    features_after: Optional[SpatialFeatures] = None
    recommendations: List[str] = field(default_factory=list)
    icon: str = ''                                # Emoji icon for the sprawl type
    classifier_type: str = 'rule-based'           # 'svm', 'random_forest', or 'rule-based'

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        result = {
            'sprawl_type': self.sprawl_type,
            'confidence': round(self.confidence, 4),
            'description': self.description,
            'icon': self.icon,
            'classifier_type': self.classifier_type,
            'recommendations': self.recommendations,
            'delta_features': self.delta_features.to_dict(),
        }
        if self.features_before:
            result['features_before'] = self.features_before.to_dict()
        if self.features_after:
            result['features_after'] = self.features_after.to_dict()
        return result


# ─── Change Detector ──────────────────────────────────────────────────

class ChangeDetector:
    """
    Computes delta features between two temporal satellite images
    and classifies the type of urban sprawl occurring.

    Supports:
        - ML-based classification (SVM/Random Forest from .joblib)
        - Rule-based fallback classification (deterministic heuristics)

    Usage:
        detector = ChangeDetector()
        result = detector.analyze(spatial_features_t1, spatial_features_t2)
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to pre-trained classifier (.joblib file).
                        If None or file doesn't exist, falls back to rule-based.
        """
        self.model = None
        self.classifier_type = 'rule-based'

        if model_path is None:
            # Try default path
            default_path = os.path.join(
                os.path.dirname(__file__), 'config', 'change_classifier.joblib'
            )
            if os.path.exists(default_path):
                model_path = default_path

        if model_path and os.path.exists(model_path) and JOBLIB_AVAILABLE:
            try:
                self.model = joblib.load(model_path)
                self.classifier_type = getattr(
                    self.model, '_classifier_type', 'ml-classifier'
                )
                print(f"[ChangeDetector] Loaded ML classifier from: {model_path}")
                print(f"[ChangeDetector] Classifier type: {self.classifier_type}")
            except Exception as e:
                print(f"[ChangeDetector] Failed to load ML model: {e}")
                print("[ChangeDetector] Falling back to rule-based classifier.")
                self.model = None
        else:
            print("[ChangeDetector] No ML model found. Using rule-based classifier.")

    # ─── Delta Feature Computation ────────────────────────────────────

    def compute_delta(
        self,
        features_before: SpatialFeatures,
        features_after: SpatialFeatures,
    ) -> DeltaFeatures:
        """
        Compute the delta (Δ) between two SpatialFeatures snapshots.

        Args:
            features_before: Spatial features from the earlier time period (T₁).
            features_after: Spatial features from the later time period (T₂).

        Returns:
            DeltaFeatures with all Δ values (T₂ - T₁).
        """
        # Safe distance delta computation (handle inf values)
        def safe_dist_delta(after_val, before_val):
            if after_val == float('inf') and before_val == float('inf'):
                return 0.0
            if before_val == float('inf'):
                return 0.0  # No meaningful delta if before was inf
            if after_val == float('inf'):
                return 0.0  # No meaningful delta if after is inf
            return after_val - before_val

        return DeltaFeatures(
            # Area percentage deltas
            delta_building_area_pct=(
                features_after.building_area_pct - features_before.building_area_pct
            ),
            delta_road_area_pct=(
                features_after.road_area_pct - features_before.road_area_pct
            ),
            delta_water_area_pct=(
                features_after.water_area_pct - features_before.water_area_pct
            ),
            delta_vegetation_area_pct=(
                features_after.vegetation_area_pct - features_before.vegetation_area_pct
            ),
            delta_barren_area_pct=(
                features_after.barren_area_pct - features_before.barren_area_pct
            ),
            delta_forest_area_pct=(
                features_after.forest_area_pct - features_before.forest_area_pct
            ),
            delta_agriculture_area_pct=(
                features_after.agriculture_area_pct - features_before.agriculture_area_pct
            ),
            delta_playground_area_pct=(
                features_after.playground_area_pct - features_before.playground_area_pct
            ),
            # Object count deltas
            delta_building_count=(
                features_after.building_count - features_before.building_count
            ),
            delta_water_body_count=(
                features_after.water_body_count - features_before.water_body_count
            ),
            delta_road_segment_count=(
                features_after.road_segment_count - features_before.road_segment_count
            ),
            # Metric deltas
            delta_building_density=(
                features_after.building_density - features_before.building_density
            ),
            delta_road_connectivity=(
                features_after.road_connectivity_score - features_before.road_connectivity_score
            ),
            delta_intersection_count=(
                features_after.intersection_count - features_before.intersection_count
            ),
            # Distance deltas
            delta_building_water_distance=safe_dist_delta(
                features_after.building_water_min_distance,
                features_before.building_water_min_distance,
            ),
            delta_building_road_distance=safe_dist_delta(
                features_after.building_road_min_distance,
                features_before.building_road_min_distance,
            ),
        )

    # ─── Classification ───────────────────────────────────────────────

    def classify(self, delta: DeltaFeatures) -> Tuple[str, float]:
        """
        Classify the type of urban sprawl from delta features.

        Uses ML model if available, otherwise falls back to rule-based.

        Args:
            delta: DeltaFeatures computed from two temporal snapshots.

        Returns:
            Tuple of (sprawl_type, confidence).
        """
        if self.model is not None:
            return self._classify_ml(delta)
        return self._classify_rule_based(delta)

    def _classify_ml(self, delta: DeltaFeatures) -> Tuple[str, float]:
        """Classify using the pre-trained ML model."""
        feature_vec = delta.to_feature_vector().reshape(1, -1)

        try:
            prediction = self.model.predict(feature_vec)[0]

            # Get confidence from predict_proba if available
            confidence = 0.85  # default
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(feature_vec)[0]
                confidence = float(np.max(probas))
            elif hasattr(self.model, 'decision_function'):
                # SVM: map decision function to pseudo-probability
                decision = self.model.decision_function(feature_vec)[0]
                if isinstance(decision, np.ndarray):
                    confidence = float(1.0 / (1.0 + np.exp(-np.max(np.abs(decision)))))
                else:
                    confidence = float(1.0 / (1.0 + np.exp(-abs(decision))))

            # Map prediction to category name
            if isinstance(prediction, (int, np.integer)):
                if 0 <= prediction < len(SPRAWL_CATEGORIES):
                    sprawl_type = SPRAWL_CATEGORIES[prediction]
                else:
                    sprawl_type = 'Stable / No Change'
            else:
                sprawl_type = str(prediction)

            return sprawl_type, confidence

        except Exception as e:
            print(f"[ChangeDetector] ML classification failed: {e}")
            return self._classify_rule_based(delta)

    def _classify_rule_based(self, delta: DeltaFeatures) -> Tuple[str, float]:
        """
        Rule-based (deterministic) classification using heuristic thresholds.

        This is the fallback when no ML model is available. It uses
        domain-expert thresholds on the Δ feature values.
        """
        d = delta

        # Compute a score for each category
        scores: Dict[str, float] = {}

        # 1. Aggressive Urbanization: buildings ↑↑, vegetation ↓↓
        building_up = max(d.delta_building_area_pct, 0) * 100  # scale to percentage points
        veg_down = max(-d.delta_vegetation_area_pct, 0) * 100
        scores['Aggressive Urbanization'] = (
            0.5 * min(building_up / 10.0, 1.0) +
            0.5 * min(veg_down / 15.0, 1.0)
        )

        # 2. Deforestation: forest ↓↓, barren/agriculture ↑
        forest_down = max(-d.delta_forest_area_pct, 0) * 100
        barren_up = max(d.delta_barren_area_pct, 0) * 100
        agri_up = max(d.delta_agriculture_area_pct, 0) * 100
        scores['Deforestation'] = (
            0.6 * min(forest_down / 15.0, 1.0) +
            0.4 * min((barren_up + agri_up) / 10.0, 1.0)
        )

        # 3. Water Encroachment: water ↑↑, buildings near water ↑
        water_up = max(d.delta_water_area_pct, 0) * 100
        water_closer = max(-d.delta_building_water_distance, 0)  # negative = closer
        scores['Water Encroachment'] = (
            0.6 * min(water_up / 8.0, 1.0) +
            0.4 * min(water_closer / 50.0, 1.0)
        )

        # 4. Sustainable Expansion: buildings ↑ moderate, green ≈ stable, infra ↑
        building_moderate = min(building_up / 5.0, 1.0) if building_up > 0 else 0.0
        green_stable = 1.0 - min(abs(d.delta_vegetation_area_pct) * 100 / 5.0, 1.0)
        infra_up = max(d.delta_road_area_pct, 0) * 100
        scores['Sustainable Expansion'] = (
            0.35 * building_moderate +
            0.35 * green_stable +
            0.30 * min(infra_up / 3.0, 1.0)
        )

        # 5. Infrastructure Development: roads ↑↑, intersections ↑↑
        road_up = max(d.delta_road_area_pct, 0) * 100
        intersection_up = max(d.delta_intersection_count, 0)
        scores['Infrastructure Development'] = (
            0.5 * min(road_up / 5.0, 1.0) +
            0.5 * min(intersection_up / 5.0, 1.0)
        )

        # 6. Stable / No Change: all deltas near zero
        total_change = (
            abs(d.delta_building_area_pct) +
            abs(d.delta_vegetation_area_pct) +
            abs(d.delta_water_area_pct) +
            abs(d.delta_road_area_pct) +
            abs(d.delta_barren_area_pct)
        ) * 100
        scores['Stable / No Change'] = max(0, 1.0 - (total_change / 10.0))

        # Find the best category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        # If no category scores significantly, it's stable
        if best_score < 0.15:
            best_category = 'Stable / No Change'
            best_score = 0.8

        # Scale confidence (0.5 - 0.95 range for rule-based)
        confidence = min(0.5 + best_score * 0.45, 0.95)

        return best_category, confidence

    # ─── Full Analysis ────────────────────────────────────────────────

    def analyze(
        self,
        features_before: SpatialFeatures,
        features_after: SpatialFeatures,
    ) -> ChangeDetectionResult:
        """
        Full change detection analysis: compute deltas → classify → report.

        Args:
            features_before: Spatial features from time period T₁.
            features_after: Spatial features from time period T₂.

        Returns:
            ChangeDetectionResult with classification, deltas, and recommendations.
        """
        # Step 1: Compute delta features
        delta = self.compute_delta(features_before, features_after)

        # Step 2: Classify the sprawl type
        sprawl_type, confidence = self.classify(delta)

        # Step 3: Get recommendations and description
        recommendations = self._get_recommendations(sprawl_type, delta)
        description = SPRAWL_DESCRIPTIONS.get(sprawl_type, '')
        icon = SPRAWL_ICONS.get(sprawl_type, '⚪')

        return ChangeDetectionResult(
            sprawl_type=sprawl_type,
            confidence=confidence,
            description=description,
            delta_features=delta,
            features_before=features_before,
            features_after=features_after,
            recommendations=recommendations,
            icon=icon,
            classifier_type=self.classifier_type,
        )

    def _get_recommendations(
        self, sprawl_type: str, delta: DeltaFeatures
    ) -> List[str]:
        """Get actionable recommendations based on the sprawl classification."""
        base_recs = SPRAWL_RECOMMENDATIONS.get(sprawl_type, [])

        # Add data-driven specifics
        extra_recs = []
        d = delta

        if d.delta_building_area_pct > 0.05:
            pct = d.delta_building_area_pct * 100
            extra_recs.append(
                f"Building area increased by {pct:.1f}pp — "
                "review construction permits and zoning compliance."
            )

        if d.delta_vegetation_area_pct < -0.05:
            pct = abs(d.delta_vegetation_area_pct) * 100
            extra_recs.append(
                f"Vegetation decreased by {pct:.1f}pp — "
                "consider mandatory green compensation requirements."
            )

        if d.delta_water_area_pct > 0.03:
            pct = d.delta_water_area_pct * 100
            extra_recs.append(
                f"Water bodies expanded by {pct:.1f}pp — "
                "conduct flood risk reassessment for nearby structures."
            )

        # Combine base + dynamic, limit to 5
        all_recs = base_recs[:3] + extra_recs[:2]
        return all_recs[:5]
