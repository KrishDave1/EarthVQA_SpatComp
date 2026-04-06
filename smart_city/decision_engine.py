"""
Decision Intelligence Layer (Module 6)

Converts spatial features into actionable planning recommendations.
Uses soft rules with configurable thresholds (not hardcoded).
Thresholds can be auto-calibrated from training set statistics.

The 4 composite metrics:
    1. Urban Density Score
    2. Green Coverage Score
    3. Flood Risk Score
    4. Infrastructure Score
"""

import yaml
import os
from typing import List, Optional

from smart_city.models import (
    SpatialFeatures, Decision, PlanningReport
)


class DecisionEngine:
    """
    Evaluates spatial features and generates planning recommendations.
    
    All thresholds and weights are loaded from a YAML config file,
    making the system configurable and adaptable to different regions.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to thresholds.yaml.
                         If None, uses default thresholds.
        """
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load threshold configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        # Inline defaults (same as thresholds.yaml)
        return self._default_config()

    def _default_config(self) -> dict:
        return {
            'density': {
                'weights': {'building_count': 0.4, 'built_area_pct': 0.6},
                'thresholds': {'low': 0.25, 'moderate': 0.5, 'high': 0.7},
                'recommendations': {
                    'high': 'Area is densely populated. Consider limiting further construction and introducing open spaces.',
                    'moderate': 'Moderate building density. Area is suitable for controlled expansion.',
                    'low': 'Low building density. Area has potential for residential or commercial development.',
                },
            },
            'green_coverage': {
                'weights': {'vegetation_pct': 1.0, 'playground_pct': 0.5},
                'thresholds': {'insufficient': 0.10, 'low': 0.15, 'adequate': 0.25, 'good': 0.40},
                'target_pct': 0.20,
                'recommendations': {
                    'insufficient': 'Critical shortage of green spaces. Urgently recommend parks and tree planting programs.',
                    'low': 'Below recommended green coverage (15-25%). Suggest introducing parks or community gardens.',
                    'adequate': 'Green coverage meets minimum urban planning standards.',
                    'good': 'Excellent green coverage. Area has strong environmental quality.',
                },
            },
            'flood_risk': {
                'weights': {'water_proximity': 0.5, 'water_area_pct': 0.3, 'building_water_ratio': 0.2},
                'thresholds': {'low': 0.25, 'moderate': 0.5, 'high': 0.7},
                'min_safe_distance_px': 50,
                'recommendations': {
                    'high': 'High flood risk detected. Buildings are dangerously close to water bodies. Recommend buffer zones and flood barriers.',
                    'moderate': 'Moderate flood risk. Some structures are near water. Consider drainage improvements.',
                    'low': 'Low flood risk. Adequate distance between buildings and water bodies.',
                },
            },
            'infrastructure': {
                'weights': {'road_coverage_pct': 0.5, 'intersection_count': 0.3, 'road_connectivity': 0.2},
                'thresholds': {'poor': 0.25, 'moderate': 0.5, 'good': 0.7},
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
                'building_count_max': 200,
                'intersection_count_max': 10,
            },
        }

    def evaluate(self, features: SpatialFeatures) -> PlanningReport:
        """
        Evaluate spatial features and generate a complete planning report.
        
        Args:
            features: SpatialFeatures extracted from a segmentation mask.
        
        Returns:
            PlanningReport with decisions and overall assessment.
        """
        decisions = []
        scores = {}

        # 1. Urban Density Score
        density_decision, density_score = self._evaluate_density(features)
        decisions.append(density_decision)
        scores['density'] = density_score

        # 2. Green Coverage Score
        green_decision, green_score = self._evaluate_green_coverage(features)
        decisions.append(green_decision)
        scores['green_coverage'] = green_score

        # 3. Flood Risk Score
        flood_decision, flood_score = self._evaluate_flood_risk(features)
        decisions.append(flood_decision)
        scores['flood_risk'] = flood_score

        # 4. Infrastructure Score
        infra_decision, infra_score = self._evaluate_infrastructure(features)
        decisions.append(infra_decision)
        scores['infrastructure'] = infra_score

        # Overall assessment
        overall_score, suitability_label = self._compute_overall_suitability(scores)
        scene_type = self._classify_scene(features)
        summary = self._generate_summary(features, decisions, scene_type)

        return PlanningReport(
            decisions=decisions,
            overall_suitability=suitability_label,
            overall_score=overall_score,
            summary=summary,
            scene_type=scene_type,
        )

    # ─── Density Assessment ───────────────────────────────────────────

    def _evaluate_density(self, features: SpatialFeatures) -> tuple:
        """Evaluate urban density."""
        cfg = self.config['density']
        weights = cfg['weights']
        thresholds = cfg['thresholds']
        norm_max = self.config.get('normalization', {}).get('building_count_max', 200)
        
        # Compute composite density score
        norm_building_count = min(features.building_count / max(norm_max, 1), 1.0)
        score = (
            weights['building_count'] * norm_building_count +
            weights['built_area_pct'] * min(features.building_area_pct / 0.5, 1.0)  # normalize: 50% built = max
        )
        score = min(score, 1.0)

        # Classify severity
        if score >= thresholds['high']:
            severity = 'high'
            title = 'High Urban Density'
        elif score >= thresholds['moderate']:
            severity = 'moderate'
            title = 'Moderate Urban Density'
        else:
            severity = 'low'
            title = 'Low Urban Density'

        factors = []
        if features.building_count > 0:
            factors.append(f"{features.building_count} buildings detected")
        factors.append(f"Built area covers {features.building_area_pct*100:.1f}% of the scene")
        if features.building_density > 0:
            factors.append(f"Building density: {features.building_density:.2f}")

        decision = Decision(
            category='density',
            severity=severity,
            score=score,
            title=title,
            description=f"Urban density score: {score:.2f}/1.00. "
                        f"The area contains {features.building_count} buildings "
                        f"covering {features.building_area_pct*100:.1f}% of the scene.",
            recommendation=cfg['recommendations'][severity],
            contributing_factors=factors,
        )
        return decision, score

    # ─── Green Coverage Assessment ────────────────────────────────────

    def _evaluate_green_coverage(self, features: SpatialFeatures) -> tuple:
        """Evaluate green space coverage."""
        cfg = self.config['green_coverage']
        weights = cfg['weights']
        thresholds = cfg['thresholds']

        # Green coverage score (higher = more green = better)
        # We combine forest+agriculture and playground separately
        veg_pct = features.forest_area_pct + features.agriculture_area_pct
        play_pct = features.playground_area_pct
        raw_green = weights['vegetation_pct'] * veg_pct + weights['playground_pct'] * play_pct
        
        # Normalize to [0, 1] — full score at target green coverage
        target = cfg.get('target_pct', 0.20)
        score = min(raw_green / target, 1.0) if target > 0 else 0.0

        # Classify
        if raw_green >= thresholds['good']:
            severity = 'low'  # low concern = good
            title = 'Good Green Coverage'
            rec_key = 'good'
        elif raw_green >= thresholds['adequate']:
            severity = 'low'
            title = 'Adequate Green Coverage'
            rec_key = 'adequate'
        elif raw_green >= thresholds['low']:
            severity = 'moderate'
            title = 'Low Green Coverage'
            rec_key = 'low'
        else:
            severity = 'high'
            title = 'Insufficient Green Coverage'
            rec_key = 'insufficient'

        factors = []
        factors.append(f"Total vegetation: {features.vegetation_area_pct*100:.1f}%")
        if features.forest_area_pct > 0:
            factors.append(f"Forest: {features.forest_area_pct*100:.1f}%")
        if features.agriculture_area_pct > 0:
            factors.append(f"Agriculture: {features.agriculture_area_pct*100:.1f}%")
        if features.playground_area_pct > 0:
            factors.append(f"Playground: {features.playground_area_pct*100:.1f}%")
        factors.append(f"Recommended green coverage: 15-25%")

        decision = Decision(
            category='green_coverage',
            severity=severity,
            score=score,
            title=title,
            description=f"Green coverage score: {score:.2f}/1.00. "
                        f"Vegetation covers {features.vegetation_area_pct*100:.1f}% of the scene "
                        f"(target: {target*100:.0f}%).",
            recommendation=cfg['recommendations'][rec_key],
            contributing_factors=factors,
        )
        return decision, score

    # ─── Flood Risk Assessment ────────────────────────────────────────

    def _evaluate_flood_risk(self, features: SpatialFeatures) -> tuple:
        """Evaluate flood risk based on water proximity to buildings."""
        cfg = self.config['flood_risk']
        weights = cfg['weights']
        thresholds = cfg['thresholds']
        min_safe_dist = cfg.get('min_safe_distance_px', 50)

        # If no water or no buildings, flood risk is zero
        if features.water_area_pct == 0 or features.building_count == 0:
            decision = Decision(
                category='flood_risk',
                severity='low',
                score=0.0,
                title='No Flood Risk',
                description='No significant flood risk detected — no water bodies near built areas.',
                recommendation='No flood-related concerns for this area.',
                contributing_factors=['No water bodies detected' if features.water_area_pct == 0 else 'No buildings detected'],
            )
            return decision, 0.0

        # Water proximity factor: inverse of distance, capped
        dist = features.building_water_min_distance
        if dist == float('inf'):
            proximity_factor = 0.0
        else:
            # Score: 1 when dist=0, approaches 0 as dist increases
            proximity_factor = max(0, 1.0 - (dist / (min_safe_dist * 3)))

        # Water area factor
        water_factor = min(features.water_area_pct / 0.15, 1.0)  # saturates at 15% water

        # Building-water ratio: what fraction of the scene has both?
        bw_ratio = min(features.building_area_pct * features.water_area_pct * 10, 1.0)

        # Composite flood risk score
        score = (
            weights['water_proximity'] * proximity_factor +
            weights['water_area_pct'] * water_factor +
            weights['building_water_ratio'] * bw_ratio
        )
        score = min(score, 1.0)

        # Classify
        if score >= thresholds['high']:
            severity = 'high'
            title = 'High Flood Risk'
        elif score >= thresholds['moderate']:
            severity = 'moderate'
            title = 'Moderate Flood Risk'
        else:
            severity = 'low'
            title = 'Low Flood Risk'

        factors = []
        if dist != float('inf'):
            factors.append(f"Nearest building-water distance: {dist:.0f} pixels")
            if dist < min_safe_dist:
                factors.append(f"⚠ Below safe distance threshold ({min_safe_dist} px)")
        factors.append(f"Water coverage: {features.water_area_pct*100:.1f}%")
        factors.append(f"{features.water_body_count} water bodies detected")

        decision = Decision(
            category='flood_risk',
            severity=severity,
            score=score,
            title=title,
            description=f"Flood risk score: {score:.2f}/1.00. "
                        f"Buildings are {dist:.0f}px from nearest water body."
                        if dist != float('inf')
                        else f"Flood risk score: {score:.2f}/1.00.",
            recommendation=cfg['recommendations'][severity],
            contributing_factors=factors,
        )
        return decision, score

    # ─── Infrastructure Assessment ────────────────────────────────────

    def _evaluate_infrastructure(self, features: SpatialFeatures) -> tuple:
        """Evaluate road infrastructure quality."""
        cfg = self.config['infrastructure']
        weights = cfg['weights']
        thresholds = cfg['thresholds']
        norm_max_intersections = self.config.get('normalization', {}).get('intersection_count_max', 10)

        # Road coverage factor (saturates at 15%)
        road_factor = min(features.road_area_pct / 0.15, 1.0)

        # Intersection density factor
        intersection_factor = min(features.intersection_count / max(norm_max_intersections, 1), 1.0)

        # Road connectivity factor
        connectivity_factor = min(features.road_connectivity_score, 1.0)

        # Composite infrastructure score
        score = (
            weights['road_coverage_pct'] * road_factor +
            weights['intersection_count'] * intersection_factor +
            weights['road_connectivity'] * connectivity_factor
        )
        score = min(score, 1.0)

        # Classify
        if score >= thresholds['good']:
            severity = 'low'  # low concern = good infra
            title = 'Good Infrastructure'
            rec_key = 'good'
        elif score >= thresholds['moderate']:
            severity = 'moderate'
            title = 'Moderate Infrastructure'
            rec_key = 'moderate'
        else:
            severity = 'high'
            title = 'Poor Infrastructure'
            rec_key = 'poor'

        factors = []
        factors.append(f"Road coverage: {features.road_area_pct*100:.1f}%")
        factors.append(f"Intersections detected: {features.intersection_count}")
        factors.append(f"Road connectivity score: {features.road_connectivity_score:.2f}")
        factors.append(f"Road segments: {features.road_segment_count}")

        decision = Decision(
            category='infrastructure',
            severity=severity,
            score=score,
            title=title,
            description=f"Infrastructure score: {score:.2f}/1.00. "
                        f"Road network covers {features.road_area_pct*100:.1f}% with "
                        f"{features.intersection_count} intersections.",
            recommendation=cfg['recommendations'][rec_key],
            contributing_factors=factors,
        )
        return decision, score

    # ─── Overall Assessment ───────────────────────────────────────────

    def _compute_overall_suitability(self, scores: dict) -> tuple:
        """Compute weighted overall suitability score."""
        cfg = self.config.get('planning', {})
        weights = cfg.get('suitability_weights', {
            'density': 0.25, 'green_coverage': 0.25,
            'flood_risk': 0.25, 'infrastructure': 0.25,
        })
        labels = cfg.get('suitability_labels', [
            {'label': 'Not Suitable', 'range': [0.0, 0.3]},
            {'label': 'Needs Improvement', 'range': [0.3, 0.5]},
            {'label': 'Moderately Suitable', 'range': [0.5, 0.7]},
            {'label': 'Suitable', 'range': [0.7, 0.85]},
            {'label': 'Highly Suitable', 'range': [0.85, 1.0]},
        ])

        # For suitability, we want:
        # - density: moderate is best (not too high, not too low) → use inverted U
        # - green_coverage: higher is better
        # - flood_risk: lower is better → invert
        # - infrastructure: higher is better

        # Transform scores for suitability
        density_suit = 1.0 - abs(scores['density'] - 0.5) * 2   # peaks at 0.5
        green_suit = scores['green_coverage']
        flood_suit = 1.0 - scores['flood_risk']  # invert: low risk = high suitability
        infra_suit = scores['infrastructure']

        overall = (
            weights['density'] * density_suit +
            weights['green_coverage'] * green_suit +
            weights['flood_risk'] * flood_suit +
            weights['infrastructure'] * infra_suit
        )
        overall = max(0, min(overall, 1.0))

        # Find label
        label = 'Unknown'
        for entry in labels:
            r = entry['range']
            if r[0] <= overall < r[1]:
                label = entry['label']
                break
        if overall >= 1.0:
            label = labels[-1]['label']

        return overall, label

    def _classify_scene(self, features: SpatialFeatures) -> str:
        """Classify scene as urban or rural based on spatial features."""
        # Simple heuristic: if building + road coverage > 30%, it's urban
        built_pct = features.building_area_pct + features.road_area_pct
        green_pct = features.vegetation_area_pct
        
        if built_pct > 0.30:
            return 'urban'
        elif green_pct > 0.50:
            return 'rural'
        elif built_pct > 0.15:
            return 'suburban'
        else:
            return 'rural'

    def _generate_summary(self, features: SpatialFeatures,
                          decisions: List[Decision], scene_type: str) -> str:
        """Generate a natural language summary of the analysis."""
        parts = []
        
        # Scene classification
        parts.append(f"This appears to be a {scene_type} area.")
        
        # Key statistics
        if features.building_count > 0:
            parts.append(
                f"The scene contains {features.building_count} buildings "
                f"covering {features.building_area_pct*100:.1f}% of the area."
            )
        
        if features.vegetation_area_pct > 0:
            parts.append(
                f"Green spaces (vegetation, agriculture, playgrounds) "
                f"cover {features.vegetation_area_pct*100:.1f}% of the area."
            )
        
        if features.water_body_count > 0:
            parts.append(
                f"There {'is' if features.water_body_count == 1 else 'are'} "
                f"{features.water_body_count} water "
                f"{'body' if features.water_body_count == 1 else 'bodies'} present."
            )
        
        # Highlight high-severity decisions
        high_severity = [d for d in decisions if d.severity == 'high']
        if high_severity:
            parts.append("\nKey concerns:")
            for d in high_severity:
                parts.append(f"• {d.title}: {d.recommendation}")
        
        # Positive aspects
        low_severity = [d for d in decisions if d.severity == 'low']
        if low_severity:
            positive = [d for d in low_severity if d.category in ('green_coverage', 'infrastructure')]
            if positive:
                parts.append("\nPositive aspects:")
                for d in positive:
                    parts.append(f"• {d.title}")
        
        return ' '.join(parts)

    def enrich_answer(self, vqa_answer: str, analysis_result, intent) -> str:
        """
        Enrich a VQA answer with decision intelligence context.
        
        This adds planning insights to the raw VQA model answer.
        """
        if analysis_result is None:
            return vqa_answer
        
        report = analysis_result.planning_report
        features = analysis_result.spatial_features
        enrichment_parts = [vqa_answer]
        
        if intent is None:
            return vqa_answer
        
        intent_type = intent.intent_type
        
        # Add relevant context based on question intent
        if intent_type in ('density', 'counting'):
            # Add density context
            density_dec = next((d for d in report.decisions if d.category == 'density'), None)
            if density_dec:
                enrichment_parts.append(f"\n\nPlanning context: {density_dec.description}")
                enrichment_parts.append(f"Recommendation: {density_dec.recommendation}")
        
        elif intent_type == 'risk':
            flood_dec = next((d for d in report.decisions if d.category == 'flood_risk'), None)
            if flood_dec:
                enrichment_parts.append(f"\n\nRisk assessment: {flood_dec.description}")
                enrichment_parts.append(f"Recommendation: {flood_dec.recommendation}")
        
        elif intent_type == 'planning':
            enrichment_parts.append(f"\n\nPlanning assessment: {report.summary}")
            enrichment_parts.append(f"Overall suitability: {report.overall_suitability} ({report.overall_score:.2f})")
        
        elif intent_type == 'relation':
            # Add relevant distance information
            for target in intent.target_objects:
                for d in features.distances:
                    if target in (d.class_a, d.class_b):
                        if d.min_distance_px != float('inf'):
                            enrichment_parts.append(
                                f"\nSpatial context: {d.class_a} to {d.class_b} "
                                f"minimum distance: {d.min_distance_px:.0f} pixels"
                            )
        
        return '\n'.join(enrichment_parts) if len(enrichment_parts) > 1 else vqa_answer
