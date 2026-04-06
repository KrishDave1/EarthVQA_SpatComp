"""
Data models for the Smart City Planning Decision Support System.
All structured data flows through these dataclasses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ClassStats:
    """Statistics for a single segmentation class."""
    class_id: int
    class_name: str
    pixel_count: int
    area_percentage: float
    component_count: int  # number of connected components
    centroids: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class DistanceMetric:
    """Distance between two segmentation classes."""
    class_a: str
    class_b: str
    min_distance_px: float        # minimum boundary-to-boundary distance
    mean_centroid_distance_px: float  # mean centroid-to-centroid distance
    proximity_score: float        # 0-1 score (1 = very close)


@dataclass
class SpatialFeatures:
    """Complete spatial feature set extracted from a segmentation mask."""
    # Per-class statistics
    class_stats: Dict[str, ClassStats] = field(default_factory=dict)

    # Area coverage percentages (quick access)
    building_area_pct: float = 0.0
    road_area_pct: float = 0.0
    water_area_pct: float = 0.0
    vegetation_area_pct: float = 0.0    # forest + agriculture + playground
    barren_area_pct: float = 0.0
    forest_area_pct: float = 0.0
    agriculture_area_pct: float = 0.0
    playground_area_pct: float = 0.0

    # Object counts
    building_count: int = 0
    water_body_count: int = 0
    road_segment_count: int = 0

    # Distance metrics
    distances: List[DistanceMetric] = field(default_factory=list)
    building_water_min_distance: float = float('inf')
    building_road_min_distance: float = float('inf')

    # Density metrics
    building_density: float = 0.0  # buildings per unit area

    # Connectivity metrics
    intersection_count: int = 0
    road_connectivity_score: float = 0.0

    # Image metadata
    image_width: int = 0
    image_height: int = 0
    total_pixels: int = 0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        result = {
            'area_coverage': {
                'building': round(self.building_area_pct, 4),
                'road': round(self.road_area_pct, 4),
                'water': round(self.water_area_pct, 4),
                'vegetation': round(self.vegetation_area_pct, 4),
                'barren': round(self.barren_area_pct, 4),
                'forest': round(self.forest_area_pct, 4),
                'agriculture': round(self.agriculture_area_pct, 4),
                'playground': round(self.playground_area_pct, 4),
            },
            'object_counts': {
                'buildings': self.building_count,
                'water_bodies': self.water_body_count,
                'road_segments': self.road_segment_count,
            },
            'distances': {
                'building_to_water_min_px': round(self.building_water_min_distance, 2) if self.building_water_min_distance != float('inf') else None,
                'building_to_road_min_px': round(self.building_road_min_distance, 2) if self.building_road_min_distance != float('inf') else None,
            },
            'density': {
                'building_density': round(self.building_density, 4),
                'intersection_count': self.intersection_count,
                'road_connectivity': round(self.road_connectivity_score, 4),
            },
            'image_info': {
                'width': self.image_width,
                'height': self.image_height,
                'total_pixels': self.total_pixels,
            },
        }
        # Add distance details
        result['distance_details'] = [
            {
                'class_a': d.class_a,
                'class_b': d.class_b,
                'min_distance_px': round(d.min_distance_px, 2),
                'mean_centroid_distance_px': round(d.mean_centroid_distance_px, 2),
                'proximity_score': round(d.proximity_score, 4),
            }
            for d in self.distances
        ]
        return result


@dataclass
class Decision:
    """A single planning decision/recommendation."""
    category: str          # e.g., "density", "green_coverage", "flood_risk", "infrastructure"
    severity: str          # "low", "moderate", "high", "critical"
    score: float           # 0-1 composite score
    title: str             # Short title
    description: str       # Detailed explanation
    recommendation: str    # Actionable suggestion
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class PlanningReport:
    """Complete planning assessment report."""
    decisions: List[Decision] = field(default_factory=list)
    overall_suitability: str = ""           # e.g., "Suitable for residential expansion"
    overall_score: float = 0.0              # 0-1 composite score
    summary: str = ""                       # Natural language summary
    scene_type: str = ""                    # "urban" or "rural"

    def to_dict(self) -> dict:
        return {
            'overall_suitability': self.overall_suitability,
            'overall_score': round(self.overall_score, 4),
            'summary': self.summary,
            'scene_type': self.scene_type,
            'decisions': [
                {
                    'category': d.category,
                    'severity': d.severity,
                    'score': round(d.score, 4),
                    'title': d.title,
                    'description': d.description,
                    'recommendation': d.recommendation,
                    'contributing_factors': d.contributing_factors,
                }
                for d in self.decisions
            ],
        }


@dataclass
class AnalysisResult:
    """Complete analysis result for an image."""
    spatial_features: SpatialFeatures
    planning_report: PlanningReport
    seg_mask: Optional[np.ndarray] = None          # H x W segmentation mask
    colorized_mask: Optional[np.ndarray] = None    # H x W x 3 RGB mask

    def to_dict(self) -> dict:
        return {
            'spatial_features': self.spatial_features.to_dict(),
            'planning_report': self.planning_report.to_dict(),
        }


@dataclass
class QuestionIntent:
    """Parsed intent from a natural language question."""
    intent_type: str              # "counting", "relation", "density", "planning", "risk", "judging", "situation"
    target_objects: List[str]     # target classes (e.g., ["building", "road"])
    relation: Optional[str] = None  # "near", "in", "around"
    raw_question: str = ""


@dataclass
class VQAResult:
    """VQA answer enriched with decision intelligence."""
    answer: str                               # Direct answer from VQA model
    confidence: float = 0.0                   # Answer confidence score
    explanation: str = ""                     # Decision-enriched explanation
    intent: Optional[QuestionIntent] = None
    analysis: Optional[AnalysisResult] = None  # Full spatial analysis

    def to_dict(self) -> dict:
        result = {
            'answer': self.answer,
            'confidence': round(self.confidence, 4),
            'explanation': self.explanation,
        }
        if self.intent:
            result['intent'] = {
                'type': self.intent.intent_type,
                'targets': self.intent.target_objects,
                'relation': self.intent.relation,
            }
        if self.analysis:
            result['analysis'] = self.analysis.to_dict()
        return result
