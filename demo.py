#!/usr/bin/env python3
"""
Demo Script — Smart City Planning Decision Support System

Demonstrates the full pipeline using synthetic segmentation masks.
No GPU, dataset, or pre-trained weights required.

This script:
1. Creates 3 synthetic masks (urban, rural, flood-risk)
2. Runs spatial feature extraction on each
3. Runs decision intelligence analysis
4. Tests the intent parser on sample questions
5. Tests the full pipeline (mask → analysis → question answering)
6. Saves results as JSON

Usage:
    python demo.py
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from smart_city.spatial_engine import SpatialFeatureEngine
from smart_city.decision_engine import DecisionEngine
from smart_city.intent_parser import IntentParser
from smart_city.pipeline import SmartCityPipeline


def create_urban_mask(size=512):
    """Create a synthetic urban scene mask."""
    mask = np.zeros((size, size), dtype=np.int32)
    
    # Dense buildings (class 1) — majority of the area
    mask[20:240, 20:240] = 1
    mask[20:240, 280:490] = 1
    mask[280:490, 20:240] = 1
    mask[280:490, 280:490] = 1
    
    # Roads (class 2) — grid pattern
    mask[240:280, :] = 2     # horizontal major road
    mask[:, 240:280] = 2     # vertical major road
    mask[120:125, 20:240] = 2  # smaller roads
    mask[380:385, 280:490] = 2
    mask[20:240, 150:155] = 2
    mask[280:490, 400:405] = 2
    
    # Small park (forest, class 5)
    mask[50:100, 300:360] = 5
    
    # Water feature (class 3) — small pond
    mask[420:460, 50:90] = 3
    
    # Barren land (class 4)
    mask[0:20, :] = 4
    mask[:, 0:20] = 4
    mask[490:, :] = 4
    mask[:, 490:] = 4
    
    return mask


def create_rural_mask(size=512):
    """Create a synthetic rural scene mask."""
    mask = np.zeros((size, size), dtype=np.int32)
    
    # Agriculture (class 6) — dominant
    mask[:, :] = 6
    
    # Forest (class 5) — large patches
    mask[0:200, 0:180] = 5
    mask[350:512, 300:512] = 5
    
    # Small village (buildings, class 1)
    mask[220:260, 220:260] = 1
    mask[225:245, 270:290] = 1
    mask[260:280, 230:250] = 1
    
    # River (water, class 3)
    for y in range(512):
        cx = int(350 + 30 * np.sin(y / 50.0))
        mask[y, max(0, cx-8):min(512, cx+8)] = 3
    
    # Dirt road (class 2)
    mask[240:244, 100:350] = 2
    mask[200:300, 195:199] = 2
    
    return mask


def create_flood_risk_mask(size=512):
    """Create a synthetic flood-risk scene mask."""
    mask = np.zeros((size, size), dtype=np.int32)
    
    # Large water body (class 3) — lake
    mask[200:400, 0:200] = 3
    
    # Buildings very close to water (class 1)
    mask[180:200, 50:180] = 1    # right at the water edge!
    mask[400:430, 50:180] = 1    # also at edge
    mask[220:380, 200:240] = 1   # very close to water
    mask[100:180, 50:180] = 1    # a bit further
    
    # Some buildings far from water
    mask[50:90, 350:450] = 1
    mask[350:450, 350:450] = 1
    
    # Roads (class 2)
    mask[195:205, :] = 2
    mask[:, 245:255] = 2
    mask[430:435, 50:450] = 2
    
    # Some farmland (class 6)
    mask[0:100, 260:340] = 6
    mask[440:512, 260:512] = 6
    
    # Forest (class 5)
    mask[0:50, 0:200] = 5
    
    # Barren (class 4)
    mask[100:180, 200:250] = 4
    
    return mask


def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_spatial_engine():
    """Demonstrate the spatial feature engine."""
    print_separator("MODULE 3: Spatial Feature Extraction Engine")
    
    engine = SpatialFeatureEngine()
    
    scenarios = [
        ("Urban Dense Scene", create_urban_mask()),
        ("Rural/Green Scene", create_rural_mask()),
        ("Flood Risk Scene", create_flood_risk_mask()),
    ]
    
    for name, mask in scenarios:
        print(f"\n--- {name} ---")
        features = engine.extract(mask)
        
        print(f"  Image size: {features.image_width}x{features.image_height}")
        print(f"  Buildings: {features.building_count} ({features.building_area_pct*100:.1f}%)")
        print(f"  Roads: {features.road_segment_count} segments ({features.road_area_pct*100:.1f}%)")
        print(f"  Water: {features.water_body_count} bodies ({features.water_area_pct*100:.1f}%)")
        print(f"  Vegetation: {features.vegetation_area_pct*100:.1f}% (forest: {features.forest_area_pct*100:.1f}%, agri: {features.agriculture_area_pct*100:.1f}%)")
        print(f"  Intersections: {features.intersection_count}")
        print(f"  Building-Water distance: {features.building_water_min_distance:.1f}px")
        print(f"  Building-Road distance: {features.building_road_min_distance:.1f}px")
        print(f"  Building density: {features.building_density:.3f}")


def demo_decision_engine():
    """Demonstrate the decision intelligence layer."""
    print_separator("MODULE 6: Decision Intelligence Layer")
    
    engine = SpatialFeatureEngine()
    decision = DecisionEngine()
    
    scenarios = [
        ("Urban Dense Scene", create_urban_mask()),
        ("Rural/Green Scene", create_rural_mask()),
        ("Flood Risk Scene", create_flood_risk_mask()),
    ]
    
    for name, mask in scenarios:
        print(f"\n--- {name} ---")
        features = engine.extract(mask)
        report = decision.evaluate(features)
        
        print(f"  Scene type: {report.scene_type}")
        print(f"  Overall suitability: {report.overall_suitability} (score: {report.overall_score:.2f})")
        print(f"\n  Decisions:")
        for d in report.decisions:
            icon = "🔴" if d.severity == "high" else ("🟡" if d.severity == "moderate" else "🟢")
            print(f"    {icon} [{d.category.upper()}] {d.title} — Score: {d.score:.2f}")
            print(f"       {d.recommendation}")
        
        print(f"\n  Summary: {report.summary[:200]}...")


def demo_intent_parser():
    """Demonstrate the intent parser."""
    print_separator("MODULE 4: Question Intent Parser")
    
    parser = IntentParser()
    
    questions = [
        "How many buildings are in this scene?",
        "Is there any water in this scene?",
        "Are there any buildings near the road?",
        "Is this area overcrowded?",
        "Is there flood risk?",
        "Is this area suitable for residential expansion?",
        "What are the land use types in this scene?",
        "How many intersections are in this scene?",
        "Are there enough green spaces?",
        "Are buildings too close to water?",
        "What is the comprehensive traffic situation?",
        "What are the needs for the renovation of villages?",
    ]
    
    for q in questions:
        intent = parser.parse(q)
        earthvqa_type = parser.classify_earthvqa_type(q)
        print(f"\n  Q: \"{q}\"")
        print(f"     Intent: {intent.intent_type} | Targets: {intent.target_objects} | Relation: {intent.relation}")
        print(f"     EarthVQA type: {earthvqa_type}")


def demo_full_pipeline():
    """Demonstrate the full pipeline with question answering."""
    print_separator("FULL PIPELINE: Mask → Analysis → Question Answering")
    
    pipeline = SmartCityPipeline()
    
    scenarios = [
        ("Urban Dense Scene", create_urban_mask(), [
            "Is this area overcrowded?",
            "Are there enough green spaces?",
            "How many intersections are in this scene?",
            "Is this area suitable for residential expansion?",
        ]),
        ("Flood Risk Scene", create_flood_risk_mask(), [
            "Is there flood risk in this area?",
            "Are buildings too close to water?",
            "How is the road connectivity?",
        ]),
        ("Rural Scene", create_rural_mask(), [
            "Is there any water in this scene?",
            "What are the main land use types?",
            "Is this area suitable for development?",
        ]),
    ]
    
    for scene_name, mask, questions in scenarios:
        print(f"\n{'='*50}")
        print(f"  Scene: {scene_name}")
        print(f"{'='*50}")
        
        # Run analysis
        analysis = pipeline.analyze_mask(mask)
        print(f"\n  Scene type: {analysis.planning_report.scene_type}")
        print(f"  Overall: {analysis.planning_report.overall_suitability} ({analysis.planning_report.overall_score:.2f})")
        
        # Answer questions
        for q in questions:
            result = pipeline.answer_question(mask, q)
            print(f"\n  Q: \"{q}\"")
            print(f"  A: {result.answer[:200]}")
            print(f"     Intent: {result.intent.intent_type} | Confidence: {result.confidence:.2f}")
    
    # Save results for one example
    output_dir = os.path.join(PROJECT_ROOT, 'demo_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save urban analysis
    urban_mask = create_urban_mask()
    urban_result = pipeline.analyze_mask(urban_mask)
    pipeline.export_results_json(urban_result, os.path.join(output_dir, 'urban_analysis.json'))
    
    # Save colorized mask
    colorized = pipeline.spatial_engine.colorize_mask(urban_mask)
    Image.fromarray(colorized).save(os.path.join(output_dir, 'urban_mask_colorized.png'))
    
    # Save flood risk analysis
    flood_mask = create_flood_risk_mask()
    flood_result = pipeline.analyze_mask(flood_mask)
    pipeline.export_results_json(flood_result, os.path.join(output_dir, 'flood_risk_analysis.json'))
    
    colorized = pipeline.spatial_engine.colorize_mask(flood_mask)
    Image.fromarray(colorized).save(os.path.join(output_dir, 'flood_risk_mask_colorized.png'))
    
    # Save rural analysis
    rural_mask = create_rural_mask()
    rural_result = pipeline.analyze_mask(rural_mask)
    pipeline.export_results_json(rural_result, os.path.join(output_dir, 'rural_analysis.json'))
    
    colorized = pipeline.spatial_engine.colorize_mask(rural_mask)
    Image.fromarray(colorized).save(os.path.join(output_dir, 'rural_mask_colorized.png'))
    
    print(f"\n\n{'='*70}")
    print(f"  Demo outputs saved to: {output_dir}/")
    print(f"  - urban_analysis.json + urban_mask_colorized.png")
    print(f"  - flood_risk_analysis.json + flood_risk_mask_colorized.png")
    print(f"  - rural_analysis.json + rural_mask_colorized.png")
    print(f"{'='*70}")


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Smart City Planning Decision Support System — Demo           ║")
    print("║   Based on EarthVQA: Towards Queryable Earth                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    demo_spatial_engine()
    demo_decision_engine()
    demo_intent_parser()
    demo_full_pipeline()
    
    print("\n\n✅ Demo completed successfully!")
