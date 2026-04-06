#!/usr/bin/env python3
"""
Phase A — Local Pipeline Test with Pretrained Weights (CPU)

Tests the full pipeline end-to-end:
    1. Load sfpnr50.pth segmentation model on CPU
    2. Run inference on a synthetic 512x512 satellite-like image
    3. Feed predicted mask into spatial engine + decision engine
    4. Generate planning report
    5. Save outputs to test_output/

No dataset required — uses a synthetic image to validate the models load correctly.
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image

# Add project root and EarthVQA to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EARTHVQA_DIR = os.path.join(PROJECT_ROOT, 'EarthVQA')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EARTHVQA_DIR)

# Paths
SEG_WEIGHTS = os.path.join(PROJECT_ROOT, 'pretrained_weights', 'sfpnr50.pth')
VQA_WEIGHTS = os.path.join(PROJECT_ROOT, 'pretrained_weights', 'soba.pth')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'test_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_test_image(size=512):
    """
    Create a synthetic satellite-like RGB image.
    Uses patterns that vaguely resemble land cover types
    so the segmentation model produces non-trivial predictions.
    """
    np.random.seed(42)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Green background (vegetation-like)
    img[:, :, 0] = 60 + np.random.randint(0, 30, (size, size)).astype(np.uint8)
    img[:, :, 1] = 120 + np.random.randint(0, 40, (size, size)).astype(np.uint8)
    img[:, :, 2] = 40 + np.random.randint(0, 20, (size, size)).astype(np.uint8)
    
    # Brown/grey patches (buildings)
    for y, x, h, w in [(50, 50, 60, 80), (180, 300, 50, 70), (350, 100, 45, 55),
                         (100, 200, 35, 45), (300, 350, 65, 50)]:
        img[y:y+h, x:x+w, 0] = 160 + np.random.randint(0, 30, (h, w)).astype(np.uint8)
        img[y:y+h, x:x+w, 1] = 140 + np.random.randint(0, 20, (h, w)).astype(np.uint8)
        img[y:y+h, x:x+w, 2] = 130 + np.random.randint(0, 20, (h, w)).astype(np.uint8)
    
    # Grey line (road)
    img[250:258, 30:480, :] = [180, 180, 180]
    img[100:400, 260:268, :] = [170, 170, 170]
    
    # Blue patch (water)
    img[380:450, 250:380, 0] = 30
    img[380:450, 250:380, 1] = 80
    img[380:450, 250:380, 2] = 180
    
    return img


def test_segmentation_model():
    """Test 1: Load segmentation model and run inference."""
    print("\n" + "=" * 60)
    print("  TEST 1: Segmentation Model (sfpnr50.pth)")
    print("=" * 60)
    
    import torch
    
    # Check weights exist
    if not os.path.exists(SEG_WEIGHTS):
        print(f"  ❌ Weights not found at: {SEG_WEIGHTS}")
        return None, None
    
    print(f"  Weights: {SEG_WEIGHTS} ({os.path.getsize(SEG_WEIGHTS) / 1e6:.1f} MB)")
    print(f"  Device: cpu")
    
    # Load model — EarthVQA needs to run from its own directory
    print("  Loading model...")
    t0 = time.time()
    
    original_cwd = os.getcwd()
    
    try:
        # Must chdir AND add to sys.path for EarthVQA imports to work
        os.chdir(EARTHVQA_DIR)
        if EARTHVQA_DIR not in sys.path:
            sys.path.insert(0, EARTHVQA_DIR)
        
        import ever as er
        from ever.core.builder import make_model
        from ever.core.config import import_config
        er.registry.register_all()
        
        # Explicitly import the model modules (they register via decorators)
        # semantic-fpn.py has a hyphen so we use importlib
        import importlib.util
        sfpn_path = os.path.join(EARTHVQA_DIR, 'module', 'semantic-fpn.py')
        spec = importlib.util.spec_from_file_location("semantic_fpn", sfpn_path)
        sfpn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sfpn_module)
        
        # Also import SOBA module (for later)
        from module.soba import SOBA  # noqa: F401 — registers via decorator
        
        cfg = import_config('sfpnr50')
        model_state_dict = torch.load(SEG_WEIGHTS, map_location='cpu')
        model = make_model(cfg['model'])
        model.load_state_dict(model_state_dict)
        model.eval()
        
        load_time = time.time() - t0
        print(f"  ✅ Model loaded in {load_time:.1f}s")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params / 1e6:.1f}M")
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)
        return None, None
    finally:
        os.chdir(original_cwd)
    
    # Create test image
    print("\n  Creating synthetic test image...")
    test_img = create_test_image(512)
    Image.fromarray(test_img).save(os.path.join(OUTPUT_DIR, 'synthetic_input.png'))
    print(f"  Saved: test_output/synthetic_input.png")
    
    # Preprocess
    img_np = test_img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_np = (img_np - mean) / std
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    
    # Run inference
    print("  Running inference on CPU (may take ~30s)...")
    t0 = time.time()
    
    with torch.no_grad():
        pred, img_feat = model(img_tensor)
    
    infer_time = time.time() - t0
    print(f"  ✅ Inference done in {infer_time:.1f}s")
    
    # Process predictions
    seg_mask = pred.argmax(dim=1).cpu().numpy()[0]  # (H, W) — 0-indexed
    img_features = img_feat.cpu().numpy()[0]  # (2048, H/32, W/32)
    
    unique_classes = np.unique(seg_mask)
    class_names = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture', 'playground']
    
    print(f"\n  Prediction shape: {seg_mask.shape}")
    print(f"  Feature shape: {img_features.shape}")
    print(f"  Unique classes predicted: {unique_classes}")
    print(f"  Class distribution:")
    for cls_id in unique_classes:
        count = (seg_mask == cls_id).sum()
        pct = count / seg_mask.size * 100
        name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        print(f"    {cls_id} ({name}): {count:,} pixels ({pct:.1f}%)")
    
    return seg_mask, img_features


def test_spatial_pipeline(seg_mask):
    """Test 2: Run spatial engine + decision engine on the predicted mask."""
    print("\n" + "=" * 60)
    print("  TEST 2: Spatial Engine + Decision Engine")
    print("=" * 60)
    
    from smart_city.spatial_engine import SpatialFeatureEngine
    from smart_city.decision_engine import DecisionEngine
    
    engine = SpatialFeatureEngine()
    decision = DecisionEngine()
    
    # Extract features
    print("  Extracting spatial features...")
    t0 = time.time()
    features = engine.extract(seg_mask)
    feat_time = time.time() - t0
    print(f"  ✅ Features extracted in {feat_time:.3f}s")
    
    # Print key metrics
    print(f"\n  Key Metrics:")
    print(f"    Buildings: {features.building_count} ({features.building_area_pct*100:.1f}%)")
    print(f"    Roads: {features.road_segment_count} segments ({features.road_area_pct*100:.1f}%)")
    print(f"    Water: {features.water_body_count} bodies ({features.water_area_pct*100:.1f}%)")
    print(f"    Vegetation: {features.vegetation_area_pct*100:.1f}%")
    print(f"    Intersections: {features.intersection_count}")
    print(f"    Building-Water dist: {features.building_water_min_distance:.0f}px")
    
    # Generate planning report
    print("\n  Generating planning report...")
    report = decision.evaluate(features)
    
    print(f"\n  Scene type: {report.scene_type}")
    print(f"  Overall suitability: {report.overall_suitability} ({report.overall_score:.2f})")
    print(f"\n  Decisions:")
    for d in report.decisions:
        icon = "🔴" if d.severity == "high" else ("🟡" if d.severity == "moderate" else "🟢")
        print(f"    {icon} [{d.category.upper()}] {d.title} — Score: {d.score:.2f}")
        print(f"       {d.recommendation}")
    
    # Colorize mask
    colorized = engine.colorize_mask(seg_mask)
    Image.fromarray(colorized).save(os.path.join(OUTPUT_DIR, 'predicted_mask_colorized.png'))
    print(f"\n  Saved: test_output/predicted_mask_colorized.png")
    
    return features, report


def test_full_pipeline(seg_mask):
    """Test 3: Run the unified pipeline with question answering."""
    print("\n" + "=" * 60)
    print("  TEST 3: Full Pipeline — Question Answering")
    print("=" * 60)
    
    from smart_city.pipeline import SmartCityPipeline
    
    pipeline = SmartCityPipeline(
        seg_weights_path=SEG_WEIGHTS,
        vqa_weights_path=VQA_WEIGHTS,
    )
    
    # Analyze the mask
    result = pipeline.analyze_mask(seg_mask)
    
    # Test questions
    questions = [
        "Is this area overcrowded?",
        "Are there enough green spaces?",
        "Is there flood risk in this area?",
        "How is the road connectivity?",
        "Is this area suitable for residential expansion?",
        "How many buildings are in this scene?",
    ]
    
    print(f"\n  Answering questions...\n")
    for q in questions:
        vqa_result = pipeline.answer_question(seg_mask, q)
        print(f"  Q: \"{q}\"")
        print(f"  A: {vqa_result.answer[:150]}")
        print(f"     Intent: {vqa_result.intent.intent_type} | Confidence: {vqa_result.confidence:.2f}")
        print()
    
    # Export results
    pipeline.export_results_json(result, os.path.join(OUTPUT_DIR, 'pipeline_analysis.json'))
    print(f"  Saved: test_output/pipeline_analysis.json")
    
    return result


def test_vqa_model_load():
    """Test 4: Verify SOBA VQA model loads (but don't run inference without features)."""
    print("\n" + "=" * 60)
    print("  TEST 4: SOBA VQA Model Load Check")
    print("=" * 60)
    
    import torch
    
    if not os.path.exists(VQA_WEIGHTS):
        print(f"  ❌ VQA weights not found at: {VQA_WEIGHTS}")
        return False
    
    print(f"  Weights: {VQA_WEIGHTS} ({os.path.getsize(VQA_WEIGHTS) / 1e6:.1f} MB)")
    
    try:
        import ever as er
        from ever.core.builder import make_model
        from ever.core.config import import_config
        
        # chdir to EarthVQA for config resolution
        original_cwd = os.getcwd()
        os.chdir(EARTHVQA_DIR)
        if EARTHVQA_DIR not in sys.path:
            sys.path.insert(0, EARTHVQA_DIR)
        
        try:
            cfg = import_config('soba')
        finally:
            os.chdir(original_cwd)
        
        model_state_dict = torch.load(VQA_WEIGHTS, map_location='cpu')
        model = make_model(cfg['model'])
        model.load_state_dict(model_state_dict)
        model.eval()
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ SOBA model loaded successfully ({n_params / 1e6:.1f}M params)")
        print(f"  ⚠️  Full VQA inference requires HDF5 feature files from Step 1")
        print(f"     (This will be done on Kaggle)")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load SOBA model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase A — Local Pipeline Test (CPU)                       ║")
    print("║  Testing pretrained weights + smart city pipeline           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Test 1: Segmentation model
    seg_mask, img_features = test_segmentation_model()
    
    if seg_mask is not None:
        # Test 2: Spatial + Decision engines
        features, report = test_spatial_pipeline(seg_mask)
        
        # Test 3: Full pipeline with Q&A
        result = test_full_pipeline(seg_mask)
    else:
        print("\n  ⚠️  Skipping pipeline tests (segmentation model not loaded)")
    
    # Test 4: VQA model load check
    vqa_ok = test_vqa_model_load()
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Segmentation model: {'✅ Working' if seg_mask is not None else '❌ Failed'}")
    print(f"  Spatial engine:     {'✅ Working' if seg_mask is not None else '⏭ Skipped'}")
    print(f"  Decision engine:    {'✅ Working' if seg_mask is not None else '⏭ Skipped'}")
    print(f"  Pipeline + Q&A:     {'✅ Working' if seg_mask is not None else '⏭ Skipped'}")
    print(f"  SOBA VQA model:     {'✅ Loads OK' if vqa_ok else '❌ Failed'}")
    print(f"\n  Outputs: {OUTPUT_DIR}/")
    print("=" * 60)
