"""
Unified Inference Pipeline (Module 5 extension)

Orchestrates the full flow:
    Image → Segmentation → Spatial Features → VQA → Decision Intelligence → Output

Supports CPU-only inference with automatic device detection.
Can work with or without the EarthVQA pre-trained models:
    - With models: full VQA + spatial analysis
    - Without models: spatial analysis only (from pre-computed or externally-provided masks)
"""

import os
import sys
import json
import numpy as np
from typing import Optional, Tuple, Dict
from PIL import Image
import yaml

from smart_city.models import (
    SpatialFeatures, AnalysisResult, VQAResult, QuestionIntent, PlanningReport
)
from smart_city.spatial_engine import SpatialFeatureEngine
from smart_city.decision_engine import DecisionEngine
from smart_city.intent_parser import IntentParser
from smart_city.change_detector import ChangeDetector, ChangeDetectionResult

# Conditional torch import (may not be needed for mask-only analysis)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. VQA model will not be available.")


def get_device():
    """Get the best available device (CPU or CUDA)."""
    if not TORCH_AVAILABLE:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


class SmartCityPipeline:
    """
    Main pipeline that ties all modules together.
    
    Usage:
        pipeline = SmartCityPipeline(config_dir='smart_city/config/')
        
        # Full analysis from image
        result = pipeline.analyze_image('path/to/satellite.png')
        
        # Answer a question
        vqa_result = pipeline.answer_question('path/to/satellite.png', 
                                              'Is this area overcrowded?')
        
        # Analysis from pre-computed mask only
        result = pipeline.analyze_mask(seg_mask_array)
    """

    def __init__(self,
                 config_dir: Optional[str] = None,
                 seg_weights_path: Optional[str] = None,
                 vqa_weights_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Args:
            config_dir: Directory containing thresholds.yaml and class_config.yaml
            seg_weights_path: Path to segmentation model weights (sfpnr50.pth)
            vqa_weights_path: Path to VQA model weights (soba.pth)
            device: 'cpu' or 'cuda'. Auto-detected if None.
        """
        self.device = device or get_device()
        
        # Resolve config paths
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
        
        thresholds_path = os.path.join(config_dir, 'thresholds.yaml')
        
        # Initialize engines (these don't need GPU)
        self.spatial_engine = SpatialFeatureEngine(thresholds_path)
        self.decision_engine = DecisionEngine(thresholds_path)
        self.intent_parser = IntentParser()
        
        # ML models (lazy-loaded)
        self.seg_model = None
        self.vqa_model = None
        self.seg_weights_path = seg_weights_path
        self.vqa_weights_path = vqa_weights_path
        
        # Change detector (lazy-loaded)
        self.change_detector = None
        
        # Model loading status
        self._seg_loaded = False
        self._vqa_loaded = False
        
        print(f"[SmartCityPipeline] Initialized on device: {self.device}")
        print(f"[SmartCityPipeline] Config dir: {config_dir}")

    # ─── Model Loading ────────────────────────────────────────────────

    def _load_seg_model(self):
        """Lazy-load the segmentation model."""
        if self._seg_loaded or not TORCH_AVAILABLE:
            return
        
        if not self.seg_weights_path or not os.path.exists(self.seg_weights_path):
            print(f"[WARNING] Segmentation weights not found at: {self.seg_weights_path}")
            print("[WARNING] Segmentation model will not be available. Use analyze_mask() with pre-computed masks.")
            return
        
        try:
            # Add EarthVQA to path for imports
            earthvqa_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EarthVQA')
            if earthvqa_dir not in sys.path:
                sys.path.insert(0, earthvqa_dir)
            
            import ever as er
            from ever.core.builder import make_model
            from ever.core.config import import_config
            er.registry.register_all()
            
            original_cwd = os.getcwd()
            os.chdir(earthvqa_dir)
            
            try:
                import importlib.util
                sfpn_path = os.path.join(earthvqa_dir, 'module', 'semantic-fpn.py')
                spec = importlib.util.spec_from_file_location("semantic_fpn", sfpn_path)
                sfpn_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sfpn_module)
                
                cfg = import_config('sfpnr50')
                model_state_dict = torch.load(
                    self.seg_weights_path,
                    map_location=torch.device(self.device)
                )
                model = make_model(cfg['model'])
                model.load_state_dict(model_state_dict)
                model.to(self.device)
                model.eval()
                
                self.seg_model = model
                self._seg_loaded = True
                print("[SmartCityPipeline] Segmentation model loaded successfully.")
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"[ERROR] Failed to load segmentation model: {e}")
            self.seg_model = None

    def _load_vqa_model(self):
        """Lazy-load the VQA model."""
        if self._vqa_loaded or not TORCH_AVAILABLE:
            return
        
        if not self.vqa_weights_path or not os.path.exists(self.vqa_weights_path):
            print(f"[WARNING] VQA weights not found at: {self.vqa_weights_path}")
            return
        
        try:
            earthvqa_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EarthVQA')
            if earthvqa_dir not in sys.path:
                sys.path.insert(0, earthvqa_dir)
            
            import ever as er
            from ever.core.builder import make_model
            from ever.core.config import import_config
            er.registry.register_all()
            
            original_cwd = os.getcwd()
            os.chdir(earthvqa_dir)
            
            try:
                from module.soba import SOBA
                
                cfg = import_config('soba')
                model_state_dict = torch.load(
                    self.vqa_weights_path,
                    map_location=torch.device(self.device)
                )
                model = make_model(cfg['model'])
                model.load_state_dict(model_state_dict)
                model.to(self.device)
                model.eval()
                
                self.vqa_model = model
                self._vqa_loaded = True
                print("[SmartCityPipeline] VQA model loaded successfully.")
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"[ERROR] Failed to load VQA model: {e}")
            self.vqa_model = None

    # ─── Core Analysis Methods ────────────────────────────────────────

    def analyze_mask(self, seg_mask: np.ndarray) -> AnalysisResult:
        """
        Analyze a pre-computed segmentation mask.
        This is the core analysis method that doesn't require GPU.
        
        Args:
            seg_mask: H x W numpy array with class IDs (0-7 or 1-8).
        
        Returns:
            AnalysisResult with spatial features and planning report.
        """
        # Extract spatial features
        spatial_features = self.spatial_engine.extract(seg_mask)
        
        # Generate planning decisions
        planning_report = self.decision_engine.evaluate(spatial_features)
        
        # Colorize mask for visualization
        colorized = self.spatial_engine.colorize_mask(seg_mask)
        
        return AnalysisResult(
            spatial_features=spatial_features,
            planning_report=planning_report,
            seg_mask=seg_mask,
            colorized_mask=colorized,
        )

    def analyze_image(self, image_path: str, scale_factor: float = 1.0) -> AnalysisResult:
        """
        Full pipeline: Image → Segmentation → Spatial Analysis → Decisions.
        
        Requires segmentation model weights.
        
        Args:
            image_path: Path to satellite image file.
            scale_factor: Artificial upscaling factor before inference to fix domain scale mismatches.
        
        Returns:
            AnalysisResult with spatial features and planning report.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for image analysis. "
                             "Use analyze_mask() with pre-computed masks instead.")
        
        self._load_seg_model()
        
        if self.seg_model is None:
            raise RuntimeError("Segmentation model not loaded. "
                             "Provide seg_weights_path or use analyze_mask().")
        
        # Preprocess image (gets tensor of shape [1, 3, H, W])
        image_tensor = self._preprocess_image(image_path)
        
        # Handle Scaling for domain shift correction
        orig_h, orig_w = image_tensor.shape[2], image_tensor.shape[3]
        if scale_factor != 1.0:
            import torch.nn.functional as F
            new_h = int(orig_h * scale_factor)
            new_w = int(orig_w * scale_factor)
            
            # SemanticFPN uses ResNet50, which has a 32x downsampling backbone.
            new_h = int(np.ceil(new_h / 32.0)) * 32
            new_w = int(np.ceil(new_w / 32.0)) * 32
            
            image_tensor = F.interpolate(image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
        # Run segmentation
        with torch.no_grad():
            pred, features = self.seg_model(image_tensor.to(self.device))
            seg_mask_tensor = pred.argmax(dim=1).cpu().float().unsqueeze(0)  # [1, 1, H_scaled, W_scaled]
            
            # Revert scale mathematically for perfectly aligned UI
            if scale_factor != 1.0:
                import torch.nn.functional as F
                seg_mask_tensor = F.interpolate(seg_mask_tensor, size=(orig_h, orig_w), mode='nearest')
                
            seg_mask = seg_mask_tensor.squeeze().numpy().astype(np.int32)  # H x W
        
        # Run spatial analysis on the mask
        return self.analyze_mask(seg_mask)

    def _post_process_mask(self, mask: np.ndarray, original_img: np.ndarray, probs: np.ndarray = None) -> np.ndarray:
        """
        Apply advanced spectral and structural priors. 
        """
        try:
            from skimage.morphology import remove_small_objects, binary_closing, disk
        except ImportError:
            return mask

        processed_mask = mask.copy()
        
        # Bridge road segments
        road_mask = (processed_mask == 2)
        if road_mask.any():
            road_mask = binary_closing(road_mask, disk(4))
            road_mask = remove_small_objects(road_mask, min_size=150)
            processed_mask[processed_mask == 2] = 0
            processed_mask[road_mask] = 2
            
        # Smooth buildings
        building_mask = (processed_mask == 1)
        if building_mask.any():
            building_mask = binary_closing(building_mask, disk(3))
            building_mask = remove_small_objects(building_mask, min_size=50)
            processed_mask[processed_mask == 1] = 0
            processed_mask[building_mask] = 1

        # Global noise removal
        for class_id in range(1, 8):
            c_mask = (processed_mask == class_id)
            if c_mask.any():
                clean_c = remove_small_objects(c_mask, min_size=50)
                processed_mask[c_mask & (~clean_c)] = 0

        return processed_mask

    def answer_question(self, image_path_or_mask, question: str) -> VQAResult:
        """
        Answer a natural language question about an image.
        
        Uses the VQA model if available, otherwise uses spatial analysis
        + decision engine to generate a rule-based answer.
        
        Args:
            image_path_or_mask: Path to image (str) or pre-computed mask (np.ndarray)
            question: Natural language question.
        
        Returns:
            VQAResult with answer, explanation, and spatial analysis.
        """
        # Parse question intent
        intent = self.intent_parser.parse(question)
        
        # Get spatial analysis
        if isinstance(image_path_or_mask, str):
            analysis = self.analyze_image(image_path_or_mask)
        else:
            analysis = self.analyze_mask(image_path_or_mask)
        
        # Try VQA model first
        vqa_answer = None
        confidence = 0.0
        
        if self.vqa_model is not None and isinstance(image_path_or_mask, str):
            try:
                vqa_answer, confidence = self._run_vqa(image_path_or_mask, question)
            except Exception as e:
                print(f"[WARNING] VQA model inference failed: {e}")
        
        # If VQA model not available, generate answer from spatial analysis
        if vqa_answer is None:
            vqa_answer = self._generate_rule_based_answer(intent, analysis)
            confidence = 0.7  # lower confidence for rule-based answers
        
        # Enrich answer with decision intelligence
        explanation = self.decision_engine.enrich_answer(
            vqa_answer, analysis, intent
        )
        
        return VQAResult(
            answer=vqa_answer,
            confidence=confidence,
            explanation=explanation,
            intent=intent,
            analysis=analysis,
        )

    # ─── Change Detection Methods ─────────────────────────────────────

    def _ensure_change_detector(self):
        """Lazy-initialize the ChangeDetector."""
        if self.change_detector is None:
            self.change_detector = ChangeDetector()

    def analyze_change(
        self,
        image_path_before: str,
        image_path_after: str,
    ) -> ChangeDetectionResult:
        """
        Compare two temporal satellite images and classify urban sprawl.
        """
        self._ensure_change_detector()
        result_before = self.analyze_image(image_path_before)
        result_after = self.analyze_image(image_path_after)
        return self.change_detector.analyze(
            result_before.spatial_features,
            result_after.spatial_features,
        )

    def analyze_change_masks(
        self,
        mask_before: np.ndarray,
        mask_after: np.ndarray,
    ) -> ChangeDetectionResult:
        """
        Compare two pre-computed segmentation masks and classify urban sprawl.
        """
        self._ensure_change_detector()
        result_before = self.analyze_mask(mask_before)
        result_after = self.analyze_mask(mask_after)
        return self.change_detector.analyze(
            result_before.spatial_features,
            result_after.spatial_features,
        )

    # ─── Rule-Based Answer Generation ─────────────────────────────────

    def _generate_rule_based_answer(self, intent: QuestionIntent,
                                     analysis: AnalysisResult) -> str:
        """
        Generate an answer using spatial features when VQA model is unavailable.
        """
        features = analysis.spatial_features
        report = analysis.planning_report
        
        if intent.intent_type == 'counting':
            return self._answer_counting(intent, features)
        elif intent.intent_type == 'judging':
            return self._answer_judging(intent, features)
        elif intent.intent_type == 'density':
            return self._answer_density(features, report)
        elif intent.intent_type == 'risk':
            return self._answer_risk(features, report)
        elif intent.intent_type == 'planning':
            return self._answer_planning(features, report)
        elif intent.intent_type == 'relation':
            return self._answer_relation(intent, features)
        elif intent.intent_type == 'situation':
            return self._answer_situation(intent, features, report)
        else:
            return report.summary

    def _answer_counting(self, intent: QuestionIntent, features: SpatialFeatures) -> str:
        """Answer counting questions."""
        for target in intent.target_objects:
            if target == 'building':
                return f"There are {features.building_count} buildings in this scene."
            elif target == 'water':
                return f"There are {features.water_body_count} water bodies in this scene."
            elif target == 'road':
                area_pct = features.road_area_pct * 100
                return f"Road coverage is {area_pct:.1f}% with {features.road_segment_count} segments and {features.intersection_count} intersections."
        
        # Generic: report non-zero classes
        parts = []
        for name, stats in features.class_stats.items():
            if stats.component_count > 0 and name != 'background':
                parts.append(f"{stats.component_count} {name}(s)")
        return "Detected: " + ", ".join(parts) if parts else "No significant objects detected."

    def _answer_judging(self, intent: QuestionIntent, features: SpatialFeatures) -> str:
        """Answer yes/no judging questions."""
        for target in intent.target_objects:
            stats = features.class_stats.get(target)
            if stats:
                if stats.pixel_count > 0:
                    return f"Yes, there {'is' if stats.component_count == 1 else 'are'} {target} in this scene ({stats.area_percentage*100:.1f}% coverage)."
                else:
                    return f"No, there is no {target} detected in this scene."
        return "Unable to determine from the available analysis."

    def _answer_density(self, features: SpatialFeatures, report: PlanningReport) -> str:
        """Answer density/overcrowding questions."""
        density_dec = next((d for d in report.decisions if d.category == 'density'), None)
        if density_dec:
            return f"{density_dec.description} {density_dec.recommendation}"
        return f"Building density: {features.building_density:.2f}, {features.building_count} buildings covering {features.building_area_pct*100:.1f}%."

    def _answer_risk(self, features: SpatialFeatures, report: PlanningReport) -> str:
        """Answer risk-related questions."""
        flood_dec = next((d for d in report.decisions if d.category == 'flood_risk'), None)
        if flood_dec:
            return f"{flood_dec.description} {flood_dec.recommendation}"
        return "No significant risk factors detected."

    def _answer_planning(self, features: SpatialFeatures, report: PlanningReport) -> str:
        """Answer planning/suitability questions."""
        return f"Overall assessment: {report.overall_suitability} (score: {report.overall_score:.2f}). {report.summary}"

    def _answer_relation(self, intent: QuestionIntent, features: SpatialFeatures) -> str:
        """Answer spatial relation questions."""
        parts = []
        for target in intent.target_objects:
            for d in features.distances:
                if target in (d.class_a, d.class_b):
                    if d.min_distance_px != float('inf'):
                        parts.append(f"{d.class_a} to {d.class_b}: {d.min_distance_px:.0f}px apart")
        if parts:
            return "Spatial relationships: " + "; ".join(parts)
        return "No significant spatial relationships found for the queried objects."

    def _answer_situation(self, intent: QuestionIntent, features: SpatialFeatures, report: PlanningReport) -> str:
        """Answer situation/status questions."""
        parts = []
        for target in intent.target_objects:
            stats = features.class_stats.get(target)
            if stats and stats.pixel_count > 0:
                parts.append(f"{target}: {stats.component_count} component(s), {stats.area_percentage*100:.1f}% coverage")
        if parts:
            return "Status: " + "; ".join(parts)
        return report.summary

    # ─── Image Preprocessing ──────────────────────────────────────────

    def _preprocess_image(self, image_path: str):
        """Preprocess an image for segmentation model input."""
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32)
        
        # Standard Normalization (LoveDA constants)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_np = (img_np - mean) / std
        
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
        return img_tensor

    def _run_vqa(self, image_path: str, question: str):
        """Run the SOBA VQA model on an image and question."""
        # This requires the full EarthVQA inference pipeline
        # For now, we return None and fall back to rule-based
        # TODO: Implement full VQA inference when models are available
        return None, 0.0

    # ─── Utility Methods ──────────────────────────────────────────────

    def get_supported_questions(self) -> list:
        """Return list of supported EarthVQA question templates."""
        try:
            earthvqa_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EarthVQA')
            if earthvqa_dir not in sys.path:
                sys.path.insert(0, earthvqa_dir)
            from data.earthvqa import EarthVQADataset
            return EarthVQADataset.QUESTIONS
        except ImportError:
            return [
                "Are there any buildings in this scene?",
                "Is there any water in this scene?",
                "How many intersections are in this scene?",
                "Is this area overcrowded?",
                "Is there flood risk?",
                "Is this area suitable for residential expansion?",
                "What are the land use types in this scene?",
            ]

    def export_results_json(self, result, output_path: str):
        """Export analysis results to a JSON file."""
        data = result.to_dict() if hasattr(result, 'to_dict') else str(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[SmartCityPipeline] Results exported to: {output_path}")
