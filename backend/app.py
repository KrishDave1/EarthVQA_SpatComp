"""
Flask REST API Backend (Module 7 - Backend)

Exposes the Smart City Planning pipeline as a REST API.
Supports:
    - POST /api/analyze     — Upload image or mask → spatial analysis + decisions
    - POST /api/ask         — Upload image/mask + question → VQA answer + explanation
    - POST /api/analyze/mask — Upload a pre-computed mask file for analysis
    - GET  /api/health      — Health check
    - GET  /api/questions   — Get list of supported question templates

All endpoints return JSON. Image/mask uploads use multipart/form-data.
"""

import os
import sys
import json
import base64
import tempfile
import traceback
import numpy as np
from io import BytesIO
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from smart_city.pipeline import SmartCityPipeline
from smart_city.spatial_engine import SpatialFeatureEngine
from smart_city.rgb_change_detector import RGBChangeDetector, colorize_rgb_mask

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ─── Configuration ────────────────────────────────────────────────────

CONFIG_DIR = os.path.join(PROJECT_ROOT, 'smart_city', 'config')
SEG_WEIGHTS = os.environ.get('SEG_WEIGHTS', os.path.join(PROJECT_ROOT, 'pretrained_weights', 'sfpnr50.pth'))
VQA_WEIGHTS = os.environ.get('VQA_WEIGHTS', os.path.join(PROJECT_ROOT, 'pretrained_weights', 'soba.pth'))

# Upload settings
UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'backend', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global pipeline instance (lazy-initialized)
_pipeline = None
_rgb_change_detector = None


def get_pipeline() -> SmartCityPipeline:
    """Lazy-initialize the pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SmartCityPipeline(
            config_dir=CONFIG_DIR,
            seg_weights_path=SEG_WEIGHTS if os.path.exists(SEG_WEIGHTS) else None,
            vqa_weights_path=VQA_WEIGHTS if os.path.exists(VQA_WEIGHTS) else None,
        )
    return _pipeline


def get_rgb_change_detector() -> RGBChangeDetector:
    """Lazy-initialize the RGB-based change detector."""
    global _rgb_change_detector
    if _rgb_change_detector is None:
        _rgb_change_detector = RGBChangeDetector()
    return _rgb_change_detector


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Convert numpy array (image) to base64-encoded PNG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_mask_from_file(file_storage) -> np.ndarray:
    """Load a segmentation mask from an uploaded file."""
    img = Image.open(file_storage)
    mask = np.array(img)
    # If RGB, take first channel (masks are single-channel)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int32)


# ─── API Endpoints ────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return jsonify({
        'status': 'ok',
        'device': pipeline.device,
        'seg_model_loaded': pipeline._seg_loaded,
        'vqa_model_loaded': pipeline._vqa_loaded,
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze a satellite image or pre-computed mask.
    
    Accepts multipart/form-data with:
        - 'image': image file (PNG/JPG) — satellite image for segmentation
        - OR 'mask': mask file (PNG) — pre-computed segmentation mask
    
    Returns JSON with spatial features and planning report.
    """
    pipeline = get_pipeline()
    
    try:
        if 'mask' in request.files:
            # Analyze pre-computed mask
            mask_file = request.files['mask']
            mask = load_mask_from_file(mask_file)
            result = pipeline.analyze_mask(mask)
            
        elif 'image' in request.files:
            # Analyze from image (requires segmentation model)
            image_file = request.files['image']
            # Save temporarily
            temp_path = os.path.join(UPLOAD_DIR, 'temp_image.png')
            image_file.save(temp_path)
            
            try:
                result = pipeline.analyze_image(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            return jsonify({'error': 'No image or mask file provided. Send as "image" or "mask" in multipart form.'}), 400
        
        # Build response
        response = result.to_dict()
        
        # Add colorized mask as base64 PNG
        if result.colorized_mask is not None:
            response['colorized_mask_base64'] = numpy_to_base64_png(result.colorized_mask)
        
        return jsonify(response)
    
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/ask', methods=['POST'])
def ask():
    """
    Answer a question about an image/mask.
    
    Accepts multipart/form-data with:
        - 'image' or 'mask': image/mask file
        - 'question': text question (form field)
    
    OR JSON body with:
        - 'mask_base64': base64-encoded mask PNG
        - 'question': text question
    
    Returns JSON with VQA answer, explanation, and analysis.
    """
    pipeline = get_pipeline()
    
    try:
        question = None
        image_or_mask = None
        
        if request.is_json:
            # JSON request
            data = request.get_json()
            question = data.get('question')
            if 'mask_base64' in data:
                mask_bytes = base64.b64decode(data['mask_base64'])
                img = Image.open(BytesIO(mask_bytes))
                image_or_mask = np.array(img)
                if image_or_mask.ndim == 3:
                    image_or_mask = image_or_mask[:, :, 0]
                image_or_mask = image_or_mask.astype(np.int32)
        else:
            # Multipart form
            question = request.form.get('question')
            
            if 'mask' in request.files:
                image_or_mask = load_mask_from_file(request.files['mask'])
            elif 'image' in request.files:
                image_file = request.files['image']
                temp_path = os.path.join(UPLOAD_DIR, 'temp_question_image.png')
                image_file.save(temp_path)
                image_or_mask = temp_path  # Pass path for image
        
        if question is None:
            return jsonify({'error': 'No question provided.'}), 400
        
        if image_or_mask is None:
            return jsonify({'error': 'No image or mask provided.'}), 400
        
        # Run VQA pipeline
        result = pipeline.answer_question(image_or_mask, question)
        
        # Clean up temp file if used
        if isinstance(image_or_mask, str) and os.path.exists(image_or_mask):
            os.remove(image_or_mask)
        
        # Build response
        response = result.to_dict()
        
        # Add colorized mask
        if result.analysis and result.analysis.colorized_mask is not None:
            response['colorized_mask_base64'] = numpy_to_base64_png(result.analysis.colorized_mask)
        
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Question answering failed: {str(e)}'}), 500


@app.route('/api/questions', methods=['GET'])
def questions():
    """Get list of supported question templates."""
    pipeline = get_pipeline()
    return jsonify({
        'questions': pipeline.get_supported_questions(),
        'custom_questions': [
            "Is this area overcrowded?",
            "Is there flood risk in this area?",
            "Is this area suitable for residential expansion?",
            "Are there enough green spaces?",
            "How is the road connectivity?",
            "What are the main land use types?",
            "Are buildings too close to water?",
        ]
    })


@app.route('/api/visualize/mask', methods=['POST'])
def visualize_mask():
    """
    Colorize a segmentation mask.
    
    Accepts: 'mask' file in multipart form
    Returns: base64-encoded RGB PNG
    """
    if 'mask' not in request.files:
        return jsonify({'error': 'No mask file provided.'}), 400
    
    try:
        mask = load_mask_from_file(request.files['mask'])
        engine = SpatialFeatureEngine()
        colorized = engine.colorize_mask(mask)
        
        return jsonify({
            'colorized_mask_base64': numpy_to_base64_png(colorized),
            'width': int(mask.shape[1]),
            'height': int(mask.shape[0]),
        })
    except Exception as e:
        return jsonify({'error': f'Visualization failed: {str(e)}'}), 500


@app.route('/api/change-detect', methods=['POST'])
def change_detect():
    """
    Detect temporal changes between two satellite images.
    
    Uses RGB-based direct pixel analysis instead of the SemanticFPN model
    (which was trained on LoveDA and fails on LEVIR-CD domain images).
    
    Accepts multipart/form-data with:
        - 'image_before' + 'image_after': satellite image files (PNG/JPG)
        - OR 'mask_before' + 'mask_after': pre-computed segmentation masks
    
    Returns JSON with sprawl classification, delta features, and colorized masks.
    """
    pipeline = get_pipeline()
    
    try:
        # Handle mask-based input (use existing pipeline for pre-computed masks)
        if 'mask_before' in request.files and 'mask_after' in request.files:
            mask_before = load_mask_from_file(request.files['mask_before'])
            mask_after = load_mask_from_file(request.files['mask_after'])
            
            result_before = pipeline.analyze_mask(mask_before)
            result_after = pipeline.analyze_mask(mask_after)
            
            pipeline._ensure_change_detector()
            change_result = pipeline.change_detector.analyze(
                result_before.spatial_features,
                result_after.spatial_features,
            )
            
            response = change_result.to_dict()
            if result_before.colorized_mask is not None:
                response['colorized_mask_before_base64'] = numpy_to_base64_png(result_before.colorized_mask)
            if result_after.colorized_mask is not None:
                response['colorized_mask_after_base64'] = numpy_to_base64_png(result_after.colorized_mask)
            
            return jsonify(response)
        
        # Handle image-based input → use RGB-based direct analysis
        elif 'image_before' in request.files and 'image_after' in request.files:
            temp_before = os.path.join(UPLOAD_DIR, 'temp_before.png')
            temp_after = os.path.join(UPLOAD_DIR, 'temp_after.png')
            
            request.files['image_before'].save(temp_before)
            request.files['image_after'].save(temp_after)
            
            try:
                # Use the RGB-based change detector (NO segmentation model needed)
                rgb_detector = get_rgb_change_detector()
                change_result, color_before, color_after, feat_before, feat_after = \
                    rgb_detector.analyze(temp_before, temp_after)
                
                # Build response
                response = change_result.to_dict()
                response['colorized_mask_before_base64'] = numpy_to_base64_png(color_before)
                response['colorized_mask_after_base64'] = numpy_to_base64_png(color_after)
                response['classifier_type'] = 'rgb-spectral'
                
                return jsonify(response)
                
            finally:
                for p in [temp_before, temp_after]:
                    if os.path.exists(p):
                        os.remove(p)
        else:
            return jsonify({
                'error': 'Provide either (image_before + image_after) or '
                         '(mask_before + mask_after) as multipart files.'
            }), 400
    
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Change detection failed: {str(e)}'}), 500


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Smart City Planning API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Smart City Planning Decision Support System — API Server")
    print("=" * 60)
    print(f"  Config dir: {CONFIG_DIR}")
    print(f"  Seg weights: {SEG_WEIGHTS}")
    print(f"  VQA weights: {VQA_WEIGHTS}")
    print(f"  Upload dir: {UPLOAD_DIR}")
    print("=" * 60)
    
    # Pre-initialize pipeline
    get_pipeline()
    
    app.run(host=args.host, port=args.port, debug=args.debug)
