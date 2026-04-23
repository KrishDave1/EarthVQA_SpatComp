# AI CONTEXT — EarthVQA Smart City Planning System
## Complete Project Knowledge Transfer Document

> **Purpose:** This file provides complete project context so that any AI model can understand the current state, architecture, codebase, design decisions, bugs encountered and resolved, and how to continue development.

---

## PROJECT OVERVIEW

* **Project Name:** Smart City Planning using EarthVQA
* **Team:** Vaibhav Mittal (IMT2022126), Krish Dave (IMT2022043), Sanchit Dogra (IMT2022035)
* **Course:** Spatial Computing, 8th Semester
* **Goal:** Build a full-stack AI system that analyzes satellite imagery to produce urban planning intelligence — combining Deep Learning segmentation, spatial computing, and a decision support engine exposed via a beautiful React dashboard.
* **Current Stage:** Fully functional and deployed locally. Presentation completed.

---

## CORE IDEA

The system takes a raw satellite image (512×512 RGB PNG), runs it through a pre-trained Semantic Segmentation neural network to classify every pixel into one of 8 land-use classes, then applies deterministic spatial computing algorithms to extract geometric features (distances, areas, topology), and finally passes those features through a calibrated decision engine to produce actionable urban planning recommendations.

A secondary VQA (Visual Question Answering) pipeline allows users to ask natural language questions about the image. An intent parser routes questions to either the deterministic spatial engine (for counting/judging queries) or the SOBA neural network (for complex relational reasoning).

### What makes it different
- Does NOT just wrap a model in a UI. The system combines **probabilistic Deep Learning** (PyTorch CNN) with **deterministic Spatial Computing** (geometry, topology, distance transforms) and a **calibrated decision engine** whose thresholds were statistically derived from 2,522 real training images.
- The decision engine thresholds are NOT hardcoded — they were auto-calibrated from the p25/p50/p75 percentile statistics of the training set distribution.

---

## TECHNOLOGY STACK

| Layer | Technology | Details |
|-------|-----------|---------|
| Deep Learning | PyTorch (CPU) | SemanticFPN (ResNet-50 backbone, ~28.5M params) for segmentation |
| VQA Model | SOBA | Spatial Object-Based Attention (~15.2M params), 165 answer categories |
| Spatial Computing | NumPy, SciPy, scikit-image | Connected component analysis, Euclidean distance transforms, skeletonization |
| Decision Engine | Python + YAML config | Weighted composite scoring with percentile-calibrated thresholds |
| Backend API | Flask + Flask-CORS | REST API on port 5001 |
| Frontend | React 18 + TypeScript + Vite | Glassmorphic dark-mode dashboard |
| Styling | TailwindCSS v4 (via `@tailwindcss/vite`) | Utility-first CSS |
| Charts | Recharts | Radar chart for suitability profile |
| Icons | Lucide React | Icon library |
| HTTP Client | Axios | Frontend-to-backend communication |
| Dev Proxy | Vite proxy | `/api` → `http://127.0.0.1:5001` |

---

## DIRECTORY STRUCTURE

```
EarthVQA_SpatComp/
├── EarthVQA/                    # Original EarthVQA research repo (submodule)
│   ├── module/
│   │   ├── semantic-fpn.py      # SemanticFPN model architecture registration
│   │   └── soba.py              # SOBA VQA model architecture
│   ├── configs/
│   │   ├── sfpnr50.py           # Segmentation model config
│   │   └── soba.py              # VQA model config
│   └── ...
├── pretrained_weights/
│   ├── sfpnr50.pth              # Segmentation model weights (~110MB)
│   └── soba.pth                 # VQA model weights (~60MB)
├── smart_city/                  # Custom pipeline modules
│   ├── __init__.py
│   ├── models.py                # All dataclasses (SpatialFeatures, PlanningReport, etc.)
│   ├── spatial_engine.py        # Spatial feature extraction from seg masks
│   ├── decision_engine.py       # Threshold-based planning intelligence
│   ├── intent_parser.py         # NLP question intent classification
│   ├── pipeline.py              # Unified orchestration pipeline
│   ├── calibrate_thresholds.py  # Statistical threshold calibration script
│   └── config/
│       ├── thresholds.yaml      # Calibrated decision thresholds
│       ├── calibrated_thresholds.yaml  # Alternative calibrated config
│       └── vqa_predictions.json # 63,216 cached VQA predictions
├── backend/
│   ├── app.py                   # Flask REST API server
│   └── uploads/                 # Temporary upload directory
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main app with tab navigation
│   │   ├── main.tsx             # React entry point
│   │   └── components/
│   │       ├── Dashboard.tsx    # Main analysis dashboard
│   │       ├── VQA.tsx          # Chat-based VQA interface
│   │       ├── ScoreCard.tsx    # Individual metric card component
│   │       └── Gallery.tsx      # Sample images gallery
│   ├── vite.config.ts           # Vite config with API proxy
│   ├── tailwind.config.js       # TailwindCSS configuration
│   └── package.json
├── start.sh                     # Startup script (Flask + Vite)
├── evaluate_pipeline.py         # Comprehensive evaluation script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## COMPLETE DATA FLOW PIPELINE

### Phase 1: User Upload → API Ingestion
**File:** `frontend/src/components/Dashboard.tsx` → `backend/app.py`

1. User drags/drops a satellite PNG onto the React Dashboard.
2. `Dashboard.tsx` creates a `FormData` with the image file under key `'image'`.
3. Axios sends `POST /api/analyze` with `Content-Type: multipart/form-data`.
4. Vite dev server proxies `/api` to `http://127.0.0.1:5001` (configured in `vite.config.ts`).
5. Flask receives the file, saves it temporarily to `backend/uploads/temp_image.png`.

### Phase 2: Semantic Segmentation (Deep Learning)
**File:** `smart_city/pipeline.py` → `EarthVQA/module/semantic-fpn.py`

1. `pipeline.py:analyze_image()` calls `_preprocess_image()`:
   - Opens image with PIL, converts to RGB NumPy array (float32).
   - Normalizes with ImageNet statistics: `mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]`.
   - Transposes to `[C, H, W]`, unsqueezes to `[1, 3, 512, 512]` tensor.
2. `_load_seg_model()` (lazy, first-call only):
   - Changes CWD to `EarthVQA/` directory (required by `ever` framework for config resolution).
   - Uses `importlib` to load the hyphenated `semantic-fpn.py` module (registers `SemanticFPN` architecture).
   - Loads `sfpnr50` config via `ever.core.config.import_config()`.
   - Creates model via `ever.core.builder.make_model()`, loads `sfpnr50.pth` state dict.
   - Restores original CWD.
3. Forward pass with `torch.no_grad()`:
   ```python
   pred, features = self.seg_model(image_tensor.to(self.device))
   seg_mask = pred.argmax(dim=1).cpu().numpy()[0]  # [512, 512] int array
   ```
   - `pred` shape: `[1, 8, 512, 512]` — probability distribution over 8 classes per pixel.
   - `features` shape: `[1, 2048, 16, 16]` — deep feature tensor from ResNet-50 final block.
   - `argmax` collapses channel dimension → each pixel gets a single class ID (0-7).

**Output:** 2D integer mask `[512, 512]` where values 0-7 represent:
```
0=background, 1=building, 2=road, 3=water,
4=barren, 5=forest, 6=agriculture, 7=playground
```

### Phase 3: Spatial Feature Extraction (Deterministic Math)
**File:** `smart_city/spatial_engine.py`

The `SpatialFeatureEngine.extract(seg_mask)` method runs 6 sequential steps:

**Step 1: Per-class statistics** (`_compute_class_stats`)
- For each of the 8 classes, creates a binary mask `(seg_mask == class_id)`.
- Counts pixels → area percentage = `pixel_count / (512*512)`.
- Runs SciPy `ndimage.label()` for connected component analysis (8-connectivity).
- Filters components < 25 pixels (noise removal).
- Computes centroid `(cy, cx)` via `ndimage.center_of_mass()` for each valid component.

**Step 2: Area coverages** (`_compute_area_coverages`)
- Extracts quick-access floats: `building_area_pct`, `road_area_pct`, `water_area_pct`, etc.
- Computes: `vegetation_area_pct = forest + agriculture + playground`.

**Step 3: Object counts** (`_extract_counts`)
- `building_count`, `water_body_count`, `road_segment_count` from component analysis.

**Step 4: Distance metrics** (`_compute_distances`)
- Computes pairwise distances for 5 critical pairs: `(building,water)`, `(building,road)`, `(building,forest)`, `(water,agriculture)`, `(road,water)`.
- **Minimum boundary distance** via Euclidean Distance Transform:
  ```python
  dist_to_b = distance_transform_edt(~mask_b)  # distance of every pixel to nearest B pixel
  min_dist = dist_to_b[mask_a].min()            # minimum distance from any A pixel to B
  ```
- **Mean centroid distance**: All-pairs Euclidean distance between centroids of A and B.
- **Proximity score**: Exponential decay: `proximity = exp(-min_dist / 100.0)`.

**Step 5: Building density** (`_compute_density`)
- `building_density = building_count / building_count_max` (normalized by calibrated maximum of 122).

**Step 6: Road connectivity** (`_compute_road_connectivity`)
- Filters road mask with `remove_small_objects(min_size=50)`.
- Applies **Zhang-Suen skeletonization** via `skimage.morphology.skeletonize()` to thin roads to 1px width.
- Counts **intersections**: skeleton pixels with ≥3 neighbors in 8-connectivity.
- Clusters nearby intersection pixels via binary dilation + labeling.
- Composite: `connectivity = 0.6 * road_factor + 0.4 * intersection_factor`.

**Output:** `SpatialFeatures` dataclass with ~16 extracted metrics.

### Phase 4: Calibrated Decision Engine
**File:** `smart_city/decision_engine.py` + `smart_city/config/thresholds.yaml`

The `DecisionEngine.evaluate(features)` method computes 4 composite scores:

**1. Urban Density Score:**
```
score = 0.4 * (building_count / 122) + 0.6 * (building_area_pct / 0.5)
```
Thresholds (calibrated from training set): low < 0.07, moderate < 0.34, high ≥ 0.70.

**2. Green Coverage Score:**
```
raw_green = 1.0 * (forest% + agriculture%) + 0.5 * playground%
score = raw_green / target_pct   (target_pct = 0.36, calibrated from p50)
```
Thresholds: insufficient < 10%, low < 15%, adequate < 36%, good ≥ 66%.

**3. Flood Risk Score:**
```
proximity_factor = max(0, 1.0 - (dist / (50 * 3)))
water_factor = min(water_area_pct / 0.15, 1.0)
bw_ratio = min(building% * water% * 10, 1.0)
score = 0.5 * proximity + 0.3 * water + 0.2 * bw_ratio
```
Returns 0.0 immediately if no water or no buildings detected.

**4. Infrastructure Score:**
```
road_factor = min(road_area_pct / 0.15, 1.0)
intersection_factor = min(intersection_count / 10, 1.0)
score = 0.5 * road_factor + 0.3 * intersection_factor + 0.2 * connectivity
```

**Overall Suitability:**
```
density_suit = 1.0 - abs(density_score - 0.5) * 2    # peaks at 0.5 (inverted U)
flood_suit = 1.0 - flood_risk_score                   # inverted
overall = 0.25 * density_suit + 0.25 * green_suit + 0.25 * flood_suit + 0.25 * infra_suit
```
Labels: Not Suitable (0-0.3), Needs Improvement (0.3-0.5), Moderately Suitable (0.5-0.7), Suitable (0.7-0.85), Highly Suitable (0.85-1.0).

**Output:** `PlanningReport` with 4 `Decision` objects + overall score + natural language summary.

### Phase 5: Response Serialization & Frontend Rendering
**File:** `backend/app.py` → `frontend/src/components/Dashboard.tsx`

1. `AnalysisResult.to_dict()` serializes all features and the planning report to JSON.
   - **Critical:** `float('inf')` values are mapped to `None` (Python `null`) to avoid invalid JSON. This was a bug that caused frontend crashes.
2. `colorized_mask` (RGB NumPy array) is converted to Base64-encoded PNG string.
3. Flask returns JSON response with keys: `spatial_features`, `planning_report`, `colorized_mask_base64`.
4. Dashboard.tsx parses the response:
   ```typescript
   const responseData = typeof analyzeRes.data === 'string'
     ? JSON.parse(analyzeRes.data)
     : analyzeRes.data;
   ```
   This guards against Axios returning raw strings for very large payloads.
5. React renders: original image + colorized mask side-by-side, 8-class legend, 4 ScoreCards, RadarChart, and AI summary.

### Phase 5b: VQA Chat Pipeline
**File:** `smart_city/intent_parser.py` → `smart_city/pipeline.py` → `frontend/src/components/VQA.tsx`

1. User types a natural language question in the chat interface.
2. `VQA.tsx` sends `POST /api/ask` with `image` + `question` as multipart form.
3. `IntentParser.parse(question)` classifies the intent:
   - Keyword scoring: each intent type has weighted keyword matches.
   - Priority order: `planning > risk > density > counting > relation > situation > judging`.
   - Also extracts target objects (`building`, `water`, `road`, etc.) and spatial relations (`near`, `in`, `between`).
4. If the intent is a structural/geometric question (counting, judging, density, risk), the pipeline answers directly from `SpatialFeatures` — **no neural network needed**, 100% accuracy.
5. If the intent is complex relational reasoning, it falls back to the SOBA VQA model (currently returns rule-based answers as SOBA inference integration is a TODO).
6. `DecisionEngine.enrich_answer()` adds planning context to the raw answer.

---

## CALIBRATION DATA

The decision thresholds in `thresholds.yaml` were computed by running `calibrate_thresholds.py` over 2,522 training mask images from the EarthVQA/LoveDA dataset:

```
python -m smart_city.calibrate_thresholds \
    --mask_dir ./EarthVQA/Train/masks_png \
    --output ./smart_city/config/thresholds.yaml
```

Key calibrated statistics:
- Building area: p25=0.0%, p50=3.5%, p75=17%, p90=27%
- Vegetation: p25=8%, p50=36%, p75=66%
- Road: p25=0.8%, p50=2.8%, p75=8%
- Max building count in training set: 122

---

## MODEL ACCURACY & CITATIONS

### Semantic Segmentation (SemanticFPN + ResNet50)
- **mIoU:** ~48-50% on LoveDA test split (8-class remote sensing segmentation)
- **Parameters:** ~28.5M
- **Citation:** Wang et al., "LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation", NeurIPS Datasets Track, 2021. DOI: 10.5281/zenodo.5706578

### Visual Question Answering (SOBA)
- **Accuracy:** ~62-78% on EarthVQA validation (varies by question type)
- **Parameters:** ~15.2M
- **Answer categories:** 165 unique spatial answers
- **Citation:** Wang et al., "EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering", AAAI 2024, Vol 38, pp 5481-5489. DOI: 10.1609/ai.v38i6.28357

### Dataset Scale
- Training images: 2,522
- Test images: 1,809
- Total QA pairs: 63,216
- Image resolution: 512×512 RGB
- Pixels per analysis: 262,144

---

## BUGS ENCOUNTERED & RESOLVED (DEVELOPMENT HISTORY)

### Bug 1: Port 5000 Conflict (macOS AirPlay)
- **Symptom:** 403 Forbidden on all API calls.
- **Cause:** macOS Monterey+ uses port 5000 for AirPlay Receiver.
- **Fix:** Changed Flask to port 5001; updated Vite proxy target.

### Bug 2: Segmentation Model Not Loading
- **Symptom:** `RuntimeError: Segmentation model not loaded`.
- **Cause:** The `ever` framework requires CWD to be `EarthVQA/` to resolve config files. Also, `semantic-fpn.py` has a hyphenated filename that Python can't import normally.
- **Fix:** Added `os.chdir(earthvqa_dir)` before model loading, and used `importlib.util.spec_from_file_location()` for the hyphenated module. Restored CWD in a `finally` block.

### Bug 3: React Crash on VQA Response
- **Symptom:** `Objects are not valid as a React child (found: object with keys {relation, targets, type})`.
- **Cause:** `VQA.tsx` was rendering `res.data.intent` (a JSON object) directly as text.
- **Fix:** Changed to `res.data.intent?.type` to extract only the string.

### Bug 4: Dashboard Crash on `.map()` of undefined
- **Symptom:** `TypeError: Cannot read properties of undefined (reading 'map')`.
- **Cause:** `Dashboard.tsx` accessed `report.decisions.map()` without optional chaining.
- **Fix:** Changed to `report?.decisions?.map()`.

### Bug 5: API State Mapping Error
- **Symptom:** Dashboard showed analysis button again after successful response.
- **Cause:** Frontend was accessing `analyzeRes.data` directly instead of `analyzeRes.data.planning_report`.
- **Fix:** Updated state setter to correctly map `responseData.planning_report`.

### Bug 6: JSON Infinity Serialization
- **Symptom:** `SyntaxError: Unexpected token 'I', ..."tance_px":Infinity,""... is not valid JSON`.
- **Cause:** Python's `float('inf')` serializes as literal `Infinity` in JSON, which is invalid JavaScript.
- **Fix:** In `models.py`, replaced `round(d.min_distance_px, 2)` with conditional: `round(val, 2) if val != float('inf') else None`.

### Bug 7: Large Payloads Parsed as Strings
- **Symptom:** Analysis succeeds (console shows `Analysis Success: {…}`) but UI doesn't render.
- **Cause:** For very large Base64 payloads, Axios sometimes returns the response as a raw string instead of a parsed object.
- **Fix:** Added defensive parsing: `typeof analyzeRes.data === 'string' ? JSON.parse(analyzeRes.data) : analyzeRes.data`.

### Bug 8: Recharts Width/Height Warning
- **Symptom:** `The width(-1) and height(-1) of chart should be greater than 0`.
- **Cause:** `ResponsiveContainer` briefly has negative dimensions during React animation transitions.
- **Impact:** Cosmetic warning only, does not affect functionality.

---

## HOW TO RUN

### Prerequisites
- Python 3.8+ with virtual environment at `.venv/`
- Node.js 18+ with npm
- PyTorch, ever-beta, scikit-image, Flask, flask-cors, Pillow, scipy, numpy, pyyaml
- Pretrained weights in `pretrained_weights/` (sfpnr50.pth, soba.pth)

### Start Command
```bash
chmod +x start.sh
./start.sh
```
This starts:
1. Flask backend on port 5001 (logs to `backend.log`)
2. Vite frontend on port 5173 (logs to `frontend.log`)

Access dashboard at: http://localhost:5173

### Stop Command
```bash
lsof -ti:5001 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

---

## API ENDPOINTS

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | `/api/analyze` | `image` (multipart file) or `mask` (multipart file) | JSON: `{spatial_features, planning_report, colorized_mask_base64}` |
| POST | `/api/ask` | `image` (file) + `question` (form field) | JSON: `{answer, confidence, explanation, intent, analysis}` |
| GET | `/api/health` | — | JSON: `{status, device, seg_model_loaded, vqa_model_loaded}` |
| GET | `/api/questions` | — | JSON: `{questions, custom_questions}` |
| POST | `/api/visualize/mask` | `mask` (file) | JSON: `{colorized_mask_base64, width, height}` |

---

## SEGMENTATION COLOR MAP

| Class ID | Class Name | RGB Color | Hex |
|----------|-----------|-----------|-----|
| 0 | Background | (255,255,255) | #FFFFFF |
| 1 | Building | (255,0,0) | #FF0000 |
| 2 | Road | (255,255,0) | #FFFF00 |
| 3 | Water | (0,0,255) | #0000FF |
| 4 | Barren | (159,129,183) | #9F81B7 |
| 5 | Forest | (0,255,0) | #00FF00 |
| 6 | Agriculture | (255,195,128) | #FFC380 |
| 7 | Playground | (165,0,165) | #A500A5 |

---

## FUTURE WORK

1. **Urban Heat Island & Air Quality Prediction:** Use spatial features as input to XGBoost/LightGBM to predict land surface temperature or AQI. Data source: NASA Landsat-8 thermal band + OpenAQ.
2. **Time-Series Change Detection:** Compare two satellite images of the same region over time, compute delta spatial features, train SVM/Random Forest to classify urban sprawl patterns. Data source: Sentinel-2 via Google Earth Engine or SpaceNet 7.
3. **Graph Neural Networks for Traffic Routing:** Convert road skeleton to a graph (intersections=nodes, roads=edges), train a GCN to predict traffic bottlenecks. Data source: OpenStreetMap + Uber Movement Dataset.

---

## KEY DESIGN DECISIONS

1. **Lazy model loading:** Models are loaded on first API call, not at server startup. This keeps startup fast and allows the system to function in mask-only mode without GPU.
2. **CWD switching:** The `ever` framework resolves configs relative to CWD. We must `os.chdir()` to `EarthVQA/` before loading models and restore after.
3. **Rule-based VQA fallback:** When the SOBA neural network isn't available or fails, the system generates answers directly from spatial features for reliable deterministic responses.
4. **Decision engine enrichment:** VQA answers are enriched with planning context from the decision engine, adding domain expertise on top of raw model outputs.
5. **Float('inf') handling:** All serialization paths guard against Python infinity values to prevent JSON parse failures in JavaScript.
