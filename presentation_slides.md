# End-Semester Presentation — New Slides Guide
## Smart City Planning using EarthVQA

> **Context:** Your mid-eval presentation had 14 slides covering: Background, Problem Statement, Project Overview, I/O Pipeline, Dataset, Methodology (×2), Complete Pipeline, Results (×2), Future Work (×2), Thank You. This document provides the **new slides to add after your existing Results slides**, replacing the old Future Work / Thank You slides.

---

## Slide Placement Map

Your existing presentation ends roughly at slide 12 (Results — model accuracy). The new slides should be inserted as follows:

| Slide # | Title | Section |
|---------|-------|---------|
| 13 | **Work Completed Since Mid-Eval** | Transition |
| 14 | **Full-Stack React Dashboard** | Frontend |
| 15 | **Dashboard — Live Analysis Flow** | Frontend |
| 16 | **VQA Chat Assistant** | Frontend |
| 17 | **Time-Series Change Detection** | New Module |
| 18 | **Change Detection — Sprawl Classification** | New Module |
| 19 | **Academically Justified Decision Engine** | Rigor |
| 20 | **Calibration from 2,522 Training Masks** | Rigor |
| 21 | **Summary of Scores & References** | Rigor |
| 22 | **Live Demo** | Demo |
| 23 | **Future Work** | Future |
| 24 | **Thank You & References** | Closing |

---

## Slide 13 — Work Completed Since Mid-Eval

**Purpose:** Transition slide showing what you promised vs. what you delivered.

### Content

**Title:** Work Completed Since Mid-Evaluation

**Body (use a two-column layout or checklist):**

| Mid-Eval Promise | Status |
|---|---|
| React-based Frontend Dashboard | ✅ Fully built (React 18 + TypeScript + Vite) |
| Chat-based VQA Interface | ✅ Implemented with intent parsing |
| Urban Heat Island Prediction | ❌ Deprioritized → replaced with Change Detection |
| Time-Series Change Detection | ✅ Full module with RF classifier |
| Academic Justification of Scores | ✅ 18 peer-reviewed references |

**Speaker Notes:**
> "Since the mid-evaluation, we have completed three major deliverables. First, a full glassmorphic React dashboard. Second, a Time-Series Change Detection module that classifies urban sprawl. Third, a rigorous academic justification for every weight and threshold in our decision engine, backed by 18 peer-reviewed references."

### Image
- No image needed. Use a clean checklist/comparison table layout.
- Optionally add a small timeline graphic: Mid-Eval → Frontend → Change Detection → Justification → End-Eval.

---

## Slide 14 — Full-Stack React Dashboard

**Purpose:** Showcase the UI you built.

### Content

**Title:** Full-Stack Smart City Dashboard

**Bullet points (left side, small font):**
- Built with React 18 + TypeScript + Vite + TailwindCSS v4
- Glassmorphic dark-mode design with smooth animations
- Flask REST API backend on port 5001
- 4 modules: Planning Dashboard, Change Detection, VQA Assistant, Sample Gallery
- Drag-and-drop satellite image upload
- Real-time semantic segmentation + spatial analysis

**Architecture mini-diagram (right side):**
```
User → React Frontend (5173)
         ↓ Axios /api/*
       Vite Proxy
         ↓
       Flask API (5001)
         ↓
    SmartCityPipeline
   ├── SemanticFPN (28.5M params)
   ├── SpatialEngine
   ├── DecisionEngine
   └── ChangeDetector
```

### Image
> [!IMPORTANT]
> **Screenshot needed:** Take a screenshot of the app's **sidebar navigation** showing all 4 tabs (Planning Dashboard, Change Detection, VQA Assistant, Sample Gallery). The dark theme with the satellite icon and blue accents will look great on the slide.

**Placement:** Full-width screenshot on the right 60%, bullet points on the left 40%.

---

## Slide 15 — Dashboard — Live Analysis Flow

**Purpose:** Show the actual analysis output — this is the money slide.

### Content

**Title:** Planning Dashboard — AI-Powered Spatial Analysis

**Subtitle:** "Drag a satellite image → get a full planning report in seconds"

**Body:** No bullet points needed — let the screenshots speak.

### Images
> [!IMPORTANT]
> **Two screenshots needed (side by side or as a carousel):**
> 1. **Upload state:** The drag-and-drop zone with a satellite image preview and the "Run Analysis (28.5M Params)" button glowing blue.
> 2. **Results state:** The full results view showing:
>    - Original image vs. 8-class SegMask side-by-side
>    - 8-class color legend
>    - 4 ScoreCards (Density, Green Coverage, Flood Risk, Infrastructure)
>    - Radar chart with suitability profile
>    - Overall verdict score (e.g., "Moderately Suitable — 62/100")
>    - AI Summary text

**Layout:** Use a "before → after" layout. Left half = upload state. Right half = results view. Or do a full-slide screenshot of the results with callout annotations.

**Speaker Notes:**
> "The dashboard accepts any 512×512 satellite PNG. The SemanticFPN model segments 262,144 pixels into 8 land-use classes. Then our Spatial Engine extracts 16 metrics using connected component analysis, Euclidean distance transforms, and skeletonization. The Decision Engine produces 4 composite scores and an overall suitability verdict."

---

## Slide 16 — VQA Chat Assistant

**Purpose:** Show the interactive Q&A interface.

### Content

**Title:** Visual Question Answering — Natural Language Interface

**Left side (small bullets):**
- Intent-parsed question routing
- 7 supported intents: counting, judging, density, risk, situation, relation, planning
- Deterministic answers from spatial features (100% reliable)
- SOBA neural network fallback for complex relational queries
- Enriched answers with planning context & confidence scores

**Example exchange (right side, styled like a chat bubble):**
```
User: "Is this area suitable for development?"
Intent: planning | Confidence: 0.95

AI: "This area is classified as 'Moderately Suitable' 
     with an overall score of 0.62. Key findings:
     - Urban Density: 0.45 (Moderate)
     - Green Coverage: 0.78 (Good — 36%+ vegetation)
     - Flood Risk: 0.31 (Low)
     - Infrastructure: 0.54 (Moderate)
     Recommendation: Consider expanding road 
     infrastructure before further development."
```

### Image
> [!IMPORTANT]
> **Screenshot needed:** The VQA chat interface with an uploaded satellite image on the left panel and a conversation on the right panel showing at least 2 Q&A exchanges. Include one where the intent tag shows (e.g., `intent: risk`, `conf: 0.95`).

**Placement:** Screenshot fills right 55%, bullet points on left 45%.

---

## Slide 17 — Time-Series Change Detection

**Purpose:** Introduce the new module that wasn't in mid-eval.

### Content

**Title:** Time-Series Change Detection (New Module)

**Subtitle:** "Compare two satellite images to track urban sprawl over time"

**Body (use a flow diagram):**
```
Image T₁ → SemanticFPN → Mask T₁ → SpatialEngine → Features T₁ ─┐
                                                                   ├→ Δ Features (16 deltas)
Image T₂ → SemanticFPN → Mask T₂ → SpatialEngine → Features T₂ ─┘
                                                                         ↓
                                                          Random Forest Classifier
                                                           (or rule-based fallback)
                                                                         ↓
                                                          Sprawl Classification
                                                          + Recommendations
```

**Key Δ Features computed:**
- Δ Building Area %, Δ Vegetation %, Δ Water %, Δ Road %
- Δ Building Count, Δ Intersection Count
- Δ Building-Water Distance (proximity change)
- 16 total delta features

### Image
> [!IMPORTANT]
> **Screenshot needed:** The Change Detection upload interface showing two side-by-side upload zones labeled "Before (T₁)" and "After (T₂)" with satellite images loaded and the "Detect Changes" button visible.

**Placement:** Flow diagram on the top half, screenshot on the bottom half.

---

## Slide 18 — Change Detection — Sprawl Classification

**Purpose:** Show the 6 sprawl categories and the results output.

### Content

**Title:** Urban Sprawl Classification — 6 Categories

**Table (left side):**

| Category | Icon | Indicators |
|---|---|---|
| Aggressive Urbanization | 🔴 | Buildings ↑↑, Vegetation ↓↓ |
| Deforestation | 🟤 | Forest ↓↓, Barren/Agriculture ↑ |
| Water Encroachment | 🔵 | Water ↑↑, Buildings near water ↑ |
| Sustainable Expansion | 🟢 | Buildings ↑, Green ≈ stable, Infra ↑ |
| Infrastructure Development | 🟡 | Roads ↑↑, Intersections ↑↑ |
| Stable / No Change | ⚪ | All Δ values ≈ 0 |

**Key point:** Each classification includes AI-generated actionable recommendations.

### Image
> [!IMPORTANT]
> **Screenshot needed:** The Change Detection results page showing:
> - The sprawl classification card (e.g., "🟢 Sustainable Expansion" with confidence %)
> - The side-by-side segmentation comparison (T₁ vs T₂)
> - The Land-Use Change (Δ) bar chart
> - The Temporal Delta Table with before/after percentages
> - The AI Recommendations panel

**Placement:** Table on the left 40%, screenshot on the right 60%.

---

## Slide 19 — Academically Justified Decision Engine

**Purpose:** This is the academic credibility slide. Show that your scores aren't arbitrary.

### Content

**Title:** Academically Justified 4-Score Decision Engine

**Subtitle:** "Every weight, threshold, and formula is traceable to peer-reviewed literature"

**Body (use a structured diagram with references):**

| Score | Formula | Primary Reference |
|---|---|---|
| **Density** | `0.4 × count/max + 0.6 × area/0.5` | Huang et al. (2007) — "two complementary metrics" |
| **Green Coverage** | `veg_area / 0.36` (target from p50) | Konijnendijk (2023) — 3-30-300 rule, ≥30% canopy |
| **Flood Risk** | `0.5×prox + 0.3×water + 0.2×ratio` | Kron (2005) — Risk = Hazard × Exposure × Vulnerability |
| **Infrastructure** | `0.5×road + 0.3×intersect + 0.2×conn` | Kansky (1963) / Boeing (2017) — α, β, γ indices |
| **Overall (WLC)** | `Σ 0.25 × scoreᵢ` | Malczewski (2004) — used in 75% of GIS-MCDM studies |

**Key callout box:**
> "18 peer-reviewed references with 60,000+ combined citations support our methodology. Weights are based on Tehrany et al. (2014), Konijnendijk (2023), and Malczewski (2004). Thresholds are empirically calibrated from the dataset, not hand-tuned."

### Image
- No screenshot needed. Use a clean table and possibly the traceability flowchart from References.md.
- Alternatively, include a small version of the mermaid flowchart: Training Masks → Calibration → thresholds.yaml → Decision Engine → Suitability.

**Speaker Notes:**
> "We want to emphasize that our decision engine is not a black box. Every single weight — why density uses 0.4/0.6 split, why flood risk prioritizes proximity at 0.5 — traces back to specific sections and pages in peer-reviewed papers. For example, our flood risk formula mirrors Kron's canonical framework used by Munich Re and the EU Floods Directive."

---

## Slide 20 — Calibration from 2,522 Training Masks

**Purpose:** Show the empirical grounding of thresholds.

### Content

**Title:** Data-Driven Threshold Calibration

**Subtitle:** "Percentile statistics from 2,522 real satellite masks"

**Raw calibration output (use a code/monospace block):**
```
=== Calibration Statistics ===
     building_count: mean=11.2   p50=7.0     p75=18.0    max=122.0
  building_area_pct: mean=0.10   p50=0.036   p75=0.168   max=0.869
      road_area_pct: mean=0.054  p50=0.027   p75=0.079   max=0.642
     water_area_pct: mean=0.062  p50=0.020   p75=0.079   max=1.000
     vegetation_pct: mean=0.390  p50=0.361   p75=0.664   max=1.000
```

**Mapping table:**

| Threshold | Value | Derivation |
|---|---|---|
| Green target | 0.36 | Directly from p50 = 0.3605 (aligns with 30% rule) |
| Green "good" | 0.66 | Directly from p75 = 0.6636 |
| Building max | 122 | Directly from max in dataset |
| Flood safe dist | 50px | Samanta et al. (2018): 0-50m = "very high risk" |
| Density "moderate" | 0.34 | round(p50 × 2) |

**Key insight:** "Our target of 36% vegetation aligns perfectly with Konijnendijk's 3-30-300 rule recommending ≥30% canopy cover."

### Image
- No screenshot needed. Use clean tables and the calibration output styled as a terminal/code block.
- Optionally include a small histogram or distribution chart of vegetation_pct across the 2,522 images with p25/p50/p75 markers.

---

## Slide 21 — Summary of Scores & References

**Purpose:** One-page academic summary — the slide a professor wants to see.

### Content

**Title:** Complete Scoring Framework — Academic Traceability

**Summary table (full width):**

| Our Score | Metrics Used | Computation | Key Reference | Where in Paper |
|---|---|---|---|---|
| Density | Count + Area% | 0.4×count + 0.6×area | Huang et al. (2007) | §2.1, p.186 |
| Green Coverage | Veg% vs target | veg / 0.36 | Konijnendijk (2023) | p.826: "≥30% canopy" |
| Flood Risk | Proximity + Area + Ratio | 0.5×prox + 0.3×water + 0.2×ratio | Kron (2005) | pp.59-62: Risk equation |
| Infrastructure | Road% + Intersections + Connectivity | 0.5×road + 0.3×int + 0.2×conn | Boeing (2017) | §4.1, Table 2 |
| Overall WLC | Weighted sum | Σ 0.25 × scoreᵢ | Malczewski (2004) | §3.2: "75% of studies" |
| Thresholds | Percentile calibration | p25/p50/p75 | Jenks (1967) | pp.186-188 |

**Footer:** "18 references, 60,000+ combined citations. Full reference document available."

### Image
- No screenshot. Clean academic table only.

---

## Slide 22 — Live Demo

**Purpose:** Transition slide before showing the live system.

### Content

**Title:** Live Demonstration

**Body (centered, large font):**
1. **Planning Dashboard** — Upload a satellite image → full analysis report
2. **VQA Assistant** — Ask: "Is this area suitable for development?"
3. **Change Detection** — Compare two temporal images → sprawl classification

**Technical details (small footer):**
- Flask backend (port 5001) + Vite dev server (port 5173)
- SemanticFPN: 28.5M params, CPU inference
- Processing time: ~3-5 seconds per image

### Image
- No image needed. Keep it clean — just the 3-step demo plan.
- This is your cue to alt-tab to the live application.

---

## Slide 23 — Future Work

**Purpose:** What comes next — show you have a research roadmap.

### Content

**Title:** Future Work

**Three research extensions (use icons + one paragraph each):**

**1. 🌡️ Urban Heat Island & Air Quality Prediction**
- Use extracted spatial features (building density, vegetation %, water %) as input to XGBoost/LightGBM
- Predict land surface temperature and AQI from the segmentation mask alone
- Data source: NASA Landsat-8 thermal band + OpenAQ
- *"Our spatial features already correlate with UHI indicators — adding a regression head is the natural next step."*

**2. 🧠 VQA-Spatial Fusion**
- Currently, rule-based answers and SOBA VQA answers operate independently
- Future: merge both into a unified reasoning pipeline that uses deep learning VQA + deterministic spatial geometry for hybrid answers
- Fine-tune SOBA with custom planning-specific questions (e.g., "What renovation is needed?")

**3. 🌐 Graph Neural Networks for Traffic Routing**
- Convert road skeleton → graph (intersections = nodes, roads = edges)
- Train a GCN to predict traffic bottlenecks and optimal routing
- Data source: OpenStreetMap + Uber Movement Dataset
- *"Our skeletonization already extracts intersection topology — this is one step away from a GNN input."*

### Image
- No screenshot needed. Use a clean 3-column layout with icons (🌡️, 🧠, 🌐).
- Optionally include a small conceptual diagram showing how the existing pipeline feeds into each extension.

**Speaker Notes:**
> "We have three concrete extensions planned. First, UHI prediction using our spatial features as input to gradient boosting — our features already capture the key drivers of urban heat islands. Second, fusing the rule-based and neural VQA pipelines for hybrid reasoning. Third, converting our road skeleton to a graph for traffic routing with GNNs."

---

## Slide 24 — Thank You & References

**Purpose:** Closing slide.

### Content

**Title:** Thank You

**Team:**
- Vaibhav Mittal (IMT2022126)
- Krish Dave (IMT2022043)
- Sanchit Dogra (IMT2022035)

**Key references (top 6 only, small font):**
1. Wang et al. (2024). EarthVQA, *AAAI 2024*
2. Wang et al. (2021). LoveDA, *NeurIPS Datasets 2021*
3. Malczewski (2004). GIS-based WLC, *Progress in Planning*
4. Konijnendijk (2023). 3-30-300 Rule, *J. Forestry Research*
5. Kron (2005). Flood Risk Framework, *Water International*
6. Boeing (2017). OSMnx, *CEUS*

**GitHub / Demo link:** *(add your repo URL here)*

---

## Screenshots Checklist

> [!IMPORTANT]
> **You need to capture these screenshots from your running application:**

| # | What to Screenshot | Used in Slide |
|---|---|---|
| 1 | Sidebar navigation (all 4 tabs visible) | Slide 14 |
| 2 | Dashboard — drag-and-drop upload zone with image preview | Slide 15 |
| 3 | Dashboard — full results view (masks + ScoreCards + radar + verdict) | Slide 15 |
| 4 | VQA chat — image loaded + 2-3 Q&A exchanges | Slide 16 |
| 5 | Change Detection — upload zone with T₁ and T₂ images | Slide 17 |
| 6 | Change Detection — full results (classification card + masks + delta chart + recommendations) | Slide 18 |

**How to capture:** Start the app with `./start.sh`, navigate to `http://localhost:5173`, and use one of the test images from `Test_Images/` (e.g., `4195.png` or `4200.png`).

---

## Slide Design Tips

- **Keep the dark theme** consistent with your app (bg `#0B1120`, blue/indigo accents)
- **Use the same color coding** as your app: 🔴 Density, 🟢 Green, 🔵 Flood, 🟡 Infrastructure
- **Font:** Use Inter or Outfit (same as your React app)
- **Each slide should have max 5-6 bullet points** — let the screenshots do the heavy lifting
- **The academic slides (19-21)** can be more text-heavy; professors expect that
- **The demo slide (22)** should be minimal — it's a cue for you to show the live system
