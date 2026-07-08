# Semantic Object Color Retouching Development Plan

## 1. Product Goal

Build a semantic color-retouching expert.

It should not only say "the photo is too yellow" or "the exposure is low". It should understand objects and regions in the image, then decide how each important region should be adjusted.

Example:

```text
Input photo:
- sky is gray and dull
- people's skin is slightly yellow
- grass is dark green
- white clothes are slightly warm

Output plans:
- Plan A: natural correction
- Plan B: commercial clean color
- Plan C: cinematic warm/cool separation

Execution:
- make the sky natural daylight blue
- keep clouds, buildings, horizon, and tree edges unchanged
- correct skin tone but preserve identity and texture
- keep clothing/logo colors faithful
```

The first model should be a "semantic color planning model", not a full image generator.

Final pipeline:

```text
photo
-> object/region recognition
-> segmentation masks
-> per-region color metrics
-> semantic color diagnosis
-> 3-5 retouch plans
-> GPT Image 2 / local masked color edit
-> quality check
-> human preference feedback
-> distilled student model
```

## 2. Core Architecture

### Layer 1: Image Intake

Input:

- JPG / PNG / HEIC photos
- optional RAW later
- current dataset: about 20,562 usable JPG photos

Tasks:

- exclude macOS `._*` resource files
- normalize EXIF orientation
- generate thumbnails
- calculate basic image stats
- assign `image_id`
- store metadata in `manifest.jsonl`

Output:

```json
{
  "image_id": "000001",
  "path": "images/000001.jpg",
  "width": 4000,
  "height": 2667,
  "orientation": "landscape",
  "split": "train"
}
```

### Layer 2: Object And Region Perception

Use a teacher stack instead of one model.

Recommended first stack:

```text
VLM: Qwen3-VL / Qwen2.5-VL / InternVL
Open-vocabulary detector: Grounding DINO / Grounding DINO 1.5
Segmentation: SAM2
Color metrics: OpenCV / Pillow / scikit-image
Optional commercial gold teacher: OpenAI vision-capable model
```

Responsibilities:

- VLM: global image understanding and scene reasoning
- Grounding DINO: find objects by text prompt, such as sky, skin, face, grass, water, clothing, building, LED screen
- SAM2: convert boxes/points into accurate masks
- color metrics: calculate objective color information inside each mask

Why this combination:

- VLM alone can hallucinate objects.
- Detector alone gives boxes, not precise editable areas.
- SAM2 gives masks but needs prompts or boxes.
- Color metrics make the diagnosis measurable.

### Layer 3: Semantic Region Inventory

For every photo, produce a region list.

Region examples:

- sky
- face / skin
- hair
- clothing
- grass / trees
- water
- building
- road
- background
- LED screen
- stage light
- white object
- product
- text / logo

Output:

```json
{
  "image_id": "000001",
  "scene_type": "outdoor_event_portrait",
  "regions": [
    {
      "region_id": "r001",
      "object_type": "sky",
      "role": "background",
      "bbox": [0.0, 0.0, 1.0, 0.32],
      "mask_path": "masks/000001/r001_sky.png",
      "confidence": 0.91,
      "editability": "high",
      "protect_level": "medium"
    },
    {
      "region_id": "r002",
      "object_type": "skin",
      "role": "primary_subject",
      "bbox": [0.42, 0.22, 0.61, 0.56],
      "mask_path": "masks/000001/r002_skin.png",
      "confidence": 0.86,
      "editability": "medium",
      "protect_level": "high"
    }
  ]
}
```

## 3. Per-Region Color Diagnosis

For each region, calculate objective metrics.

Recommended metrics:

- average RGB
- average LAB
- hue distribution
- saturation distribution
- luminance percentiles
- highlight clipping
- shadow clipping
- color temperature proxy
- green/magenta cast proxy
- colorfulness
- local contrast

Example:

```json
{
  "region_id": "r001",
  "object_type": "sky",
  "color_metrics": {
    "avg_lab": [71.2, -2.1, -8.5],
    "brightness": 0.72,
    "saturation": 0.18,
    "dominant_hue": "cyan_gray",
    "clip_high_pct": 1.7,
    "clip_low_pct": 0.0
  },
  "diagnosis": {
    "state": "dull_gray_sky",
    "severity": 0.64,
    "recommended_action": "shift_to_natural_daylight_blue"
  }
}
```

## 4. Object-Specific Color Rules

Build a semantic color rule library. This is the heart of the product.

### Sky

Rules:

- If daytime sky is gray/cyan and not sunset/night, suggest natural blue.
- Preserve clouds, skyline, branches, buildings, and horizon edges.
- Avoid neon blue or fake HDR.
- Do not make sunset/night/stage backgrounds blue unless user asks.

Actions:

- natural blue sky
- slightly deeper blue
- bright commercial blue
- keep overcast mood

Prompt fragment:

```text
Only adjust the sky region. Make the sky a natural daylight blue with realistic brightness and soft saturation. Preserve cloud shape, building edges, trees, horizon, and the original lighting direction. Do not change people, clothing, signs, or background structure.
```

### Skin

Rules:

- Highest protection priority.
- Correct yellow/green/red pollution gently.
- Preserve identity, face shape, pores, makeup, age, and expression.
- Avoid plastic skin.

Actions:

- clean natural skin
- warm healthy skin
- reduce yellow-green cast
- reduce red/orange oversaturation

### Grass / Trees

Rules:

- Natural greens should not become neon.
- Yellow-green pollution can be reduced.
- Keep local texture.

Actions:

- fresh natural green
- darker cinematic green
- reduce yellow cast

### Water

Rules:

- Can be shifted to cleaner blue/cyan only if scene supports it.
- Preserve reflections and wave texture.

### Clothing / Logos / Product Colors

Rules:

- Usually protected.
- Do not change brand colors unless user explicitly requests.
- White clothes can be neutralized.

### LED Screen / Stage Light

Rules:

- Do not blindly neutralize; stage color is often intentional.
- Reduce color pollution on faces separately.
- Preserve atmosphere.

### Food

Rules:

- Warm appetizing color is often better.
- Avoid making food gray or too cool.

## 5. Retouch Plan Generator

For each image, generate 3-5 plans.

Recommended V1 plans:

1. `safe_natural`
   - Correct color while preserving realism.
   - Best for batch event delivery.

2. `commercial_clean`
   - Brighter, cleaner, more pleasing.
   - Best for family, school, event albums.

3. `cinematic_separation`
   - Warm subject, cooler background, stronger mood.
   - Best for portrait/social media.

4. `object_enhancement`
   - Object-targeted enhancement, such as blue sky, cleaner grass, whiter clothes.

5. `restore_original_intent`
   - For stage/night/sunset, preserve mood and only fix pollution.

Output:

```json
{
  "image_id": "000001",
  "global_diagnosis": {
    "scene_type": "outdoor_event_portrait",
    "main_problem": "gray sky and yellow-green skin pollution",
    "edit_risk": "skin identity and clothing color must be protected"
  },
  "plans": [
    {
      "plan_id": "p1",
      "name": "safe_natural",
      "summary": "Correct white balance and make skin natural, keep sky mostly realistic.",
      "risk": "low",
      "region_actions": [
        {
          "region_id": "r001",
          "object_type": "sky",
          "action": "slightly_clean_blue_gray",
          "strength": 0.35
        },
        {
          "region_id": "r002",
          "object_type": "skin",
          "action": "remove_yellow_green_cast",
          "strength": 0.45
        }
      ],
      "gpt_image_prompt": "Only adjust color and tone. Preserve identity, clothing, composition, background structure, and object shapes. Correct yellow-green cast on skin, keep skin texture natural, gently clean the sky toward a believable daylight blue-gray, protect highlights, and avoid artificial HDR."
    },
    {
      "plan_id": "p2",
      "name": "object_enhancement_blue_sky",
      "summary": "Make the sky natural blue while protecting people and buildings.",
      "risk": "medium",
      "region_actions": [
        {
          "region_id": "r001",
          "object_type": "sky",
          "action": "natural_daylight_blue",
          "strength": 0.7
        }
      ],
      "gpt_image_prompt": "Only adjust the sky and overall color harmony. Make the sky a natural daylight blue with realistic saturation. Preserve clouds, tree edges, buildings, horizon, people, faces, clothing, signs, and composition. Do not replace the scene or alter object shapes."
    }
  ]
}
```

## 6. Editing Engine

Use two execution modes.

### Mode A: GPT Image 2 Semantic Edit

Best for:

- natural-looking semantic edits
- complex mixed lighting
- sky, background, mood, and overall photo harmony
- edits that require understanding context

Use:

- original image
- optional mask
- high input fidelity
- detailed preservation constraints

Important:

- GPT Image 2 can edit images from text and image inputs.
- Masked editing should be treated as guidance, not pixel-perfect Photoshop masking.
- Always run post-edit quality checks.

### Mode B: Local Masked Color Adjustment

Best for:

- precise color shifts
- blue sky from detected mask
- saturation/brightness/hue changes
- batch previews
- low cost

Use:

- SAM2 mask
- LAB/HSL color transforms
- edge feathering
- subject-protection masks

For example:

```text
sky mask -> shift hue toward natural blue -> increase saturation modestly -> preserve luminance gradient -> feather edge -> composite
```

Recommendation:

- Use local masked edit for preview and cheap deterministic color correction.
- Use GPT Image 2 for final high-quality semantic edit or when local edit looks fake.

## 7. Quality Check System

Every edited image should be checked automatically.

Checks:

- target object changed correctly
- protected objects stayed stable
- face identity preserved
- skin texture not plastic
- clothing/logo colors not damaged
- text/signs not corrupted
- sky/tree/building edges not broken
- no new objects added
- no over-filtered look

Metrics:

- color delta in target mask
- color delta in protected masks
- face embedding similarity
- OCR/text similarity for signs/logos
- VLM before/after critique
- human preference score

Example QA output:

```json
{
  "image_id": "000001",
  "plan_id": "p2",
  "target_success": true,
  "sky_blue_score": 0.82,
  "protected_skin_delta": 0.11,
  "identity_risk": "low",
  "artifact_risk": "medium",
  "reject": false,
  "notes": "Sky improved; slight halo around tree edge."
}
```

## 8. Distillation Strategy

Do not first train a model to generate edited pixels.

First train:

```text
image -> semantic_color_plan.json
```

This is cheaper, safer, and easier to evaluate.

### Stage 0: No-Training Baseline

Run the full teacher pipeline on 100-300 images:

```text
Grounding DINO -> SAM2 -> color metrics -> VLM diagnosis -> plans -> GPT Image 2/local edit -> QA
```

Goal:

- prove the system can identify objects and produce useful color plans
- create demo outputs
- collect failure cases

### Stage 1: Pseudo-Label Dataset

Run 2,000 images:

- 1,000 random
- 500 face/person-heavy
- 300 outdoor/sky/greenery
- 200 dark/stage/LED/mixed-light

For each image save:

```text
images/
masks/
metrics/
teacher_labels/
plans/
edited_candidates/
qa/
preferences/
```

### Stage 2: Student Planner Model

Train a student model to predict:

- scene type
- object list
- important regions
- color diagnosis
- object-specific actions
- plan ranking
- preservation constraints

Student options:

1. Small VLM LoRA
   - Qwen-VL / InternVL / MiniCPM-V class model
   - Input: image
   - Output: JSON plan
   - Best for natural-language reasoning

2. Vision encoder + classifier heads
   - SigLIP / CLIP / DINOv2 / ConvNeXt
   - Output: structured labels and scores
   - Best for cheap production inference

3. Hybrid
   - Detector/segmenter still handles masks
   - student predicts diagnosis and plan ranking
   - DS turns structured output into polished prompts

Recommended V1:

```text
Hybrid:
Grounding DINO + SAM2 + color metrics
+ small student planner
+ DS prompt generator
+ GPT Image 2 editor
```

## 9. Dataset Schema

Directory:

```text
dataset/
  images/
  masks/
  metrics/
  teacher_labels/
  plans/
  edited_candidates/
  qa/
  preferences/
  manifest.jsonl
```

Manifest row:

```json
{
  "image_id": "000001",
  "image_path": "images/000001.jpg",
  "split": "train",
  "width": 4000,
  "height": 2667,
  "teacher_label": "teacher_labels/000001.json",
  "metrics": "metrics/000001.json"
}
```

Teacher label:

```json
{
  "image_id": "000001",
  "scene": {
    "type": "outdoor_portrait",
    "lighting": "daylight_overcast",
    "mood": "documentary"
  },
  "regions": [
    {
      "region_id": "sky_001",
      "object_type": "sky",
      "confidence": 0.91,
      "mask_path": "masks/000001/sky_001.png",
      "color_state": "dull_gray",
      "recommended_actions": ["natural_blue", "preserve_clouds"]
    }
  ],
  "plans": [],
  "protected_objects": ["faces", "skin", "clothing", "text", "logos"],
  "reject_conditions": []
}
```

## 10. Development Milestones

### Milestone 1: Dataset Cleanup And Baseline

Deliverables:

- clean manifest excluding `._*`
- 300-image representative sample
- color metrics extraction
- first 30 before/after demo edits

Acceptance:

- images load correctly
- sky/person/skin/LED cases represented
- generated plans are understandable and useful

### Milestone 2: Object + Mask Pipeline

Deliverables:

- object prompt list
- Grounding DINO box detection
- SAM2 mask generation
- mask visualization contact sheets
- per-region color metrics

Acceptance:

- sky masks usable on outdoor images
- people/face/skin protection masks usable on portrait/event images
- common false positives logged

### Milestone 3: Plan Generator

Deliverables:

- structured JSON plan
- 3-5 plan variants per image
- GPT Image 2 prompt builder
- local masked preview editor

Acceptance:

- sky-blue prompt is only generated when sky edit is contextually valid
- skin/clothing protection constraints appear in every prompt
- plans differ meaningfully

### Milestone 4: Editing And QA Loop

Deliverables:

- run GPT Image 2 edits for selected images
- run local masked color edits for comparison
- before/after contact sheets
- QA JSON
- human preference UI or review CSV

Acceptance:

- at least 60-70% of selected outputs are usable without manual prompt rewrite
- failure cases are categorized

### Milestone 5: Distilled Student

Deliverables:

- 2,000-5,000 pseudo-labeled samples
- train/val/test split
- student planner model
- evaluation report

Acceptance:

- student predicts object-color actions close to teacher labels
- GPT Image 2 prompts from student output remain usable
- inference cost lower than teacher pipeline

## 11. Evaluation Metrics

Perception:

- object detection precision for sky/person/grass/water/clothing/LED/text
- mask usability score
- false object hallucination rate

Color diagnosis:

- exposure label accuracy
- white-balance label accuracy
- object color-state accuracy
- skin-tone diagnosis accuracy

Editing:

- target object color success rate
- protected region color-delta limit
- identity preservation score
- artifact rejection rate
- human preference win rate

Business/product:

- cost per photo
- latency per photo
- percentage of photos needing human intervention
- percentage of photos with at least one acceptable generated plan

## 12. Cost Control

Strategies:

- Run VLM teachers on compressed images first.
- Use local detection/segmentation where possible.
- Cache every intermediate result.
- Only send selected plans to GPT Image 2.
- Generate local previews before expensive semantic edits.
- Use commercial/expensive teachers only for gold sets and difficult samples.
- Use student planner after enough pseudo-labels exist.

## 13. Key Risks

1. GPT Image 2 may alter non-target objects.
   - Mitigation: use masks, high-fidelity input, strict preservation prompts, QA.

2. VLM may hallucinate objects.
   - Mitigation: require detector/mask confirmation.

3. Sky-blue edits may look fake.
   - Mitigation: context rules for weather, sunset, night, stage, and reflections.

4. Skin may be over-smoothed or recolored poorly.
   - Mitigation: skin protection rules and identity QA.

5. Brand/logo/clothing colors may change.
   - Mitigation: protected-object mask and reject checks.

6. Dataset may be scene-biased.
   - Mitigation: tag scene distribution and add missing scenes deliberately.

## 14. Recommended Immediate Next Step

Start with 100 photos:

- 30 outdoor/sky photos
- 30 event/person photos
- 20 stage/LED/mixed-light photos
- 20 difficult dark/overexposed photos

Build:

```text
clean manifest
-> region detection/masks
-> per-region metrics
-> semantic color plans
-> 3 prompt variants
-> 1-2 edited outputs per image
-> review sheet
```

This will quickly show whether the object-aware color expert is commercially promising before training anything large.
