# Photo Retouching Teacher Model Plan

## Goal

Build a low-cost "visual eye" for photo retouching. The first student model should not be a general VLM. It should be a focused photo-diagnosis model that outputs structured labels for tone, color, subject, composition, and edit direction.

Pipeline:

```text
photo
-> teacher model ensemble
-> structured diagnosis JSON
-> distilled student model
-> DS reasoning / prompt generation
-> GPT Image 2 edit
```

## Recommended Teacher Stack

### Tier 1: Main Vision-Language Teachers

Use these to generate rich natural-language and JSON diagnosis labels.

1. Qwen2.5-VL / Qwen-VL family
   - Role: best first open-source teacher for Chinese/English photo understanding.
   - Strengths: strong image understanding, object/scene reasoning, OCR, Chinese prompts, local deployment options.
   - Use for: scene type, subject, lighting, color cast, composition critique, retouching direction.

2. InternVL family
   - Role: second open-source teacher for cross-checking labels.
   - Strengths: strong general VLM performance, good visual detail reasoning, large model range.
   - Use for: independent diagnosis, disagreement detection, confidence scoring.

3. MiniCPM-V family
   - Role: smaller/cheaper teacher and future student candidate.
   - Strengths: lightweight deployment, decent multimodal reasoning.
   - Use for: fast batch labeling, edge/mobile feasibility tests.

4. Commercial teacher option: OpenAI vision-capable model
   - Role: high-quality teacher for a smaller, cleaner golden set.
   - Strengths: reliable instruction following and structured outputs.
   - Use for: 500-2000 high-quality verified labels, evaluation set, prompt-quality baseline.

### Tier 2: Aesthetic / Image Quality Teachers

Use these for scores that VLMs often describe inconsistently.

1. LAION aesthetic predictor
   - Role: general aesthetic score.
   - Output: aesthetic_score 0-10.

2. NIMA / MUSIQ / CLIP-IQA style models
   - Role: image quality and aesthetic assessment.
   - Output: technical_quality_score, aesthetic_score, sharpness/quality estimates.

3. Classical image metrics
   - Role: cheap deterministic labels.
   - Output: exposure histogram, contrast, saturation, white-balance estimate, blur score, noise estimate.

### Tier 3: Subject / Composition Tools

Use these to make composition labels concrete.

1. Segment Anything 2
   - Role: subject masks and region separation.
   - Output: main_subject_mask, background regions, skin/body/object regions when possible.

2. Grounding DINO / OWL-style open-vocabulary detection
   - Role: locate people, faces, products, food, sky, horizon, clutter.
   - Output: bounding boxes and object tags.

3. DINOv2 / SigLIP / CLIP embeddings
   - Role: image clustering, duplicate detection, style grouping.
   - Output: embedding vectors for data curation and retrieval.

## Recommended First Teacher Setup

Start with:

```text
Qwen2.5-VL or InternVL
+ classical image metrics
+ LAION/NIMA-style aesthetic scoring
+ optional SAM2 for subject masks
```

Do not start by training a full VLM. Start by creating a strong pseudo-labeled dataset.

## First Student Model

Best first target:

```text
image -> retouch_diagnosis.json
```

Possible student choices:

1. CLIP/SigLIP image encoder + small MLP heads
   - Fastest and cheapest.
   - Good for classification and scores.
   - Output: scene, exposure label, white balance label, style label, quality scores.

2. Small VLM fine-tune, such as Qwen2.5-VL small / InternVL small / MiniCPM-V
   - Better if you need natural-language diagnosis.
   - More expensive than encoder-head training.

3. Hybrid
   - Encoder-head model outputs structured tags.
   - DS turns those tags into final retouch prompt.
   - This is the recommended first product version.

## Label Schema V1

Each image should produce:

```json
{
  "scene_type": "portrait | landscape | street | food | product | indoor | night | wedding | travel | other",
  "main_subject": "short description",
  "subject_location": "center | left | right | upper | lower | off_center",
  "exposure": {
    "label": "underexposed | normal | overexposed | high_contrast | flat",
    "severity": 0.0
  },
  "white_balance": {
    "label": "neutral | too_warm | too_cool | green_cast | magenta_cast | mixed_light",
    "severity": 0.0
  },
  "skin_tone": {
    "present": true,
    "label": "natural | too_yellow | too_red | too_gray | uneven | not_applicable"
  },
  "color_palette": {
    "dominant_colors": ["green", "yellow", "blue"],
    "saturation": "low | natural | high | oversaturated"
  },
  "composition": {
    "label": "strong | acceptable | cluttered | subject_too_small | tilted | too_empty",
    "crop_suggestion": "none | tighter | vertical | horizontal | straighten | expand"
  },
  "aesthetic_score": 0.0,
  "technical_quality_score": 0.0,
  "recommended_edit_style": "natural_clean | cinematic | film | bright_airy | commercial_clean | moody | japanese_light | black_white",
  "edit_prompt_brief": "one sentence retouching direction",
  "preserve_constraints": [
    "preserve identity",
    "preserve clothing",
    "preserve background structure"
  ],
  "risk_flags": [
    "face_identity_sensitive",
    "text_or_logo_present",
    "hands_visible",
    "low_resolution"
  ]
}
```

## Training Data Strategy

1. Put all source photos into one input directory.
2. Remove duplicates and near-duplicates using CLIP/SigLIP/DINO embeddings.
3. Cluster images by scene and style.
4. Sample a balanced subset for high-quality teacher labeling.
5. Generate pseudo-labels with two VLM teachers.
6. Add deterministic image metrics.
7. Flag disagreements between teachers for human review.
8. Manually review 500-2000 important samples.
9. Train the first student model on structured JSON targets.
10. Evaluate by comparing the student output to teacher/human labels and by running GPT Image 2 edits.

## Data Size Targets

- 500-1000 photos: prototype and prompt validation.
- 3000-10000 photos: useful first diagnosis classifier.
- 10000-50000 photos: stable vertical retouching diagnosis model.
- 100000+ photos with before/after edits: serious style-specific model.

## First Milestone

Create a dataset like:

```text
dataset/
  images/
    000001.jpg
    000002.jpg
  labels/
    000001.json
    000002.json
  manifest.jsonl
```

Each `manifest.jsonl` row:

```json
{"image":"images/000001.jpg","label":"labels/000001.json","split":"train"}
```

## Practical Recommendation

For the first run, use Qwen2.5-VL or InternVL as the main open-source teacher, use a commercial vision model for a smaller golden evaluation set, and train a CLIP/SigLIP encoder-head student first. This will produce a practical low-cost "eye" faster than full VLM distillation.
