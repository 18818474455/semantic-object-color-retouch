# Color Expert V1 Plan

## Positioning

The first model should be a color diagnosis and retouching-direction expert, not a full photo editor and not a full general VLM.

Core task:

```text
photo -> color diagnosis -> 3-5 color grading plans -> GPT Image 2 edit prompts
```

This model should answer:

- What is wrong with the current color?
- What should be preserved?
- Which color directions are suitable for this image?
- Which direction is safest, which is more stylized, and which is most commercially appealing?
- How should GPT Image 2 be instructed to edit only color and tone?

## Why This Is Better As Step One

Color is more controllable than full semantic editing.

The current dataset has enough photos for:

- exposure and brightness distribution learning
- white-balance and color-cast diagnosis
- skin-tone cleanliness diagnosis
- saturation/contrast/palette diagnosis
- scene-specific color direction suggestions
- prompt generation for GPT Image 2 color edits

It is not yet enough to directly learn a top photographer's private style unless there are before/after edited pairs or high-quality human preference labels.

## Teacher Setup

Use a small teacher ensemble, but only for color.

### Main Color Reasoning Teacher

Use one strong VLM to read the image and produce structured color analysis.

Good candidates:

- Qwen-VL / Qwen2.5-VL / Qwen3-VL family
- InternVL family
- OpenAI vision-capable model for a smaller gold set

Teacher output should be JSON, not free-form comments.

### Objective Color Metrics Teacher

Use deterministic image algorithms to give the VLM hard facts:

- RGB / LAB / HSV histogram statistics
- brightness percentile distribution
- highlight clipping and shadow clipping
- average saturation and colorfulness
- gray-world white-balance estimate
- warm/cool and green/magenta cast estimate
- dominant palette
- local face/skin color if face boxes exist

This reduces hallucination and makes the teacher more consistent.

### Optional Aesthetic Teacher

Use aesthetic/IQA models only for scoring, not for final color decisions.

- LAION aesthetic predictor
- MUSIQ / NIMA / CLIP-IQA style models

## Student Model V1

Recommended first student:

```text
image encoder + small prediction heads
```

Good architecture:

```text
SigLIP / CLIP / DINOv2 / ConvNeXt image encoder
-> MLP heads
-> color diagnosis labels and scores
```

The student does not need to write beautiful language. It only needs to output stable structured tags and scores. DS or another language model can turn those tags into the final GPT Image 2 prompt.

## Label Schema V1

Each photo should produce this JSON:

```json
{
  "image_id": "000001",
  "color_diagnosis": {
    "exposure_state": "underexposed | normal | overexposed | high_contrast | flat",
    "exposure_severity": 0.0,
    "white_balance_state": "neutral | too_warm | too_cool | green_cast | magenta_cast | mixed_light",
    "white_balance_severity": 0.0,
    "contrast_state": "flat | natural | harsh | high_key | low_key",
    "contrast_severity": 0.0,
    "saturation_state": "desaturated | natural | oversaturated | color_polluted",
    "saturation_severity": 0.0,
    "skin_tone_state": "not_present | natural | too_yellow | too_red | too_gray | uneven | polluted_by_background",
    "skin_tone_priority": 0.0,
    "dominant_color_problem": "none | yellow_green_cast | red_orange_cast | cyan_blue_cast | mixed_stage_light | dirty_shadow | blown_highlight",
    "overall_color_quality": 0.0
  },
  "preserve": {
    "identity": true,
    "skin_texture": true,
    "clothing_color": true,
    "background_structure": true,
    "original_lighting_direction": true
  },
  "color_plans": [
    {
      "name": "safe_natural_clean",
      "intent": "Clean and realistic color with natural skin tone.",
      "risk_level": "low",
      "best_for": "event delivery, portrait proofing, documentary photos",
      "prompt": "Only adjust color and tone. Keep the original identity, clothing, background, composition, and lighting direction unchanged. Correct the white balance, remove yellow-green color pollution, make skin tone natural and clean, slightly lift shadow detail, protect highlights, keep saturation realistic, and avoid changing facial features or object shapes."
    },
    {
      "name": "cinematic_warm_cool",
      "intent": "Cinematic color separation with warm skin and cooler background.",
      "risk_level": "medium",
      "best_for": "portrait, stage, fashion, social media",
      "prompt": "Only adjust color and tone. Preserve identity, texture, clothes, and composition. Create subtle cinematic color separation: natural warm skin tones, slightly cooler and cleaner background, controlled contrast, deeper but detailed shadows, soft highlights, and polished color without over-saturation."
    },
    {
      "name": "bright_commercial_clean",
      "intent": "Bright, clean, commercial-looking color.",
      "risk_level": "medium",
      "best_for": "commercial, family, school, event albums",
      "prompt": "Only adjust color and tone. Preserve all people and scene details. Make the image brighter and cleaner, neutralize color cast, improve skin clarity, reduce muddy shadows, keep whites clean, keep colors pleasant and realistic, and avoid plastic skin or artificial HDR."
    }
  ],
  "reject_conditions": [
    "If face identity changes",
    "If skin texture becomes plastic",
    "If clothes or logos change color incorrectly",
    "If background objects are regenerated",
    "If the result looks like a filter instead of professional retouching"
  ]
}
```

## Training Strategy

### Stage 0: No Training Baseline

Before training, create a color expert pipeline:

```text
image
-> objective color metrics
-> VLM color diagnosis
-> DS generates 3 color plans
-> GPT Image 2 edits
-> human chooses best result
```

This gives immediate outputs and generates preference data.

### Stage 1: Pseudo-Label Dataset

Use 2000-5000 images first.

For each image:

1. Extract objective color metrics.
2. Ask a strong VLM for color diagnosis JSON.
3. Ask it to generate 3 color grading plans.
4. Run sanity checks on the JSON.
5. Manually review 300-500 difficult samples.

### Stage 2: Train Student Color Diagnoser

Train the student to predict:

- exposure_state
- white_balance_state
- contrast_state
- saturation_state
- skin_tone_state
- dominant_color_problem
- overall_color_quality
- recommended color plan ranking

### Stage 3: Preference Loop

Generate 3 GPT Image 2 edits per selected image.

Human picks:

- best color
- most natural skin
- closest to professional delivery
- reject reasons

Use this to improve plan ranking.

## First Batch Recommendation

Start with 2000 images from the current dataset:

- 1000 random event/person photos
- 500 images with detected faces
- 300 dark or stage-light photos
- 200 bright or high-key photos

This batch is enough to build and test the first color expert.

## What Success Looks Like

V1 is successful if it can reliably say:

- this image is too warm / too green / too dark / too flat
- skin tone is the main priority or not
- safe color plan vs stylized color plan
- what GPT Image 2 should preserve
- which generated result should be rejected

V1 does not need to understand every object in the image. It needs to be accurate about color.
