"""Plan generator.

Stage 0 uses a transparent rule-based planner as a stand-in for the VLM /
student model. It only ever SELECTS frozen action ids from actions.v1.json and
attaches a strength; it never invents parameters or prompts. When the VLM is
wired in (M3+), replace `select_region_actions` with the model call; everything
downstream (gating, routing, prompt assembly) stays the same.

Output (plans/<image_id>.json):
{ "image_id": ..., "plans": [ {plan}, ... ] }
"""
from __future__ import annotations

from typing import Any

LOCK_BLOCK = (
    "This is a photo retouching task, NOT image generation. "
    "Strictly preserve: person identity, facial features, face shape, skin texture, "
    "pores, expression, age, hair, body shape, pose, clothing design and brand colors, "
    "logos, text and signs, background structure, object shapes, composition, and framing. "
    "Do not add, remove, move, or replace any object. Do not beautify or slim faces/bodies."
)

PRESERVE_DEFAULT = [
    "identity", "skin_texture", "clothing_color", "logos", "text",
    "background_structure", "object_shapes", "composition",
]


def _action(actions_cfg: dict, object_type: str, action_id: str) -> dict | None:
    obj = actions_cfg["objects"].get(object_type)
    if not obj:
        return None
    return obj["actions"].get(action_id)


def _region_by_type(region_metrics: dict, object_type: str) -> dict | None:
    for r in region_metrics["regions"]:
        if r["object_type"] == object_type:
            return r
    return None


# ------------------------------------------------------------------ semantic gates
def sky_blue_allowed(region_metrics: dict, thresholds: dict) -> tuple[bool, str]:
    scene = region_metrics["scene"]
    if scene["type"] in ("sunset", "night_or_dark", "stage_led_mixed"):
        return False, f"scene={scene['type']} blocks forced blue sky"
    sky = _region_by_type(region_metrics, "sky")
    if sky is None:
        return False, "no sky region"
    gates = thresholds["confidence_gates"]
    if sky["confidence"] < gates["region_min_confidence"]:
        return False, f"sky confidence {sky['confidence']} < gate"
    if sky["mask_quality"] < gates["mask_min_quality"]:
        return False, f"sky mask_quality {sky['mask_quality']} < gate"
    cm = sky.get("color_metrics", {})
    h = thresholds["scene_gate_heuristics"]
    if cm.get("saturation_mean", 1.0) >= h["dull_sky_saturation_below"]:
        return False, "sky already saturated (not dull)"
    if cm.get("brightness", 0.0) <= h["dull_sky_brightness_above"]:
        return False, "sky too dark to be daylight blue"
    return True, "dull daytime sky, blue allowed"


# ------------------------------------------------------------- action selection
def select_region_actions(region_metrics: dict, thresholds: dict) -> dict[str, dict]:
    """Return a menu of candidate actions keyed by plan intent.

    Placeholder heuristics. Each value is a list of region_action dicts:
      {object_type, region_id, action_id, strength}
    """
    menu: dict[str, list[dict]] = {
        "safe_natural": [],
        "commercial_clean": [],
        "object_enhancement": [],
        "restore_original_intent": [],
    }
    g = region_metrics["global_metrics"]
    scene = region_metrics["scene"]

    # global white balance from Lab a/b cast magnitude
    cast = abs(g.get("green_magenta_a", 0.0)) + abs(g.get("warm_cool_b", 0.0))
    if cast > 4.0:
        wb = {"object_type": "global", "region_id": "global",
              "action_id": "white_balance_correct", "strength": min(0.6, 0.2 + cast / 30.0)}
        menu["safe_natural"].append(wb)
        menu["commercial_clean"].append(dict(wb))

    # commercial clean: brighten + contrast
    if g.get("brightness", 128.0) < 150:
        menu["commercial_clean"].append(
            {"object_type": "global", "region_id": "global", "action_id": "exposure_lift", "strength": 0.4})
    menu["commercial_clean"].append(
        {"object_type": "global", "region_id": "global", "action_id": "contrast_boost", "strength": 0.3})

    # sky enhancement (gated)
    allowed, _reason = sky_blue_allowed(region_metrics, thresholds)
    sky = _region_by_type(region_metrics, "sky")
    if sky is not None and allowed:
        menu["safe_natural"].append(
            {"object_type": "sky", "region_id": sky["region_id"], "action_id": "slight_clean", "strength": 0.35})
        menu["object_enhancement"].append(
            {"object_type": "sky", "region_id": sky["region_id"], "action_id": "natural_daylight_blue", "strength": 0.7})
        menu["commercial_clean"].append(
            {"object_type": "sky", "region_id": sky["region_id"], "action_id": "natural_daylight_blue", "strength": 0.55})

    # restore original intent for protected scenes
    if scene["type"] in ("stage_led_mixed", "sunset", "night_or_dark"):
        menu["restore_original_intent"].append(
            {"object_type": "global", "region_id": "global", "action_id": "none", "strength": 0.0})

    return menu


# ------------------------------------------------------------------ plan assembly
def _resolve(actions_cfg: dict, ra: dict) -> dict | None:
    a = _action(actions_cfg, ra["object_type"], ra["action_id"])
    if a is None:
        return None
    return {
        "object_type": ra["object_type"],
        "region_id": ra["region_id"],
        "action": f"{ra['object_type']}.{ra['action_id']}",
        "action_id": ra["action_id"],
        "strength": round(float(ra["strength"]), 3),
        "executor": a.get("executor", "local_cpp"),
        "local_params": a.get("local_params", {}),
        "gpt_prompt_fragment": a.get("gpt_prompt_fragment", ""),
    }


def _route(plan_actions: list[dict], routing_metrics: dict, thresholds: dict) -> tuple[str, bool, str]:
    r = thresholds["routing"]
    reasons = []
    force_gpt = any(pa["executor"] == "gpt_image_2" for pa in plan_actions)
    if routing_metrics.get("clip_high_pct", 0.0) > r["clip_high_pct_gpt"]:
        force_gpt = True
        reasons.append(f"clip_high_pct={routing_metrics['clip_high_pct']}>{r['clip_high_pct_gpt']}")
    if routing_metrics.get("clip_low_pct", 0.0) > r["clip_low_pct_gpt"]:
        force_gpt = True
        reasons.append(f"clip_low_pct={routing_metrics['clip_low_pct']}>{r['clip_low_pct_gpt']}")
    if routing_metrics.get("sharpness_proxy", 1.0) < r["sharpness_proxy_gpt_below"]:
        force_gpt = True
        reasons.append(f"sharpness_proxy={routing_metrics['sharpness_proxy']}<{r['sharpness_proxy_gpt_below']}")
    if routing_metrics.get("mixed_light_score", 0.0) > r["mixed_light_score_gpt_above"]:
        force_gpt = True
        reasons.append(f"mixed_light_score={routing_metrics['mixed_light_score']}>{r['mixed_light_score_gpt_above']}")
    executor = "gpt_image_2" if force_gpt else "local_cpp"
    reason = "; ".join(reasons) if reasons else "color/tone only; no severe clipping"
    return executor, force_gpt, reason


def _build_gpt_prompt(plan_actions: list[dict], quality_fragment: str = "") -> str:
    frags = [pa["gpt_prompt_fragment"] for pa in plan_actions if pa["gpt_prompt_fragment"]]
    parts = [LOCK_BLOCK]
    if frags:
        parts.append(" ".join(frags))
    if quality_fragment:
        parts.append(quality_fragment)
    return "\n".join(parts)


def generate_plans(region_metrics: dict, actions_cfg: dict, thresholds: dict) -> dict:
    menu = select_region_actions(region_metrics, thresholds)
    routing_metrics = region_metrics["routing_metrics"]
    plans: list[dict] = []

    plan_specs = [
        ("safe_natural", "Correct white balance and skin, keep sky mostly realistic.", "low"),
        ("commercial_clean", "Brighter, cleaner, more pleasing delivery color.", "low"),
        ("object_enhancement", "Targeted enhancement of key objects (e.g. blue sky).", "medium"),
        ("restore_original_intent", "Preserve stage/sunset/night mood, only fix pollution.", "low"),
    ]

    pid = 0
    for name, summary, risk in plan_specs:
        raw = menu.get(name, [])
        resolved = [x for x in (_resolve(actions_cfg, ra) for ra in raw) if x]
        if not resolved:
            continue
        pid += 1
        executor, two_stage_gpt, route_reason = _route(resolved, routing_metrics, thresholds)
        plan = {
            "plan_id": f"p{pid}",
            "name": name,
            "summary": summary,
            "risk": risk,
            "executor": executor,
            "routing_reason": route_reason,
            "two_stage": bool(two_stage_gpt and any(pa["executor"] == "local_cpp" for pa in resolved)),
            "region_actions": [
                {k: v for k, v in pa.items() if k != "gpt_prompt_fragment"} for pa in resolved
            ],
            "preserve": PRESERVE_DEFAULT,
        }
        if executor == "gpt_image_2":
            plan["gpt_image_prompt"] = _build_gpt_prompt(resolved)
        plans.append(plan)

    # always offer a latitude_recovery GPT plan when clipping is severe
    r = thresholds["routing"]
    if (routing_metrics.get("clip_high_pct", 0.0) > r["clip_high_pct_gpt"]
            or routing_metrics.get("clip_low_pct", 0.0) > r["clip_low_pct_gpt"]):
        pid += 1
        lr = _action(actions_cfg, "global", "latitude_recovery")
        plans.append({
            "plan_id": f"p{pid}",
            "name": "latitude_recovery",
            "summary": "Recover highlight/shadow detail and clarity beyond local latitude.",
            "risk": "medium",
            "executor": "gpt_image_2",
            "routing_reason": f"clip_high={routing_metrics['clip_high_pct']} clip_low={routing_metrics['clip_low_pct']}",
            "two_stage": True,
            "local_pre_correction": {"white_balance": "auto"},
            "region_actions": [
                {"object_type": "global", "region_id": "global",
                 "action": "global.latitude_recovery", "action_id": "latitude_recovery",
                 "strength": lr.get("default_strength", 0.7), "executor": "gpt_image_2",
                 "local_params": {}}
            ],
            "preserve": PRESERVE_DEFAULT,
            "gpt_image_prompt": _build_gpt_prompt(
                [{"gpt_prompt_fragment": lr["gpt_prompt_fragment"]}],
                quality_fragment=lr["gpt_prompt_fragment"]),
        })

    # no-edit fallback guarantee
    if not plans:
        plans.append({
            "plan_id": "p1", "name": "no_edit", "summary": "No confident edit; keep original.",
            "risk": "none", "executor": "local_cpp", "routing_reason": "low confidence / no actionable region",
            "two_stage": False, "region_actions": [], "preserve": PRESERVE_DEFAULT,
        })

    return {"image_id": region_metrics["image_id"], "scene": region_metrics["scene"], "plans": plans}
