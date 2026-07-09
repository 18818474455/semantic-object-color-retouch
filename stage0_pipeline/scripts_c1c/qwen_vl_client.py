"""C1c — minimal Qwen-VL chat client via API易's OpenAI-compatible endpoint.

Separate from gpt_image2_client.py (which targets the images/edits endpoint
for gpt-image-2-all). This one is /v1/chat/completions with image_url
content parts, used to ask a real VLM to judge whether a detected region
actually matches its label (e.g. "is this really sky?").

API易 proxies Alibaba's hosted Qwen3-VL tiers (qwen3-vl-plus/flash/...), not
the exact self-hosted Apache-2.0 open weights named in the plan
(Qwen3-VL-8B-Instruct) — see outputs/phase-c2-reference-self-distill-design.md
§9 and the v3.1 addendum §2 for why a hosted tier is used for this
validation experiment instead of self-deploying (16GB local RAM, no MPS in
the current torch build, an 8B VLM would be impractically slow on CPU).

Stdlib only (urllib), same secrets file as gpt_image2_client.py.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gpt_image2_client import load_secrets, _mask_key  # noqa: E402

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _image_data_uri(path: str | Path) -> str:
    data = Path(path).read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def vl_model(cfg: dict) -> str:
    return cfg.get("vl_model") or "qwen3-vl-plus"


def chat_vision(cfg: dict, prompt: str, image_paths: list[str | Path],
                model: str | None = None, max_tokens: int = 300,
                temperature: float = 0.0) -> str:
    """Send one user turn with text + N images, return the text reply."""
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_data_uri(p)}})

    body = json.dumps({
        "model": model or vl_model(cfg),
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    url = cfg["base_url"].rstrip("/") + "/v1/chat/completions"
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        raise RuntimeError(f"chat/completions HTTP {e.code}: {raw[:500]!r}") from None

    parsed = json.loads(raw)
    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError(f"chat/completions: no choices in response: {parsed}")
    return choices[0]["message"]["content"]


def main() -> int:
    cfg = load_secrets()
    print(f"endpoint={cfg['base_url']} key={_mask_key(cfg.get('api_key',''))} vl_model={vl_model(cfg)}")
    img = sys.argv[1] if len(sys.argv) > 1 else None
    if not img:
        print("usage: qwen_vl_client.py <image_path> [prompt]")
        return 1
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in one sentence."
    reply = chat_vision(cfg, prompt, [img])
    print("reply:", reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
