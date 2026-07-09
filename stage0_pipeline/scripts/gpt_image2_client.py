"""GPT Image 2 client for API易 (apiyi.com) OpenAI-compatible endpoints.

Stdlib only (urllib) so no extra deps. The API key is read from
secrets/api.local.json (gitignored) or env vars; it is never logged.

Docs: https://docs.apiyi.com/api-capabilities/gpt-image-2-all/image-edit

Capabilities:
  list_models()                    -> connectivity + available models
  edit_image(image_path, prompt)   -> semantic edit, returns PNG bytes

Usage:
  ../.venv/bin/python scripts/gpt_image2_client.py --check
  ../.venv/bin/python scripts/gpt_image2_client.py --edit <image> --prompt "..."
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import ssl
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS = PROJECT_ROOT / "secrets" / "api.local.json"


def load_secrets() -> dict:
    if SECRETS.exists():
        with open(SECRETS, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "base_url": os.environ.get("APIYI_BASE_URL", os.environ.get("FUNAI_BASE_URL", "https://api.apiyi.com")),
        "api_key": os.environ.get("APIYI_API_KEY", os.environ.get("FUNAI_API_KEY", "")),
        "image_model": os.environ.get("APIYI_IMAGE_MODEL", os.environ.get("FUNAI_IMAGE_MODEL", "gpt-image-2-all")),
        "response_format": os.environ.get("APIYI_RESPONSE_FORMAT", "url"),
    }


def _mask_key(k: str) -> str:
    return (k[:6] + "..." + k[-4:]) if len(k) > 12 else "***"


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _req(url: str, headers: dict, data: bytes | None = None, method: str = "GET") -> tuple[int, bytes]:
    ctx = ssl.create_default_context()
    headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json", **headers}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=300) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def list_models(cfg: dict) -> tuple[int, list[str], str]:
    url = cfg["base_url"].rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    status, body = _req(url, headers)
    try:
        parsed = json.loads(body)
        ids = [m.get("id", "") for m in parsed.get("data", [])]
        return status, ids, ""
    except Exception:
        return status, [], body.decode("utf-8", "replace")[:500]


def _multipart(fields: dict[str, str], files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "----chroma" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, filename, content in files:
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        out += f"Content-Type: {ctype}\r\n\r\n".encode()
        out += content + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), boundary


def _decode_b64_json(raw: str) -> bytes:
    """API易 b64_json may include a data:image/...;base64, prefix."""
    if raw.startswith("data:"):
        _, _, payload = raw.partition(",")
        raw = payload or raw
    return base64.b64decode(raw)


def _decode_image_response(body: bytes) -> bytes | None:
    """Handle OpenAI images response (b64_json or url) or chat markdown image."""
    try:
        parsed = json.loads(body)
    except Exception:
        return None
    data = parsed.get("data")
    if isinstance(data, list) and data:
        item = data[0]
        if item.get("b64_json"):
            return _decode_b64_json(item["b64_json"])
        if item.get("url"):
            status, img = _req(item["url"], {})
            return img if status == 200 else None
    return None


def _edit_fields(cfg: dict, prompt: str, model: str | None = None) -> dict[str, str]:
    fields = {
        "model": model or cfg.get("image_model", "gpt-image-2-all"),
        "prompt": prompt,
        "n": "1",
    }
    response_format = cfg.get("response_format")
    if response_format:
        fields["response_format"] = response_format
    return fields


def edit_image_via_images_edits(cfg: dict, image_path: str, prompt: str, model: str | None = None) -> bytes | None:
    url = cfg["base_url"].rstrip("/") + "/v1/images/edits"
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    fields = _edit_fields(cfg, prompt, model)
    body, boundary = _multipart(fields, [("image", Path(image_path).name, img_bytes)])
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    status, resp = _req(url, headers, data=body, method="POST")
    if status != 200:
        raise RuntimeError(f"images/edits HTTP {status}: {resp.decode('utf-8','replace')[:400]}")
    img = _decode_image_response(resp)
    if img is None:
        raise RuntimeError(f"images/edits: could not decode image from response: {resp[:300]!r}")
    return img


_COLOR_REF_PROMPT = (
    "修改图1，把图1的整体色调和色彩风格对齐到图2的参考风格。"
    "保持图1的人物结构、构图、皮肤质感和所有文字/logo完全不变，"
    "只调整颜色、曝光和局部对比，不要改变画面内容。"
)


def edit_with_reference(cfg: dict, target_path: str, reference_path: str, prompt: str,
                        model: str | None = None) -> bytes | None:
    """Regrade target (图1) using reference look (图2). Sends both images."""
    url = cfg["base_url"].rstrip("/") + "/v1/images/edits"
    with open(target_path, "rb") as f:
        tgt = f.read()
    with open(reference_path, "rb") as f:
        ref = f.read()
    fields = _edit_fields(cfg, prompt or _COLOR_REF_PROMPT, model)
    # API易: repeat the same `image` field for multi-image fusion (图1/图2 order).
    files = [
        ("image", Path(target_path).name, tgt),
        ("image", Path(reference_path).name, ref),
    ]
    body, boundary = _multipart(fields, files)
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    status, resp = _req(url, headers, data=body, method="POST")
    if status != 200:
        raise RuntimeError(f"images/edits HTTP {status}: {resp.decode('utf-8','replace')[:400]}")
    img = _decode_image_response(resp)
    if img is None:
        raise RuntimeError(f"images/edits: could not decode image: {resp[:300]!r}")
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="list models (connectivity test)")
    ap.add_argument("--edit", help="image path to edit")
    ap.add_argument("--ref", help="reference look image (enables 仿色 transfer)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_secrets()
    print(f"endpoint={cfg['base_url']} key={_mask_key(cfg.get('api_key',''))} model={cfg.get('image_model')}")

    if args.check:
        status, ids, err = list_models(cfg)
        print(f"GET /v1/models -> HTTP {status}, {len(ids)} models")
        img_like = [m for m in ids if "image" in m.lower() or "dall" in m.lower() or "flux" in m.lower()]
        if img_like:
            print("image-capable models:", ", ".join(sorted(img_like)[:20]))
        elif ids:
            print("sample models:", ", ".join(sorted(ids)[:20]))
        if err:
            print("body:", err)
        return 0 if status == 200 else 1

    if args.edit:
        stem = Path(args.edit).stem.split("-")[0]
        if args.ref:
            out = args.out or str(PROJECT_ROOT / "outputs" / "color_transfer" / (stem + "_gpt.png"))
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            img = edit_with_reference(cfg, args.edit, args.ref, args.prompt, args.model)
        else:
            out = args.out or str(PROJECT_ROOT / "outputs" / "gpt_smoke" / (stem + "_gpt.png"))
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            img = edit_image_via_images_edits(cfg, args.edit, args.prompt, args.model)
        with open(out, "wb") as f:
            f.write(img)
        print(f"saved -> {out} ({len(img)} bytes)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
