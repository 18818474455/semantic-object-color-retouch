"""C2.3 — PerClassHead v0: ridge-regression baseline that predicts the
6-number Lab-affine param target (see fit_region_params.py) from a feature
vector built out of the cached style profile + target-side class fraction.

Why ridge, not an MLP, for v0: the bootstrap manifest currently has very few
class-rows (single digits until the source photo volume is mounted and the
full 20/100-image regression set is exported). A closed-form regularized
linear model is the right complexity for that regime — it cannot silently
overfit into a black box the way an under-supervised MLP would, and it is
trivial to inspect (just a weight matrix). Once `param_targets.jsonl` has
enough rows (see MIN_ROWS_FOR_SPLIT below) this script should be upgraded to
a small torch MLP with a proper held-out split; that upgrade path is noted
inline where the swap point is.

Reads:  dataset/c2/manifest.jsonl, dataset/c2/param_targets.jsonl
Writes: dataset/c2/head_ridge_v0.json   (weights + normalization stats)
        dataset/c2/head_ridge_v0_report.json  (fit quality)

Run:
  cd stage0_pipeline
  ../.venv-m2/bin/python scripts_c2/train_per_class_head.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import common

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PIPELINE_ROOT / "dataset" / "c2"

TARGET_KEYS = ["lab_L_scale", "lab_L_shift", "lab_a_scale", "lab_a_shift", "lab_b_scale", "lab_b_shift"]
HASH_EMBED_DIM = 16
RIDGE_LAMBDA = 1.0
# Below this many class-rows, skip the held-out split and report in-sample
# fit only (a real generalization number would be meaningless noise).
MIN_ROWS_FOR_SPLIT = 8


def _class_hash_embed(name: str, dim: int = HASH_EMBED_DIM) -> np.ndarray:
    """Stable pseudo-embedding for open-vocabulary class names. Placeholder
    for a learned embedding table once the class vocabulary stabilizes
    (tracked by the region_provider_v2 detection prompt list)."""
    h = hashlib.sha1(name.encode("utf-8")).digest()
    vec = np.zeros(dim, dtype=np.float32)
    vec[h[0] % dim] = 1.0
    vec[dim - 1 - (h[1] % dim)] += 0.5
    return vec


def build_feature(profile: dict, cls: str, tgt_frac: float) -> np.ndarray | None:
    stats = profile.get("classes", {}).get(cls)
    if stats is None:
        return None
    ref_feat = np.array([
        stats["frac"], stats["l_mean"], stats["l_lo"], stats["l_hi"],
        stats["mean_ab"][0], stats["mean_ab"][1],
        stats["std_ab"][0], stats["std_ab"][1],
        stats["c_std"], stats["h_mean"],
    ], dtype=np.float32)
    return np.concatenate([ref_feat, [tgt_frac], _class_hash_embed(cls)]).astype(np.float32)


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    manifest_rows = {r["sample_id"]: r for r in common.read_jsonl(OUT_ROOT / "manifest.jsonl")}
    target_rows = common.read_jsonl(OUT_ROOT / "param_targets.jsonl")

    X, Y, meta = [], [], []
    for row in target_rows:
        sample = manifest_rows.get(row["sample_id"])
        if sample is None:
            continue
        profile_path = PIPELINE_ROOT / sample["style_profile_path"]
        if not profile_path.is_file():
            continue
        profile = json.loads(profile_path.read_text())
        tgt_frac = sample.get("matched_info", {}).get(row["class"], {}).get("tgt_frac", row["frac"])
        feat = build_feature(profile, row["class"], tgt_frac)
        if feat is None:
            continue
        y = np.array([row[k] for k in TARGET_KEYS], dtype=np.float32)
        X.append(feat)
        Y.append(y)
        meta.append({"sample_id": row["sample_id"], "class": row["class"], "bucket": row["bucket"]})
    if not X:
        return np.zeros((0, 0)), np.zeros((0, 0)), []
    return np.stack(X), np.stack(Y), meta


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = RIDGE_LAMBDA) -> dict:
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
    Xn = (X - mu) / sigma
    Xb = np.concatenate([Xn, np.ones((Xn.shape[0], 1), dtype=np.float32)], axis=1)
    d = Xb.shape[1]
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(d, dtype=np.float32), Xb.T @ Y)
    return {"W": W, "mu": mu, "sigma": sigma}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    Xn = (X - model["mu"]) / model["sigma"]
    Xb = np.concatenate([Xn, np.ones((Xn.shape[0], 1), dtype=np.float32)], axis=1)
    return Xb @ model["W"]


def main() -> int:
    X, Y, meta = load_dataset()
    n = X.shape[0]
    print(f"loaded {n} class-rows from param_targets.jsonl")
    if n == 0:
        print("no rows available; run export_bootstrap_dataset.py then fit_region_params.py first")
        return 1

    model = fit_ridge(X, Y)
    pred = predict(model, X)
    mae_in_sample = float(np.abs(pred - Y).mean())
    baseline_mae = float(np.abs(Y - Y.mean(axis=0, keepdims=True)).mean())

    report = {
        "n_rows": n,
        "feature_dim": int(X.shape[1]),
        "target_keys": TARGET_KEYS,
        "in_sample_mae": round(mae_in_sample, 5),
        "predict_mean_baseline_mae": round(baseline_mae, 5),
        "note": (
            "n < recommended minimum; this is a pipeline smoke test, not a "
            "generalization estimate. Mount the source photo volume, rerun "
            "export_bootstrap_dataset.py (full) + fit_region_params.py, and "
            "re-train once n_rows >= ~40 for a meaningful eval."
            if n < MIN_ROWS_FOR_SPLIT else
            "in-sample only; upgrade to held-out split once class diversity grows."
        ),
        "rows": meta,
    }

    if n >= MIN_ROWS_FOR_SPLIT:
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        cut = max(1, n // 5)
        test_idx, train_idx = idx[:cut], idx[cut:]
        held_model = fit_ridge(X[train_idx], Y[train_idx])
        held_pred = predict(held_model, X[test_idx])
        held_mae = float(np.abs(held_pred - Y[test_idx]).mean())
        report["held_out_mae"] = round(held_mae, 5)
        print(f"held-out MAE={held_mae:.4f} (baseline in-sample MAE={baseline_mae:.4f})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    weights_path = OUT_ROOT / "head_ridge_v0.json"
    weights_path.write_text(json.dumps({
        "W": model["W"].tolist(), "mu": model["mu"].tolist(), "sigma": model["sigma"].tolist(),
        "target_keys": TARGET_KEYS, "feature_dim": int(X.shape[1]),
    }, ensure_ascii=False, indent=2))
    report_path = OUT_ROOT / "head_ridge_v0_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"in-sample MAE={mae_in_sample:.4f} (predict-mean baseline={baseline_mae:.4f})")
    print(f"weights -> {weights_path}")
    print(f"report  -> {report_path}")
    if n < MIN_ROWS_FOR_SPLIT:
        print(f"\nNOTE: {report['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
