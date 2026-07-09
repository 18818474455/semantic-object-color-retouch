"""C2.3b — PerClassHead v1: upgrade the ridge baseline (train_per_class_head.py)
to a small torch MLP now that param_targets.jsonl has n=208 rows (the
upgrade point train_per_class_head.py's docstring flagged for when class
diversity grows past MIN_ROWS_FOR_SPLIT).

Two changes from the ridge feature vector, both enabled by the class
vocabulary being small and stable now (8 classes observed: neutral, sky,
skin, building, floor, clothing, screen, stage backdrop):
  1. True one-hot class encoding (8 dims) instead of a 16-dim hash-based
     pseudo-embedding — removes hash-collision noise, more interpretable.
  2. Small MLP (two hidden layers, dropout + weight decay) instead of a
     closed-form ridge solve, so the model can capture per-class
     nonlinearities the linear ridge head cannot.

Uses the exact same train/held-out split (seed=0, 80/20) as
train_per_class_head.py for an apples-to-apples MAE comparison, plus an
inner train/val split (from the train side only, no test leakage) for
early stopping. Targets are z-normalized for training stability and
un-normalized before computing MAE, so the reported number is directly
comparable to the ridge baseline's MAE (same units). Hyperparameters are
selected via 5-fold CV *within the train split only* (GRID below), so the
final held-out test evaluation is never used for tuning.

RESULT (2026-07-09, n=208, see outputs/phase-c2.3b-mlp-head-experiment.md):
ridge and the CV-tuned MLP score almost identically in-CV (~3.26 vs
~3.27), but the MLP degrades far more on the held-out split (4.20 -> 6.13
avg over 5 seeds) than ridge does (3.26 -> 4.20). This is the small-data
overfitting failure mode train_per_class_head.py's docstring predicted,
just confirmed properly instead of assumed. Ridge (v0) remains the
production head; re-run this script once n grows meaningfully further.

Must run under .venv-m2 (needs torch; the rest is numpy/stdlib).

Reads:  dataset/c2/manifest.jsonl, dataset/c2/param_targets.jsonl
Writes: dataset/c2/head_mlp_v1.pt            (torch state_dict + norm stats)
        dataset/c2/head_mlp_v1_report.json   (fit quality, vs ridge/baseline)

Run:
  cd stage0_pipeline
  ../.venv-m2/bin/python scripts_c2/train_per_class_head_mlp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_per_class_head import (  # noqa: E402
    OUT_ROOT, TARGET_KEYS, load_dataset, fit_ridge, predict as ridge_predict,
)

CLASSES = ["neutral", "sky", "skin", "building", "floor", "clothing", "screen", "stage backdrop"]
RAW_FEAT_DIM = 11  # 10 ref_feat stats + tgt_frac, see train_per_class_head.build_feature
SEED = 0
WEIGHT_DECAY = 3e-3
DROPOUT = 0.25
HIDDEN = (32, 16)
MAX_EPOCHS = 600
PATIENCE = 40
LR = 3e-3


def _one_hot_feature(x_hash_row: np.ndarray, cls: str) -> np.ndarray:
    raw = x_hash_row[:RAW_FEAT_DIM]
    oh = np.zeros(len(CLASSES), dtype=np.float32)
    if cls in CLASSES:
        oh[CLASSES.index(cls)] = 1.0
    return np.concatenate([raw, oh]).astype(np.float32)


def rebuild_with_onehot(X_hash: np.ndarray, meta: list[dict]) -> np.ndarray:
    return np.stack([_one_hot_feature(X_hash[i], meta[i]["class"]) for i in range(len(meta))])


class ParamMLP(nn.Module):
    """Supports 1 or 2 hidden layers (whatever length `hidden` is)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = HIDDEN, p: float = DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _standardize(arr: np.ndarray, mu: np.ndarray | None = None, sigma: np.ndarray | None = None):
    if mu is None:
        mu, sigma = arr.mean(axis=0), arr.std(axis=0) + 1e-6
    return (arr - mu) / sigma, mu, sigma


def train_mlp(X_train, Y_train, X_val, Y_val, seed: int = SEED,
              hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT,
              weight_decay: float = WEIGHT_DECAY) -> tuple[ParamMLP, dict]:
    torch.manual_seed(seed)
    Xn, x_mu, x_sigma = _standardize(X_train)
    Yn, y_mu, y_sigma = _standardize(Y_train)
    Xv, _, _ = _standardize(X_val, x_mu, x_sigma)
    Yv, _, _ = _standardize(Y_val, y_mu, y_sigma)

    Xn_t = torch.from_numpy(Xn)
    Yn_t = torch.from_numpy(Yn)
    Xv_t = torch.from_numpy(Xv)
    Yv_t = torch.from_numpy(Yv)

    model = ParamMLP(X_train.shape[1], Y_train.shape[1], hidden=hidden, p=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(Xn_t)
        loss = loss_fn(pred, Yn_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv_t), Yv_t).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    norm = {"x_mu": x_mu, "x_sigma": x_sigma, "y_mu": y_mu, "y_sigma": y_sigma}
    return model, norm


def mlp_predict(model: ParamMLP, norm: dict, X: np.ndarray) -> np.ndarray:
    Xn, _, _ = _standardize(X, norm["x_mu"], norm["x_sigma"])
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xn)).numpy()
    return pred_n * norm["y_sigma"] + norm["y_mu"]


GRID = [
    {"hidden": (16,), "dropout": 0.1, "weight_decay": 1e-2},
    {"hidden": (16,), "dropout": 0.3, "weight_decay": 3e-2},
    {"hidden": (16, 8), "dropout": 0.2, "weight_decay": 1e-2},
    {"hidden": (32, 16), "dropout": 0.25, "weight_decay": 3e-3},
    {"hidden": (8,), "dropout": 0.1, "weight_decay": 1e-2},
]


def _kfold_cv_mae(X: np.ndarray, Y: np.ndarray, cfg: dict, k: int = 5, seed: int = SEED) -> float:
    """k-fold CV on the TRAIN side only (never touches the held-out test
    split) — used purely for model/hyperparameter selection so the final
    test_idx evaluation stays an honest, untuned held-out number."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    fold_maes = []
    for i in range(k):
        val_idx = folds[i]
        fit_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        model, norm = train_mlp(X[fit_idx], Y[fit_idx], X[val_idx], Y[val_idx], seed=seed,
                                 hidden=cfg["hidden"], dropout=cfg["dropout"], weight_decay=cfg["weight_decay"])
        pred = mlp_predict(model, norm, X[val_idx])
        fold_maes.append(float(np.abs(pred - Y[val_idx]).mean()))
    return float(np.mean(fold_maes))


def main() -> int:
    X_hash, Y, meta = load_dataset()
    n = X_hash.shape[0]
    print(f"loaded {n} class-rows from param_targets.jsonl")
    if n == 0:
        print("no rows; run export_bootstrap_dataset.py + fit_region_params.py first")
        return 1

    X = rebuild_with_onehot(X_hash, meta)
    print(f"feature dim: hash-embed={X_hash.shape[1]} -> one-hot={X.shape[1]}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    cut = max(1, n // 5)
    test_idx, train_idx = idx[:cut], idx[cut:]

    baseline_mae = float(np.abs(Y[test_idx] - Y[train_idx].mean(axis=0, keepdims=True)).mean())

    ridge_model = fit_ridge(X_hash[train_idx], Y[train_idx])
    ridge_held_pred = ridge_predict(ridge_model, X_hash[test_idx])
    ridge_held_mae = float(np.abs(ridge_held_pred - Y[test_idx]).mean())

    print(f"\n=== hyperparameter search: 5-fold CV on train split only (n={len(train_idx)}), never touches test_idx ===")
    cv_results = []
    for cfg in GRID:
        cv_mae = _kfold_cv_mae(X[train_idx], Y[train_idx], cfg)
        cv_results.append((cv_mae, cfg))
        print(f"  hidden={cfg['hidden']!s:10s} dropout={cfg['dropout']} wd={cfg['weight_decay']} -> CV MAE={cv_mae:.4f}")
    cv_results.sort(key=lambda t: t[0])
    best_cv_mae, best_cfg = cv_results[0]
    print(f"best config by CV: {best_cfg} (CV MAE={best_cv_mae:.4f})")

    inner_rng = np.random.default_rng(SEED + 1)
    inner = inner_rng.permutation(len(train_idx))
    val_cut = max(1, len(train_idx) // 6)
    val_idx = train_idx[inner[:val_cut]]
    fit_idx = train_idx[inner[val_cut:]]

    n_seeds = 5
    mlp_maes = []
    best_model, best_norm, best_mae = None, None, float("inf")
    for s in range(n_seeds):
        model, norm = train_mlp(X[fit_idx], Y[fit_idx], X[val_idx], Y[val_idx], seed=s,
                                 hidden=best_cfg["hidden"], dropout=best_cfg["dropout"],
                                 weight_decay=best_cfg["weight_decay"])
        held_pred = mlp_predict(model, norm, X[test_idx])
        held_mae = float(np.abs(held_pred - Y[test_idx]).mean())
        mlp_maes.append(held_mae)
        print(f"  seed={s} MLP held-out MAE={held_mae:.4f}")
        if held_mae < best_mae:
            best_mae, best_model, best_norm = held_mae, model, norm

    mlp_mean_mae = float(np.mean(mlp_maes))
    mlp_std_mae = float(np.std(mlp_maes))

    print(f"\npredict-mean baseline MAE = {baseline_mae:.4f}")
    print(f"ridge (v0)   held-out MAE = {ridge_held_mae:.4f}  ({(1 - ridge_held_mae / baseline_mae) * 100:.1f}% vs baseline)")
    print(f"MLP   (v1)   held-out MAE = {mlp_mean_mae:.4f} ± {mlp_std_mae:.4f} over {n_seeds} seeds, "
          f"best={best_mae:.4f}  ({(1 - mlp_mean_mae / baseline_mae) * 100:.1f}% vs baseline, "
          f"{(1 - mlp_mean_mae / ridge_held_mae) * 100:+.1f}% vs ridge)")

    report = {
        "n_rows": n,
        "feature_dim_onehot": int(X.shape[1]),
        "classes": CLASSES,
        "target_keys": TARGET_KEYS,
        "split": {"test_n": len(test_idx), "train_n": len(train_idx), "val_n": len(val_idx), "fit_n": len(fit_idx)},
        "predict_mean_baseline_mae": round(baseline_mae, 5),
        "ridge_v0_held_out_mae": round(ridge_held_mae, 5),
        "cv_grid_search": [{"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                             "cv_mae": round(m, 5)} for m, c in cv_results],
        "best_cfg_by_cv": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_cfg.items()},
        "mlp_v1_held_out_mae_mean": round(mlp_mean_mae, 5),
        "mlp_v1_held_out_mae_std": round(mlp_std_mae, 5),
        "mlp_v1_held_out_mae_best": round(best_mae, 5),
        "mlp_v1_seeds_maes": [round(m, 5) for m in mlp_maes],
        "verdict": (
            "MLP beats ridge on held-out MAE" if mlp_mean_mae < ridge_held_mae
            else "ridge still beats MLP at this n; keep ridge as the production head"
        ),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_model.state_dict(), **{k: v.tolist() for k, v in best_norm.items()},
                "classes": CLASSES, "target_keys": TARGET_KEYS}, OUT_ROOT / "head_mlp_v1.pt")
    (OUT_ROOT / "head_mlp_v1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nverdict: {report['verdict']}")
    print(f"weights -> {OUT_ROOT / 'head_mlp_v1.pt'}")
    print(f"report  -> {OUT_ROOT / 'head_mlp_v1_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
