"""
Shared utilities: seeding, metrics, logging helpers.
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


# ─── reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def smoke_test_config(cfg: dict) -> dict:
    """Overlay the smoke_test block onto the full config for CPU dry-runs."""
    import copy
    c = copy.deepcopy(cfg)
    st = c.get("smoke_test", {})
    if st:
        c["data"]["n_customers_limit"] = st.get("n_customers", 20)
        c["data"]["lookback_hours"] = st.get("lookback_hours", 48)
        c["training"]["epochs"] = st.get("epochs", 2)
        c["training"]["batch_size"] = st.get("batch_size", 8)
        c["model"]["d_model"] = st.get("d_model", 32)
        c["model"]["n_layers"] = st.get("n_layers", 1)
        c["model"]["gat_in"] = st.get("d_model", 32)
        c["model"]["gat_hidden"] = st.get("d_model", 32)
        c["model"]["gat_out"] = st.get("d_model", 32)
        c["model"]["fusion_d_model"] = st.get("d_model", 32)
    return c


# ─── metrics ──────────────────────────────────────────────────────────────────

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    mask = true > eps
    return float(np.mean(np.abs((pred[mask] - true[mask]) / (true[mask] + eps)))) * 100.0


def skill_score(pred: np.ndarray, true: np.ndarray, baseline: np.ndarray) -> float:
    """
    Skill score relative to a naive baseline (e.g., seasonal naive).
    SS = 1 - RMSE(model) / RMSE(baseline).
    Positive = better than baseline; 0 = same; negative = worse.
    """
    rmse_model = rmse(pred, true)
    rmse_base = rmse(baseline, true)
    if rmse_base < 1e-10:
        return 0.0
    return float(1.0 - rmse_model / rmse_base)


def compute_metrics(pred: np.ndarray, true: np.ndarray,
                    baseline: np.ndarray | None = None) -> dict:
    m = {
        "MAE": mae(pred, true),
        "RMSE": rmse(pred, true),
        "MAPE": mape(pred, true),
    }
    if baseline is not None:
        m["SkillScore"] = skill_score(pred, true, baseline)
    return m


# ─── paths ────────────────────────────────────────────────────────────────────

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
