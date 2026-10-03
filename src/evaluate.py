"""
Evaluation for the Multimodal Energy Transformer.

All metrics are computed in PHYSICAL units (kWh), i.e. after inverting the
train-fit standardisation.  Generation metrics are additionally reported for
the daytime mask only (clear-sky-like hours), because night-time PV is
identically ~0 and otherwise flatters every model's MAPE.

Metric conventions
------------------
  MAE   : mean absolute error (kWh)
  RMSE  : root mean squared error (kWh)
  MAPE  : mean absolute percentage error (%), masked to |y| > eps
  skill : 1 - MSE_model / MSE_seasonal_naive   (higher is better; >0 beats naive)

Usage:
    from src.evaluate import evaluate_model
    results = evaluate_model(model, dm, cfg, device, split="test")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import DEMAND_CH, GEN_CH
from src.utils import compute_metrics


# ─── de-normalisation ────────────────────────────────────────────────────────

def _invert(scaler, arr: np.ndarray) -> np.ndarray:
    """Invert a StandardScaler on a (..., 1)-shaped last axis or flat array."""
    flat = arr.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(arr.shape)


def _metrics(true: np.ndarray, pred: np.ndarray, ref=None) -> dict:
    """
    Thin wrapper over utils.compute_metrics that (a) takes (true, pred) order,
    and (b) returns stable lowercase keys used by the reporting tables.
    """
    m = compute_metrics(pred, true, baseline=ref)
    out = {"mae": m["MAE"], "rmse": m["RMSE"], "mape": m["MAPE"]}
    if "SkillScore" in m:
        out["skill"] = m["SkillScore"]
    else:
        out["skill"] = float("nan")
    return out


# ─── run inference over a split ──────────────────────────────────────────────

@torch.no_grad()
def predict_split(model, dm, cfg, device: str, split: str = "test"):
    """
    Returns dict of numpy arrays in PHYSICAL units:
        demand_pred, gen_pred, y_demand, y_gen   each (M, n_horizons)
        anchor_hour  (M,)  absolute hour index of the last observed step
        node_id      (M,)  building/postcode index
    """
    model.eval()
    ds = dm.test_ds if split == "test" else dm.train_ds
    loader = (dm.test_dataloader(batch_size=cfg["training"]["batch_size"])
              if split == "test"
              else dm.train_dataloader(batch_size=cfg["training"]["batch_size"]))

    dp_all, gp_all, yd_all, yg_all, anchors, nodes = [], [], [], [], [], []
    for batch in loader:
        x_b = batch["x_building"].to(device)
        x_w = batch["x_weather_now"].to(device)
        x_t = batch["x_temporal"].to(device)
        sub_t = batch["sub_temporal"].to(device)
        ei = batch["sub_edge_index"][0].to(device)
        ew = batch["sub_edge_weight"].to(device)
        eb = batch["sub_edge_bias"].to(device)
        if getattr(model, "_drop_graph", False):
            from src.train import _mask_all_cross_edges
            eb = _mask_all_cross_edges(eb)

        dp, gp = model(x_b, x_w, x_t, sub_t, ei, ew, eb)
        dp_all.append(dp.cpu().numpy())
        gp_all.append(gp.cpu().numpy())
        yd_all.append(batch["y_demand"].numpy())
        yg_all.append(batch["y_gen"].numpy())
        meta = batch["meta"].numpy()          # (B, 2) = [node, anchor]
        nodes.append(meta[:, 0])
        anchors.append(meta[:, 1])

    out = {
        "demand_pred": _invert(dm.demand_scaler, np.concatenate(dp_all, 0)),
        "gen_pred": _invert(dm.gen_scaler, np.concatenate(gp_all, 0)),
        "y_demand": _invert(dm.demand_scaler, np.concatenate(yd_all, 0)),
        "y_gen": _invert(dm.gen_scaler, np.concatenate(yg_all, 0)),
        "node_id": np.concatenate(nodes, 0),
        "anchor_hour": np.concatenate(anchors, 0),
    }
    return out


# ─── metrics ─────────────────────────────────────────────────────────────────

def _seasonal_naive_baseline(dm, split: str, horizons) -> dict:
    """
    Seasonal-naive: y_hat(t+h) = y(t+h-24*ceil(h/24)).  Built directly from the
    19-channel aligned tensor so it is exactly the 'same hour, previous day/ week'
    persistence rule.  Returns arrays aligned to predict_split()'s (M, H) layout.
    """
    ds = dm.test_ds if split == "test" else dm.train_ds
    data = ds.data                         # (N, T, C) aligned tensor
    # Built from the *scaled* aligned tensor, then inverted with the same
    # scalers.  Seasonal-naive persistence is invariant under the per-channel
    # affine standardisation, so this equals the rule applied to raw kWh.
    nd, ad = [], []
    for (n, a) in ds._index:
        row_d, row_g = [], []
        for h in horizons:
            lag = 24 * max(1, int(np.ceil(h / 24)))   # 24h for h<=24, 168h for h>24
            src = a + h - lag
            if src < 0:
                src = a                        # fallback: persist last observed
            row_d.append(data[n, src, DEMAND_CH])
            row_g.append(data[n, src, GEN_CH])
        nd.append(row_d)
        ad.append(row_g)
    nd = _invert(dm.demand_scaler, np.array(nd, dtype=np.float32))
    ad = _invert(dm.gen_scaler, np.array(ad, dtype=np.float32))
    return {"demand_naive": nd, "gen_naive": ad}


def evaluate_model(model, dm, cfg, device: str, split: str = "test",
                   night_mask: bool = True) -> dict:
    """
    Compute the full metric table for one model on one split.

    Returns a nested dict:
      {split: {"demand": {h: {...}}, "gen": {h: {...}}, "gen_daytime": {...}}}
    plus an overall (all-horizons) block.
    """
    horizons = list(cfg["data"]["horizons"])
    pred = predict_split(model, dm, cfg, device, split)
    naive = _seasonal_naive_baseline(dm, split, horizons)

    # Daytime mask for generation: hours 07:00–18:00 local at the target time.
    # Target absolute hour = anchor_hour + h; convert to hour-of-day via a
    # midnight offset.  The aligned tensor starts at data.t0 (hour 0 = 00:00).
    hod_pred = (pred["anchor_hour"][:, None] + np.array(horizons)[None, :]) % 24

    results = {"demand": {}, "gen": {}, "gen_daytime": {}, "overall": {}}

    # per-horizon
    for hi, h in enumerate(horizons):
        yd = pred["y_demand"][:, hi]
        dp = pred["demand_pred"][:, hi]
        nd = naive["demand_naive"][:, hi]
        results["demand"][h] = _metrics(yd, dp, ref=nd)

        yg = pred["y_gen"][:, hi]
        gp = pred["gen_pred"][:, hi]
        ng = naive["gen_naive"][:, hi]
        results["gen"][h] = _metrics(yg, gp, ref=ng)

        if night_mask:
            day = (hod_pred[:, hi] >= 7) & (hod_pred[:, hi] <= 18)
            if day.sum() > 0:
                results["gen_daytime"][h] = _metrics(yg[day], gp[day], ref=ng[day])

    # overall (flatten horizons)
    yd = pred["y_demand"].reshape(-1)
    dp = pred["demand_pred"].reshape(-1)
    nd = naive["demand_naive"].reshape(-1)
    yg = pred["y_gen"].reshape(-1)
    gp = pred["gen_pred"].reshape(-1)
    ng = naive["gen_naive"].reshape(-1)
    results["overall"]["demand"] = _metrics(yd, dp, ref=nd)
    results["overall"]["gen"] = _metrics(yg, gp, ref=ng)
    if night_mask:
        day = ((hod_pred.reshape(-1) >= 7) & (hod_pred.reshape(-1) <= 18))
        results["overall"]["gen_daytime"] = _metrics(yg[day], gp[day], ref=ng[day])
    results["split"] = split
    results["n_samples"] = int(pred["y_demand"].shape[0])
    return results


def print_table(results: dict):
    """Human-readable per-horizon table."""
    print(f"\n=== {results['split'].upper()} (n={results['n_samples']}) ===")
    print(f"{'target':<12}{'h':>6}{'MAE':>10}{'RMSE':>10}{'MAPE%':>9}{'skill':>9}")
    for target in ("demand", "gen", "gen_daytime"):
        block = results.get(target, {})
        for h, m in block.items():
            print(f"{target:<12}{h:>6}{m['mae']:>10.3f}{m['rmse']:>10.3f}"
                  f"{m['mape']:>9.2f}{m['skill']:>9.3f}")


def save_results(results: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
