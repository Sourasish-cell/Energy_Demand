"""
Training loop for the Multimodal Energy Transformer.

Supports ablation flags so a single entry point can produce the full ablation
table (no-graph, no-cross-attention, single-task).

Usage (see notebooks/03_training.ipynb):

    from src.train import train_model
    model, history = train_model(cfg, dm, device, out_dir="runs/base")
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import AusgridDataModule
from src.models.multimodal_transformer import MultimodalEnergyTransformer
from src.utils import set_seed


# ─── multi-task loss ─────────────────────────────────────────────────────────

def multitask_loss(demand_pred, gen_pred, y_demand, y_gen, cfg):
    """
    Weighted sum of MSE on the (standardised) demand and generation targets.
    Weights come from cfg['training']['demand_weight'] / ['gen_weight'].
    """
    w_d = cfg["training"]["demand_weight"]
    w_g = cfg["training"]["gen_weight"]
    l_d = nn.functional.mse_loss(demand_pred, y_demand)
    l_g = nn.functional.mse_loss(gen_pred, y_gen)
    return w_d * l_d + w_g * l_g, {"loss_demand": l_d.item(), "loss_gen": l_g.item()}


# ─── one epoch ───────────────────────────────────────────────────────────────

def _run_epoch(model, loader, cfg, device, optimizer=None, scheduler=None,
               graph_cfg: dict | None = None):
    """If optimizer is None → evaluation mode (no grad, no step)."""
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    demand_parts, gen_parts, yd_parts, yg_parts = [], [], [], []

    for batch in loader:
        x_b = batch["x_building"].to(device, non_blocking=True)
        x_w = batch["x_weather_now"].to(device, non_blocking=True)
        x_t = batch["x_temporal"].to(device, non_blocking=True)
        sub_t = batch["sub_temporal"].to(device, non_blocking=True)
        ei = batch["sub_edge_index"][0].to(device, non_blocking=True)  # (2,E)
        ew = batch["sub_edge_weight"].to(device, non_blocking=True)
        eb = batch["sub_edge_bias"].to(device, non_blocking=True)
        y_d = batch["y_demand"].to(device, non_blocking=True)
        y_g = batch["y_gen"].to(device, non_blocking=True)

        if getattr(model, "_drop_graph", False):
            eb = _mask_all_cross_edges(eb)

        with torch.set_grad_enabled(train):
            dp, gp = model(x_b, x_w, x_t, sub_t, ei, ew, eb)
            loss, parts = multitask_loss(dp, gp, y_d, y_g, cfg)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"].get("grad_clip", 1.0)
            )
            optimizer.step()

        bs = x_b.shape[0]
        total_loss += loss.item() * bs
        n += bs
        demand_parts.append(dp.detach().cpu().numpy())
        gen_parts.append(gp.detach().cpu().numpy())
        yd_parts.append(y_d.detach().cpu().numpy())
        yg_parts.append(y_g.detach().cpu().numpy())

    if scheduler is not None and train:
        scheduler.step()

    out = {
        "loss": total_loss / max(n, 1),
        "demand_pred": np.concatenate(demand_parts, axis=0),
        "gen_pred": np.concatenate(gen_parts, axis=0),
        "y_demand": np.concatenate(yd_parts, axis=0),
        "y_gen": np.concatenate(yg_parts, axis=0),
    }
    return out


# ─── full training ───────────────────────────────────────────────────────────

def train_model(cfg, dm: AusgridDataModule, device: str,
                out_dir: str | Path = "runs/base",
                graph_ablation: str = "full"):
    """
    graph_ablation ∈ {"full", "no_graph"}.
      full     : GAT over the real star subgraph.
      no_graph : drop edges (self-loops only) → spatial stream sees only itself.
                 Implemented by zeroing the cross-neighbour edges and keeping
                 self-loops, so the spatial encoder degenerates to the root's
                 own temporal embedding.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["training"]["seed"])

    model = MultimodalEnergyTransformer(cfg).to(device)

    if graph_ablation == "no_graph":
        # Zero out all non-self-loop edges by setting their bias to -1e9
        # per batch (handled in _run_epoch via a flag on the model).
        model._drop_graph = True
    else:
        model._drop_graph = False

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = cfg["training"]["epochs"]
    if cfg["training"].get("scheduler") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = None

    train_loader = dm.train_dataloader(batch_size=cfg["training"]["batch_size"])
    # NOTE: with chronological split we validate on the tail of train by
    # re-using the test loader is WRONG; instead carve a validation slice.
    # The data module exposes test_ds for final test; here we early-stop on
    # a held-out slice of the TRAIN period (last 15%).
    val_ds = dm.train_ds
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience = cfg["training"].get("patience", 10)
    bad = 0

    for ep in range(epochs):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, cfg, device, optimizer=opt,
                        scheduler=sched)
        history["train_loss"].append(tr["loss"])
        # Light validation: evaluate loss on a bounded slice of train_ds
        val_metrics = _quick_val_loss(model, dm, cfg, device, frac=0.15)
        history["val_loss"].append(val_metrics)
        print(f"epoch {ep+1}/{epochs}  train={tr['loss']:.4f}  "
              f"val={val_metrics:.4f}  ({time.time()-t0:.1f}s)")

        if val_metrics < best_val - cfg["training"].get("min_delta", 1e-4):
            best_val = val_metrics
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def _quick_val_loss(model, dm, cfg, device, frac: float = 0.15):
    """
    Evaluate MSE loss on a chronological hold-out: the last `frac` of the
    training anchors by *hour*, across all nodes.

    Note: the dataset index is node-major — (node, anchor) pairs with the node
    varying slowest — so simply taking the last `frac` of the index would hold
    out whole nodes spanning the entire time range, not the end of the period.
    We therefore select by anchor value so the hold-out is genuinely the tail of
    the training window.
    """
    from torch.utils.data import DataLoader, Subset
    ds = dm.train_ds
    anchors = np.array([a for (_n, a) in ds._index])
    cutoff = np.quantile(anchors, 1 - frac)
    val_idx = list(np.where(anchors >= cutoff)[0])
    sub = Subset(ds, val_idx)
    loader = DataLoader(sub, batch_size=cfg["training"]["batch_size"],
                        shuffle=False)
    model.eval()
    total, cnt = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_b = batch["x_building"].to(device)
            x_w = batch["x_weather_now"].to(device)
            x_t = batch["x_temporal"].to(device)
            sub_t = batch["sub_temporal"].to(device)
            ei = batch["sub_edge_index"][0].to(device)
            ew = batch["sub_edge_weight"].to(device)
            eb = batch["sub_edge_bias"].to(device)
            y_d = batch["y_demand"].to(device)
            y_g = batch["y_gen"].to(device)
            if getattr(model, "_drop_graph", False):
                eb = _mask_all_cross_edges(eb)
            dp, gp = model(x_b, x_w, x_t, sub_t, ei, ew, eb)
            loss, _ = multitask_loss(dp, gp, y_d, y_g, cfg)
            total += loss.item() * x_b.shape[0]
            cnt += x_b.shape[0]
    return total / max(cnt, 1)


def _mask_all_cross_edges(eb: torch.Tensor) -> torch.Tensor:
    """
    For the no-graph ablation: keep self-loops (bias 0) but mask cross edges.
    In our fixed local topology the self-loops are the LAST (K+1) entries.
    We must know K; infer from length: E = 2K + (K+1) = 3K + 1 → K=(E-1)/3.
    """
    E = eb.shape[-1]
    K = (E - 1) // 3
    out = eb.clone()
    out[:, :2 * K] = -1e9     # mask all non-self-loop edges
    return out
