"""
Baselines for Paper 1.

Three families, all evaluated on the SAME chronological test split and the
SAME leakage-safe window convention as the transformer:

  1. seasonal_naive  — persistence at lag 24 h (h<=24) / 168 h (h>24).
                       No parameters; reference floor in the skill score.
  2. DLinear         — Zeng et al. 2023, "Are Transformers Effective for
                       Time Series Forecasting?"  Decomposition + linear map.
                       Trained per-target (demand, generation) on flattened
                       lookback windows.  Cheap, strong short-horizon baseline.
  3. LightGBM        — gradient-boosted trees on engineered tabular features
                       (calendar + weather + lag/rolling statistics of the
                       target + building features).  Adds exogenous inputs a
                       pure time-series model cannot see; the fairest test of
                       whether the multimodal transformer adds value over a
                       strong feature-engineered tabular model.

Every baseline returns predictions as (M, H) arrays in PHYSICAL units, aligned
row-for-row with src.evaluate.predict_split, so the same metric code applies.

NOTE: LightGBM and the DLinear torch module are imported lazily so that the
numpy-only smoke tests can import this module without heavy deps present.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import DEMAND_CH, GEN_CH
from src.evaluate import _invert


# ─── shared feature extraction from the aligned tensor ───────────────────────

def _split_dataset(dm, split: str):
    return dm.test_ds if split == "test" else dm.train_ds


def _target_series(ds, n, a, h):
    """Physical-unit target at horizon h for node n anchored at a."""
    return ds.data[n, a + h, DEMAND_CH], ds.data[n, a + h, GEN_CH]


# ─── 1. seasonal naive ───────────────────────────────────────────────────────

def seasonal_naive_predict(dm, split: str, horizons: list[int]) -> dict:
    """
    y_hat(t+h) = y(t+h-24*ceil(h/24)).  Returns physical-unit (M, H) arrays.
    (Same rule as evaluate._seasonal_naive_baseline — exposed here as a
     standalone baseline so all three families share one interface.)
    """
    ds = _split_dataset(dm, split)
    data = ds.data
    nd, ad = [], []
    for (n, a) in ds._index:
        rd, ra = [], []
        for h in horizons:
            lag = 24 * max(1, int(np.ceil(h / 24)))
            src = a + h - lag
            if src < 0:
                src = a
            rd.append(data[n, src, DEMAND_CH])
            ra.append(data[n, src, GEN_CH])
        nd.append(rd)
        ad.append(ra)
    return {
        "demand_pred": _invert(dm.demand_scaler, np.array(nd, dtype=np.float32)),
        "gen_pred": _invert(dm.gen_scaler, np.array(ad, dtype=np.float32)),
    }


# ─── 2. DLinear ──────────────────────────────────────────────────────────────

class DLinear:
    """
    Minimal DLinear: for each channel, decompose the lookback window into a
    moving-average trend and a residual, then apply one linear map per
    component to the horizon.  One instance per target.

    Only needs the target's own history (univariate, channel-independent),
    which is the standard DLinear configuration.
    """

    def __init__(self, lookback: int, horizons: list[int], kernel: int = 25,
                 device: str = "cpu"):
        import torch
        import torch.nn as nn

        self.lookback = lookback
        self.horizons = horizons
        self.H = len(horizons)
        self.kernel = kernel
        self.device = device

        class _Net(nn.Module):
            def __init__(self, L, H, k):
                super().__init__()
                self.seq = nn.Sequential(nn.Linear(L, H))          # trend map
                self.res = nn.Sequential(nn.Linear(L, H))          # residual map
                self.k = k
                self.avg = nn.AvgPool1d(kernel_size=k, stride=1)

            def forward(self, x):                                  # x: (B, L)
                pad = (self.k - 1) // 2
                xp = nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
                trend = self.avg(xp).squeeze(1)                    # (B, L)
                resid = x - trend
                return self.seq(trend) + self.res(resid)           # (B, H)

        self.net = _Net(lookback, self.H, kernel).to(device)

    def _windows(self, ds, ch):
        """Return (X, Y) scaled arrays for one target over the dataset index."""
        L = ds.L
        X, Y = [], []
        for (n, a) in ds._index:
            start = a - L + 1
            X.append(ds.data[n, start:a + 1, ch])
            Y.append([ds.data[n, a + h, ch] for h in self.horizons])
        return (np.array(X, dtype=np.float32),
                np.array(Y, dtype=np.float32))

    def fit(self, dm, split: str, ch: int, epochs: int = 60, lr: float = 1e-3,
            batch_size: int = 256):
        import torch

        ds = _split_dataset(dm, split)
        X, Y = self._windows(ds, ch)
        Xt = torch.from_numpy(X).to(self.device)
        Yt = torch.from_numpy(Y).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossf = torch.nn.MSELoss()
        n = Xt.shape[0]
        self.net.train()
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = self.net(Xt[idx])
                loss = lossf(pred, Yt[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * idx.numel()
            if (ep + 1) % 20 == 0:
                print(f"  DLinear ch={ch} ep{ep+1} mse={tot/n:.5f}")
        return self

    def predict(self, dm, split: str, ch: int):
        import torch

        ds = _split_dataset(dm, split)
        X, _ = self._windows(ds, ch)
        self.net.eval()
        preds = []
        Xt = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            for i in range(0, Xt.shape[0], 512):
                preds.append(self.net(Xt[i:i + 512]).cpu().numpy())
        return np.concatenate(preds, 0)          # (M, H) scaled


def dlinear_baseline(dm, cfg, split: str = "test", device: str = "cpu") -> dict:
    """Fit one DLinear per target, return physical-unit (M, H) predictions."""
    horizons = list(cfg["data"]["horizons"])
    L = dm.test_ds.L
    out = {}
    for ch, scaler, name in ((DEMAND_CH, dm.demand_scaler, "demand_pred"),
                             (GEN_CH, dm.gen_scaler, "gen_pred")):
        m = DLinear(L, horizons, device=device)
        m.fit(dm, "train", ch)
        p = m.predict(dm, split, ch)
        out[name] = _invert(scaler, p)
    return out


# ─── 3. LightGBM ─────────────────────────────────────────────────────────────

def _tabular_features(dm, ds, split: str) -> tuple[np.ndarray, dict]:
    """
    Build one feature row per (node, anchor) sample.

    Features (all computed from data at hours <= anchor, plus the known-future
    calendar/weather at the target hour — the same information the transformer
    encoder receives):
      - building features (16, already scaled)
      - calendar: hour-of-day, day-of-week, is_weekend, month, sin/cos of hour
        and doy, for BOTH anchor and each target hour
      - weather at anchor (9) and a simple mean over the lookback window
      - target history stats: last value, mean/std/min/max over lookback,
        lags 1,2,3,24,168 for demand and generation
    """
    data = ds.data
    L = ds.L
    horizons = ds.horizons
    C = data.shape[2]

    # channel layout: 0 demand, 1 gen, 2..10 weather(9), 11..18 calendar(8)
    # (see dataset.py: WEATHER_COLS then 8 calendar channels)
    from src.data.dataset import N_CALENDAR, WEATHER_COLS
    cal_start = 2 + len(WEATHER_COLS)

    def cal_vec(t):
        # the 8 calendar channels are already sin/cos-encoded in the tensor
        return data[0, t, cal_start:cal_start + N_CALENDAR]

    # Precompute a calendar matrix once (shared across nodes) from channel 1 of
    # any node — calendar features are node-independent.
    cal_matrix = data[0, :, cal_start:cal_start + N_CALENDAR]   # (T, 8)

    X, meta = [], {"node": [], "anchor": []}
    for (n, a) in ds._index:
        start = a - L + 1
        win = data[n, start:a + 1, :]                  # (L, C)
        bfeat = ds.building_scaled[n]                  # (16,)
        anch = win[-1]
        feats = [
            *bfeat,
            *anch[2:2 + len(WEATHER_COLS)],            # weather at anchor
            *win[:, 2:2 + len(WEATHER_COLS)].mean(0),  # weather mean over window
            *cal_matrix[a],                            # calendar at anchor
        ]
        # target-hour calendar (known future) for each horizon
        for h in horizons:
            t = min(a + h, cal_matrix.shape[0] - 1)
            feats.extend(cal_matrix[t])
        # target history stats
        d_hist = win[:, DEMAND_CH]
        g_hist = win[:, GEN_CH]
        for hist in (d_hist, g_hist):
            feats.extend([
                hist[-1], hist.mean(), hist.std(),
                hist.min(), hist.max(),
                hist[-2], hist[-3],
                hist[-25] if L > 24 else hist[0],
                hist[0],
            ])
        X.append(feats)
        meta["node"].append(n)
        meta["anchor"].append(a)
    return np.array(X, dtype=np.float32), meta


def lightgbm_baseline(dm, cfg, split: str = "test") -> dict:
    """
    Train one LightGBM regressor per (target, horizon).  Returns physical-unit
    (M, H) predictions aligned with the other baselines.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover - depends on user env
        raise ImportError(
            "LightGBM baseline requires `pip install lightgbm`."
        ) from e

    horizons = list(cfg["data"]["horizons"])
    tr = dm.train_ds
    te = dm.test_ds
    Xtr, _ = _tabular_features(dm, tr, "train")
    Xte, _ = _tabular_features(dm, te, "test")

    # scaled targets, then invert with the scalers (same as the NN path)
    def targets(ds):
        d = np.array([[ds.data[n, a + h, DEMAND_CH] for h in horizons]
                      for (n, a) in ds._index], dtype=np.float32)
        g = np.array([[ds.data[n, a + h, GEN_CH] for h in horizons]
                      for (n, a) in ds._index], dtype=np.float32)
        return d, g

    Ytr_d, Ytr_g = targets(tr)

    out = {}
    for name, Ytr, scaler in (("demand_pred", Ytr_d, dm.demand_scaler),
                              ("gen_pred", Ytr_g, dm.gen_scaler)):
        cols = []
        for hi in range(len(horizons)):
            reg = lgb.LGBMRegressor(
                n_estimators=800, learning_rate=0.05, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                n_jobs=-1, verbose=-1,
            )
            reg.fit(Xtr, Ytr[:, hi])
            cols.append(reg.predict(Xte))
        pred_scaled = np.stack(cols, axis=1)          # (M, H)
        out[name] = _invert(scaler, pred_scaled)
    return out


# ─── unified entry point ─────────────────────────────────────────────────────

def run_baselines(dm, cfg, split: str = "test", device: str = "cpu",
                  which: tuple = ("seasonal_naive", "dlinear", "lightgbm")) -> dict:
    """
    Run the requested baselines and return {name: {demand_pred, gen_pred}}.
    Each prediction array is (M, H) in physical units, aligned with the NN's
    test predictions for a like-for-like metric comparison.
    """
    out = {}
    if "seasonal_naive" in which:
        out["seasonal_naive"] = seasonal_naive_predict(dm, split, list(cfg["data"]["horizons"]))
    if "dlinear" in which:
        out["dlinear"] = dlinear_baseline(dm, cfg, split, device=device)
    if "lightgbm" in which:
        out["lightgbm"] = lightgbm_baseline(dm, cfg, split)
    return out
