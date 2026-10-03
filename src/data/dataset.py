"""
Ausgrid Solar Home Electricity Data — loader, feature engineering, graph
builder, and leakage-safe subgraph dataset.

Real dataset schema (Ausgrid Solar Home Electricity Data, 2010-2013):
  Columns: Customer | Generator Capacity | Postcode | date |
           Consumption Category | 0:30 | 1:00 | 1:30 | ... | 24:00
  - Customer            : integer ID
  - Generator Capacity  : installed PV size in kW
  - Postcode            : Sydney postcode
  - date                : "dd/mm/yyyy"
  - Consumption Category: "GC" (grid consumption, kWh / 30 min) or
                          "GG" (gross generation, kWh / 30 min)
  - 48 half-hour columns: energy (kWh) in each 30-min slot

Three annual files:
  2010-2011 Solar home electricity data.csv
  2011-2012 Solar home electricity data.csv
  2012-2013 Solar home electricity data.csv

Temporal split (chronological, no shuffling):
  Train: 2010-07-01 → 2012-06-30
  Test : 2012-07-01 → 2013-06-30

Temporal tensor channels (19):
  0 demand_kwh            1 gen_kwh
  2 temperature_2m        3 relative_humidity_2m   4 precipitation
  5 surface_pressure      6 wind_speed_10m         7 shortwave_radiation
  8 direct_radiation      9 diffuse_radiation     10 cloud_cover
  11 hour_sin  12 hour_cos  13 dow_sin  14 dow_cos  15 is_weekend
  16 is_holiday  17 season_sin  18 season_cos

Building features (16): raw PV capacity, log capacity, lat, lon, per-customer
  train-period demand mean/std, and postcode-level aggregates (count, median/
  std demand, PV penetration, mean PV capacity), 4 area-class one-hots, coastal
  flag.  All computed from TRAINING data only — no test-window leakage.

Weather source: Open-Meteo historical archive API (ERA5-backed, keyless),
  Sydney CBD coordinates, hourly, 2010-07-01 → 2013-06-30.

IMPORTANT (leakage design):
  Every training sample is anchored at an hour t.  Its spatial context is a
  STAR SUBGRAPH built from the train-period similarity graph, and every node's
  window ENDS AT t.  A prediction at t therefore never sees neighbour data from
  hours > t.  Edge weights come from the train-period similarity matrix only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler


# ─── known Sydney postcode centroids (subset present in Ausgrid data) ─────────
POSTCODE_COORDS: dict[int, tuple[float, float]] = {
    2000: (-33.8688, 151.2093), 2010: (-33.8876, 151.2155),
    2020: (-33.9360, 151.1646), 2030: (-33.9271, 151.2619),
    2037: (-33.8795, 151.1786), 2040: (-33.8770, 151.1520),
    2044: (-33.9056, 151.1542), 2045: (-33.8717, 151.1359),
    2046: (-33.8647, 151.1062), 2049: (-33.8469, 151.1523),
    2060: (-33.8330, 151.2050), 2065: (-33.8171, 151.1941),
    2066: (-33.8109, 151.1395), 2067: (-33.8020, 151.1703),
    2068: (-33.8050, 151.2219), 2073: (-33.7751, 151.1239),
    2075: (-33.7521, 151.1263), 2077: (-33.7231, 151.0991),
    2080: (-33.6977, 151.0922), 2081: (-33.6826, 151.1109),
    2085: (-33.7692, 151.2334), 2088: (-33.8278, 151.2581),
    2092: (-33.7836, 151.2575), 2093: (-33.7669, 151.2730),
    2094: (-33.7463, 151.2759), 2099: (-33.7315, 151.2891),
    2100: (-33.7603, 151.2809), 2101: (-33.7347, 151.2638),
    2102: (-33.7503, 151.2612), 2103: (-33.7403, 151.2524),
    2107: (-33.6917, 151.3041), 2108: (-33.6509, 151.3218),
    2110: (-33.8326, 151.0921), 2111: (-33.8177, 151.1026),
    2112: (-33.8106, 151.0781), 2113: (-33.7940, 151.1162),
    2114: (-33.8161, 151.0562), 2115: (-33.8011, 151.0387),
    2116: (-33.8338, 151.0197), 2117: (-33.8183, 151.0029),
    2118: (-33.7858, 151.0027), 2119: (-33.7624, 151.0200),
    2120: (-33.7407, 151.0373), 2121: (-33.8037, 151.0847),
    2122: (-33.7995, 151.0647), 2125: (-33.7516, 151.0607),
    2126: (-33.7243, 151.0587), 2127: (-33.8213, 151.1399),
    2128: (-33.8392, 151.1132), 2130: (-33.8822, 151.1327),
    2131: (-33.8986, 151.1261), 2132: (-33.9074, 151.1129),
    2133: (-33.9159, 151.1081), 2134: (-33.8981, 151.0935),
    2135: (-33.8834, 151.0833), 2136: (-33.8908, 151.0663),
    2137: (-33.8735, 151.0897), 2138: (-33.8439, 151.1028),
    2140: (-33.8595, 151.0681), 2141: (-33.8473, 151.0515),
    2142: (-33.8367, 151.0293), 2143: (-33.8742, 151.0328),
    2144: (-33.8605, 151.0152), 2145: (-33.8040, 150.9778),
    2146: (-33.7898, 150.9659), 2147: (-33.7749, 150.9420),
    2148: (-33.7592, 150.9103), 2150: (-33.8077, 151.0001),
    2151: (-33.7927, 150.9806), 2152: (-33.7724, 150.9622),
    2153: (-33.7439, 150.9380), 2154: (-33.7218, 150.9618),
    2155: (-33.6997, 150.9248), 2156: (-33.6818, 150.9635),
    2157: (-33.6630, 151.0169), 2158: (-33.6812, 150.9882),
    2159: (-33.6564, 151.0545), 2160: (-33.8658, 151.0068),
    2161: (-33.8793, 151.0218), 2162: (-33.8984, 151.0121),
    2163: (-33.9205, 151.0101), 2164: (-33.8702, 150.9901),
    2165: (-33.8866, 150.9810), 2166: (-33.9046, 150.9676),
    2167: (-33.9294, 150.9749), 2168: (-33.9201, 150.9379),
    2170: (-33.9550, 150.9352), 2171: (-33.9389, 150.9110),
    2172: (-33.9702, 150.9599), 2173: (-33.9896, 150.9441),
    2174: (-33.9956, 150.9235), 2175: (-33.8570, 150.8936),
    2176: (-33.8789, 150.9087), 2177: (-33.8958, 150.9125),
    2178: (-33.9158, 150.9143), 2179: (-33.9363, 150.8926),
    2195: (-33.9187, 151.0489), 2196: (-33.9317, 151.0573),
    2197: (-33.9430, 151.0451), 2198: (-33.9560, 151.0353),
    2199: (-33.9665, 151.0454), 2200: (-33.9518, 151.0617),
    2203: (-33.9062, 151.1219), 2204: (-33.9210, 151.1291),
    2205: (-33.9362, 151.1317), 2206: (-33.9187, 151.1484),
    2207: (-33.9444, 151.1428), 2208: (-33.9341, 151.1622),
    2209: (-33.9476, 151.1560), 2210: (-33.9644, 151.1479),
    2211: (-33.9773, 151.1269), 2212: (-33.9669, 151.1061),
    2213: (-33.9835, 151.1027), 2214: (-33.9945, 151.1155),
    2216: (-33.9514, 151.1826), 2217: (-33.9648, 151.1776),
    2218: (-33.9776, 151.1680), 2219: (-33.9914, 151.1617),
    2220: (-34.0026, 151.1248), 2221: (-34.0143, 151.1028),
    2222: (-34.0163, 151.0793), 2223: (-34.0282, 151.1118),
    2224: (-34.0288, 151.1337), 2225: (-34.0449, 151.1116),
    2226: (-34.0492, 151.1417), 2227: (-34.0622, 151.1326),
    2228: (-34.0768, 151.1426), 2229: (-34.0517, 151.1564),
    2230: (-34.0420, 151.1688), 2231: (-34.0249, 151.1736),
    2232: (-34.0105, 151.1867), 2233: (-34.0293, 151.1878),
    2234: (-34.0542, 151.2093), 2250: (-33.4222, 151.3416),
    2251: (-33.4503, 151.3723), 2256: (-33.5204, 151.3440),
    2257: (-33.5519, 151.3278), 2258: (-33.3880, 151.4397),
}
_DEFAULT_LAT, _DEFAULT_LON = -33.8688, 151.2093


def _postcode_coords(pc: int) -> tuple[float, float]:
    return POSTCODE_COORDS.get(int(pc), (_DEFAULT_LAT, _DEFAULT_LON))


# ─── channel bookkeeping ─────────────────────────────────────────────────────
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "surface_pressure", "wind_speed_10m",
    "shortwave_radiation", "direct_radiation",
    "diffuse_radiation", "cloud_cover",
]
# Order of channels in the temporal tensor (must match configs temporal_features)
N_CALENDAR = 8   # hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_holiday, season_sin, season_cos
TEMPORAL_CHANNELS = 2 + len(WEATHER_COLS) + N_CALENDAR   # 2 + 9 + 8 = 19
DEMAND_CH = 0
GEN_CH = 1


# ─── NSW public holidays 2010-2013 (real dates; NSW observed) ────────────────
_NSW_HOLIDAYS: set[str] = {
    # 2010
    "2010-01-01", "2010-01-26", "2010-04-02", "2010-04-05", "2010-04-25",
    "2010-06-14", "2010-10-04", "2010-12-25", "2010-12-27", "2010-12-28",
    # 2011
    "2011-01-01", "2011-01-03", "2011-01-26", "2011-04-22", "2011-04-25",
    "2011-04-26", "2011-06-13", "2011-10-03", "2011-12-25", "2011-12-26",
    "2011-12-27",
    # 2012
    "2012-01-01", "2012-01-02", "2012-01-26", "2012-04-06", "2012-04-09",
    "2012-04-25", "2012-06-11", "2012-10-01", "2012-12-25", "2012-12-26",
    # 2013
    "2013-01-01", "2013-01-28", "2013-03-29", "2013-04-01", "2013-04-25",
    "2013-06-10", "2013-10-07", "2013-12-25", "2013-12-26",
}


def calendar_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """Return (N, 8) calendar/holiday/season channels."""
    h = ts.hour.values
    dow = ts.dayofweek.values
    doy = ts.dayofyear.values
    is_weekend = (dow >= 5).astype(np.float32)
    is_holiday = np.array(
        [1.0 if d.strftime("%Y-%m-%d") in _NSW_HOLIDAYS else 0.0
         for d in ts], dtype=np.float32
    )
    season_ang = 2 * np.pi * (doy - 1) / 365.25
    return np.stack([
        np.sin(2 * np.pi * h / 24).astype(np.float32),
        np.cos(2 * np.pi * h / 24).astype(np.float32),
        np.sin(2 * np.pi * dow / 7).astype(np.float32),
        np.cos(2 * np.pi * dow / 7).astype(np.float32),
        is_weekend,
        is_holiday,
        np.sin(season_ang).astype(np.float32),
        np.cos(season_ang).astype(np.float32),
    ], axis=1).astype(np.float32)


# ─── raw loader ──────────────────────────────────────────────────────────────

_AUSGRID_FILES = [
    "2010-2011 Solar home electricity data.csv",
    "2011-2012 Solar home electricity data.csv",
    "2012-2013 Solar home electricity data.csv",
]


def load_ausgrid_raw(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load and concatenate the three Ausgrid annual CSVs, melt the 48 half-hour
    columns, pivot GC/GG, resample to hourly.

    Returns a tidy DataFrame with columns:
      customer_id (int), pv_capacity_kw (float), postcode (int),
      timestamp (datetime64), demand_kwh (float), gen_kwh (float)
    """
    raw_dir = Path(raw_dir)
    frames = []
    for fname in _AUSGRID_FILES:
        fp = raw_dir / fname
        if not fp.exists():
            raise FileNotFoundError(
                f"Expected Ausgrid file not found: {fp}\n"
                "Download the 'Solar home electricity data' ZIP from:\n"
                "  https://www.ausgrid.com.au/Industry/Our-Research/"
                "Data-to-share/Solar-home-electricity-data\n"
                "and unzip the three annual CSVs into data/raw/."
            )
        frames.append(pd.read_csv(fp))
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip() for c in raw.columns]

    required = {"Customer", "Generator Capacity", "Postcode",
                "date", "Consumption Category"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Ausgrid file missing expected columns: {missing}")

    slot_cols = [c for c in raw.columns if re.match(r"^\d{1,2}:\d{2}$", c)]
    if len(slot_cols) != 48:
        raise ValueError(
            f"Expected 48 half-hour columns, found {len(slot_cols)}. "
            f"First few: {slot_cols[:5]}"
        )

    id_cols = ["Customer", "Generator Capacity", "Postcode", "date",
               "Consumption Category"]
    melted = raw.melt(id_vars=id_cols, value_vars=slot_cols,
                      var_name="slot", value_name="kwh")

    melted["slot_clean"] = melted["slot"].str.replace("24:00", "00:00")
    melted["date_adj"] = pd.to_datetime(melted["date"], dayfirst=True,
                                         errors="coerce")
    is_mid = melted["slot"] == "24:00"
    melted.loc[is_mid, "date_adj"] = (
        melted.loc[is_mid, "date_adj"] + pd.Timedelta(days=1)
    )
    melted["timestamp"] = pd.to_datetime(
        melted["date_adj"].dt.strftime("%Y-%m-%d") + " " + melted["slot_clean"],
        errors="coerce",
    )
    melted = melted.dropna(subset=["timestamp"])

    melted = melted.rename(columns={
        "Customer": "customer_id",
        "Generator Capacity": "pv_capacity_kw",
        "Postcode": "postcode",
    })
    melted["Consumption Category"] = melted["Consumption Category"].str.strip()

    gc = (melted[melted["Consumption Category"] == "GC"]
          [["customer_id", "pv_capacity_kw", "postcode", "timestamp", "kwh"]]
          .rename(columns={"kwh": "demand_kwh_30min"}))
    gg = (melted[melted["Consumption Category"] == "GG"]
          [["customer_id", "timestamp", "kwh"]]
          .rename(columns={"kwh": "gen_kwh_30min"}))

    merged = gc.merge(gg, on=["customer_id", "timestamp"], how="inner")
    merged["demand_kwh_30min"] = merged["demand_kwh_30min"].clip(lower=0.0)
    merged["gen_kwh_30min"] = merged["gen_kwh_30min"].clip(lower=0.0)

    merged = merged.set_index("timestamp").sort_index()
    parts = []
    for cust_id, grp in merged.groupby("customer_id"):
        meta = grp[["pv_capacity_kw", "postcode"]].iloc[0]
        h = (grp[["demand_kwh_30min", "gen_kwh_30min"]]
             .resample("1h").sum()
             .rename(columns={"demand_kwh_30min": "demand_kwh",
                              "gen_kwh_30min": "gen_kwh"}))
        h["customer_id"] = int(cust_id)
        h["pv_capacity_kw"] = float(meta["pv_capacity_kw"])
        h["postcode"] = int(meta["postcode"])
        parts.append(h)
    hourly = pd.concat(parts).reset_index()
    hourly = hourly.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    return hourly


# ─── weather ─────────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, start: str, end: str,
                  cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch hourly weather from the Open-Meteo historical archive API
    (ERA5-backed, no API key).  Returns a DataFrame indexed by local timestamp
    with the 9 WEATHER_COLS columns.
    """
    if cache_path is not None and Path(cache_path).exists():
        wdf = pd.read_parquet(cache_path)
        wdf.index = pd.to_datetime(wdf.index)
        return wdf

    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
    except ImportError as e:
        raise ImportError(
            "Weather packages missing. Run: "
            "pip install openmeteo-requests requests-cache retry-requests"
        ) from e

    cache_session = requests_cache.CachedSession(".cache_openmeteo",
                                                 expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat, "longitude": lon,
        "hourly": WEATHER_COLS,
        "start_date": start, "end_date": end,
        "timezone": "Australia/Sydney",
    }
    resp = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )[0].Hourly()

    timestamps = pd.date_range(
        start=pd.to_datetime(resp.Time(), unit="s"),
        end=pd.to_datetime(resp.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=resp.Interval()),
        inclusive="left",
    )
    wdf = pd.DataFrame({"timestamp": timestamps})
    for i, var in enumerate(WEATHER_COLS):
        wdf[var] = resp.Variables(i).ValuesAsNumpy()
    wdf = wdf.set_index("timestamp")
    if cache_path is not None:
        wdf.to_parquet(cache_path)
    return wdf


# ─── building features ───────────────────────────────────────────────────────

BUILDING_FEATURE_COLS = [
    "pv_capacity_kw", "log_pv_capacity", "lat", "lon",
    "mean_demand", "std_demand",
    "postcode_n_customers", "postcode_median_demand",
    "postcode_std_demand", "postcode_pv_penetration",
    "postcode_mean_pv_capacity",
    "area_0", "area_1", "area_2", "area_3",
    "coastal",
]   # exactly 16


def build_building_features(hourly_df: pd.DataFrame,
                            train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 16 building/postcode features from the TRAINING slice only.
    `train` must be hourly_df filtered to the train window.
    Indexed by customer_id.
    """
    cust_stats = (train.groupby("customer_id")
                  .agg(pv_capacity_kw=("pv_capacity_kw", "first"),
                       postcode=("postcode", "first"),
                       mean_demand=("demand_kwh", "mean"),
                       std_demand=("demand_kwh", "std"))
                  .reset_index())

    pc_stats = (cust_stats.groupby("postcode")
                .agg(postcode_n_customers=("customer_id", "count"),
                     postcode_median_demand=("mean_demand", "median"),
                     postcode_std_demand=("std_demand", "mean"),
                     postcode_pv_penetration=("pv_capacity_kw",
                                              lambda x: float((x > 0).mean())),
                     postcode_mean_pv_capacity=("pv_capacity_kw", "mean"))
                .reset_index())

    feats = cust_stats.merge(pc_stats, on="postcode", how="left")
    feats["lat"] = feats["postcode"].map(lambda pc: _postcode_coords(pc)[0])
    feats["lon"] = feats["postcode"].map(lambda pc: _postcode_coords(pc)[1])
    feats["log_pv_capacity"] = np.log1p(feats["pv_capacity_kw"])

    feats["area_class"] = pd.cut(feats["postcode"],
                                 bins=[0, 2100, 2150, 2200, 9999],
                                 labels=[0, 1, 2, 3]).astype(int)
    area_ohe = pd.get_dummies(feats["area_class"], prefix="area").astype(float)
    for c in ["area_0", "area_1", "area_2", "area_3"]:
        if c not in area_ohe.columns:
            area_ohe[c] = 0.0
    feats = pd.concat([feats, area_ohe[["area_0", "area_1", "area_2", "area_3"]]],
                      axis=1)
    feats["coastal"] = (feats["lat"] > -33.9).astype(float)

    out = feats[["customer_id"] + BUILDING_FEATURE_COLS].copy().fillna(0.0)
    return out.set_index("customer_id")


# ─── graph ───────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def build_graph(building_feats: pd.DataFrame,
                train: pd.DataFrame,
                radius_km: float = 30.0,
                profile_sim_threshold: float = 0.7,
                max_neighbors: int = 8) -> dict:
    """
    Build the train-period similarity graph and return fixed-size neighbour
    lists (top-K by edge weight) for leakage-safe star subgraphs.

    Edge rule: customers i,j connected iff
      haversine(postcode centroids) < radius_km
      AND cosine similarity of their TRAIN-period mean hour-of-day demand
      profiles ≥ profile_sim_threshold.
    Edge weight = that cosine similarity (in [-1, 1] → clipped to [0,1]).

    Returns
    -------
    dict:
      node_ids       : list[int]                 customer IDs in node order
      neighbor_ids   : list[list[int]]           per node, up to max_neighbors
                                                 node indices, padded with -1
      neighbor_weight: list[np.ndarray]          per node, matching weights
                                                 (padded positions → 0.0)
      n_nodes        : int
    """
    custs = building_feats.index.tolist()
    n = len(custs)
    lats = building_feats["lat"].values
    lons = building_feats["lon"].values

    # Train-period hour-of-day mean demand profile per customer
    train = train.copy()
    train["hour"] = train["timestamp"].dt.hour
    profiles = np.zeros((n, 24), dtype=np.float64)
    for i, cid in enumerate(custs):
        sub = train[train["customer_id"] == cid]
        prof = (sub.groupby("hour")["demand_kwh"].mean()
                .reindex(range(24), fill_value=0.0).values)
        profiles[i] = prof
    norms = np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8
    pnorm = profiles / norms
    sim = pnorm @ pnorm.T   # (n, n) cosine similarity

    neighbor_ids: list[list[int]] = []
    neighbor_weight: list[np.ndarray] = []
    for i in range(n):
        cands = []
        for j in range(n):
            if i == j:
                continue
            if haversine_km(lats[i], lons[i], lats[j], lons[j]) > radius_km:
                continue
            s = float(sim[i, j])
            if s < profile_sim_threshold:
                continue
            cands.append((j, max(0.0, min(1.0, s))))
        cands.sort(key=lambda x: x[1], reverse=True)
        cands = cands[:max_neighbors]
        ids = [c[0] for c in cands]
        wts = np.array([c[1] for c in cands], dtype=np.float32)
        # Pad to max_neighbors
        pad = max_neighbors - len(ids)
        ids = ids + [-1] * pad
        wts = np.concatenate([wts, np.zeros(pad, dtype=np.float32)])
        neighbor_ids.append(ids)
        neighbor_weight.append(wts)

    return {
        "node_ids": custs,
        "neighbor_ids": neighbor_ids,
        "neighbor_weight": neighbor_weight,
        "n_nodes": n,
    }


def build_local_edge_index(max_neighbors: int,
                           device: str = "cpu") -> torch.Tensor:
    """
    Fixed local star-graph topology: node 0 (root) ↔ nodes 1..K (neighbours),
    plus self-loops on all K+1 nodes.  Same for every sample, so the batched
    GAT can share one edge_index.

    Returns edge_index of shape (2, E).
    """
    K = max_neighbors
    src, dst = [], []
    for j in range(1, K + 1):
        src += [0, j]     # root → neighbour, neighbour → root
        dst += [j, 0]
    for i in range(K + 1):   # self-loops
        src.append(i)
        dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


# ─── aligned array container ─────────────────────────────────────────────────

def build_aligned_arrays(hourly: pd.DataFrame, building_feats: pd.DataFrame,
                         weather: pd.DataFrame,
                         demand_scaler: StandardScaler,
                         gen_scaler: StandardScaler,
                         weather_scaler: StandardScaler,
                         ) -> dict:
    """
    Build a single aligned tensor `data` of shape (N_nodes, T, 19), plus the
    master timestamp index and the raw (unscaled) demand/gen for target
    construction and de-normalisation of metrics.

    Missing hours are forward-filled then zero-filled (documented limitation).
    """
    master = pd.date_range(hourly["timestamp"].min(),
                           hourly["timestamp"].max(), freq="1h")
    T = len(master)
    node_ids = building_feats.index.tolist()
    N = len(node_ids)

    weather_aligned = weather.reindex(master, method="nearest")
    wcols = weather_aligned[WEATHER_COLS].values
    if np.isnan(wcols).any():
        wcols = pd.DataFrame(wcols, index=master,
                             columns=WEATHER_COLS).ffill().fillna(0.0).values
    weather_scaled = weather_scaler.transform(wcols)     # (T, 9)

    cal = calendar_features(master)                      # (T, 8)

    data = np.zeros((N, T, TEMPORAL_CHANNELS), dtype=np.float32)
    raw_demand = np.zeros((N, T), dtype=np.float32)
    raw_gen = np.zeros((N, T), dtype=np.float32)

    hourly = hourly.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    by_cust = {cid: g for cid, g in hourly.groupby("customer_id")}

    for i, cid in enumerate(node_ids):
        if cid not in by_cust:
            continue
        g = by_cust[cid].set_index("timestamp").sort_index()
        d = g["demand_kwh"].reindex(master)
        gn = g["gen_kwh"].reindex(master)
        d = d.ffill().fillna(0.0).values
        gn = gn.ffill().fillna(0.0).values
        raw_demand[i] = d.astype(np.float32)
        raw_gen[i] = gn.astype(np.float32)
        d_s = demand_scaler.transform(d.reshape(-1, 1)).ravel()
        g_s = gen_scaler.transform(gn.reshape(-1, 1)).ravel()
        data[i, :, DEMAND_CH] = d_s
        data[i, :, GEN_CH] = g_s
        data[i, :, 2:2 + len(WEATHER_COLS)] = weather_scaled
        data[i, :, 2 + len(WEATHER_COLS):] = cal

    return {
        "data": data,               # (N, T, 19) scaled
        "raw_demand": raw_demand,   # (N, T) unscaled kWh
        "raw_gen": raw_gen,         # (N, T) unscaled kWh
        "master_index": master,
        "node_ids": node_ids,
    }


# ─── leakage-safe subgraph dataset ───────────────────────────────────────────

class AusgridSubgraphDataset(Dataset):
    """
    Lazy, leakage-safe dataset.

    Each item is (root_node, anchor_idx).  It returns:
      x_building     : (16,)
      x_weather_now  : (9,)          root weather at anchor
      x_temporal     : (L, 19)       root window ending at anchor
      sub_temporal   : (K+1, L, 19)  root (idx 0) + neighbours, all ending at anchor
      sub_edge_index : (2, E)        shared local star topology
      sub_edge_weight: (E,)          per-item real edge weights (0 at padding)
      sub_edge_bias  : (E,)          0 for real edges, -1e9 for padded neighbours
      y_demand       : (H,)
      y_gen          : (H,)
      meta           : (node, anchor)  for evaluation bookkeeping
    """

    def __init__(self, aligned: dict, graph: dict, edge_index: torch.Tensor,
                 lookback: int, horizons: list[int],
                 anchor_range: tuple[int, int]) -> None:
        super().__init__()
        self.data = aligned["data"]              # (N, T, C)
        self.raw_demand = aligned["raw_demand"]
        self.raw_gen = aligned["raw_gen"]
        self.node_ids = graph["node_ids"]
        self.neighbor_ids = graph["neighbor_ids"]
        self.neighbor_weight = graph["neighbor_weight"]
        self.n_nodes = graph["n_nodes"]
        self.edge_index = edge_index
        self.L = lookback
        self.horizons = horizons
        self.H = len(horizons)
        self.K = len(self.neighbor_ids[0]) if self.neighbor_ids else 0
        self.T = self.data.shape[1]

        # Window convention (no leakage):
        #   inputs  = data[n, i-L+1 : i+1]        (last observed hour = i)
        #   target  = data[n, i+h]  for horizon h  (strictly after i)
        # anchor `i` ranges so that i-L+1 >= 0 and i+h_max <= T-1.
        h_max = max(horizons)
        self.lo = max(anchor_range[0], self.L - 1)
        self.hi = min(anchor_range[1], self.T - 1 - h_max)
        self._n_anchors = max(0, self.hi - self.lo + 1)
        self._index = [(n, a) for n in range(self.n_nodes)
                       for a in range(self.lo, self.hi + 1)]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        n, a = self._index[idx]          # a = anchor = last observed hour i
        L, K = self.L, self.K
        start = a - L + 1

        root_win = self.data[n, start:a + 1, :]                    # (L, C)
        sub = np.zeros((K + 1, L, self.data.shape[2]), dtype=np.float32)
        sub[0] = root_win

        ew = np.zeros(2 * K + (K + 1), dtype=np.float32)   # real weights
        eb = np.zeros_like(ew)                             # bias mask
        # local edge order matches build_local_edge_index:
        #   [0,1],[1,0],[0,2],[2,0],..., then K+1 self-loops
        pos = 0
        for j in range(1, K + 1):
            nb = self.neighbor_ids[n][j - 1]
            if nb < 0:
                # padded neighbour → mask its two edges
                eb[pos] = -1e9
                eb[pos + 1] = -1e9
            else:
                sub[j] = self.data[nb, start:a + 1, :]
                w = float(self.neighbor_weight[n][j - 1])
                ew[pos] = w
                ew[pos + 1] = w
            pos += 2
        # self-loops keep weight 1.0, bias 0
        ew[pos:pos + K + 1] = 1.0

        x_weather_now = root_win[-1, 2:2 + len(WEATHER_COLS)].copy()

        # Targets strictly AFTER the anchor: index i + h - 1 with anchor==i
        # is equivalent to (anchor+1) + (h-1) - 1 = anchor + h - 1.  We define
        # target_idx = a + h to make "1-hour ahead" the very next hour.
        y_d = np.array([self.data[n, a + h, DEMAND_CH]
                        for h in self.horizons], dtype=np.float32)
        y_g = np.array([self.data[n, a + h, GEN_CH]
                        for h in self.horizons], dtype=np.float32)

        return {
            "x_building": torch.from_numpy(self.building_scaled[n].copy()),
            "x_weather_now": torch.from_numpy(x_weather_now),
            "x_temporal": torch.from_numpy(root_win.astype(np.float32)),
            "sub_temporal": torch.from_numpy(sub),
            "sub_edge_index": self.edge_index,
            "sub_edge_weight": torch.from_numpy(ew),
            "sub_edge_bias": torch.from_numpy(eb),
            "y_demand": torch.from_numpy(y_d),
            "y_gen": torch.from_numpy(y_g),
            "meta": torch.tensor([n, a], dtype=torch.long),
        }

    # building features are supplied separately as a fixed (N, 16) matrix and
    # indexed by node in the training loop; we attach it here for convenience.
    def attach_building(self, building_scaled: np.ndarray) -> None:
        self.building_scaled = building_scaled

    def get_building(self, n: int) -> torch.Tensor:
        return torch.from_numpy(self.building_scaled[n])


# ─── data module ─────────────────────────────────────────────────────────────

class AusgridDataModule:
    """
    Orchestrates loading, chronological split, train-only scaling, graph
    construction, and leakage-safe subgraph dataset creation.

        dm = AusgridDataModule(cfg)
        dm.setup()
        train_loader = dm.train_dataloader()
        test_loader  = dm.test_dataloader()
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.graph = None

    def setup(self) -> None:
        dc = self.cfg["data"]
        raw_dir = Path(dc["raw_dir"])
        processed_dir = Path(dc.get("processed_dir", "data/processed"))
        processed_dir.mkdir(parents=True, exist_ok=True)

        # ── load raw ──────────────────────────────────────────────────────────
        pq = processed_dir / "ausgrid_hourly.parquet"
        if pq.exists():
            hourly = pd.read_parquet(pq)
            hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
        else:
            hourly = load_ausgrid_raw(raw_dir)
            hourly.to_parquet(pq, index=False)

        # Optional customer subsample (smoke tests)
        limit = dc.get("n_customers_limit")
        if limit is not None:
            keep = sorted(hourly["customer_id"].unique())[:int(limit)]
            hourly = hourly[hourly["customer_id"].isin(keep)].copy()

        # ── chronological split ───────────────────────────────────────────────
        train_end = pd.to_datetime(dc["train_end"])
        test_start = pd.to_datetime(dc["test_start"])
        train_slice = hourly[hourly["timestamp"] <= train_end]
        train_mask = hourly["timestamp"] <= train_end

        # ── weather ───────────────────────────────────────────────────────────
        weather = fetch_weather(
            lat=_DEFAULT_LAT, lon=_DEFAULT_LON,
            start="2010-07-01", end="2013-06-30",
            cache_path=processed_dir / "weather_sydney.parquet",
        )

        # ── building features (train-only) ────────────────────────────────────
        building_feats = build_building_features(hourly, train_slice)

        # ── scalers fit on TRAIN only ─────────────────────────────────────────
        self.demand_scaler = StandardScaler().fit(train_slice[["demand_kwh"]].values)
        self.gen_scaler = StandardScaler().fit(train_slice[["gen_kwh"]].values)
        # Fit the weather scaler on the TRAIN slice only, consistent with the
        # demand/gen/building scalers — never on the full period.
        weather_train = weather[weather.index <= train_end]
        self.weather_scaler = StandardScaler().fit(
            weather_train[WEATHER_COLS].values
        )

        # ── aligned arrays ────────────────────────────────────────────────────
        aligned = build_aligned_arrays(
            hourly, building_feats, weather,
            self.demand_scaler, self.gen_scaler, self.weather_scaler,
        )

        # ── building feature matrix (standardised) ────────────────────────────
        bf = building_feats.loc[aligned["node_ids"]].values
        self.building_scaler = StandardScaler().fit(bf)
        building_scaled = self.building_scaler.transform(bf).astype(np.float32)

        # ── graph (train only) ────────────────────────────────────────────────
        self.graph = build_graph(
            building_feats, train_slice,
            radius_km=dc.get("graph_radius_km", 30.0),
            profile_sim_threshold=dc.get("graph_profile_sim_threshold", 0.7),
            max_neighbors=dc.get("max_neighbors", 8),
        )
        max_neighbors = dc.get("max_neighbors", 8)
        edge_index = build_local_edge_index(max_neighbors)

        # ── anchor ranges (chronological) ─────────────────────────────────────
        # A training sample anchored at `a` reads targets up to a + h_max.  To
        # keep the training window strictly disjoint from the test period, cap
        # the last training anchor at (test_start - h_max hours), so no training
        # target reaches into the test window.
        master = aligned["master_index"]
        horizons = dc["horizons"]
        h_max = max(horizons)
        train_cutoff = test_start - pd.Timedelta(hours=h_max)
        train_hi = int((master <= train_cutoff).sum()) - 1
        test_lo = int((master < test_start).sum())
        test_hi = len(master) - 1

        lookback = dc["lookback_hours"]

        self.train_ds = AusgridSubgraphDataset(
            aligned, self.graph, edge_index, lookback, horizons,
            anchor_range=(0, train_hi),
        )
        self.train_ds.attach_building(building_scaled)
        self.test_ds = AusgridSubgraphDataset(
            aligned, self.graph, edge_index, lookback, horizons,
            anchor_range=(test_lo, test_hi),
        )
        self.test_ds.attach_building(building_scaled)

        self.aligned = aligned
        self.building_scaled = building_scaled
        self.max_neighbors = max_neighbors

    # building features are static per node; the training loop reads them via
    # the dataset's building_scaled array using meta[:,0].

    def train_dataloader(self, batch_size: int = 32,
                         num_workers: int = 2) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)

    def test_dataloader(self, batch_size: int = 64,
                        num_workers: int = 2) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
