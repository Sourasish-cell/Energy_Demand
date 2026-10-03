# A Multimodal Spatiotemporal Transformer for Joint Forecasting of Urban Energy Demand and Renewable Generation

Paper 1 of the programme *AI-Driven Digital Twin for Carbon-Aware, Climate-Resilient Urban Energy Systems*.

This repository contains the complete, runnable pipeline for the paper: a four-stream
multimodal transformer that forecasts residential electricity **demand** and rooftop solar
**generation** jointly, at 1 h, 6 h, 24 h and 168 h ahead, over a graph of co-metered
households. It ships with the data pipeline, the model, the training and evaluation code,
three baselines, three ablations, and Colab-ready notebooks.

Everything here trains and evaluates on **real, public data**. No synthetic stand-in is used
at any point, and no result number is hard-coded — every table in the paper is produced by
running the notebooks.

---

## Data

**Ausgrid Solar Home Electricity Data** (2010-07-01 → 2013-06-30, Sydney, Australia).
Half-hourly electricity consumption and gross solar generation, co-metered on the same
household, with postcodes. This is the public dataset that co-meters both energy quantities,
which is exactly what a *joint* demand-and-generation model needs. The pipeline resamples to
hourly by summing the two half-hourly energy readings within each hour (30-min values are
energy in kWh, so their sum is the hourly energy).

**Open-Meteo historical weather archive** (ERA5-backed, no API key). Nine hourly channels per
postcode centroid: temperature, humidity, precipitation, pressure, wind speed, shortwave /
direct / diffuse radiation, cloud cover.

**Split (chronological, never random):** train 2010-07-01 → 2012-06-30, test
2012-07-01 → 2013-06-30. All scalers and the whole graph are fit on the **training slice only**.

> The dataset is not redistributed here. Notebook `01` downloads / loads it from the source you
> point it at and fails loudly if the data is missing — it never substitutes simulated values.

---

## Layout

```
configs/default.yaml          single source of truth for every hyper-parameter
src/
  data/dataset.py             AusgridDataModule + AusgridDataset (leakage-safe star subgraphs)
  models/
    building_encoder.py       MLP over tabular building/postcode attributes
    temporal_encoder.py       PatchTST-style patched Transformer over the 168 h window
    spatial_gnn.py            multi-head Graph Attention Network (pure PyTorch, no PyG)
    fusion.py                 cross-attention fusion of the four streams
    multimodal_transformer.py the assembled model (Wiring + leakage-safe forward)
  train.py                    training loop, multi-task loss, ablations, checkpointing
  evaluate.py                 metrics in physical kWh, skill score, day-masked PV metrics
  baselines.py                seasonal-naive, DLinear, LightGBM
  utils.py                    config loading, seeding, metric primitives
tests/test_leakage_and_graph.py   4 dependency-light invariants (numpy only)
notebooks/
  01_data_preparation.ipynb   load real data, sanity-check, run tests, cache the dataset
  02_eda.ipynb                exploratory figures
  03_training.ipynb           smoke test → full run → three ablations
  04_evaluation.ipynb         results tables: model vs baselines vs ablations
scripts/gen_methodology_docx.js   regenerates docs/Paper1_Methodology.docx
docs/Paper1_Methodology.docx      the paper's methodology write-up
```

---

## Setup

```bash
pip install -r requirements.txt
```

The full configuration targets a single GPU (Google Colab is the intended environment).
Add the repo root to `sys.path` before importing `src.*` (the notebooks do this already).

## How to run

Run the notebooks in order on Colab:

1. **`01_data_preparation.ipynb`** — point it at the raw Ausgrid files, load and sanity-check
   the real data, re-run the leakage tests against the real tensors, and cache the assembled
   `AusgridDataModule` to `data/processed/dataset.pkl`.
2. **`02_eda.ipynb`** — generate the exploratory figures.
3. **`03_training.ipynb`** — run a CPU smoke test (shape/correctness only), then the full GPU
   training run, then the three ablations (`no_graph`, `single_task`, `no_cross_attn`).
   Checkpoints land in `runs/<name>/model.pt`.
4. **`04_evaluation.ipynb`** — score the full model, the baselines and the ablations on the
   held-out year and write the paper's tables to `runs/<name>/*.csv`.

To regenerate the methodology PDF/docx: `node scripts/gen_methodology_docx.js`.

---

## Model, in one pass

Each sample is a (node `n`, anchor hour `t`) pair. Four encoders are computed:

| Stream | Module | Input → output |
|---|---|---|
| Building | `BuildingEncoder` | 16 attributes → 64-d |
| Weather  | `WeatherEncoder` | 9-channel snapshot at `t` → 64-d |
| Temporal | `PatchTemporalEncoder` | `(168, 19)` window → 128-d |
| Spatial  | `SpatialGAT` | local star subgraph → 128-d (root read) |

`CrossAttentionFusion` projects all four to a common width and runs two cross-attention steps
(temporal+spatial attend over building+weather, and back), concatenates the four tokens, and
projects to a 256-d shared representation. Two linear heads then emit demand and generation at
the four horizons. Full details in `docs/Paper1_Methodology.docx`.

### The one design decision that matters most: leakage-safe star subgraphs

A graph model is trivially easy to make leaky: encode every node over the whole timeline once,
then index the embedding per sample — and a prediction anchored at hour `t` silently receives
context from hours after `t`. This model does not do that. **Every sample supplies a local
star subgraph whose node windows all end at that sample's anchor hour `t`** (the root at index
0 plus up to K = 8 neighbours), the batched GAT runs over just that subgraph, and the root's
output is read as the spatial embedding. So the spatial stream is a function of data available
at `t` only. This invariant is encoded in `tests/test_leakage_and_graph.py` and re-checked
against the real tensors in notebook `01`.

---

## Evaluation

All metrics are reported in physical kWh (scalers inverted). For each horizon and target:
MAE, RMSE, MAPE, and a **skill score** against a seasonal-naive reference
(`1 − RMSE_model / RMSE_baseline`). Generation is also reported day-masked (07:00–18:00),
because night-time PV is ≈ 0 and otherwise inflates percentage error.

Baselines (`src/baselines.py`): **seasonal-naive**, **DLinear** (Zeng et al. 2023), and
**LightGBM** on calendar + weather + lag/rolling + building features — the strongest
feature-engineered tabular comparison, and the one that most fairly tests whether the
multimodal coupling earns its complexity.

Ablations (`src/train.py`): **no_graph** (cross-neighbour edges masked), **single_task**
(generation loss weight 0), **no_cross_attn** (streams concatenated instead of attending).
Each is scored against the full model and reported as % degradation.

---

## Reporting integrity

Two commitments govern this repo, and they are load-bearing:

1. **No synthetic data stands in for real data.** If a source is unavailable the pipeline
   halts rather than substituting simulated values.
2. **No metric is reported unless the code produced it.** Results tables are filled by running
   the notebooks on the real test split, and left blank where a model has not been trained —
   never filled with placeholders.

Consequently, this repository states no measured results. They are produced by executing the
pipeline on the real data and reported in the paper.

---

## Known limitations (stated up front)

- **Building features are postcode-level proxies**, not per-building records from a
  building-performance database (e.g. DOE BPD). They encode location and consumption behaviour,
  which bounds how interpretable the building stream is.
- **Geographic and temporal scope** is residential Sydney over three years; transferability to
  other climates or feeder topologies is untested.
- **Graph scale** is small fixed-degree star subgraphs within 30 km; multi-scale or hierarchical
  graphs are out of scope here.
- **PV MAPE** is ill-conditioned at night; the daytime mask mitigates but does not remove this.
