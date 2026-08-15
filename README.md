# Hormuz Disruption Engine

A supply-chain disruption early-warning system for the Strait of Hormuz and UAE ports, built as a production-quality deep-learning prototype.

Roughly a fifth of global seaborne oil transits Hormuz. Eleven of the UAE's thirteen major ports sit *inside* the strait and are cut off from deep-sea trade if it closes; only **Fujairah** and Khor Fakkan bypass it. This system fuses satellite imagery, market price signals, and geopolitical event text into a single chokepoint risk index — and explains, in plain language, why that index moved.

> **Validated against a real event.** The 2026 Hormuz crisis (onset 2026-03-02, Brent +117% peak) is in the training window. The system detects it **43 days before the onset date**, derived purely from market and text anomalies — not from the June headline announcement.

---

## What is built

| Component | Status | Key result |
|-----------|--------|------------|
| GDELT 2.0 Translingual live collector | ✅ Running in Docker | 934,948 historical rows, 3-way Gulf filter validated |
| FRED + yfinance market panel | ✅ Collected | 2,204 days × 11 series, 2018–2026 |
| Sentinel-1 SAR chip pipeline | ✅ Collected | 449 chips, 4 ports, 4 anchor windows, 5,947 PU |
| Arm A — CA-CFAR + water mask | ✅ Built & measured | 371 full passes, orbit-tagged, land baseline validated |
| Arm B — Sequence VAE (LSTM) | ✅ Trained & evaluated | AUC 0.680; 2026 Hormuz 38-day lead |
| Arm C — Zero-shot multilingual Transformer | ✅ Built & evaluated | AUC 0.848; top day 2019-06-14 (Gulf of Oman attacks) |
| Feature panel | ✅ Built | 3,129 days × 19 features, leakage-checked |
| Causal LOEO backtest | ✅ Run | 2/3 scoreable events detected, median 36-day lead |
| Weighted rule fusion | ✅ Built | Arm B 0.50 · Arm C 0.50; beats either arm alone |
| FastAPI backend | ✅ Live | 24 endpoints, DuckDB, SAR chip renderer, semantic registry |
| Claude analyst brief | ✅ Built | 22-metric registry, 25 prohibitions injected as guardrails |
| UAE ports viewer | ✅ Built | 13 ports, Hormuz-dependency flag, bypass split |
| GDELT historical backfill | ✅ Built | Route A (raw files, no billing), Route B (BigQuery, code ready) |

**Start the full system in one command:**
```bash
python -m backend.launch
```

See [RUNNING.md](RUNNING.md) for the complete setup guide.

---

## Architecture

Three arms, each handling a genuinely different data modality. Every arm earns its place or is excluded from fusion.

```
Sentinel-1 SAR (GRD, dual-pol)  ──> [CA-CFAR + MobileNetV2]  ──> congestion_index  (weight 0.00*)
FRED Brent/WTI + ETF series     ──> [Sequence VAE, LSTM]      ──> market_anomaly    (weight 0.50)
GDELT 2.0 Translingual slugs    ──> [Zero-shot Transformer]   ──> geo_tension        (weight 0.50)
                                              │
                                              ▼
                                   Causal expanding percentile
                                   Weighted rule fusion → risk index (0–100)
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                             Claude brief          Dashboard + alerts
```

\* Arm A (SAR) is measured at weight 0.00 in the current fusion because CA-CFAR at 1024 px (~15.6 m/px) shows no reroute signal at the port level — a quantified finding, not a placeholder. The CFAR baseline is retained as the benchmark the CNN must beat when higher-resolution collection scales up.

**Why each arm exists**
- **Arm A (SAR)** — Sentinel-1 sees through Gulf haze, cloud, and darkness. All-weather, transponder-independent vessel detection. Physical ground truth.
- **Arm B (Market VAE)** — Markets price chokepoint tension before ships physically pile up. The VAE detects sustained Brent/WTI divergence — the Gulf-specific signal — and fired 38 days before the 2026 onset.
- **Arm C (Text)** — Geopolitical text at 15-minute cadence identifies *where* the tension is. Zero-shot contrastive scoring means no labels are consumed in training; all five events are held for evaluation.

**Why Arms B and C are complementary — measured, not assumed**

| | Sharp incidents (2019) | Sustained dislocation (2026) |
|---|---|---|
| Arm B (market VAE) | Misses — market moved less than 1σ | ✅ 38-day lead |
| Arm C (text) | ✅ z = 3.4–4.7 on 2019 events | ⚠️ 2-day lead alone |
| **Fused** | ✅ Abqaiq 30 d | ✅ **43 d** |

---

## Measured backtest results

Causal leave-one-event-out · threshold from quiet days · 10-day embargo

| Event | Onset | Status | Peak score | Lead |
|-------|-------|--------|-----------|------|
| 2019-05 Fujairah attacks | 2019-05-12 | — unscoreable (prior history < 365 d) — | | |
| 2019-06 Gulf of Oman | 2019-06-13 | — unscoreable (prior history < 365 d) — | | |
| 2019-09 Abqaiq strike | 2019-09-14 | **Detected** | 97.0 | **30 days** |
| 2024-01 Red Sea/Houthi | 2024-01-15 | Undetected | 79.9 | — |
| **2026-03 Hormuz closure** | 2026-03-02 | **Detected** | 96.8 | **43 days** |

**2/3 scoreable events detected · median lead 36 days**

The two earliest 2019 events are unscoreable under causal rules — there is not enough prior panel history to have generated a meaningful forecast at the time. This is reported rather than padded.

---

## Arm C — text scoring highlights

Zero-shot contrastive multilingual scoring via `paraphrase-multilingual-MiniLM-L12-v2`. No labels consumed during scoring.

- **AUC 0.848** on point events vs quiet days
- **Precision@20: 1.71× lift** over base rate
- Highest-scoring day in the 366-day corpus: **2019-06-14** — the day after Gulf of Oman tanker attacks
- Top 12 days all map to real maritime incidents (Abqaiq ×5, Red Sea ×3, Fujairah ×1, Sabiti tanker ×1)

---

## Data sources

| Source | Use | Status |
|--------|-----|--------|
| [Copernicus Data Space](https://dataspace.copernicus.eu) | Sentinel-1 SAR chips | ✅ 449 chips collected |
| [FRED / St. Louis Fed](https://fred.stlouisfed.org) | Brent, WTI, gas (official EIA redistribution) | ✅ 2,204 days |
| [GDELT 2.0 Translingual](https://www.gdeltproject.org) | 15-min global event stream | ✅ 934,948 rows, live poller running |
| [Global Fishing Watch](https://globalfishingwatch.org) | Satellite AIS — Gulf confirmed (258K datapoints) | ✅ Backtest layer |
| [aisstream.io](https://aisstream.io) | Terrestrial AIS WebSocket | Key valid; no Gulf coverage (terrestrial receivers absent) |
| [xView3](https://iuu.xview.us) | Labelled SAR vessel dataset | Pre-training reference for CNN |
| [yfinance](https://github.com/ranaroussi/yfinance) | Futures + shipping ETFs | ✅ Supplement to FRED |

**Satellite is fresher than AIS for the Gulf.** Sentinel-1 revisits Fujairah every ~1.4 days (three satellites: S1A, S1C, S1D). Global Fishing Watch satellite AIS lags 4 days. For Gulf waters, Arm A provides the most timely physical observation.

---

## Repository layout

```
.
├── CLAUDE.md                        # full project context and measured findings
├── README.md
├── RUNNING.md                       # setup and run guide
├── ARCHITECTURE.md
├── hormuz_execution_plan.html       # full spec, results, and live demo widget
├── hormuz_architecture_diagram.svg  # system architecture diagram
├── .env.example                     # copy to .env; fill in API keys
├── docker-compose.yml               # GDELT poller service
├── backend/
│   ├── launch.py                    # python -m backend.launch — starts everything
│   ├── api/                         # FastAPI — 24 endpoints
│   ├── arms/
│   │   ├── sar/                     # CA-CFAR + MobileNetV2 CNN
│   │   ├── market/                  # Sequence VAE (LSTM encoder/decoder)
│   │   └── text/                    # zero-shot contrastive scoring
│   ├── fusion/                      # weighted rule + causal LOEO backtest
│   ├── features/                    # feature panel builder + leakage check
│   ├── ingest/
│   │   ├── gdelt/                   # live poller + historical backfill
│   │   ├── market/                  # FRED + yfinance collector
│   │   └── satellite/               # Sentinel Hub chip acquisition
│   ├── agent/                       # Claude analyst brief
│   ├── semantic/                    # metric registry (22 metrics, 25 prohibitions)
│   └── streaming/                   # Kafka demo (profiled, not required)
├── ports/
│   ├── ais_client.py                # AIS collector (validated on English Channel)
│   ├── uae_ports.csv                # 13 UAE ports + offshore terminals
│   └── uae_ports.geojson
└── data/                            # gitignored — regenerable from pipeline
    ├── raw/                         # collected GDELT, market, SAR
    └── derived/                     # feature panel, fusion risk index
```

---

## Quick start

```bash
git clone https://github.com/rd51/Strait-Disruption-DL.git
cd Strait-Disruption-DL
pip install pandas pyarrow requests websockets fastapi uvicorn \
    pydantic scikit-learn scipy torch tensorflow sentence-transformers \
    duckdb Pillow rasterio tifffile yfinance
cp .env.example .env          # add your API keys
python -m backend.launch      # starts Docker poller + API on localhost:8000
```

Open `http://localhost:8000` for the dashboard, `/docs` for the full API reference.

Full credential setup, rebuild steps, and troubleshooting: **[RUNNING.md](RUNNING.md)**

---

## UAE ports — the coastline split

The Hormuz-dependency split is the geographic thesis of this project.

| Side | Ports | Characteristic |
|------|-------|---------------|
| Inside strait (11) | Jebel Ali, Khalifa, Zayed, Port Rashid, Khalid, Hamriyah, Ajman, Mina Saqr, Jebel Dhanna/Ruwais, Das Island, Zirku | Cut off if Hormuz closes |
| **Outside — bypass (2)** | **Fujairah ★**, Khor Fakkan | World's #2 bunkering hub; absorbs rerouted traffic |

**Fujairah is the single most important signal.** Congestion spiking at Fujairah is the earliest *physical* confirmation of disruption — traffic reroutes there when the strait is stressed.

All 13 ports and terminals are covered by the bounding box `[[[23.5, 51.0], [27.0, 57.5]]]`, verified against the port CSV.

---

## Design principles

1. **Genuine multimodal justification.** Every arm handles a fundamentally different modality. No branch exists for decoration; Arm A is in the schema at zero weight rather than silently dropped.
2. **Unsupervised arms — labels for evaluation only.** With ~5 events, supervised training cannot be robustly validated. The VAE and zero-shot scorer train without labels; all five anchors are held for the backtest.
3. `TimeSeriesSplit` everywhere. Causal leave-one-event-out. 10-day embargo. Leakage test fails the build.
4. **Synthetic data never enters the backtest.** Real data drives all validation figures.
5. **Named scope boundaries over optimistic framing.** The measured null on Arm A rerouting and the GBM degenerate result are documented in the architecture, not hidden.

---

## Stack

FastAPI · DuckDB · Keras 3 (MobileNetV2) · PyTorch (Sequence VAE) · sentence-transformers · scikit-learn · Docker · Sentinel Hub · GDELT · FRED

---

*Built 2026-07-28 · all numbers from real runs against live data*
