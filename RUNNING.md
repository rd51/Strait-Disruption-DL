# Hormuz Disruption Engine — How to Run

A step-by-step guide for anyone receiving this project who wants to see it working.

**Read this first:** the system has two deployment halves that run differently. The GDELT collector runs in Docker; the API and ML pipeline run directly in Python on the host. This split is intentional — the collector image is lean (4 packages, always-on), and the ML environment is heavy (~6 GB of TensorFlow/PyTorch/sentence-transformers). They share the same `data/` volume.

---

## Prerequisites

| Tool | Version tested | Notes |
|------|---------------|-------|
| Python | **3.12** | Other 3.10+ versions likely work; 3.12 is what this was built on |
| Docker Desktop | 29.6+ | Required only for the GDELT live collector and Kafka demo |
| Git | any | For cloning |

No GPU is required. The CNN arm uses MobileNetV2 and runs on CPU (TF ≥ 2.11 has no native Windows GPU support).

---

## 1. Clone and install

```bash
git clone <repo-url>
cd "Final Software Prototype"
```

Install the full backend environment (includes ML libraries):

```bash
pip install pandas>=2.0 pyarrow>=14 requests websockets fastapi uvicorn \
    pydantic scikit-learn scipy torch tensorflow sentence-transformers \
    duckdb Pillow rasterio tifffile yfinance
```

Or, if a `requirements-backend.txt` is present:

```bash
pip install -r backend/requirements-backend.txt
```

> **Windows note:** File Explorer silently appends `.txt` to extensionless files.
> The `.env` file is just `.env` — check it was not saved as `.env.txt`.

---

## 2. Configure credentials

Copy the example environment file and fill in any keys you hold:

```bash
cp .env.example .env
```

Open `.env` — the only required field for the core engine is:

```
ANTHROPIC_API_KEY=sk-ant-...      # needed ONLY for /brief (the LLM analyst report)
```

Everything else is optional depending on which parts you want to run:

| Variable | What it enables | Where to get it |
|----------|----------------|-----------------|
| `ANTHROPIC_API_KEY` | `/brief` — LLM-generated analyst report | console.anthropic.com |
| `FRED_API_KEY` | Re-running market data collection | fredaccount.stlouisfed.org/apikeys (free) |
| `CDSE_CLIENT_ID` + `CDSE_CLIENT_SECRET` | Re-running SAR chip collection | dataspace.copernicus.eu → User Settings → OAuth clients |
| `AISSTREAM_KEY` | AIS vessel stream — **works perfectly in European/global waters; zero coverage in the Gulf** (see `/vessels`) | aisstream.io (free tier) |
| `GFW_TOKEN` | Global Fishing Watch backtest AIS | globalfishingwatch.org/our-apis/tokens |
| `GOOGLE_APPLICATION_CREDENTIALS` | GDELT BigQuery backfill (optional Route B) | Google Cloud Console |

**To just run the API on the data that is already on disk, no keys are needed at all.** GDELT raw files are open. The market panel, SAR chips, trained models, and fusion index are pre-collected and committed. Keys are only required if you want to refresh or extend the data.

> **Credential security:** keys are read from env vars first, then from local files (e.g. `secrets/gfw_token.txt`). The root `.gitignore` blocks `aisstream.*`, `.env`, `secrets/`, and `data/raw/` — never narrow those rules.

---

## 3. What is already on disk

This repository ships with pre-collected data. You do **not** need to re-run the full collection pipeline to see results:

| What | Location | Size | Notes |
|------|----------|------|-------|
| GDELT live slots (2026-07-27) | `data/raw/gdelt/live/` | ~100 MB | 24 slots from the live run |
| GDELT historical windows | `data/raw/gdelt/historical/` | ~2.8 GB | 934,948 rows across 3 anchor events |
| Market panel (Arm B) | `data/raw/market/` | small | 2,204 days × 11 series, 2018–2026 |
| SAR chips (Arm A) | `data/raw/satellite/chips/` | ~3.7 GB | 449 chips × 4 ports × 4 windows |
| Daily feature panel | `data/derived/features_daily.parquet` | small | 3,129 days × 19 features |
| Arm B VAE model | `data/models/arm_b_vae/` | small | trained weights + scores |
| Arm A CNN model | `data/models/arm_a_cnn/` | small | MobileNetV2 transfer weights |
| Fusion risk index | `data/derived/fusion/risk_index.parquet` | small | LOEO backtest results |

If any of these are missing (e.g. after a fresh clone without LFS), the relevant section below explains how to rebuild them.

---

## 4. Start the API — the fastest path to seeing everything

The launcher handles both Docker (collector) and the API in one command:

```bash
python -m backend.launch
```

This will:
1. Start the GDELT poller container (Docker must be running)
2. Start the FastAPI server on `http://127.0.0.1:8000`
3. Preload the ML models into memory (takes ~30s on first run)
4. Print a status report

Then open:

| URL | What you see |
|-----|-------------|
| `http://127.0.0.1:8000/` | Live dashboard |
| `http://127.0.0.1:8000/docs` | Interactive API explorer (Swagger UI) |
| `http://127.0.0.1:8000/health` | Quick sanity check |

If you want just the API without Docker:

```bash
uvicorn backend.api.main:app --reload --port 8000
```

**To stop everything:**
```bash
python -m backend.launch --down
```

---

## 5. Key API endpoints — what each one shows

All endpoints are documented in the Swagger UI at `/docs`. The most informative ones:

### System state
```
GET /freshness       → how old each arm's data is (staleness per source)
GET /sources         → which series are official (FRED, GDELT, Copernicus) vs unofficial (yfinance)
GET /pipeline        → DAG state derived from filesystem timestamps
GET /events          → the 5 backtest label anchors, with onset dates from price data
GET /models          → trained model reports with full evaluation metrics inline
```

### Arm data
```
GET /arms/gdelt                  → daily GDELT aggregates, keyed on DATEADDED
GET /arms/market                 → Brent, WTI, spread, ETF series
GET /arms/sar                    → CFAR vessel detections per port per chip
GET /arms/market/anomaly         → Arm B VAE reconstruction error (anomaly score)
GET /features                    → the full 3,129-day × 19-feature daily panel
```

### Live inference (runs the models on new input)
```
POST /predict/text               → zero-shot multilingual disruption scoring on any text
GET  /predict/market             → Arm B VAE on the trailing 20-trading-day window
GET  /predict/live               → both arms on the most recent available data
POST /predict/warm               → preload models (avoids 30s cold start on first call)
```

### The risk index and brief
```
GET /risk            → fused chokepoint risk index (0–100) with LOEO backtest
GET /brief           → LLM-generated analyst brief (requires ANTHROPIC_API_KEY; dry_run=true by default)
GET /brief?dry_run=false  → calls the API and returns the actual report
```

### Geography
```
GET /ports           → the 13 UAE ports with Hormuz-dependency flag and bypass caveat
GET /chips           → Sentinel-1 chip catalogue (port, date, coverage class, orbit)
GET /chips/{port}/{date}.png  → render one SAR chip as a viewable PNG with CFAR detections circled
```

### Satellite vessel monitoring and risk index
```
GET /vessels         → vessel monitoring endpoint — see the satellite AIS note below
GET /risk            → composite risk index — see honest_reading field for basis
```

> **AIS coverage note:** the `AISSTREAM_KEY` is valid and the collector code is fully functional. Pointed at a European bounding box it pulls live vessel data — verified at ~101 messages/second on a comparably-sized box. aisstream aggregates **volunteer terrestrial receivers**, which are absent in the Gulf. The system uses **Sentinel-1 SAR (Arm A)** for transponder-independent vessel detection in the Hormuz region, and Global Fishing Watch satellite AIS for the backtest layer (Gulf coverage confirmed: 258,281 datapoints in the Hormuz box). The `/vessels` endpoint documents this architecture rather than returning an empty result.

---

## 6. Rebuild from scratch (if data is missing)

Run these in order. Each step takes care of a different arm. **Steps 6a–6c can run in parallel once you have the dependencies.**

### 6a. GDELT live collection (Docker, runs permanently)

```bash
docker compose up -d gdelt-poller
docker compose logs -f gdelt-poller
```

Expected output per slot: `~800–1050 rows total, ~20–80 Gulf rows, ~4–21 conflict rows`. A slot with 0 Gulf rows is suspicious but not always wrong (it can happen). If you see it repeatedly, the filter has broken.

For a single slot without Docker:
```bash
python -m backend.ingest.gdelt.poller --once
```

### 6b. GDELT historical backfill (Route A — no billing, no auth)

Downloads the three anchor event windows (2019 Gulf of Oman attacks, 2019 Abqaiq strike, 2024 Red Sea/Houthi). Total: ~35,000 files, ~3.5 GB, a few hours on a home line.

```bash
# See what windows are available
python -m backend.ingest.gdelt.backfill_raw --list

# Dry run (no downloads)
python -m backend.ingest.gdelt.backfill_raw --window 2019_gulf_of_oman --dry-run

# Download one window
python -m backend.ingest.gdelt.backfill_raw --window 2019_gulf_of_oman

# Download all three at once (8 parallel workers)
python -m backend.ingest.gdelt.backfill_raw --all --workers 8
```

Safe to Ctrl-C and restart — already-downloaded slots are skipped.

### 6c. Market data (Arm B)

```bash
python -m backend.ingest.market.collect
```

Pulls from FRED (official, citable) and yfinance. FRED series include Brent, WTI, and natural gas; yfinance adds futures and ETFs. Requires `FRED_API_KEY` in `.env` (free at fredaccount.stlouisfed.org). Takes a few minutes. Output: `data/raw/market/market_daily.parquet`.

### 6d. SAR chip collection (Arm A — costs Sentinel Hub quota)

> **Quota warning:** each chip costs 13.3 processing units (PU) at 1024 px. You have 30,000 PU/month. The full 449-chip collection used **5,947 PU (20% of the monthly allowance)**. Check your balance first:
> `GET https://sh.dataspace.copernicus.eu/api/v1/accounting/usage`

Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu) and add your credentials to `.env`.

```bash
python -m backend.ingest.satellite.collect --port fujairah --dry-run   # preview
python -m backend.ingest.satellite.collect --port fujairah             # pull one port
python -m backend.ingest.satellite.collect --all                       # all 4 ports
```

If the data is already on disk, skip this — it's expensive.

### 6e. Build the feature panel

```bash
python -m backend.features.build
```

Produces `data/derived/features_daily.parquet` — 3,129 days × 19 features, 2018–2026. Includes a leakage check:

```bash
python -m backend.features.build --check-leakage
```

The leakage check asserts that no derived feature outlives or out-counts its source — it has caught two separate pandas fabrication bugs in this project (`pct_change()` padding NaNs by default, and `rolling()` running past source end).

### 6f. Run CFAR (Arm A baseline)

```bash
python -m backend.arms.sar.run_cfar
```

Runs CA-CFAR + water mask over all collected SAR chips. No quota cost; this is CPU computation on the chips already on disk.

### 6g. Train Arm B VAE

```bash
python -m backend.arms.market.vae
```

LSTM encoder → 8-d latent → LSTM decoder on 20-trading-day windows. Trained only on non-event periods; events are evaluated, never used for training. Saves weights and evaluation report to `data/models/arm_b_vae/`.

Expected results:
- AUC 0.680 (Hormuz windows vs. calm)
- 2026 Hormuz detection with **38-day lead** (first p95 crossing 2026-01-23, onset 2026-03-02); reconstruction error 4.9× the p95 threshold
- The VAE is calibrated for sustained market dislocation; the four shorter anchors did not produce anomalous pricing conditions — complementary detection for those events is the role of Arm C

### 6h. Run Arm C (text embedding, no training)

```bash
python -m backend.arms.text.slugs
python -m backend.arms.text.embed
```

Zero-shot multilingual scoring on URL slugs extracted from the historical GDELT corpus. No labels consumed. AUC 0.848 on point events; highest-scoring day in the 366-day corpus is 2019-06-14 — the day after the Gulf of Oman tanker attacks.

### 6i. Build the fusion risk index

```bash
python -m backend.fusion.combine
```

Combines Arm B and Arm C via a weighted rule. Weights: Arm B 0.50, Arm C 0.50, Arm A 0.00 (CFAR baseline; SAR arm scoped for future high-resolution collection). Saves `data/derived/fusion/risk_index.parquet` and a full LOEO backtest report.

Expected backtest:
- **2 of 3 scoreable events detected** with usable lead (Abqaiq 30 days, 2026 Hormuz 43 days)
- Red Sea: undetected in the current configuration (identified as an area for Arm C enrichment)
- The two earliest 2019 events are unscoreable under causal rules (insufficient prior history at the time)

---

## 7. The optional extras

### LLM analyst brief

Requires `ANTHROPIC_API_KEY` in `.env`. Dry-run (no API cost) by default:

```bash
# Via the API
curl http://127.0.0.1:8000/brief                       # dry run, returns the grounded prompt
curl "http://127.0.0.1:8000/brief?dry_run=false"       # calls Claude, returns the brief

# Directly
python -m backend.agent.brief --dry-run
python -m backend.agent.brief                          # actual API call
```

Every number is computed by the store and passed in; the semantic registry's 25 prohibitions are injected as guardrails so the model cannot assert things this project has measured to be false.

### Kafka streaming demo

Not required by the project (measured ingest is 1.0 row/sec vs. a broker's 1,000,000 msg/s design point — 0.0001% utilisation). Built as a scale demonstration:

```bash
docker compose --profile streaming up -d kafka
python -m backend.streaming.pipeline --limit 5000
```

The pipeline benchmarks itself and reports the numbers. `/streaming/facts` serves them via the API.

### BigQuery GDELT backfill (Route B — billable)

Only needed if you want the full 2015–2026 series rather than the three anchor windows. Requires a Google Cloud account:

```bash
pip install -r backend/requirements-backfill.txt
gcloud auth application-default login
python -m backend.ingest.gdelt.backfill --probe-schema          # free, verifies column names
python -m backend.ingest.gdelt.backfill --window 2019_gulf_of_oman   # dry run (default)
python -m backend.ingest.gdelt.backfill --window 2019_gulf_of_oman --execute
```

Read the cost warning in `backend/ingest/gdelt/backfill.py` before running `--execute`. Dry-run cost estimates are the authoritative guide; the module refuses scans above 50 GB.

---

## 8. Verified environment

Tested on:

```
Windows 11 Home (10.0.26200)
Python 3.12.10
pandas 2.3.3  pyarrow 24  requests  websockets
scikit-learn  torch  tensorflow  xgboost  fastapi  uvicorn
Docker 29.6.1 + Compose v5.3.0
```

Absent from the original build environment: `gcloud`, `bq`, `google-cloud-bigquery`, `sentinelhub`, `lightgbm`. These are only needed for the BigQuery backfill, cloud deploys, or future GBM fusion work — not for running the API on existing data. (`yfinance`, `duckdb`, `Pillow`, `scipy` are in the install command above.)

---

## 9. What each endpoint does

| Endpoint | What it returns |
|----------|----------------|
| `/risk` | Weighted rule fusion of Arm B + Arm C. 2 of 3 scoreable events detected, median 36-day lead. The `honest_reading` field in the response explains the score's basis. |
| `/arms/market/anomaly` | Sequence VAE reconstruction error for the current 20-day market window. Calibrated for sustained Brent/WTI divergence — the chokepoint-specific pricing signal. 2026 Hormuz: 38-day lead. |
| `/arms/sar` | CA-CFAR vessel detection from Sentinel-1 SAR — transponder-independent, all-weather. 449 chips collected, water-masked, orbit-tagged. |
| `/predict/text` | Zero-shot multilingual contrastive score on GDELT URL slugs. AUC 0.848 on point events; highest-scoring day in the corpus is 2019-06-14 (Gulf of Oman tanker attacks). |
| `/vessels` | Vessel monitoring via satellite sources. aisstream terrestrial AIS has no Gulf receiver coverage; the endpoint explains the architecture rather than returning an empty map. Satellite SAR (Arm A) and GFW provide the vessel picture. |
| `/brief?dry_run=false` | Calls the Anthropic API to generate an analyst brief. Requires `ANTHROPIC_API_KEY`. |
| `/chips/{port}/{date}.png` | Renders a real Sentinel-1 chip. Partial chips (swath-edge clips) are flagged in the sidecar JSON. |

---

## 10. Troubleshooting

**`docker: daemon DOWN`** — Start Docker Desktop, wait ~10s, re-run.

**`uvicorn: command not found`** — `pip install uvicorn`.

**GDELT poller returns 0 Gulf rows for every slot** — check that the three-way filter (bbox OR FIPS country code OR keyword) is intact in `backend/ingest/gdelt/transform.py`. A pure geofence drops all articles datelined outside the Gulf that are still about it.

**SAR chips all return zeros** — a Sentinel Hub request matching no acquisition returns a valid zero-filled TIFF. Always call `catalog.find_passes()` first. The code does this already; if you see zeros, check the date range.

**Arm B VAE gives AUC ~0.5 after retraining** — check that β warm-up is enabled. β=1.0 from step 0 collapses the posterior and the model reconstructs every window to the training mean.

**`UnicodeEncodeError` on Windows when printing GDELT slugs** — the text corpus contains Arabic, Farsi and Cyrillic. Call `backend.common.secrets.safe_stdout()` before printing; the code does this, but interactive REPL sessions may not.

**`TypeError: 'TrackedList' object is not callable`** in Keras — a method named `_losses` on a `keras.Model` subclass shadows an internal attribute. Rename the method.

**API returns 404 for `/features`** — run `python -m backend.features.build` first. The panel is derived, not stored in git.

**API returns 503 for `/arms/market/anomaly`** — train Arm B first: `python -m backend.arms.market.vae`.

---

## 11. Architecture in one sentence per component

| Component | What it does | Validated? |
|-----------|-------------|-----------|
| `ingest/gdelt/` | Polls GDELT 2.0 (Translingual) every 15 min; writes parquet | ✅ Live-tested, 8 slots, Docker clean exit |
| `ingest/market/` | Pulls FRED + yfinance market panel | ✅ 2,204 days × 11 series |
| `ingest/satellite/` | Fetches Sentinel-1 GRD chips via Sentinel Hub Process API | ✅ 449 chips, 5,947 PU used |
| `arms/sar/cfar.py` | CA-CFAR + water mask vessel detector; orbit-tagged, land-baseline validated | ✅ 449 chips, water-masked |
| `arms/market/vae.py` | LSTM sequence VAE, anomaly by reconstruction error | ✅ AUC 0.680, 1/5 events |
| `arms/text/` | Zero-shot multilingual scoring on URL slugs | ✅ AUC 0.848, 3/3 sharp incidents |
| `features/build.py` | Daily feature panel; leakage gate enforced | ✅ 3,129 days × 19 features |
| `fusion/combine.py` | Weighted rule (B+C); LOEO backtest | ✅ 2/3 scoreable events, median 36d |
| `api/main.py` | FastAPI serving all arms, semantic layer, inference | ✅ Runs locally |
| `agent/brief.py` | Claude-generated brief, grounded in registry + store | ✅ Dry-run works; live needs key |
| `streaming/pipeline.py` | Kafka demo (not required) | ✅ Built; 16,557x slower than direct |
| `ingest/ais/client.py` | AIS WebSocket collector | ✅ Code valid; **zero Gulf coverage** |
