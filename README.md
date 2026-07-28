# Hormuz Disruption Engine

A supply-chain disruption early-warning system for the Strait of Hormuz and UAE ports,
with a live per-vessel safety layer on top.

Roughly a fifth of global seaborne oil transits Hormuz. Eleven of the UAE's thirteen
major ports and terminals sit *inside* the strait and are cut off from deep-sea trade
if it closes; only Fujairah and Khor Fakkan bypass it. This system fuses satellite
imagery, market signals, and geopolitical event text into a single chokepoint risk
index, and explains — in plain language — why that index moved.

---

## Two systems, one screen

This is a **hybrid**, and the two halves have different epistemic status. Keeping them
separate is a design requirement, not a stylistic preference.

| | Disruption engine | Vessel-safety layer |
|---|---|---|
| **Question** | "Is Hormuz about to seize up — should a shipper reroute?" | "What is this ship doing right now?" |
| **Scope** | Port-level, macro | Per-vessel, real-time |
| **Method** | Three DL arms → GBM fusion → risk index | Rule-based geofence alerts on live AIS |
| **Validated?** | Yes — backtested on historical events, strict temporal splits | **No** — live monitoring demo, makes no forecast claim |
| **Priority** | Primary. Carries all ML validity. | Secondary. Sits on top. |

**The only permitted coupling:** the disruption risk index modulates the vessel-alert
threshold (high macro risk → tighter per-vessel sensitivity). Everything else stays
walled off. If the vessel layer starts being presented as predictive, the project has
regressed into the "dashboard over substance" failure mode it exists to avoid.

---

## Architecture

Three arms, each handling a genuinely different data modality — no architectural
decoration. Each arm earns its place.

```
Sentinel-1 SAR + Sentinel-2 optical ──> [CNN]        ──> congestion_index
EIA Brent / tanker + freight series ──> [TFT/LSTM+VAE] ──> market_anomaly_score
GDELT 2.0 events + ACLED            ──> [Transformer] ──> geo_tension_score
                                              │
                                              ▼
                                     [Gradient Boosting fusion]
                                              │
                                              ▼
                                  chokepoint risk index (0–100)
                                              │
                        ┌─────────────────────┴────────────────────┐
                        ▼                                          ▼
              LLM brief (NVIDIA NIM)                    dashboard + alert feed
                                                                   ▲
                        live AIS (aisstream) ──> [rule engine] ─────┘
```

**Why each arm exists**
- **CNN** — SAR sees through Gulf haze, cloud, and night, so it does vessel *detection*;
  optical provides berth-occupancy *context* on clear days.
- **TFT/LSTM + VAE** — markets price tension before ships physically pile up, so this
  arm typically fires first. VAE flags reconstruction-error anomalies; TFT/LSTM forecasts.
- **Transformer** — extracts escalation signal from event text at 15-minute cadence.

---

## Data sources

| Source | Use | Access |
|---|---|---|
| [Copernicus Data Space](https://dataspace.copernicus.eu) | Sentinel-1 SAR + Sentinel-2 optical | Free, key required |
| [xView3](https://iuu.xview.us) | Labelled SAR dark-vessel dataset | Free |
| [aisstream.io](https://aisstream.io) | Live AIS WebSocket | Free tier, key required |
| [NOAA Marine Cadastre](https://marinecadastre.gov/ais) | Historical AIS (backtest) | Free |
| [Global Fishing Watch](https://globalfishingwatch.org) | Global research AIS | Free, registration |
| [U.S. EIA Open Data](https://www.eia.gov/opendata) | Brent, crude flows, Hormuz stats | Free, key required |
| [GDELT 2.0](https://www.gdeltproject.org) | Events + GKG, 15-min cadence | Free, no key |
| [ACLED](https://acleddata.com) | Curated conflict events (ground truth) | Free academic, key required |
| [UKMTO](https://www.ukmto.org) | Gulf maritime incident advisories | Public |

**Live-capable vs backtest-only.** Only GDELT and aisstream are true real-time feeds.
Satellite revisits the Gulf every ~1–6 days (the "live" congestion signal is really
"latest available pass"). EIA is daily/weekly. ACLED and UKMTO have reporting lag and
are **ground truth for validation, not live inputs.**

---

## Repository layout

```
.
├── CLAUDE.md                  # session context for Claude Code — read this first
├── README.md
├── docs/
│   ├── hormuz_execution_plan.html   # full spec: workflow, deliverables, stack, demo
│   ├── hormuz_architecture.html
│   └── banking_fraud_aml_usecase.html  # prior project, style reference
├── data/
│   ├── uae_ports.csv          # 13 ports + offshore terminals
│   └── uae_ports.geojson      # same, for Leaflet/MapLibre
├── ports/                     # ⚠️ misnamed — collectors AND port reference data
│   ├── ais_client.py          # live AIS WebSocket collector
│   └── gdelt_poller.py        # 15-min GDELT Events poller
├── arms/                      # (not built) cnn/, market/, nlp/
├── fusion/                    # (not built) GBM + risk index
├── api/                       # (not built) FastAPI
└── dashboard/                 # (not built) React + Leaflet
```

---

## Setup

```bash
pip install websockets pandas requests
```

**Keys.** Never commit them. Each collector reads an env var first, then a local
key file:

```
ports/aisstream.key       # aisstream key, one line, no quotes
```

Add to `.gitignore`:
```
*.key
.env
vessels_snapshot.json
```

**Run the collectors:**
```bash
python ports/gdelt_poller.py --once    # single pull
python ports/gdelt_poller.py            # 15-min loop
python ports/ais_client.py             # live vessel stream
```

---

## Current status

| Component | State |
|---|---|
| UAE ports dataset (13 ports + terminals) | ✅ Done |
| Execution plan / architecture docs | ✅ Done |
| GDELT poller | ⚠️ Built, logic unit-tested against synthetic data — **not yet run against live GDELT** |
| AIS client | ⚠️ Built, parsing tested against mock messages — **never confirmed against the live feed** |
| BigQuery historical pull | ❌ Not started — next step |
| CNN arm (SAR congestion) | ❌ Not started |
| Market arm (VAE/TFT) | ❌ Not started |
| Transformer NLP arm | ❌ Not started (poller currently uses a placeholder heuristic score) |
| GBM fusion + risk index | ❌ Not started |
| FastAPI backend | ❌ Not started |
| Dashboard | ❌ Mockup only (in execution plan HTML) |
| Backtest | ❌ Not started |

---

## Honest limitations

These are documented, not glossed. They belong in any writeup of this system.

- **Label scarcity (~5 real events).** 2012, 2019 ×2, 2024–25, and a 2026 window.
  This cannot support conventional supervised training. The engine is framed as
  **anomaly detection / early warning, not a classifier** with robust event odds.
- **Backtest leakage risk.** With so few events it is trivially easy to leak future
  information and produce a flattering, meaningless backtest. Mitigated by
  `TimeSeriesSplit` everywhere and a CI leakage test that fails the build.
- **Vessel layer is not a predictor.** Rule-based live monitoring, not backtested.
- **Transponder-dark detection is by absence.** A dark vessel transmits nothing, so
  it cannot appear in an AIS feed — it is inferred from last-seen gaps downstream.
- **Satellite cadence ≠ real-time.** ~1–6 day revisit. The dashboard must not imply
  second-by-second imagery.
- **Container OOM on rasters.** Full Sentinel scenes exceed container memory; tile
  and stream, never load whole scenes.
- **GDELT noise.** Same incident re-reported hundreds of times; tone coding imperfect;
  geographic tagging misfires. De-noise against ACLED. A raw count spike is a signal
  to corroborate, not a fact.
- **2012 anchor predates GDELT 2.0** (starts Feb 2015), so that event has no GDELT
  signal. Either drop it as a label or source it separately.

---

## Design principles

1. **Genuine multimodal justification.** Every arm handles a fundamentally different
   modality. No branch exists for decoration.
2. **Working system over dashboard.** The deliverable is a backtested, functional
   system. Visual polish is secondary to model validity.
3. **Honest limitations over optimistic framing.** Named failure modes with mitigations.
4. **Leakage prevention as a CI gate.** Temporal splits enforced; the leakage test
   fails the build.
5. **Verify before trust.** Public repos and datasets get cloned and inspected before
   they become dependencies.
