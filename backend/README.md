# Backend — Hormuz Disruption Engine

Everything server-side: collectors, model arms, feature layer, and the API.
The frontend consumes this over HTTP and holds no data logic of its own.

```
backend/
├── common/          project paths + credential loading (shared)
├── semantic/        metric registry — definitions, provenance, PROHIBITIONS
├── ingest/          collectors — one subpackage per source
│   ├── gdelt/       Arm C · 15-min events ✅ · DOC headlines ⚠️ rate-limited
│   ├── market/      Arm B · FRED + yfinance          ✅ collected
│   ├── satellite/   Arm A · Sentinel-1 SAR chips     ✅ 449 chips collected
│   ├── gfw/         Global Fishing Watch (backtest)  ⚠️ 4-day lag
│   └── ais/         vessel layer                     ❌ no Gulf coverage
├── arms/
│   ├── sar/         CFAR + water mask + patch CNN (MobileNetV2)
│   ├── market/      sequence VAE (LSTM) + evaluation
│   └── text/        multilingual sentence embeddings
├── features/        common daily index + temporal splits
├── api/             FastAPI + DuckDB serving layer
├── Dockerfile
└── requirements.txt
```

Data lives **outside** this folder, at `<root>/data/`, because it is shared
state rather than backend source. `common/paths.py` resolves the root by
searching upward for a marker file, so no module depends on how deep it sits.

---

## Run it

```bash
# API  (from the project root)
uvicorn backend.api.main:app --reload --port 8000     # docs at /docs

# collectors
docker compose up -d gdelt-poller                     # 24/7, every 15 min
docker compose up -d api                              # containerised API

# one-shot jobs
python -m backend.ingest.market.collect               # Arm B
python -m backend.ingest.satellite.catalog --port fujairah --days 30
python -m backend.ingest.satellite.fetch  --port fujairah --date 2026-07-27
python -m backend.arms.sar.run_cfar                   # CFAR baseline
python -m backend.features.build                      # feature panel
python -m backend.features.splits                     # fold audit
```

---

## API surface

| Endpoint | Returns |
|---|---|
| `GET /health` | liveness |
| `GET /freshness` | **per-arm as-of + staleness** |
| `GET /sources` | provenance: official vs unofficial |
| `GET /features` | daily feature panel (`start`, `end`, `columns`) |
| `GET /arms/gdelt` | daily GDELT aggregates on `DATEADDED` |
| `GET /arms/market` | Brent, WTI, spread, ETFs |
| `GET /arms/sar` | CFAR congestion per port/date |
| `GET /events` | backtest label calendar |
| `GET /risk` | **501 — fusion not built** |
| `GET /vessels` | **501 — no Gulf AIS exists** |

### Why two endpoints deliberately fail

`/risk` returns 501 because the GBM fusion does not exist. A placeholder index
would be indistinguishable from a real one in a UI and would end up quoted in a
writeup. It fails loudly until there is a validated model behind it.

`/vessels` returns 501 because no free source provides live Gulf vessel
positions. Measured 2026-07-27: aisstream returned 0 messages in 30s for the
Hormuz box while a comparably sized European box returned ~101 messages/second
on the same key. This matters more than a blank panel would suggest —
transponder-dark detection works by *absence*, so with no tracks at all every
vessel reads as dark. An empty map would be confidently wrong.

### Freshness is a first-class endpoint

Only **one** arm is genuinely live:

| Arm | Cadence | Live? |
|---|---|---|
| GDELT | 15 minutes | ✅ |
| Market | daily close | ❌ ~1 day behind |
| SAR | 1.4–6 day revisit | ❌ "latest pass" |
| Vessel | — | ❌ no data |

A dashboard that renders all four identically implies a real-time physical read
of the Gulf that does not exist. Every panel must show its own age.

---

## Design rules the code enforces

**Index on knowability, not occurrence.** GDELT aggregates on `DATEADDED`
(publication), never `Day` (alleged event date) — `Day` trails publication by
up to 365 days, so using it places information before anyone could have had it.

**Never fill silently.** Weekends and holidays stay `NaN` all the way to the
API. Two pandas defaults fabricate data and both were caught here:
`pct_change()` pads NaNs unless given `fill_method=None`, and `rolling()` over a
reindexed calendar keeps emitting values after its source ends. `check_leakage`
asserts no derived feature may outlive or out-count its source.

**Dedup before counting.** GDELT emits one row per actor pair per article, a
60–75% overcount. Volume features count distinct `SOURCEURL`.

**Water-mask SAR before counting vessels.** Land detections are the same cranes
and quays every pass (measured near-constant at 39/37/38 on Fujairah). Masking
turned a 53→57 day-over-day change (+7.5%) into 14→19 (+36%) by removing a
large fixed offset.

**Embargo the splits.** Plain `TimeSeriesSplit` still bleeds: `brent_vol_7d` on
the first test day is computed from days sitting in train. Default embargo is
10 days, and `audit_folds` rejects any fold where an event straddles the
boundary.

---

## Deep-learning layer

```bash
python -m backend.arms.market.vae          # Arm B  sequence VAE
python -m backend.arms.market.evaluate     #        AUC / lead time / specificity
python -m backend.ingest.satellite.collect --plan-only   # cost from FREE catalogue
python -m backend.arms.sar.patches         # Arm A  CNN patch dataset
python -m backend.arms.sar.cnn             #        MobileNetV2 transfer learning
python -m backend.ingest.gdelt.headlines   # Arm C  DOC 2.0 headlines
python -m backend.arms.text.embed          #        multilingual embeddings
```

**Everything unsupervised or zero-shot, on purpose.** With ~5 labelled events a
supervised model cannot be validated — any accuracy would be an artefact of
which fold the events fell in. So Arm B trains only on non-event windows and
scores by reconstruction error; Arm C expresses the target concept as reference
sentences and scores by cosine similarity. Neither consumes a label, so the
labels stay available for evaluation. Only the GBM fusion and the backtest
depend on the fold decision.

**Measured, with the caveats attached:**

| Arm | Result | The caveat that must travel with it |
|---|---|---|
| B (VAE) | AUC 0.680, **38-day lead** on 2026 | Detects **1 of 5** anchors; fires on COVID and Ukraine too |
| A (CFAR) | land counts near-constant → mask validated | **No reroute signal**: Fujairah +0.0% across the onset |
| A (CNN) | separates infrastructure / clutter / vessels | vessel labels are CFAR's own output — agreement is circular |
| C (embed) | EN↔AR cosine 0.866 vs 0.073 noise | headline collection blocked by DOC rate limits |

## Known gaps

- **GBM fusion, risk index, backtest, leakage gate** — not built.
- **Fold strategy undecided.** Events cluster in 2019/2024/2026, so a uniform
  5-fold split leaves folds 1 and 2 testing on windows with *no* labelled event.
- **SAR sampling density correlates with the label.** 2026 has ~2× the revisit
  of historical windows *and* is an event window; a model could learn
  "dense sampling ⇒ crisis". Mitigate by aggregating to a weekly cadence and
  never exposing observation counts as a feature.
- **Arm A is thin** — 3 chips at 1 port. Full collection is 674 passes ≈ 8,986
  PU at 1024px (30% of the 30,000/month allowance).
