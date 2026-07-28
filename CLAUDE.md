# CLAUDE.md — Hormuz Disruption Engine

Context for Claude Code sessions on this project. Read before acting.

---

## What this is

A supply-chain disruption early-warning system for the Strait of Hormuz and UAE ports,
plus a live per-vessel safety layer. Three deep-learning arms (CNN on satellite, TFT/LSTM+VAE
on market series, Transformer on news/event text) fuse via gradient boosting into a
chokepoint risk index (0–100), which drives a dashboard and an LLM-generated brief.

Owner is a deep-learning student in the UAE building this as a production-quality
prototype, not an academic demo. **No deadline.** Quality over speed.

---

## ⚠️ Open remediation items — do these first

**1. ✅ RESOLVED — the key lives at `ports/aisstream.txt`** (not `.key`), 40-char lowercase
hex, verified valid against the live server. `ingest/ais/constants.py` searches
`aisstream.key`, `aisstream.txt` and `aisstream.key.txt`, env var first.
⚠️ **The root `.gitignore` now matches `aisstream.*` by name, not by extension.** A
`*.key`-only rule would have let `aisstream.txt` be committed — the credential has already
been saved under two different extensions, and Windows File Explorer appends `.txt`
silently. Never narrow that rule back to an extension match.
(The key being valid does **not** mean the feed works — see the Gulf coverage gotcha.)

**2. `git init` still not done.** Root `.gitignore` now exists (covers `*.key`, `.env`,
`data/raw/`, `vessels_snapshot.json`) but there is still no `.git`, so every ignore rule
in the tree remains inert. Run `git init` → `git status` → confirm no key appears → commit.

**3. Delete `ports/ais_test.py`.** Throwaway diagnostic. Findings recorded in gotchas below.

**4. Restructure — HALF DONE.** GDELT now lives in `ingest/gdelt/` (2026-07-27). The AIS
client and the port reference data are still in `ports/`, so the layout is currently
split-brain. Finish it: `ais_client.py` → `ingest/ais/`, `uae_ports.*` →
`data/reference/`, HTML → `docs/`. **Update this file in the same commit.**

---

## Actual layout (as on disk — not aspirational)

```
Final Software Prototype/
├── CLAUDE.md
├── README.md                        # ⚠️ stale: still documents ports/gdelt_poller.py
├── ARCHITECTURE.md
├── WORKFLOW.md
├── .gitignore                       # root — still inert, no .git yet
├── .dockerignore
├── .env.example                     # copy to .env; real .env is gitignored
├── docker-compose.yml               # gdelt-poller service
├── hormuz_execution_plan.html       # full spec: workflow, deliverables, stack, demo
├── hormuz_architecture.html
├── ingest/                          # collectors (containerised services)
│   ├── Dockerfile                   # poller image; non-root, healthchecked
│   ├── requirements.txt             # lean — the 24/7 container
│   ├── requirements-backfill.txt    # + GCP SDK, host-only one-shot job
│   └── gdelt/
│       ├── schema.py                # 61-col schema, FIPS, filter constants — VALIDATED
│       ├── transform.py             # three-way Gulf filter, dedup, placeholder score
│       ├── storage.py               # atomic parquet writes, partitioned raw store
│       ├── poller.py                # live 15-min service (docker entrypoint)
│       └── backfill.py              # BigQuery historical, dry-run cost gate
├── data/
│   └── raw/gdelt/                   # GITIGNORED — collected data
│       ├── live/events/dt=YYYY-MM-DD/<stamp>.parquet
│       ├── live/scores/dt=YYYY-MM-DD/scores.jsonl
│       ├── historical/<window>.parquet
│       └── _state/{poller.json,heartbeat}
└── ports/                           # ⚠️ still misnamed — AIS + reference data left here
    ├── ais_client.py                # → move to ingest/ais/
    ├── ais_test.py                  # DELETE
    ├── aisstream.key                # ⚠️ PLACEHOLDER TEXT — see remediation 1
    ├── .gitignore                   # inert
    ├── uae_ports.csv                # → move to data/reference/
    ├── uae_ports.geojson            # → move to data/reference/
    └── uae_ports_viewer.html        # → move to docs/
```

**Remaining target** (see remediation 4):
`ingest/ais/` · `data/reference/` (ports) · `arms/` · `fusion/` · `api/` · `dashboard/`
· `docs/` (HTML)

---

## Non-negotiable design rules

**1. The wall between the two systems.**
The disruption engine (backtested, carries all ML validity) and the vessel-safety layer
(live, rule-based, NOT backtested) are separate systems that share a screen. The *only*
permitted coupling: risk index modulates vessel-alert threshold. Never present the vessel
layer as predictive. Never let it into the backtest.

**2. Synthetic data never enters the backtest gate.**
Real data drives validation. Synthetic fills only sparse labels and live-window gaps, is
always tagged, and is barred from the gate. If synthetic leaks in, the gate is void — say
so rather than shipping a flattering number.

**3. `TimeSeriesSplit` everywhere. Leakage test fails the build.**

**4. Each arm must earn its place.** Every branch handles a genuinely different modality.
If a component can't be justified by the data it consumes, cut it.

**5. Honest limitations over polish.** Distinguish "built" from "verified against live
data." Right now that distinction matters enormously — see Current state.

**6. Verify before trust.** Inspect public repos/datasets before depending on them.

---

## Hard-won gotchas — do not re-break these

**🔴 aisstream.io HAS NO AIS COVERAGE IN THE GULF. Measured, not assumed.**
This is the single most consequential finding so far. Verified 2026-07-27 with a valid
40-char key, after ruling out every other explanation:

| Bounding box | Result |
|---|---|
| World, no filter | **6758 msgs / 45s** — key and connection fine |
| Europe-wide `[35,-10]-[60,30]` (Gulf-sized area) | **2533 / 25s ≈ 101 msg/s** |
| North Sea `[50,6]-[52,8]` | 23 / 25s |
| **Hormuz `[23.5,51]-[27,57.5]`** | **0 / 30s** |
| **Gulf + Arabian Sea `[10,40]-[32,70]`** | **0 / 60s** |

Ruled out: corner ordering (all four orderings silent; the same SW/NE convention works
in Europe), bbox nesting (three levels, validated in code), the message-type filter
(world box with the same filter yields 2544/30s), rate limiting (Gulf probed *first* for
60s → 0, then Europe on the same key immediately after → 101 msg/s), and a bad key.

**Why:** aisstream aggregates volunteer *terrestrial* AIS receivers. Coverage is dense
in Europe/North America and absent in the Gulf. No amount of code fixes this.

**Consequences.** The vessel-safety layer cannot run on aisstream for UAE waters, and the
"live vessel map" deliverable is not achievable from this source. Options, in order of
honesty: (a) lean on the **CNN/SAR arm**, which detects vessels regardless of transponders
or receiver coverage — this is exactly why that arm exists, and the finding *strengthens*
the architecture; (b) **Global Fishing Watch — TESTED, see below**;
(c) satellite AIS (Spire/MarineTraffic/exactEarth) — commercial; (d) demo the vessel layer
over a covered region, clearly labelled as not-the-Gulf. **Do not** show an empty Gulf map
and imply it means calm water. Note `NOAA Marine Cadastre` is US waters only — also no Gulf.

**✅ Global Fishing Watch DOES cover the Gulf — but lags 4 days.** Tested 2026-07-27 with
a live token (806-char JWT). Its AIS is satellite-sourced, not volunteer-terrestrial,
which is exactly why it succeeds where aisstream fails:

| Probe (`public-global-presence`, 90d) | Datapoints |
|---|---|
| Hormuz box | 258,281 |
| Gulf + Arabian Sea | 1,631,217 |
| English Channel control | 1,432,365 |

**Freshness is the catch.** Day-by-day probing of the Hormuz box: 2026-07-23 and earlier
return data (~6–7k/day); 07-24 through 07-27 are **empty**. That is a **4-day publication
lag**. So GFW is a **backtest and congestion-ground-truth source, NOT a live feed.**
Never label a GFW-driven panel "live".

**API contract gotchas** (both cost a debugging round, both return 422 not 401 — i.e. the
token was fine and the request was wrong): `/v3/datasets` rejects the call unless **both**
`limit` and `offset` are supplied; `/v3/4wings/report` will **not** accept `geojson` as a
query parameter — the geometry must go in the **POST body**.

**The bounding box still needs THREE bracket levels.**
```python
UAE_HORMUZ_BBOX = [[[23.5, 51.0], [27.0, 57.5]]]   # CORRECT: list of boxes
UAE_HORMUZ_BBOX = [[23.5, 51.0], [27.0, 57.5]]     # WRONG: two zero-area boxes, silent
```
The server accepts the malformed version and returns silence with no error — identical
symptom to the coverage gap above, which is why `constants.load_bbox()` now validates
nesting depth at startup and the stall warning names coverage as cause #1.

**The AIS collector code itself is VALIDATED.** Pointed at the English Channel via
`AIS_BBOX`, it pulled 1023 messages / 963 vessels / 878 persisted positions in 50s, with
MMSI merge, parquet flush and snapshot all correct. The code is not the problem.
(That European data was deleted afterwards — wrong-region data must never sit unlabeled
in the raw store.)

**GDELT uses FIPS country codes, NOT ISO.**
`Oman = MU` · `Iraq = IZ` · `Kuwait = KU` · `Bahrain = BA` · `UAE = AE` · `Iran = IR`
· `Saudi = SA` · `Qatar = QA`. Filtering on ISO silently drops Oman.

**GDELT Events files: 61 columns, tab-separated, NO header row.** Schema hardcoded as
`EVENT_COLS` in `gdelt_poller.py`. Never let pandas infer.

**Gulf filtering must be three-way: bbox OR country code OR keyword.** A London-datelined
article about the Strait of Hormuz is relevant but geolocates to the UK. Confirmed live
2026-07-27: the keyword leg uniquely caught a Yemen-datelined report of Houthi strikes on
three Saudi tankers. Pure geofencing drops exactly the reporting that matters most.

**🔴 Index time series on `DATEADDED`, NEVER on `Day`.** `Day` is when the event
allegedly happened as coded from the article; `DATEADDED` is when GDELT actually published
the record. Measured on the consolidated 2019_abqaiq window (258,948 rows fetched from
publication slots 2019-08-01→10-31):

| | `Day` | `DATEADDED` |
|---|---|---|
| range | 2018-08-01 → 2019-10-31 | 2019-08-01 → 2019-10-31 |
| distinct days | 209 | **92 — exactly the slots fetched** |
| max lag behind publication | **365 days** | — |

98.4% of rows are same-day, so the effect is small (0.4% of rows fall outside the window
by `Day`), but indexing on `Day` places information **before it was knowable** — textbook
look-ahead bias, and precisely what the CI leakage gate exists to catch. `DATEADDED` also
carries 15-minute granularity (`YYYYMMDDHHMMSS`) versus `Day`'s date-only.
`backfill_raw.consolidate()` sorts on `DATEADDED` for this reason.

**One article becomes several GDELT event rows — dedup on `SOURCEURL` before counting.**
GDELT emits a row per actor pair extracted from an article, each with its own
GlobalEventID, so raw row counts measure *re-reporting volume*, not event volume.
Measured live: 21 conflict rows from 12 distinct URLs (75% overcount); another slot,
16 rows from 10 URLs (60%). A volume feature built on raw rows will spike when wire
services repeat a story and stay flat when a genuinely new incident is reported once.

**Use GDELT Translingual, not English-only.** Arabic/Farsi coverage matters for the Gulf.

**Transponder-dark vessels cannot appear in an AIS feed.** A dark ship transmits nothing;
detection is by *absence* (last-seen gaps) downstream. Don't solve this in the collector.

**Satellite is not real-time.** Sentinel revisits the Gulf every ~1–6 days. "Live"
congestion means "latest available pass." Don't let the dashboard imply otherwise.

**Never commit keys.** Env var first, then a local `*.key` file. See remediation item 1.

---

## Environment notes

- Windows/PowerShell dev machine; VS Code. Watch for File Explorer appending `.txt` to
  extensionless files like `aisstream.key`.
- Collectors are containerised services on EC2 in production, not laptop workloads —
  a live AIS collector holds a persistent WebSocket 24/7.

---

## Geography reference

Bounding box `[[[23.5, 51.0], [27.0, 57.5]]]` covers **all 13 ports/terminals**, the
Hormuz lanes, and the Gulf of Oman approaches. Verified against `ports/uae_ports.csv`.

**The coastline split is the thesis:**
- **Inside the strait (Hormuz-dependent, 11):** Jebel Ali, Khalifa, Zayed, Port Rashid,
  Khalid, Hamriyah, Ajman, Mina Saqr, Jebel Dhanna/Ruwais, Das Island, Zirku
- **Outside (bypass, 2):** **Fujairah**, Khor Fakkan

**Fujairah is the single most important signal.** World's #2 bunkering hub, outside the
strait. When Hormuz is stressed, traffic reroutes *there* — congestion spiking at Fujairah
is the earliest *physical* confirmation of disruption.

---

## Backtest anchors (the scarce labels)

Availability verified against the live archive 2026-07-27 (HTTP HEAD on each):

| Anchor | GDELT 2.0 raw file | Note |
|---|---|---|
| 2011–12 Hormuz closure threat | ❌ 404 | **but see GDELT 1.0 below** |
| 2019-06-13 Gulf of Oman attacks | ✅ 200, 138 KB | |
| 2019-09-14 Abqaiq strike | ✅ 200, 50 KB | |
| 2024-01-15 Red Sea / Houthi | ✅ 200, 100 KB | |
| 2026 window | — | **CANDIDATE DATES FOUND — see below** |

**⚠️ UKMTO PUBLISHES NO SCRAPEABLE INCIDENT FEED.** Probed 2026-07-27: `ukmto.org` home
returns 200 but `/incidents`, `/indian-ocean/incidents`, `/rss` and `/api/incidents` all
404, and the home page contains zero parseable dates. Sourcing labels from UKMTO is a
manual reading job, not an automated one. **Use GDELT DOC 2.0 instead** — free, no key,
and it pinned the window in minutes.

**🔴 THE 2026 ANCHOR — CANDIDATE TIMELINE (from GDELT DOC 2.0 headlines, NOT yet
corroborated).** The project has been collecting live data *during an active Hormuz
crisis* without that being written down anywhere:

| Date | Headline evidence |
|---|---|
| 2026-03-12→03-24 | Iran sets transit conditions; mining allegations; US pressure builds |
| 2026-04-06→04-14 | **peak reporting**; US warships clear mines; blockade talk; IRGC communiqué 59 |
| 2026-05-06 | *"Kuwait Exports Zero Barrels of Oil for the First Time Since 1991 Gulf War"* |
| 2026-05-20 | *"first strait transit since the Iran war"* |
| 2026-05-22 | EU prepares sanctions over the closure |
| **2026-06-11** | *"Third tanker hit near Oman after US attack kills three Indian sailors"* |
| **2026-06-12** | **"Iran maritime authority announces Hormuz closure until further notice"** |
| 2026-06-14 | *"Trump Says US Reached a Deal With Iran to Reopen the Strait of Hormuz"* |
| 2026-06-19 | IEA: closure "has altered perceptions of energy security" |
| 2026-06-26 | Iran MFA says Hormuz open after IRGC warning "sparks confusion" |
| 2026-07-21 | Bab el-Mandeb declared closed; Houthi naval blockade on Saudi Arabia |

**🔴 CORRECTED BY PRICE DATA — DO NOT LABEL 2026-06-12.** The headline timeline above is
real but the June closure announcement sits in the **de-escalation**, not the onset.
Measured on FRED Brent (`DCOILBRENTEU`) against a Q4-2025 baseline of **$63.65**:

| 2026 | Brent mean | max | Brent-WTI spread |
|---|---|---|---|
| Jan | 66.60 | 72.25 | 6.55 |
| Feb | 70.89 | 73.17 | 6.38 |
| **Mar** | **103.13** | 126.69 | 11.75 |
| **Apr** | **117.29** | **138.21** | **17.66** |
| May | 107.14 | 118.26 | 5.41 |
| Jun | 85.40 | 101.69 | **0.83** |

· first +15% cross **2026-03-02** · +50% by **2026-03-06** · peak **$138.21 on 2026-04-07**
(+117% over baseline) · spread peak **25.94** (normal 3-5) · Brent **fell 24%** across the
06-05→06-25 "closure" window.

**USE THIS LABEL: onset 2026-03-02, peak 2026-04-07.** Escalation days by size:
03-12 (+12.53%), 03-18 (+8.95%), 03-05 (+8.62%), 04-07 (+8.31%), 03-02 (+8.30%).
An early-warning system labelled on the June headline peak would be scored for detecting
an event that had already ended, and its "lead time" would be meaningless.

**The Brent-WTI spread is the Gulf-specific discriminator — but ONLY SMOOTHED.**
WTI is landlocked US crude, so a widening spread prices *chokepoint* risk whereas a
parallel Brent+WTI move is a global oil story.

**🔴 CORRECTED 2026-07-28 — the RAW DAILY spread does not rank shocks correctly.** Its
all-time maximum across the whole 2018-2026 sample is **54.34 on 2020-04-20**, the day
WTI settled at **-$36.98** because the May contract expired with Cushing storage full.
That is a US futures mechanic with zero Gulf content, and it beats the 2026 Hormuz peak
(25.94) by more than 2x. Exactly one day in the entire series exceeds 25.94, and it is
that one — so **excluding it, 2026 Hormuz IS the all-time high**, but a feature ranking
on the raw daily value puts a storage squeeze above a strait closure.

**Persistence is the real discriminator.** Measured over a 10-day trailing mean:

| Shock | smoothed spread max | days raw > 15 |
|---|---|---|
| **2026 Hormuz** | **22.98** | **19** |
| 2019 Gulf of Oman | 11.87 | 0 |
| COVID crash | 11.22 | **1** |
| 2022 Ukraine | 10.21 | 0 |
| 2023-24 Red Sea | 7.04 | 0 |
| 2019 Abqaiq | 6.38 | 0 |

(baseline smoothed spread: p50 4.37 · p90 8.37 · p99 15.24)

`spread_days_gt15_30d` is the cleanest single separator found so far — 19 versus 1 versus
zero everywhere else. The $15 cut sits at the raw spread's p99 (14.46), so it is
calibrated to the sample rather than fitted to the label. Both `spread_smooth_10d` and
`spread_days_gt15_30d` are now in the feature panel.

**⚠️ The spread separates SUSTAINED SEABORNE dislocation, not "Hormuz".** 2022 Ukraine
widened it too (max 12.63) for the same structural reason — Brent is the waterborne
benchmark and Russian seaborne exports were disrupted. Geography comes from Arm C, not
from the price. Do not claim the spread alone identifies the Gulf.

**✅ THESIS VALIDATED, MEASURED.** Brent's first major move was 2026-03-02; GDELT's
article-volume cluster starts ~2026-03-12. **The market led the news by ~10 days**, which
is exactly the "markets price tension before ships physically pile up" premise that makes
Arm B fire first. Not an assumption any more.

(Retained for reference — the headline sequence, useful for narrative but NOT as the
label: 2026-06-11 tanker strike, 2026-06-12 closure announcement, 2026-06-14 reopening.)

**Treat these as CANDIDATES, not ground truth.** They are headlines GDELT indexed —
subject to every GDELT noise mode in this file (re-reporting inflation, coding errors,
mistranslation). "Announced closed" is also not the same as "physically closed", and the
06-26 confusion story shows the reporting itself contradicted. Corroborate before any of
this enters the backtest gate.

**GDELT DOC 2.0 rate limit: one request per 5 seconds.** Exceeding it returns HTTP 429
with the limit stated in the body. Sleep >=5s between calls or the run silently loses
windows.

**⚠️ CORRECTION — the 2012 anchor is NOT unreachable, only 2.0-unreachable.**
The earlier note ("predates GDELT 2.0; no GDELT signal") is right about 2.0 and wrong in
general. GDELT 1.0 serves pre-2013 data as **monthly** archives, which is why the daily
URL pattern 404s and the anchor looked dead:
```
http://data.gdeltproject.org/events/201201.zip                     # 88.8 MB, verified
http://data.gdeltproject.org/events/GDELT.MASTERREDUCEDV2.1979-2013.zip   # 1.04 GB
```
Downloaded and inspected: `201201.csv`, **57 columns** (not 61 — different schema),
`SQLDATE` as `20120101`, 114 Gulf-FIPS rows in a 1667-row sample.

**Before using it, weigh the hazard.** GDELT 1.0 is English-only — no Translingual, so
the Arabic/Farsi Gulf reporting that makes this project's filter load-bearing is absent
exactly where it matters. 1.0 and 2.0 also differ in coding pipeline and tone semantics,
so splicing them into one backtest with ~5 labels introduces a regime shift that is easy
to mistake for signal. **Recommended: pull 2012 as a tagged out-of-sample sanity check,
never into the primary gate** — the same rule that bars synthetic data.

**The promised 1979 backfile in 2.0 format never shipped.** The 2015 blog post said the
full backfile would land "in late Spring 2015". The 2.0 master list's earliest export is
`20150218230000`, so it did not. Don't plan around it.

---

## Current state

**✅ VALIDATED AGAINST LIVE DATA (2026-07-27) — the first real data in this project:**
- `ingest/gdelt/` — replaces the old `ports/gdelt_poller.py` demo script (deleted). Now a
  containerised service: slot-aligned scheduling (no drift), idempotent restarts, bounded
  catch-up after downtime, SIGTERM-aware, parquet persistence. Verified end to end —
  ran in Docker, wrote 8 slots to the host volume, healthcheck green, clean 2.5s SIGTERM
  exit. Its tension score is still the **placeholder heuristic**; Arm C replaces it.
- Live validation evidence (file `20260727110000.translation.export.CSV`, 841 rows):
  61 columns on 841/841 rows, zero ragged; semantic checks 100% (IDs numeric, URLs are
  URLs, Goldstein in range); FIPS confirmed live — a Slovakia row carried `LO`, not `SK`,
  and ISO look-alikes `OM`/`IQ`/`KW`/`BH` matched 0 rows each.
- Three-way Gulf filter confirmed **load-bearing**: the keyword leg uniquely caught a
  Yemen-datelined report of Houthi strikes on three Saudi tankers that pure geofencing
  would have dropped. Across windows the bbox leg has so far added no rows the country
  leg did not already catch — keep it anyway for open-water events with no country code,
  but know it is not currently what earns the filter its keep.

**✅ CODE VALIDATED, ❌ BLOCKED ON DATA SOURCE — `ingest/ais/`:**
- Rewritten from `ports/ais_client.py` into a service: persistence (append-only parquet
  position history + atomic snapshot), stall detection, fatal-vs-transient error handling,
  SIGTERM flush, `AIS_BBOX` override. Proven end to end on the English Channel —
  1023 msgs, 963 vessels, 878 positions persisted in 50s.
- **It returns zero for the Gulf, and that is not a bug in this code.** aisstream has no
  receiver coverage there. See the coverage gotcha. The vessel-safety layer needs a
  different data source or a different framing before it can ship.
- The old `ports/ais_client.py` is superseded. Its docstring claimed an "atomic JSON
  snapshot" it never had — in-memory dict plus print, nothing reached disk.
- `ports/uae_ports.csv` / `.geojson` — 13 ports, coast side, Hormuz-dependency flag.
  *Reference* data, not extracted data.

**✅ SATELLITE ARM — FIRST REAL IMAGERY (2026-07-27).** `ingest/satellite/` works end to
end: OAuth (`expires_in` = **1800s**, not the 10 min assumed), Process API chip request,
atomic write. Pulled a Fujairah VV+VH float32 chip for the same day's pass:

| | |
|---|---|
| chip size | **1.85 MB** vs 0.9–1.6 GB for the full GRDH product — ~500x smaller |
| cost | 3.33 processing units for 512x512 |
| VV | median 0.030 (calm water), p99.9 17.16, max 740.7 — **564x** ratio |
| VH | median 0.0073, p99.9 1.07, max 44.6 — 147x ratio |
| bright tail | 263 px above p99.9 in VV — candidate vessels |

That dynamic range — near-zero sea with a bright metal tail — **is** the detection signal.
Verify chips with pixel statistics, never with "valid TIFF": a request matching no
acquisition still returns a well-formed TIFF full of zeros.
`rasterio` is NOT installed (`tifffile` was added for inspection); the CNN arm will need
it for georeferencing.

**✅ ARM C HISTORICAL COMPLETE (Route A, raw files — no BigQuery, no billing):**

| Window | Rows | Conflict | Articles |
|---|---|---|---|
| `2019_gulf_of_oman` | 363,431 | 66,937 | 160,384 |
| `2019_abqaiq` | 258,948 | 50,215 | 113,167 |
| `2024_red_sea` | 312,569 | 63,888 | 150,930 |
| **total** | **934,948** | | **424,481** |

Sorted on `DATEADDED` (monotonic — verified), so the series is safe to index.

**The three-way filter, settled at scale.** Legs across all 934,948 rows:
`country` 653,369 · `bbox+country` 252,792 · **`keyword`-only 18,238** ·
`country+keyword` 5,557 · `bbox+country+keyword` 4,830 · **`bbox+keyword` 162**.
The keyword leg uniquely contributes 18,238 rows pure geofencing would drop. The 162
`bbox+keyword` rows carry **no country code at all** — open-water chokepoint events — which
is exactly the case the bbox leg exists for. An early 8-slot sample suggested bbox was
redundant; at scale it is not. Do not drop either leg.

**🔴 FIXED BUG — `filter_gulf` crashed on any slot with ZERO Gulf rows.** The provenance
tag was built with `DataFrame.apply(..., axis=1)`, which on an **empty** frame returns a
DataFrame rather than a Series, so assigning it raised
`ValueError: Cannot set a DataFrame with multiple columns to the single column gulf_match`.
A 15-minute slot with no Gulf events is uncommon but entirely normal, so this failed
**deterministically on exactly those slots** — in both the backfill and the live poller —
while looking like a transient network fault because it only hit ~0.4% of slots.
Two wrong diagnoses were made before the real one (first "transient HTTP", then
"rate limiting"); the low-concurrency retry disproved the second. **The lesson: a failure
that reproduces identically is not transient — surface the exception instead of
re-running.** `fetch_slot` swallowed it at `log.debug`, so it was never seen.
Now built with a list comprehension and covered by an empty-frame regression test.

**⚠️ Concurrency causes fake "permanent" failures.** 63 red_sea slots failed at
`--workers 6` and failed again identically on retry, which looked like corrupt files. They
were rate-limiting: fetched one at a time the same slots download fine. Retry stragglers at
`--workers 2`. Genuine gaps do exist too (HEAD returns a real 404 — GDELT never published
those slots); the two are only distinguishable at low concurrency.

**✅ ARM B COLLECTED (2026-07-27) — `ingest/market/`.** 2,204 days x 11 series, 2018-2026.
**A FRED key removes the need for an EIA key**: `DCOILBRENTEU` is EIA's Europe Brent Spot
FOB redistributed by the St. Louis Fed (series notes point at eia.doe.gov — verified). So
Arm B has an official, citable primary and yfinance is a legitimate supplement rather than
a rule violation.

· FRED (official=True): `DCOILBRENTEU`, `DCOILWTICO`, `DHHNGSP` — ~2,150 obs each
· yfinance (official=False): `BZ=F`, `CL=F`, and ETFs KSA/FXI/INDA/EWJ/EWY/EIS
· every row carries `source` + `official`, so downstream can filter to defensible data
· coverage: 86 / 66 / 105 / 56 days across the four anchor windows

**Weekends are NaN, never forward-filled at ingest.** Filling is a modelling decision that
belongs in the feature layer where it can be made consistently with the time splits.

**✅ ARM A BASELINE (`arms/sar/cfar.py`).** CA-CFAR with guard band, dual-pol agreement
(a detection must appear in VV **and** VH — VH suppresses sea clutter), size filtering.
On Fujairah: 53 detections (07-24, 100% cover), 57 (07-27, 100%), 44 (07-26, **59% cover
— flagged non-comparable**). Gives the CNN a baseline to beat, per design rule 4.
⚠️ Land is bright in SAR and there is no water mask yet, so counts include cranes and
quays — treat as RELATIVE day-over-day at one port, never an absolute vessel census.

**Extracted so far:** GDELT live + all three historical windows (934,948 rows), Arm B
market panel, three Fujairah SAR chips.

**Not started:** BigQuery historical execution (code written, never authenticated),
satellite acquisition, market series, CNN arm, market arm, real transformer arm,
GBM fusion, FastAPI, dashboard, backtest.

**Environment (verified 2026-07-27):** Python 3.12.10; pandas 2.3.3, pyarrow 24, requests,
websockets, sklearn, torch, tensorflow, xgboost, fastapi all present. Docker 29.6.1 +
Compose v5.3.0 working. **Absent:** `gcloud`, `bq`, `google-cloud-bigquery`, `sentinelhub`,
`yfinance`, `lightgbm`. The Docker daemon dropped one build mid-run (`rpc error … EOF`)
and succeeded on retry — if a build dies unexplained, just retry before debugging.

---

## Data extraction — next steps

Order matters: cheapest and most certain first, so failures surface early.

### Step 1 — GDELT live collection ✅ DONE (2026-07-27)
```
python -m ingest.gdelt.poller --once      # single slot, host
docker compose up -d gdelt-poller         # the 24/7 service
docker compose logs -f gdelt-poller
```
No key needed. Typical Gulf yield per 15-min slot: ~800–1050 rows total, ~20–80 Gulf,
~4–21 conflict. If a slot returns 0 Gulf rows or the entire file, the filter has broken.

**Dedup before counting anything.** GDELT emits one event row per actor pair extracted
from an article, so a single report becomes several rows with distinct GlobalEventIDs.
Measured live: 21 conflict rows from 12 distinct `SOURCEURL`s — a 75% overcount. Any
volume-based signal must dedup on `SOURCEURL` first or it spikes on re-reporting rather
than on escalation. `transform.dedup_articles` does this; the placeholder score uses it.

### Step 1b — live feed facts (verified 2026-07-27)

| | English | Translingual |
|---|---|---|
| `lastupdate.txt` | ✅ live | ✅ live |
| Master list export slots | 393,665 | 388,405 |
| Archive starts | `20150218230000` | `20150218224500` |
| Freshness when probed | slot 13:45 | slot 13:30 |

**Translingual lags English by ~15 minutes** — one slot. The poller defaults to
Translingual (Arabic/Farsi coverage is not optional here), so a slot that looks "missing"
may simply not be published yet. That is why the poller treats a 404 on the newest slot
as normal and retries next cycle rather than erroring.

**GDELT 2.0 epoch is 2015-02-18 23:00 UTC** — probed directly: 02-17 and 02-18 12:00 are
404, 02-19 12:00 is 200. The "late morning February 19" in the blog post is US local time.

### Step 2 — GDELT historical: TWO routes, and BigQuery is now optional

**Route A — raw 15-minute files (no account, no billing, no auth).** All three usable
anchors were confirmed downloadable (table above). The same validated 61-column parser
handles them, so live and historical paths cannot drift. Cost is bandwidth and wall-clock,
not money: the three anchor windows total ~366 days × 96 slots ≈ **35,000 files, ~3.5 GB**.
Parallelise modestly and it is a background job, not an interactive one. **Start here** —
it has no setup and no way to produce a surprise bill.

**Route B — BigQuery.** Warranted when you want the whole 2015→now series rather than
three windows, or GKG/Mentions joins that would mean re-downloading everything. Needs a
Google Cloud account with billing; the free query allowance is monthly (verify terms).

### Step 2b — GDELT historical via BigQuery (code written, never authenticated)
Implemented in `ingest/gdelt/backfill.py`. Public tables: `gdelt-bq.gdeltv2.events`,
`.gkg`, `.eventmentions`. Needs a Google Cloud account; **nothing here has been run**, so
the table name and column spellings are unconfirmed.
```
pip install -r ingest/requirements-backfill.txt
gcloud auth application-default login
python -m ingest.gdelt.backfill --probe-schema                     # free, verify names
python -m ingest.gdelt.backfill --window 2019_gulf_of_oman         # dry run (default)
python -m ingest.gdelt.backfill --window 2019_gulf_of_oman --execute
```
**Correction to the earlier cost note.** "Constrain on date first — date filtering is what
controls cost" is only true for a *partitioned* table. BigQuery bills by the bytes of the
**columns referenced**, and row filters prune nothing unless the table is partitioned or
clustered on that column. If `gdeltv2.events` is not date-partitioned, narrowing the date
range will **not** reduce the bill — dropping columns will. Trust the dry-run estimate
over any assumption, including this paragraph.

Guardrails already in the module: dry run is the default and `--execute` is required;
scans above `BQ_MAX_SCAN_GB` (default 50) are refused; columns are named explicitly, never
`SELECT *`; `--probe-schema` checks column names via INFORMATION_SCHEMA at no scan cost.

Conflict filtering is deliberately **not** done in SQL — the same Python
`transform.filter_conflict` runs on live and historical paths so the backtest and
inference cannot drift apart. Output: one parquet per anchor into
`data/raw/gdelt/historical/` (gitignored).

Windows in `ANCHOR_WINDOWS`: `2019_gulf_of_oman`, `2019_abqaiq`, `2024_red_sea` — each
padded before the event so lead time is measurable. The 2012 anchor is excluded (predates
the GDELT 2.0 epoch, 2015-02-18) and so is the 2026 window (dates never pinned).

### Step 3 — Satellite imagery (Copernicus) — RECONNAISSANCE DONE 2026-07-27

**The catalogue is queryable with NO credentials.** Verified live — search and discovery
are open; only *download* needs OAuth. So AOI/date planning can happen before registering:
```
https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=...   # HTTP 200, no auth
https://catalogue.dataspace.copernicus.eu/stac                            # HTTP 200
```

**Revisit at Fujairah is ~1.4 days, NOT 1–6.** Measured via the Sentinel Hub Catalog API
(`ingest/satellite/catalog.py`), 30-day window: **22 acquisitions on 22 distinct days**,
revisit gaps `[1,1,1,2,1,2,1,2,...]` — **mean 1.4, min 1, max 2**. Three satellites are
flying (**S1A, S1C, S1D**). Every pass is IW dual-pol VV+VH.

**Two orbit geometries, at two fixed local times:**
- **ascending** ≈ 14:15–14:24 UTC
- **descending** ≈ 02:06–02:15 UTC

They view the same water from opposite sides, so backscatter differs systematically
between them. Record orbit direction with every chip — a CNN trained on a mix without it
will partly learn viewing geometry rather than vessels. (`backCoeff: GAMMA0_TERRAIN`
normalises terrain, not look direction.)

**🔴 THE PARTIAL-CHIP TRAP — the most dangerous thing found in this arm so far.**
A chip clipped by the swath edge is a *valid, real* SAR image covering only part of the
AOI. Measured at Fujairah on 2026-07-26: **59% coverage**. If the congestion index counts
bright targets, that chip reports ~59% of the vessels — which reads as *"congestion fell
at Fujairah"*, the exact signal this project exists to detect, caused by nothing but the
satellite's footprint. `fetch.assess_coverage` now classifies every chip FULL / PARTIAL /
EMPTY and writes a `.json` sidecar next to the pixels. **Never compare a partial chip's
count against a full one without normalising.**

**Empty requests still cost full price.** A date with no acquisition returns a well-formed
4,603-byte TIFF of zeros and still spends the full **3.333 PU**. Over 30 days that is ~27
PU wasted on non-pass days. Catalogue search is free — always call
`catalog.find_passes()` first. `fetch` exits 2 on an empty chip so a loop cannot silently
fill the store with zeros.

**Resolution note.** 512 px over a ~16 km AOI gives **~35 m/px**, but Sentinel-1 GRDH is
natively ~10 m. At 35 m a small vessel is 1–3 px; at 10 m it is 10–30 px. Request ~1600 px
for the same AOI when cutting real detection chips — at proportionally higher PU cost.

**⚠️ Scene sizes confirm the OOM rule empirically** — do not load whole scenes:

| Product | Size | Use |
|---|---|---|
| `IW_SLC` | **7.1 GB** | interferometry — not needed, far too big |
| `IW_GRDH` (VV+VH) | **0.9–1.6 GB** | ✅ **this is the detection product** |
| `IW_RAW` | 1.7 GB | unprocessed |
| `IW_OCN` | 63–78 MB | ocean wind/wave/current, small |

**Notable inversion: SAR is FRESHER than GFW for the Gulf** — a same-day pass every 1–3
days, versus GFW's flat 4-day lag. That makes Arm A, not AIS, the timeliest source of
*physical* Gulf observation, and further strengthens the case for building it properly.

Register at dataspace.copernicus.eu for OAuth credentials (download only).
- **Sentinel-1 GRD (IW mode, VV/VH)** — SAR, all-weather vessel *detection*. Primary.
- **Sentinel-2 L2A** — optical, berth-occupancy *context*, clear days only.
- Access via `sentinelhub-py` (★906) or the STAC/OData APIs.
- **Start with Fujairah only, one date, and prove a single download completes** before
  scaling to other ports or date ranges.
- **Tile and stream. Never load a whole scene** — full rasters blow container memory.
- Define small AOI polygons per port (anchorage + berths), not whole-port bounding boxes.
- Labels/pretraining: **xView3** (iuu.xview.us) and `allenai/sar_vessel_detect` (Allen AI
  model from that challenge) — clone and inspect before depending on it.

### Step 4 — Ports / geospatial reference
`uae_ports.csv` gives *point* locations. The CNN arm needs *polygons*.
- Source or draw anchorage/berth polygons per port — OpenStreetMap (Overpass API) carries
  harbour/berth features; geojson.io works for hand-drawing.
- Check `coord_precision` in the CSV: Hamriyah and the three offshore terminals
  (Jebel Dhanna, Das Island, Zirku) are **approximate** — verify against a nautical chart
  before using them as AOI centres.
- Port-authority throughput stats (DP World, AD Ports, Fujairah) = congestion ground truth.
  Periodic release, backtest-only.

### Step 5 — Market series (Arm B)
**Primary: U.S. EIA Open Data** (eia.gov/opendata) — official, documented, free key.
Brent, crude flows, Hormuz chokepoint statistics. Daily/weekly resolution, not intraday.

**Supplement: `yfinance`** for what EIA doesn't cover — `BZ=F` (Brent futures), `CL=F`
(WTI), tanker/shipping equities as freight proxies.
⚠️ `yfinance` is an **unofficial scraper** of Yahoo's private endpoints: it breaks without
warning and its ToS position is ambiguous. Keep EIA as the defensible primary; treat
yfinance as convenience. **Never let yfinance be the sole source behind a backtest claim.**

Freight indices (Baltic Dry, tanker spot rates) are largely commercial. Use public proxies
and **document the gap** rather than implying the real series is in hand.

### Step 6 — News / event text (Arm C)
GDELT is the backbone. Supplement, don't duplicate:
- **GDELT DOC 2.0 API** — free, no key, fulltext search. Best first supplement.
- **Maritime trade RSS** — gCaptain, Splash247, Maritime Executive publish free RSS and
  carry far better signal-to-noise than general news for chokepoint events.
- **UKMTO advisories** (ukmto.org) — the Gulf's real incident reporting body. Ground truth,
  and the source for pinning the 2026 anchor dates.
- **ACLED** (acleddata.com) — curated conflict events, free academic key. **Reporting lag:
  ground truth and GDELT de-noiser, NOT a live source.**
- Commercial news APIs (NewsAPI, NewsData, Mediastack) have free tiers with meaningful
  limits and delays. **Verify current terms before designing around them** — tier terms
  change often. Only add one if GDELT + RSS demonstrably leaves a gap.

**Note:** Yahoo Finance is a *market data* source, not a news source — it belongs to Arm B.

---

## Sentinel Hub quota + the SAR sampling trap (measured 2026-07-27)

**Quota: 30,000 PU/month, 30,000 requests/month, overage = 0.** The zero overage is a
HARD CAP, not a bill — exceeding it makes requests fail. No surprise-cost risk, but a job
that overruns dies mid-run. Check with:
`GET https://sh.dataspace.copernicus.eu/api/v1/accounting/usage`

**PU scales exactly with pixel count** (measured: 256px=0.833, 512px=3.333, 1024px=13.333
— clean 4x per doubling). 1600px ≈ 32.6 PU.

**Arm A full collection = 674 passes** (4 ports x 4 windows, from the free catalogue):
512px → 2,246 PU (7%) · **1024px → 8,986 PU (30%) ← use this** · 1600px → 21,948 PU (73%,
too little headroom).

**⚠️ Catalogue `limit` caps at 100** and returns HTTP 400 above it — a long window can be
silently truncated. Use `catalog.find_passes_chunked` (45-day chunks) for anything over
~2 months.

**🔴 SAMPLING DENSITY IS CORRELATED WITH THE LABEL — a leakage vector TimeSeriesSplit
cannot catch.** Revisit per window, measured:

| Window | Fujairah | Jebel Ali | Platforms |
|---|---|---|---|
| 2019 Gulf of Oman | 3.1d | 6.8d | S1A, S1B |
| 2019 Abqaiq | 3.1d | 5.8d | S1A |
| 2024 Red Sea | 3.0d | 6.1d | S1A |
| **2026 Hormuz** | **1.4d** | **2.7d** | S1A, **S1C, S1D** |

2026 has ~2x the sampling density of every historical window (S1C/S1D are new) **and** is
an event window. A model can learn "dense sampling ⇒ 2026 ⇒ crisis" without ever seeing a
future value, so the temporal split will not flag it.
**Mitigate: aggregate SAR features to a common WEEKLY cadence; never expose raw
observation counts as a feature.**
Secondary: Fujairah/Khor Fakkan (Gulf of Oman coast) get ~2x the passes of
Jebel Ali/Khalifa (Gulf coast) in EVERY window — same swath geometry, stable over time, so
less dangerous, but per-port counts are still not directly comparable.

## Feature layer + splits (built 2026-07-27)

`features/build.py` -> `data/derived/features_daily.parquet` — 3,129 days x 19 features,
2018-01-02 → 2026-07-27. `features/splits.py` — TimeSeriesSplit with embargo + fold audit.

**🔴 PANDAS SILENTLY FABRICATES DATA — TWO SEPARATE WAYS, BOTH HIT THIS PROJECT.**
Caught only because a derived column had MORE non-nulls than its own source:
1. `pct_change()` pads NaNs **by default** → invents 0% returns on non-trading days and
   runs past the last close. Always pass `fill_method=None`.
2. `rolling()` over a REINDEXED daily calendar keeps emitting values after the source
   ends, because a trailing window still finds enough points. Mask every rolling output
   with `.where(source.notna())`.
`check_leakage` now asserts **no derived feature may outlive or out-count its source**.
That single check caught both; keep it.

**Feature notes.** GDELT aggregates on `DATEADDED` only. `brent_wti_spread` is the
Gulf-specific discriminator (WTI is landlocked; a widening spread prices chokepoint risk,
a parallel move is a global oil story). SAR uses `n_on_water` and **drops** partial chips
rather than rescaling — vessels cluster in the anchorage, so rescaling by covered area
assumes a uniformity that does not hold.

**Splits carry an EMBARGO, not just ordered folds.** `brent_vol_7d` on the first test day
is computed from days that are in TRAIN, so plain `TimeSeriesSplit` still bleeds. Default
embargo 10 days. `audit_folds` additionally rejects any fold where an event **straddles**
the boundary — with ~5 labels, a model that has seen half a crisis and is then scored on
the other half looks skilful and is not.

**✅ RESOLVED 2026-07-28 — the fold strategy is CAUSAL LEAVE-ONE-EVENT-OUT.**
`leave_one_event_out_splits()` gives one fold per event by construction, so the
empty-fold problem cannot arise. `audit_loeo()` PASSES with 3 of 5 folds scoreable;
the two earliest 2019 events are unscoreable for want of prior history. The uniform
split is retained in `main()` as a printed COUNTER-EXAMPLE, never used — keeping its
failure visible is what stops the lesson being lost. The pipeline gate is now green.

**🔴 CAUSAL vs POOLED LOEO IS A REAL CHOICE.** Pooled LOEO (train on every other event,
including LATER ones) is standard for classifiers and indefensible for an early-warning
system: a 2019 event scored by a model that has seen 2026 is not a forecast. `causal=True`
is the default; `causal=False` tags every fold `anti_causal: True` so a retrospective
number cannot be quoted by accident.

**⚠️ SUPERSEDED — the original problem, kept for context:** Events cluster in 2019, 2024 and 2026,
so an even 5-fold split gives folds 1 and 2 test windows (2020-02→2021-10, 2021-10→2023-06)
containing **no labelled event at all**. Scoring them says nothing about early-warning
skill. Decide deliberately: fewer folds, event-aware splitting (each test fold must contain
>=1 event), or report only event-bearing folds and say so. Do not average across empty
folds and quote it as accuracy.

## Deep-learning layer

### ✅ Semantic layer (`backend/semantic/registry.py`, 2026-07-28)
22 metrics, **100% of the feature panel defined**, 22 encoded prohibitions. Each metric
carries definition · provenance · caveats · **forbidden uses**. The `forbidden` field is
the valuable one — every entry was learned by getting it wrong and measuring the cost.
Served at `/semantic/metrics`, `/semantic/forbidden`, `/semantic/audit`. An undefined
column is a number nobody can defend; `audit_frame` refuses to let one reach a model
silently.

### ✅ ARM B — sequence VAE (`backend/arms/market/vae.py`, 2026-07-28)
LSTM encoder → 8-d latent → LSTM decoder, 20-trading-day windows, 9 stationary market
features. **Trained ONLY on non-event windows** (1,302 of 1,896, 60-day buffer), scored
by reconstruction error. Unsupervised by design: with ~5 labels a supervised classifier
cannot be validated, so labels are used to EVALUATE and never to train.

| | |
|---|---|
| AUC (Hormuz windows vs calm) | **0.680** |
| Hormuz windows above p95 | **22.0%** vs 5.1% calm — 4.3x enrichment |
| **2026 Hormuz lead time** | **38 days** (first p95 cross 2026-01-23, onset 03-02) |
| false-positive rate | 0.74% (14 windows, mostly 2022 Ukraine) |
| recon 0.532 · KL 2.40 | no posterior collapse |

**🔴 THE BUFFER MUST EXCEED THE LEAD TIME BEING CLAIMED.** At the original 30-day buffer
the 38-day-early crossing sat *outside* the buffer, so the flagged windows were in the
TRAINING set — the model was being credited with detecting a run-up it had trained on.
Widened to 60 days and retrained: the detection **survives**, and the windows are now
genuinely held out. AUC fell 0.746→0.680 (more buffer/run-up days now count as
"positive", diluting the class) but the detection is unchanged and now defensible.
Held-out scores across the crossing: 0.313 (01-22) → **1.548** (01-23) → 2.42 (01-27),
against a p95 threshold of 1.089 — a 4.9x single-day step.

**🔴 ONLY 1 OF 5 EVENTS IS DETECTED.** Peak scores in the 60-day pre-onset window:
2026 Hormuz **2.92** (fires); Abqaiq 0.95, Red Sea 0.67, Gulf of Oman 0.62, Fujairah 0.42
— all below the p95 threshold of 1.089. Arm B detects *large market dislocations*, and
four of the five anchors simply did not move the market enough to be anomalous. **Do not
report "38 days of lead time" without this sentence beside it.** It is one event.
This is the strongest argument in the project for why Arm C exists.

**🔴 COVID CONTAMINATES "NORMAL" IF NOT DECLARED.** First training run put the top 10
"normal" windows all in April–May 2020, which set p99 at **17.0** against a p95 of 1.42 —
a detector calibrated on a pandemic would miss every Hormuz event. COVID is now a
**declared, reasoned exclusion** (`NON_HORMUZ_EXCLUSIONS`), justified by an exogenous
cause (demand collapse, no chokepoint component), still scored, and reported: it fires
harder than any Hormuz window (median 4.74 vs 0.60). Keep that list SHORT — excluding
every large move that isn't the target defines "normal" as "everything unlike our signal"
and guarantees detection.

**β warm-up is load-bearing.** β=1.0 from step 0 gave KL **0.134 nats across 8 dims**
(~0.017/dim) — near-total posterior collapse, latent carrying nothing. β annealed 0→0.1
over the first third of training gives KL 2.64 and recon 0.92→0.53.

**Keras 3 gotcha:** naming a method `_losses` on a `keras.Model` subclass silently
shadows `Layer.__init__`'s internal `self._losses = []`, and the call fails with
`TypeError: 'TrackedList' object is not callable`.

### ✅ ARM A DATA COLLECTED — and it produces a NEGATIVE RESULT (2026-07-28)
`backend/ingest/satellite/collect.py` — **449 chips** at 1024px across 4 ports x 4 anchor
windows, **5,947 PU (20% of the monthly allowance)**, 371 FULL / 78 PARTIAL, orbit
direction recorded on every chip (223 ascending / 219 descending). CFAR + water mask run
over all of them.

**✅ The water mask is validated by land counts.** Land detections are near-constant per
port across all four windows — Fujairah 157/158/164/171, Jebel Ali 175/176/178/179. They
really are the same cranes and quays every pass, exactly as the masking rationale claimed.

**🔴 THE FUJAIRAH REROUTE THESIS IS NOT VISIBLE IN CFAR VESSEL COUNTS.** Median vessels
on water across the 2026 onset (2026-03-02):

| Port | pre-onset | post-onset | change |
|---|---|---|---|
| Fujairah (bypass) | 30.0 | 30.0 | **+0.0%** |
| Khor Fakkan (bypass) | 8.0 | 8.0 | **+0.0%** |
| Jebel Ali (inside strait) | 109.0 | 105.0 | −3.7% |
| Khalifa (inside strait) | 38.0 | 36.5 | −3.9% |

Aggregated, the result runs **backwards** from the thesis: bypass ports −12.5%, inside-strait
ports +6.1%. The detector is not broken — Fujairah counts range 14–63 with std 11.5, and
the water mask is stable (water_frac std 0.037). There is simply no reroute signal in the
*count* of bright targets.

**Confirmed with orbit held fixed and on four different measures** (count, total detected
area, large-vessel count, large-vessel area). Selected orbit-matched changes across the
2026 onset:

| Port | measure | ascending | descending |
|---|---|---|---|
| Fujairah | count | −12.5% | −3.3% |
| Fujairah | area | −2.8% | **+4.8%** |
| Khor Fakkan | area | −20.1% | −30.1% |
| Khalifa | count | +3.3% | +18.8% |

**The signs flip between orbits for the same port and measure**, which puts the effect
firmly inside noise. Bypass ports fall, inside-strait ports rise — the opposite of the
thesis — on no consistent measure. This is a well-controlled null, not a missing feature.

**Quantified with Mann-Whitney U** (one-sided, H1 = bypass ports increase), orbit-matched:

| Port | orbit | n pre/post | p | effect r |
|---|---|---|---|---|
| Fujairah | descending | 15/20 | 0.674 | **−0.087** |
| Fujairah | ascending | 6/9 | 0.616 | **−0.074** |
| Khor Fakkan | descending | 15/20 | 0.887 | **−0.237** |
| Khalifa | descending | 7/10 | 0.263 | +0.200 |

Nothing significant, all effects small (|r| ≤ 0.24), and **both bypass ports are negative
in both orbits** while inside-strait ports are mildly positive in 3 of 4 cells — the data
leans consistently *against* the thesis, not merely "no signal".

**Power: this rules out a LARGE effect, not a subtle one.** With n≈15/20 per group and
sd≈11 vessels, the detectable effect at 80% power is roughly a **10-vessel (33%) shift**.
A dramatic reroute would have shown; a 10-15% one would not. State the null with that
bound attached.

**⚠️ THE "AOI IS TOO SMALL" EXPLANATION IS WEAKENED BY ITS OWN TEST.** If the 8 km box
truncated the anchorage, detections would pile against the boundary. Measured share of
water detections in the outer 20% ring (a uniform distribution gives 36.0%):

| Port | outer ring | reading |
|---|---|---|
| Fujairah | **30.5%** | slightly CENTRE-heavy — box not obviously truncating |
| Khalifa | 26.7% | centre-heavy |
| Jebel Ali | 36.2% | uniform |
| Khor Fakkan | 44.2% | edge-heavy — this one may be clipped |

Fujairah, the port the thesis rests on, is centre-concentrated. So re-cutting larger
anchorage AOIs (~14,000 PU, ~47% of the monthly allowance at 1600 px) is **not** clearly
justified by the evidence — only Khor Fakkan looks clipped. Do not spend that quota on a
hunch this test does not support.

Secondary: at 1024 px over a ~16 km AOI the resolution is ~15.6 m/px, so a small vessel is
1–2 px and only a ~300 m tanker reaches 20 px. The `n_big`/`area_big` measures above are
therefore built on 0–1 detections per chip and carry no weight.

As measured, Arm A contributes **no crisis signal**, and the "earliest physical
confirmation" claim for Fujairah is unsupported by this project's own data. CFAR remains
the baseline the CNN must beat — a null baseline is still a baseline.

**🔴 ORBIT GEOMETRY IS A 2x ARTIFACT AT KHALIFA.** Median vessels by look direction:

| Port | ascending | descending | desc/asc |
|---|---|---|---|
| **Khalifa** | **51.0** | **26.0** | **0.51** |
| Jebel Ali | 100.0 | 94.0 | 0.94 |
| Fujairah | 30.0 | 30.0 | 1.00 |
| Khor Fakkan | 7.0 | 8.0 | 1.14 |

Same port, same water, **half the vessels** depending only on which side the satellite
looked from. Comparing an ascending pass against a descending one at Khalifa manufactures
a 2x "congestion change" out of nothing. **Normalise per port per orbit before any
comparison**, and never mix orbits in a single series. The effect is absent at the other
three ports, so it cannot be handled with one global correction.

### 🔴 ARM A CNN — BUILT, MEASURED, DOES NOT EARN ITS PLACE YET
`backend/arms/sar/patches.py` + `cnn.py`. 12,687 patches (64x64x3) cut from CFAR
detections across 371 acquisitions, three classes: `vessel_candidate` (CFAR detection on
water), `infrastructure` (CFAR detection on land — **reliable labels**, they are fixed
cranes and quays), `sea_clutter` (random water, no detection). Split by ACQUISITION, never
by patch — patches from one chip share sea state, speckle and vessels.

MobileNetV2 (ImageNet) transfer, two-stage: frozen-conv head training, then top-30
fine-tune. Chosen over EfficientNet because there is no GPU (TF >= 2.11 has no native
Windows GPU support at all) and depthwise-separable convolutions cost ~8-9x fewer MACs:
`1/N + 1/D_k²` = ~0.115 at `D_k`=3, N=256.

| Model | Test accuracy |
|---|---|
| **Random forest on brightness statistics** | **0.946** |
| MobileNetV2 transfer (final) | 0.837 |
| MobileNetV2, frozen BatchNorm | 0.787 |
| MobileNetV2, one NaN patch in the data | 0.350 |

**The CNN LOSES to a trivial baseline, and design rule 4 says say so.** The likely reason
is structural, not a hyperparameter: at 1024 px over a ~16 km AOI the resolution is
~15.6 m/px, so a vessel is **1–3 pixels** — a point source with no shape. CNNs exploit
spatial structure; here there is almost none, and centre-brightness statistics capture
essentially all the available information. This matches the resolution note already in
this file (`~1600 px for real detection chips`); 1024 px was collected instead, at
13.3 PU/chip versus 32.6.

**🔴 ONE NaN PATCH IN 12,687 DESTROYED TRAINING.** `np.clip` PROPAGATES NaN, so a single
non-finite pixel survived normalisation, made one batch's loss NaN, and one NaN gradient
step turned every weight NaN permanently. The model then emitted a constant: 35%
"accuracy" with recall 1.0 on one class and 0.0 on the other two — which reads like a hard
task, not a corrupted model. **The hand-feature baseline is what exposed it** (94% on
identical splits). Both `patches.to_rgb` and `cnn.load` now sanitise/refuse non-finite
input. Always run a trivial baseline beside a network; it is a correctness check, not just
a comparison.

**🔴 FROZEN BATCHNORM COSTS 5 POINTS.** MobileNetV2's BN layers carry ImageNet moving
statistics, which are wrong for SAR dB imagery. `trainable=False` plus `base(x,
training=False)` pins them there. Letting BN adapt while keeping conv weights frozen:
0.787 -> 0.837.

### ✅ ARM C — ZERO-SHOT SEMANTIC SCORING. THE BEST-PERFORMING ARM (2026-07-28)
`backend/arms/text/slugs.py` + `embed.py`.

**🔴 THE TEXT WAS ALREADY ON DISK.** GDELT Events carries no article body, and the DOC
2.0 headline API is not reliably reachable. But **228,299 of 424,473 archived SOURCEURLs
(53.8%) yield four or more alphabetic words** — news CMSs put the headline in the URL
path. Free, multilingual, no API, already collected:

    tbmm baskani mustafa sentop bagdata gidecek
    arabia saudi iran amenazar seguridad regional mundial

Scoring is **zero-shot and contrastive**: cosine similarity to 10 disruption reference
sentences MINUS similarity to 6 ordinary maritime/energy sentences, aggregated as the mean
of each day's top 10. The contrast is what stops it being a volume proxy. No labels are
consumed, so none are burned — same discipline as the Arm B VAE.

**Results — 43,920 slugs sampled across 366 days (120/day cap so busy days do not
dominate):**

| Test | Result |
|---|---|
| **AUC, point events vs quiet days** | **0.848** |
| precision@20 (all events) | 0.90 vs 0.53 base rate — **1.71x lift** |
| AUC, all event days incl. Red Sea campaign | 0.602 — see caveat |

Per-event peak against that event's OWN trailing-30d baseline:

| Event | peak | z |
|---|---|---|
| 2019-05-12 Fujairah attacks | 0.4474 | **+3.39** |
| 2019-06-13 Gulf of Oman | 0.5108 | **+4.68** |
| 2019-09-14 Abqaiq | 0.4596 | **+3.51** |
| 2023-11-19 Red Sea | 0.3937 | +0.86 (missed) |

**The single highest-scoring day in the whole 366-day corpus is 2019-06-14 — the day after
the Gulf of Oman tanker attacks.** Every one of the top 12 days maps to a real maritime
incident (Abqaiq takes five slots, Red Sea three, Fujairah one, and 2019-10-11 is the
Sabiti tanker explosion).

**⚠️ THE 0.602 AUC IS AN ARTEFACT OF THE LABEL, NOT A WEAKNESS.** The Red Sea window spans
2023-11-19→2024-03-31, so **53% of the corpus counts as "event"** and the discrimination
task becomes nearly vacuous. Report the point-event AUC (0.848) with the label definition
stated, never the 0.602 alone.

**🔴 ARMS B AND C ARE COMPLEMENTARY — MEASURED, NOT ASSUMED.** This is the empirical
justification for the multi-arm architecture that design rule 4 demands:

| | sharp incidents (2019) | sustained dislocation (2026) |
|---|---|---|
| **Arm B** (market VAE) | ❌ all below threshold | ✅ 38-day lead |
| **Arm C** (semantic) | ✅ z = 3.4–4.7 | to be tested |

Each arm detects what the other misses. Neither alone covers the event set.

**Honest limitations.** Slugs are a DEGRADED proxy for headlines — lowercased, stripped of
function words, often truncated. The 46% of URLs that are opaque numeric IDs are
**missing-not-at-random**: outlets on numeric-ID CMSs are systematically excluded and skew
by region and language. Check the domain mix before calling coverage representative.

### Fold strategy blocks LESS than previously stated (corrected 2026-07-28)
It gates **GBM fusion and the backtest** — anything trained or scored on the ~5 event
labels. It does **not** gate Arm A (xView3 carries its own labels and splits), Arm B
(unsupervised — needs event *dates* to define "normal", not folds), or Arm C
(self-supervised embeddings). Those three can be built before the fold decision is made.

### ✅ FUSION + LOEO BACKTEST (`backend/fusion/combine.py`, 2026-07-28)
**A WEIGHTED RULE, NOT A TRAINED MODEL.** Five labels cannot validate a fitted
combiner, so weights come from each arm's separately measured performance and are not
tuned. Arm B 0.50 · Arm C 0.50 · **Arm A 0.00** (zero weight, kept in the schema so the
null is visible rather than quietly dropped). Scores are converted to **causal expanding
percentiles** — a full-sample rank would rank 2019 against 2026 data that did not exist.

**CAUSAL leave-one-event-out, threshold from quiet days excluding all events:**

| Event | status | peak pre-onset | lead |
|---|---|---|---|
| 2019 Fujairah | **UNSCOREABLE** | — | — |
| 2019 Gulf of Oman | **UNSCOREABLE** | — | — |
| 2019 Abqaiq | ✅ detected | 97.01 | **30 d** |
| 2024 Red Sea | ❌ missed | 79.91 | — |
| **2026 Hormuz** | ✅ detected | 96.84 | **43 d** |

**HEADLINE: 2 of 3 scoreable events detected with usable lead, median 36 days.**

**🔴 "UNSCOREABLE" IS NOT "MISSED".** Under causal LOEO an event needs enough PRIOR
history to have been predictable at the time. The two earliest 2019 events have only
298 and 321 days of prior panel history, so no honest score exists for them. An earlier
version of this backtest scored all five and reported **"3 of 5 detected, median lead
30 days"** — that credited the system with a 2019 Gulf of Oman detection at 0 days lead
that it could not have made in 2019, and 0 days is not early warning anyway. Excluding
them is the honest move even though it shrinks the denominator.

**⚠️ THE MINIMUM-HISTORY BAR DEPENDS ON WHETHER ANYTHING IS FITTED.** `min_train_days`
=365 is right for the GBM (which is fitted) and wrong for the weighted rule (which
trains nothing). Applying the fitted-model bar to the rule made three events unscoreable
for a reason that did not apply. The rule's real constraint is the causal percentile
warm-up (120 days), already enforced inside `causal_rank`.

**🔴 FUSION BEATS EITHER ARM ALONE — measured, and the justification for the
architecture.** On 2026 Hormuz: Arm B alone 38 d, Arm C alone **2 d**, fused **43 d**.
The complementarity is now measured on the SAME event:

| | sharp incidents (2019) | sustained dislocation (2026) |
|---|---|---|
| **Arm B** (market VAE) | ❌ all missed | ✅ 38 d lead |
| **Arm C** (semantic) | ✅ z = 3.4–4.7 | ⚠️ only 2 d lead |
| **Fused** | ✅ Abqaiq 30 d | ✅ **43 d** |

**The GBM comparison is degenerate, and that IS the finding.** Labelling event windows
+30 d makes **233 of 282 rows positive (83%)**, so 2 of 3 folds are single-class and
un-scoreable; the one that scored gave AUC 0.643. Reported per-fold, never averaged.
This is the concrete demonstration that 5 labels cannot support a fitted combiner.

### ✅ AGENT / BRIEF LAYER (`backend/agent/brief.py`)
Claude-generated analyst brief. **The registry's 25 prohibitions are injected into the
system prompt as guardrails** — the payoff for building the semantic layer. Every number
is computed here and passed in; the model is asked to explain and prioritise, never to
recall. Nulls and unbuilt components are passed in explicitly so absence of evidence is a
fact it can state rather than a gap it fills. `--dry-run` inspects the grounded prompt
without an API call. Needs `ANTHROPIC_API_KEY` in `.env`.

### ✅ KAFKA — BUILT AND HONESTLY LABELLED (`backend/streaming/pipeline.py`)
**Not justified by this project's volume, and the code says so.** Measured ingest is
**1.0 row/sec** (88,800 rows/day) against a single broker's ~1,000,000 msg/s design point
— **0.0001% utilisation**. Benchmarked against the direct path on real rows: Kafka
measured **16,557x slower**.

**Be fair when quoting that number:** most of the 8.4 s is FIXED cost (topic creation,
consumer-group join, metadata fetch, rebalance) that does not scale with message count.
It is not Kafka's throughput. The honest conclusion is narrower and still decisive: at
1 row/sec the fixed overhead dominates and the durability/replay/fan-out a broker buys
are all unused. Profile-gated (`--profile streaming`) so nobody starts a broker by
accident. `/streaming/facts` serves the numbers so the framing travels with the component.

## Honest limitations (carry into any writeup)

- **Label scarcity (~5 events).** Cannot support conventional supervised training. Framed
  as anomaly detection / early warning, **not** a classifier with robust event odds.
- **Backtest leakage risk.** Trivially easy with so few events. `TimeSeriesSplit` + CI gate.
- **Vessel layer is not a predictor.** Rule-based live monitoring, never backtested.
- **Satellite cadence ≠ real-time.** ~1–6 day revisit.
- **Container OOM on rasters.** Tile and stream.
- **GDELT noise.** Same incident re-reported hundreds of times; tone coding imperfect;
  geotagging misfires. A raw count spike is a signal to corroborate, not a fact.
- **2012 anchor predates GDELT 2.0.** No GDELT signal for it.
- **Freight-rate series are commercial.** Public proxies only; state the gap.
- **Arm B detects 1 of 5 anchors.** Only the 2026 Hormuz event moved the market enough
  to be anomalous. Any lead-time claim must say "on one event".
- **The market arm is not Gulf-specific.** It fires on Ukraine and COVID too. Only Arm C
  supplies geography.
- **Arm A shows no reroute signal.** Measured on 449 chips: Fujairah vessel counts are
  unchanged (+0.0%) across the 2026 onset, and the bypass-vs-strait split runs backwards
  from the thesis. State this rather than implying the satellite arm confirms disruption.
- **Orbit direction is a 2x confound at Khalifa.** Never compare passes of different
  look direction without per-port normalisation.
- **The SAR CNN loses to a brightness baseline** (0.837 vs 0.946). At ~15.6 m/px a vessel
  is 1-3 px, so there is no spatial structure for convolution to exploit. Do not present
  the CNN as a working detector on this data.
- **GDELT DOC 2.0 is not reliably reachable.** Measured: 30s spacing 1/5 success, 60s
  spacing 2/4. The documented "one request per 5 seconds" is not achievable on the shared
  public endpoint under load, which caps how much headline text Arm C can obtain.

---

## Working style

- Push back on scope creep and on framing that overstates what's validated.
- Prefer real public datasets with implementation evidence over synthetic placeholders.
- Advanced math welcome where genuinely justified (VAE ELBO/KL, LSTM gates, scaled
  dot-product attention, convolution dimensions) — not as decoration.
- HTML deliverables: dark-themed, self-contained, navigation, diagrams, tables, numbered
  references with live URLs. Style reference: `hormuz_execution_plan.html`.
- When something can't be verified from the current environment, say so explicitly.
