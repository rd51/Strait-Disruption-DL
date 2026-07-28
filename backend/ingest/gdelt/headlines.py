"""
Arm C text source — article HEADLINES via the GDELT DOC 2.0 API.

WHY THIS EXISTS. The GDELT Events store this project already holds (934,948
rows) carries no article text at all — its columns are actor names, CAMEO
codes, tone scores and a SOURCEURL. Checked directly: the URLs are mostly
opaque numeric IDs (`lratvakan.com/news/619886.html`), not headline slugs, so
there is no usable text to embed hiding in the data already collected.

DOC 2.0 supplies the missing modality: real article titles with timestamps,
free, no key. That is what makes Arm C a genuine THIRD modality rather than a
re-encoding of the same event metadata — which design rule 4 requires.

RATE LIMIT IS ONE REQUEST PER 5 SECONDS and exceeding it returns HTTP 429 with
the limit stated in the body (not a JSON error — parsing it as JSON fails with
a confusing "Expecting value" message). This collector sleeps 6s and treats a
429 as a back-off signal rather than a failure.

ONE REQUEST PER DAY. `maxrecords` caps at 250, so a multi-day query silently
returns only the top slice. Querying day by day keeps each request inside the
cap and gives an even sample across the window rather than a burst around the
loudest day.

⚠️ THE SAME LOOK-AHEAD RULE AS THE EVENTS STORE APPLIES. `seendate` is when
GDELT saw the article — the publication timestamp. Index on it. There is no
"event date" field here to be tempted by, which is a small mercy.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from ...common.paths import repo_root
from ...common.secrets import safe_stdout

log = logging.getLogger(__name__)

DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# Documented limit is one request per 5s, but 6s still drew repeated 429s in
# practice — the budget is shared across all users of the public endpoint, so
# the stated rate is a ceiling under contention, not a guarantee. 9s measured
# clean. Do NOT tune this down to make a run finish faster: a 429 returns a
# plain-text body, so an unhandled one parses as "Expecting value: line 1" and
# looks exactly like an empty day. That misread cost one wrong conclusion
# already ("DOC 2.0 has no 2019 coverage" — it has 250 headlines on 2019-06-13).
# 🔴 THE DOCUMENTED 5s LIMIT IS NOT ACHIEVABLE. Measured directly against the
# live endpoint, 5 requests per spacing:
#
#     30s spacing -> 1/5 succeeded
#     60s spacing -> 2/4 succeeded
#
# The limiter is a stateful rolling window that tightens under load on the
# shared public endpoint, not a simple per-request spacing rule. Request SHAPE
# is not the cause — a 20-record query 429'd while a 250-record one succeeded
# 20 seconds later. This is an external constraint; no client-side change makes
# the endpoint faster.
#
# The response: fetch MULTI-DAY chunks so the job needs ~5x fewer requests,
# and space them generously with long retries.
# Measured behaviour: an ISOLATED request succeeds (verified with a single
# probe returning HTTP 200), but any sustained series is blocked within one or
# two requests — including a series started 45s after a successful probe, which
# suggests the probe itself re-arms the limiter. The sustainable rate is
# therefore roughly one request every few MINUTES, not seconds.
#
# 180s x 71 chunks makes this a ~3.5 hour unattended job. That is acceptable:
# it costs only wall-clock, it checkpoints every 3 chunks, and it resumes.
# Do not shorten this to make a run finish sooner — a blocked run finishes
# faster and collects nothing.
RATE_LIMIT_S = 180.0
PENALTY_BACKOFF_S = 300.0
# Days per request. maxrecords caps at 250 regardless, so a 5-day chunk yields
# ~50 headlines/day — ample for a top-10 daily score, and 71 requests instead
# of 351 for the same span.
CHUNK_DAYS = 5
MAX_RECORDS = 250           # hard API cap

# Chokepoint / Gulf maritime vocabulary. Deliberately broader than the Events
# keyword filter: a headline is ~10 words, so a narrow query returns almost
# nothing, whereas the Events filter runs against structured actor+geo fields.
QUERY = (
    '("strait of hormuz" OR hormuz OR "persian gulf" OR "arabian gulf" OR '
    '"gulf of oman" OR "bab el-mandeb" OR "red sea" OR tanker OR '
    '"oil shipping" OR "strait" OR fujairah OR "jebel ali") '
)

WINDOWS = {
    "2019_gulf_of_oman": ("2019-05-01", "2019-07-15"),
    "2019_abqaiq":       ("2019-08-15", "2019-10-15"),
    "2024_red_sea":      ("2023-11-01", "2024-02-15"),
    "2026_hormuz":       ("2026-01-15", "2026-04-30"),
}


def out_dir():
    d = repo_root() / "data" / "raw" / "gdelt" / "headlines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_day(day: date, lang: str | None = None,
              retries: int = 5, span_days: int = 1) -> list[dict]:
    """
    Headlines for [day, day+span_days). Returns [] for a genuinely empty span.

    `sort=hybridrel` returns the most RELEVANT articles rather than the most
    recent, which matters for a multi-day chunk: date-sorted would fill the 250
    slots from the newest end and leave the earlier days of the chunk empty.
    Relevance-sorted, coverage spreads across the whole span — verify with the
    per-day distribution that `collect` logs.
    """
    q = QUERY + (f" sourcelang:{lang}" if lang else "")
    last = day + timedelta(days=span_days - 1)
    params = {
        "query": q,
        "mode": "artlist",
        "format": "json",
        "startdatetime": f"{day:%Y%m%d}000000",
        "enddatetime": f"{last:%Y%m%d}235959",
        "maxrecords": str(MAX_RECORDS),
        "sort": "hybridrel",
    }
    for attempt in range(retries):
        r = requests.get(DOC_URL, params=params, timeout=90,
                         headers={"User-Agent": "hormuz-research/1.0"})
        if r.status_code == 429:
            # 🔴 THE PENALTY WINDOW IS MINUTES, NOT SECONDS. Backing off in
            # 18/27/36s steps kept the collector permanently blocked: it burned
            # all three retries on the FIRST day and would have failed all 351
            # the same way, while a hand-issued request minutes earlier had
            # returned 250 headlines fine. Rapid retry is what SUSTAINS the
            # block — each attempt re-arms it. Back off in minutes.
            wait = PENALTY_BACKOFF_S * (attempt + 1)
            log.warning("429 on %s — penalty backoff %.0fs (attempt %d/%d)",
                        day, wait, attempt + 1, retries)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            log.error("HTTP %d on %s: %s", r.status_code, day, r.text[:160])
            return []
        if not r.content.strip():
            return []
        try:
            arts = r.json().get("articles", [])
        except ValueError:
            # A non-JSON 200 is how DOC 2.0 reports some soft errors.
            log.warning("non-JSON 200 on %s: %s", day, r.text[:120])
            return []
        for a in arts:
            a["query_date"] = str(day)
        return arts
    log.error("gave up on %s after %d attempts", day, retries)
    return []


def collect(window: str, lang: str | None = None) -> pd.DataFrame:
    lo, hi = WINDOWS[window]
    start, end = date.fromisoformat(lo), date.fromisoformat(hi)
    path = out_dir() / f"{window}.parquet"
    if path.exists():
        log.info("%s already collected (%s) — skipping", window, path.name)
        return pd.read_parquet(path)

    # Checkpoint so an interruption costs one day, not the whole window. At
    # ~12s per request a full window is 15-20 minutes of wall clock, and the
    # endpoint can penalty-box the IP at any point.
    ckpt = out_dir() / f"_partial_{window}.parquet"
    rows: list[dict] = []
    done_days: set[str] = set()
    if ckpt.exists():
        prev = pd.read_parquet(ckpt)
        rows = prev.to_dict("records")
        done_days = set(prev["query_date"].astype(str))
        log.info("resuming %s from checkpoint: %d headlines, %d days done",
                 window, len(rows), len(done_days))

    day = start
    n_chunks = ((end - start).days // CHUNK_DAYS) + 1
    i = 0
    while day <= end:
        i += 1
        if str(day) in done_days:
            day += timedelta(days=CHUNK_DAYS)
            continue
        span = min(CHUNK_DAYS, (end - day).days + 1)
        arts = fetch_day(day, lang=lang, span_days=span)
        rows.extend(arts)
        done_days.add(str(day))
        log.info("%s  chunk %d/%d (%s +%dd) · %d new · %d total",
                 window, i, n_chunks, day, span, len(arts), len(rows))
        if rows and i % 3 == 0:
            pd.DataFrame(rows).to_parquet(ckpt, index=False)
        time.sleep(RATE_LIMIT_S)
        day += timedelta(days=CHUNK_DAYS)
    if rows:
        pd.DataFrame(rows).to_parquet(ckpt, index=False)

    if not rows:
        log.error("%s produced ZERO headlines — check the query or coverage", window)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # seendate is the PUBLICATION timestamp — the only date safe to index on.
    df["seendate"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ",
                                    utc=True, errors="coerce")
    before = len(df)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    log.info("%s: %d headlines, %d after url dedup (%.1f%% dupes)",
             window, before, len(df), 100 * (1 - len(df) / max(before, 1)))
    df["window"] = window
    df.to_parquet(path, index=False)
    ckpt.unlink(missing_ok=True)
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect GDELT DOC 2.0 headlines")
    p.add_argument("--windows", default="all")
    p.add_argument("--lang", default=None,
                   help="sourcelang filter, e.g. english. Default: ALL languages "
                        "(Arabic/Farsi coverage is not optional for the Gulf).")
    a = p.parse_args(argv)

    names = list(WINDOWS) if a.windows == "all" else a.windows.split(",")
    total = 0
    for w in names:
        df = collect(w, lang=a.lang)
        total += len(df)
        if not df.empty:
            print(f"{w:22s} {len(df):6d} headlines  "
                  f"{df.seendate.min()} -> {df.seendate.max()}")
    print(f"{'TOTAL':22s} {total:6d}")
    return 0


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
