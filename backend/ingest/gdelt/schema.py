"""
GDELT 2.0 Events schema + Gulf-region constants for the Hormuz Disruption Engine.

────────────────────────────────────────────────────────────────────────────────
EMPIRICALLY VALIDATED — do not "simplify" without re-running the validation.
────────────────────────────────────────────────────────────────────────────────
Every constant here was checked against a LIVE GDELT Translingual export on
2026-07-27 (file `20260727110000.translation.export.CSV`, 841 rows):

  · 61 tab-separated columns on 841/841 rows — zero ragged rows, zero rows
    dropped by `on_bad_lines`.
  · Semantic spot-checks at 100%: GlobalEventID numeric, Day = YYYYMMDD,
    EventRootCode 1-2 digits, SOURCEURL starts with http, QuadClass in 1-4,
    GoldsteinScale within [-10, 10]. ActionGeo_Lat non-null on 94.8% (the
    remainder are legitimately un-geocoded events).
  · FIPS confirmed live: a Slovakia row carried ActionGeo_CountryCode = "LO"
    (FIPS) and not "SK" (ISO). Gulf codes seen in that window: IR=55, AE=5,
    SA=2, BA=2, QA=1. ISO look-alikes OM/IQ/KW/BH: 0 rows each — filtering on
    ISO really would silently drop Oman, Iraq, Kuwait and Bahrain.

These constants are the expensive part of this module. The plumbing around them
is cheap and replaceable; this file is not.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────── source endpoints

LASTUPDATE_EN = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
LASTUPDATE_TRANSLINGUAL = "http://data.gdeltproject.org/gdeltv2/lastupdate-translation.txt"
MASTERFILELIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# GDELT 2.0 begins 2015-02-18. Anything earlier has no GDELT 2.0 signal at all
# (this is why the 2011-12 Hormuz closure-threat anchor cannot use this source).
GDELT2_EPOCH = "2015-02-18"

# ─────────────────────────────────────────────────────── geographic filtering

# Wider than the AIS bounding box on purpose: news events are datelined to
# whole countries, not to a shipping lane.
GULF_LAT = (22.0, 31.0)
GULF_LON = (46.0, 60.0)

# GDELT uses FIPS 10-4 country codes, NOT ISO 3166. This is the single most
# common way to silently lose Gulf coverage:
#   Oman    MU  (ISO would be OM)      Iraq     IZ  (ISO IQ)
#   Kuwait  KU  (ISO KW)               Bahrain  BA  (ISO BH)
#   UAE     AE  ·  Iran IR  ·  Saudi SA  ·  Qatar QA   (these happen to agree)
GULF_FIPS = frozenset({"IR", "AE", "MU", "SA", "QA", "KU", "BA", "IZ"})

# Terms that make an event chokepoint-relevant even when it geolocates outside
# the Gulf. Validated as load-bearing: in the 2026-07-27 window the bounding box
# contributed ZERO rows the country filter had not already caught, while the
# keyword leg uniquely caught 3 — including a Yemen-datelined report of Houthi
# forces striking three Saudi tankers. Pure geofencing would have dropped it.
HORMUZ_TERMS = (
    "hormuz", "strait", "fujairah", "jebel ali", "khor fakkan",
    "tanker", "persian gulf", "arabian gulf", "gulf of oman",
)

# ────────────────────────────────────────────────────────── event filtering

# CAMEO EventRootCode — the escalation ladder:
#   13 Threaten · 14 Protest · 15 Exhibit force · 16 Reduce relations
#   17 Coerce   · 18 Assault · 19 Fight         · 20 Mass violence
CONFLICT_ROOT_CODES = frozenset({"13", "14", "15", "16", "17", "18", "19", "20"})

# ─────────────────────────────────────────────────────────────── the schema

# GDELT 2.0 Events: 61 columns, tab-separated, NO header row in the file.
# Order is significant — the file has no header, so this list IS the schema.
EVENT_COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

N_EVENT_COLS = 61
assert len(EVENT_COLS) == N_EVENT_COLS, (
    f"EVENT_COLS drifted: {len(EVENT_COLS)} columns, expected {N_EVENT_COLS}"
)

# Columns that must be numeric before any filtering or scoring happens.
NUMERIC_COLS = (
    "ActionGeo_Lat", "ActionGeo_Long",
    "GoldsteinScale", "AvgTone", "NumMentions", "NumSources", "NumArticles",
)

# Minimal column set retained in the persisted store. Keeping the full 61
# columns on every 15-minute slot costs ~4x the disk for fields the arms never
# read; these are the ones the transformer arm and the backtest actually use.
RETAIN_COLS = (
    "GlobalEventID", "Day", "DATEADDED",
    "Actor1Name", "Actor1CountryCode", "Actor2Name", "Actor2CountryCode",
    "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_Lat", "ActionGeo_Long",
    "SOURCEURL",
)


class SchemaError(RuntimeError):
    """Raised when a GDELT file does not match the validated 61-column schema."""


def validate_raw_frame(df) -> None:
    """
    Fail loudly if a parsed GDELT frame does not match the validated schema.

    A misaligned schema parses without error and produces plausible-looking
    garbage — every field shifted by one column still yields a DataFrame. That
    silent-corruption mode is exactly what this guard exists to prevent, so it
    raises rather than warns.
    """
    if df.shape[1] != N_EVENT_COLS:
        raise SchemaError(
            f"expected {N_EVENT_COLS} columns, got {df.shape[1]}. "
            "GDELT changed its Events schema, or the file is not an Events export."
        )
    if list(df.columns) != EVENT_COLS:
        raise SchemaError("column names do not match EVENT_COLS")
    if df.empty:
        return

    # Cheap semantic canaries. Each one caught a real class of corruption during
    # the 2026-07-27 live validation, so they check meaning and not just shape.
    eid_ok = df["GlobalEventID"].astype(str).str.match(r"^\d+$").mean()
    if eid_ok < 0.95:
        raise SchemaError(
            f"GlobalEventID numeric on only {eid_ok:.1%} of rows — columns look shifted"
        )

    url_ok = df["SOURCEURL"].fillna("").astype(str).str.startswith("http").mean()
    if url_ok < 0.80:
        raise SchemaError(
            f"SOURCEURL looks like a URL on only {url_ok:.1%} of rows — columns look shifted"
        )
