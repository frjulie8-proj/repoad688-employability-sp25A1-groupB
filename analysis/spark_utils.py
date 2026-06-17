# ── Spark mirror of utils.py (Julie) ──────────────────────────────
# Reusable Spark helpers so the KMeans + state-join pipeline stays DRY
# and matches the conventions in analysis/utils.py (same paths, same IND
# mapping). Both Model-A steps flow through these functions.
import os
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
FIGURES = BASE / "assets" / "figures"

# Same industry sector bins as utils.map_ind_to_industry(), expressed
# as (low_exclusive, high_inclusive, label) — one source of truth.
_IND_BINS = [
    (0, 290, "Agriculture"), (290, 490, "Mining"), (490, 770, "Construction"),
    (770, 3990, "Manufacturing"), (3990, 4090, "Wholesale Trade"),
    (4090, 4590, "Retail Trade"), (4590, 4690, "Transportation"),
    (4690, 5790, "Information"), (5790, 6390, "Finance & Insurance"),
    (6390, 6470, "Real Estate"), (6470, 6780, "Professional Services"),
    (6780, 7190, "Management"), (7190, 7580, "Administrative Services"),
    (7580, 7790, "Education"), (7790, 8470, "Healthcare"),
    (8470, 8590, "Arts & Entertainment"), (8590, 9290, "Accommodation & Food"),
    (9290, 9590, "Other Services"), (9590, 9870, "Public Administration"),
    (9870, 9999, "Military"),
]

# Mirrors utils._STATE_ABBREV (kept here so this module has no pandas/plotly
# dependency). The one join key shared by IPUMS and Lightcast.
_STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


# ── session ────────────────────────────────────────────────────────
def get_spark(app_name: str = "ad688", cores: str = "local[4]") -> SparkSession:
    """One place to build a quiet local Spark session."""
    os.environ.setdefault(
        "JAVA_HOME",
        os.popen("dirname $(dirname $(readlink -f $(which java)))").read().strip(),
    )
    spark = (
        SparkSession.builder.master(cores)
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ── shared column expressions (reused by BOTH steps) ───────────────
def map_ind_to_industry(col):
    """Spark column expr mirroring utils.map_ind_to_industry()."""
    expr = F.when(F.lit(False), None)
    for low, high, label in _IND_BINS:
        expr = expr.when((col > low) & (col <= high), F.lit(label))
    return expr


def weighted_mean(value_col: str, weight_col: str, alias: str):
    """PERWT-weighted mean as a single reusable agg expression."""
    return (F.sum(F.col(value_col) * F.col(weight_col)) / F.sum(weight_col)).alias(alias)


def weighted_share(when_col: str, equals, weight_col: str, alias: str):
    """PERWT-weighted share of rows where when_col == equals (e.g. % female)."""
    return (
        F.sum(F.when(F.col(when_col) == equals, F.col(weight_col)).otherwise(0))
        / F.sum(weight_col)
    ).alias(alias)


def with_state_code(df, name_col: str = "STATE_NAME"):
    """Attach STATE_CODE to an IPUMS frame from its full state name."""
    mapping = F.create_map([F.lit(x) for kv in _STATE_ABBREV.items() for x in kv])
    return df.withColumn("STATE_CODE", mapping[F.col(name_col)])


def parse_state_code(col):
    """Pull the trailing 2-letter state from a 'City, ST' Lightcast string."""
    return F.regexp_extract(col, r",\s*([A-Za-z]{2})\s*$", 1)


# ── generic aggregation + join (the DRY core of the pipeline) ──────
def aggregate_to_state(df, group_keys, *agg_exprs):
    """One aggregation function applied to BOTH the IPUMS and Lightcast
    sides — pass the grouping keys and whatever agg expressions each needs."""
    return df.groupBy(*group_keys).agg(*agg_exprs)


def join_on_state(supply, demand, keys=("STATE_CODE",)):
    """Single state-key join linking worker supply to employer demand."""
    return supply.join(demand, on=list(keys), how="inner")


# ── loaders ────────────────────────────────────────────────────────
def load_employed_spark(spark):
    """Spark twin of utils.load_employment(): positive wages, valid sex/state,
    industry sector attached, military/unmapped industries dropped."""
    df = spark.read.parquet(str(PROCESSED / "employed_only.parquet"))
    return (
        df.filter(F.col("INCWAGE") > 0)
        .dropna(subset=["SEX_LABEL", "STATE_NAME"])
        .withColumn("INDUSTRY", map_ind_to_industry(F.col("IND")))
        .dropna(subset=["INDUSTRY"])
    )


def load_lightcast_postings_spark(spark):
    """Posting-level Lightcast frame with parsed STATE_CODE + advertised salary,
    built by joining the location and relational tables on the posting ID."""
    loc = spark.read.parquet(str(PROCESSED / "lightcast_location.parquet"))
    rel = spark.read.parquet(str(PROCESSED / "lightcast_job_postings_relational.parquet"))
    loc = loc.withColumn("STATE_CODE", parse_state_code(F.col("CITY_NAME")))
    loc = loc.filter(F.col("STATE_CODE") != "")
    return loc.join(rel.select("ID", "SALARY_FROM"), on="ID", how="left")
