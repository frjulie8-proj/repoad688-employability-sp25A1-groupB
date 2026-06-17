# ── Step 2: link clusters to Lightcast demand via the state key (Julie) ──
# Reads Step 1's clustered IPUMS output, rolls the worker SUPPLY side up to
# state x cluster, rolls the Lightcast DEMAND side up to state, and joins on
# the single shared key (STATE_CODE). Both sides use the SAME
# aggregate_to_state() helper, so there is no duplicated groupBy logic.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_utils as su
from pyspark.sql import functions as F

spark = su.get_spark("state_join_julie")

# ── 1. SUPPLY side: clustered workers -> state x cluster ───────────
clustered = spark.read.parquet(str(su.PROCESSED / "ipums_clustered.parquet"))
clustered = su.with_state_code(clustered, "STATE_NAME")

supply = su.aggregate_to_state(
    clustered, ["STATE_CODE", "cluster"],
    F.count("*").alias("n_workers"),
    su.weighted_mean("INCWAGE", "PERWT", "avg_wage"),
    su.weighted_share("SEX_LABEL", "Female", "PERWT", "pct_female"),
)

# ── 2. DEMAND side: Lightcast postings -> state ────────────────────
postings = su.load_lightcast_postings_spark(spark)
demand = su.aggregate_to_state(
    postings, ["STATE_CODE"],
    F.count("*").alias("posting_count"),
    F.expr("percentile_approx(SALARY_FROM, 0.5)").alias("median_adv_salary"),
)

# ── 3. JOIN on the single shared key ───────────────────────────────
joined = su.join_on_state(supply, demand, keys=["STATE_CODE"]).orderBy("STATE_CODE", "cluster")
print(f"State x cluster rows: {joined.count()}  |  states covered: "
      f"{joined.select('STATE_CODE').distinct().count()}")

# ── 4. Save joined table for the report ────────────────────────────
out = su.PROCESSED / "state_cluster_demand.csv"
joined.toPandas().to_csv(out, index=False)
print(f"Saved state x cluster x demand table -> {out}")
joined.show(20, truncate=False)

# ── 5. Quick insight: does the gender gap track demand? ────────────
# Compare female share in the top-pay cluster across high- vs low-demand states.
med_demand = demand.approxQuantile("posting_count", [0.5], 0.01)[0]
top_cluster = (clustered.groupBy("cluster")
               .agg(su.weighted_mean("INCWAGE", "PERWT", "w"))
               .orderBy(F.desc("w")).first()["cluster"])
insight = (
    joined.filter(F.col("cluster") == top_cluster)
    .withColumn("demand_tier", F.when(F.col("posting_count") >= med_demand, "high-demand")
                .otherwise("low-demand"))
    .groupBy("demand_tier")
    .agg(F.round(F.avg("pct_female"), 3).alias("avg_pct_female_top_cluster"),
         F.count("*").alias("n_states"))
)
print(f"\nTop-pay cluster = {top_cluster}. Female share in it, by state demand tier:")
insight.show(truncate=False)

spark.stop()
