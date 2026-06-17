# ── Step 2 data: state x segment x sex shares via Spark SQL (Julie) ──
# Produces the table that powers the interactive dropdown map: for every
# state and cluster, the PERWT-weighted female and male share. Written in
# Spark SQL (rather than the DataFrame API) so the aggregation reads as
# plain SQL for the report.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_utils as su

spark = su.get_spark("segment_sex_sql")

clustered = su.with_state_code(
    spark.read.parquet(str(su.PROCESSED / "ipums_clustered.parquet")), "STATE_NAME"
)
clustered.createOrReplaceTempView("clustered")

result = spark.sql(
    """
    SELECT
        STATE_CODE,
        cluster,
        ROUND(100 * SUM(CASE WHEN SEX_LABEL = 'Female' THEN PERWT ELSE 0 END)
              / SUM(PERWT), 1)                              AS pct_female,
        ROUND(100 * SUM(CASE WHEN SEX_LABEL = 'Male' THEN PERWT ELSE 0 END)
              / SUM(PERWT), 1)                              AS pct_male,
        ROUND(SUM(PERWT), 0)                                AS weighted_workers
    FROM clustered
    WHERE STATE_CODE IS NOT NULL
    GROUP BY STATE_CODE, cluster
    ORDER BY STATE_CODE, cluster
    """
)

out = su.PROCESSED / "state_segment_sex.csv"
result.toPandas().to_csv(out, index=False)
print(f"Saved state x segment x sex shares -> {out}")
result.show(8, truncate=False)
spark.stop()
