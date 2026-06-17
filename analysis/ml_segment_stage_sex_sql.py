# ── Step 2 data: state x segment x career-stage x sex via Spark SQL (Julie) ──
# Powers the interactive Live Demo dashboard and the animated map. Adds a
# career-stage band derived from AGE, on top of the existing segment and sex
# dimensions, all PERWT-weighted. Written in Spark SQL for the report.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_utils as su

spark = su.get_spark("segment_stage_sex_sql")

clustered = su.with_state_code(
    spark.read.parquet(str(su.PROCESSED / "ipums_clustered.parquet")), "STATE_NAME"
)
clustered.createOrReplaceTempView("clustered")

result = spark.sql(
    """
    SELECT
        STATE_CODE,
        cluster,
        CASE WHEN AGE < 35 THEN 'Early (<35)'
             WHEN AGE < 55 THEN 'Mid (35-54)'
             ELSE 'Late (55+)' END                AS career_stage,
        SEX_LABEL                                  AS sex,
        ROUND(SUM(PERWT), 0)                       AS weighted
    FROM clustered
    WHERE STATE_CODE IS NOT NULL
    GROUP BY STATE_CODE, cluster,
             CASE WHEN AGE < 35 THEN 'Early (<35)'
                  WHEN AGE < 55 THEN 'Mid (35-54)'
                  ELSE 'Late (55+)' END,
             SEX_LABEL
    ORDER BY STATE_CODE, cluster, career_stage, sex
    """
)

out = su.PROCESSED / "state_segment_stage_sex.csv"
result.toPandas().to_csv(out, index=False)
print(f"Saved state x segment x stage x sex -> {out}  ({result.count()} rows)")
result.show(8, truncate=False)
spark.stop()
