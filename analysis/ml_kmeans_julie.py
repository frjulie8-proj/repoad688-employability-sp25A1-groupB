# ── Model A: KMeans on IPUMS (Julie) ───────────────────────────────
# Required unsupervised model. Clusters workers by earnings + age, then
# interprets the clusters against industry (NAICS-style sector) and sex.
# Feeds the gender-gap story: which wage/age segments are male- vs
# female-dominated, and what industries do they map to.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import spark_utils`
import spark_utils as su
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

FEATURES = ["INCWAGE", "AGE"]
K_RANGE = range(2, 8)
SEED = 42

spark = su.get_spark("kmeans_julie")
df = su.load_employed_spark(spark).cache()
print(f"Rows after cleaning: {df.count():,}")

# ── 1. Feature pipeline: assemble -> standardize ───────────────────
assembled = VectorAssembler(inputCols=FEATURES, outputCol="raw_features").transform(df)
scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True).fit(assembled)
data = scaler.transform(assembled).cache()

# ── 2. Choose k: silhouette + elbow on a 10% sample (fast & stable) ─
sample = data.sample(fraction=0.1, seed=SEED).cache()
evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
print("\n k | silhouette |   within-cluster cost (elbow)")
print("-" * 48)
sweep = {}
for k in K_RANGE:
    km = KMeans(featuresCol="features", k=k, seed=SEED)
    m = km.fit(sample)
    sil = evaluator.evaluate(m.transform(sample))
    cost = m.summary.trainingCost
    sweep[k] = sil
    print(f" {k} |   {sil:6.3f}   |   {cost:14,.0f}")

best_k = max(sweep, key=sweep.get)
print(f"\nChosen k = {best_k} (highest silhouette = {sweep[best_k]:.3f})")

# Standardized metrics row for the DRY model-evaluation section.
import pandas as _pd
_pd.DataFrame([
    {"model": "K-Means clustering", "type": "clustering",
     "metric": "Silhouette", "value": round(sweep[best_k], 4)},
    {"model": "K-Means clustering", "type": "clustering",
     "metric": "k", "value": float(best_k)},
]).to_csv(su.PROCESSED / "kmeans_metrics.csv", index=False)
print(f"Saved kmeans metrics -> {su.PROCESSED / 'kmeans_metrics.csv'}")

# ── 3. Fit final model on full data ────────────────────────────────
final = KMeans(featuresCol="features", k=best_k, seed=SEED).fit(data)
clustered = final.transform(data).withColumnRenamed("prediction", "cluster").cache()

# ── 4. Profile clusters (PERWT-weighted, per project convention) ────
profile = su.aggregate_to_state(
    clustered, ["cluster"],
    F.count("*").alias("n"),
    su.weighted_mean("INCWAGE", "PERWT", "wgt_avg_wage"),
    su.weighted_mean("AGE", "PERWT", "wgt_avg_age"),
    su.weighted_share("SEX_LABEL", "Female", "PERWT", "pct_female"),
).orderBy("cluster")
print("\n=== CLUSTER PROFILE (PERWT-weighted) ===")
profile.show(truncate=False)

# Dominant industry per cluster (weighted by PERWT)
ind_counts = (
    clustered.groupBy("cluster", "INDUSTRY")
    .agg(F.sum("PERWT").alias("w"))
)
w = Window.partitionBy("cluster").orderBy(F.desc("w"))
top_ind = (
    ind_counts.withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") <= 3)
    .groupBy("cluster")
    .agg(F.collect_list("INDUSTRY").alias("top3_industries"))
    .orderBy("cluster")
)
print("=== TOP-3 INDUSTRIES PER CLUSTER ===")
top_ind.show(truncate=False)

# ── 5. Save profile table for the notebook/report ──────────────────
out = su.PROCESSED / "kmeans_cluster_profile.csv"
prof_pd = profile.join(top_ind, "cluster").orderBy("cluster").toPandas()
prof_pd["top3_industries"] = prof_pd["top3_industries"].apply(lambda x: "; ".join(x))
prof_pd.to_csv(out, index=False)
print(f"\nSaved cluster profile -> {out}")
print(prof_pd.to_string(index=False))

# ── 6. Persist clustered person-level data for Step 2 (state join) ──
clustered_out = su.PROCESSED / "ipums_clustered.parquet"
(clustered.select("STATE_NAME", "cluster", "SEX_LABEL", "INCWAGE", "AGE", "PERWT", "INDUSTRY")
 .write.mode("overwrite").parquet(str(clustered_out)))
print(f"Saved clustered person-level data -> {clustered_out}")

# ── 7. Small sample for the cluster scatter ("neighbor graph") ─────
scatter_sample = (
    clustered.select("INCWAGE", "AGE", "cluster")
    .sample(fraction=0.01, seed=SEED).limit(4000).toPandas()
)
scatter_out = su.PROCESSED / "kmeans_scatter_sample.csv"
scatter_sample.to_csv(scatter_out, index=False)
print(f"Saved scatter sample ({len(scatter_sample)} rows) -> {scatter_out}")

spark.stop()
