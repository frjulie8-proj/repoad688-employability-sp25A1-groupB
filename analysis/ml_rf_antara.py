"""
PySpark Random Forest: Gender Wage Gap Analysis
Author: Antara Sudhir
Course: AD688 Applied Business Analytics

Follows the same pattern as analysis/ml_kmeans_julie.py:
- Uses shared spark_utils.py helpers
- Saves precomputed results to data/processed/
- Pages load CSVs; no model retraining on render
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "analysis"))

from spark_utils import get_spark, load_employed_spark, PROCESSED
from pyspark.sql import functions as F
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    # ── 1. Start Spark session ─────────────────────────────────────
    spark = get_spark(app_name="rf_wage_gap_antara", cores="local[*]")

    # ── 2. Show parallelism settings ───────────────────────────────
    print("=" * 55)
    print("SPARK CONFIGURATION")
    print("=" * 55)
    default_parallelism = spark.sparkContext.defaultParallelism
    default_partitions = spark.conf.get("spark.sql.shuffle.partitions")
    print(f"Default parallelism (cores):     {default_parallelism}")
    print(f"Default shuffle partitions:      {default_partitions}")

    # ── 3. Load data ───────────────────────────────────────────────
    print("\nLoading IPUMS employed_only data...")
    df = load_employed_spark(spark)
    df = df.select("AGE", "SEX_LABEL", "RACE_LABEL", "STATE_NAME",
                   "OCC", "IND", "INCWAGE", "PERWT")

    # Add broad occupation and industry groups
    df = df.withColumn("OCC_GROUP", ((F.col("OCC") / 100).cast("int") * 100).cast("string"))
    df = df.withColumn("IND_GROUP", ((F.col("IND") / 1000).cast("int") * 1000).cast("string"))

    print(f"Rows loaded: {df.count():,}")
    print(f"Default partitions: {df.rdd.getNumPartitions()}")

    # ── 4. Sample 200k rows for computational efficiency ───────────
    total = df.count()
    fraction = 200000 / total
    df_sample = df.sample(fraction=fraction, seed=42)
    print(f"\nSampled {df_sample.count():,} rows for training")

    # ── 5. Show repartitioning ─────────────────────────────────────
    print("\n--- WITHOUT repartitioning ---")
    print(f"Partitions before: {df_sample.rdd.getNumPartitions()}")

    df_repartitioned = df_sample.repartition(default_parallelism)
    print(f"\n--- WITH repartitioning (matched to cores) ---")
    print(f"Partitions after:  {df_repartitioned.rdd.getNumPartitions()}")
    print(f"Better utilizes all {default_parallelism} available cores")

    df_sample = df_repartitioned

    # ── 6. Build ML Pipeline ───────────────────────────────────────
    print("\nBuilding ML Pipeline...")

    # String indexers for categorical columns
    cat_cols = ["SEX_LABEL", "RACE_LABEL", "STATE_NAME", "OCC_GROUP", "IND_GROUP"]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in cat_cols]

    # One-hot encode indexed columns
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
                for c in cat_cols]

    # Assemble all features into a vector
    feature_cols = ["AGE"] + [f"{c}_ohe" for c in cat_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="keep")

    # Random Forest Regressor
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="INCWAGE",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

    # ── 7. Train/test split ────────────────────────────────────────
    train_df, test_df = df_sample.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train_df.count():,} | Test: {test_df.count():,}")

    # ── 8. Fit model ───────────────────────────────────────────────
    print("\nTraining Random Forest (PySpark MLlib)...")
    model = pipeline.fit(train_df)
    rf_model = model.stages[-1]

    # ── 9. Evaluate ────────────────────────────────────────────────
    evaluator = RegressionEvaluator(labelCol="INCWAGE", predictionCol="prediction",
                                    metricName="r2")
    train_r2 = evaluator.evaluate(model.transform(train_df))
    test_r2 = evaluator.evaluate(model.transform(test_df))

    print(f"\nTrain R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")

    # ── 10. Feature importance ─────────────────────────────────────
    print("\nExtracting feature importance...")
    importances = rf_model.featureImportances.toArray()

    # Get feature names from StringIndexer labels + OneHotEncoder sizes
    # This maps Spark OHE positions back to the original category values.
    feature_names = ["AGE"]

    for i, col in enumerate(cat_cols):
        indexer_model = model.stages[i]
        encoder_model = model.stages[len(indexers) + i]

        labels = list(indexer_model.labels)
        encoded_size = encoder_model.categorySizes[0] - 1

        for j in range(encoded_size):
            if j < len(labels):
                label_value = labels[j]
                feature_names.append(f"{col}_{label_value}")
            else:
                feature_names.append(f"{col}_unknown_{j}")

    # Pad or trim to match importance array length
    while len(feature_names) < len(importances):
        feature_names.append(f"feature_{len(feature_names)}")
    feature_names = feature_names[:len(importances)]

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df["is_gender"] = imp_df["feature"].str.contains("SEX_LABEL")
    sex_rank = imp_df[imp_df["is_gender"]].index[0] + 1 if imp_df["is_gender"].any() else "N/A"

    print(f"\nTop 15 features:")
    print(imp_df.head(15).to_string(index=False))
    print(f"\nGender rank: #{sex_rank} of {len(imp_df)}")

    # ── 11. Show decision tree structure ───────────────────────────
    print("\n--- DECISION TREE STRUCTURE (Tree 0 of 100) ---")
    print(rf_model.trees[0].toDebugString[:1000])

    # ── 12. Partial dependence by age and gender ───────────────────
    print("\nComputing partial dependence...")
    age_range = list(range(20, 66, 2))
    pdp_rows = []

    # Use mode values for categorical features
    mode_race = df_sample.groupBy("RACE_LABEL").count().orderBy("count", ascending=False).first()[0]
    mode_state = df_sample.groupBy("STATE_NAME").count().orderBy("count", ascending=False).first()[0]
    mode_occ = df_sample.groupBy("OCC_GROUP").count().orderBy("count", ascending=False).first()[0]
    mode_ind = df_sample.groupBy("IND_GROUP").count().orderBy("count", ascending=False).first()[0]

    for sex in ["Male", "Female"]:
        for age in age_range:
            pdp_rows.append({
                "AGE": age, "SEX": sex, "SEX_LABEL": sex,
                "RACE_LABEL": mode_race, "STATE_NAME": mode_state,
                "OCC_GROUP": mode_occ, "IND_GROUP": mode_ind, "INCWAGE": 0
            })

    pdp_spark = spark.createDataFrame(pdp_rows)
    pdp_preds = model.transform(pdp_spark)
    pdp_pd = pdp_preds.select("AGE", "SEX", "prediction").toPandas()
    pdp_pd = pdp_pd.rename(columns={"prediction": "predicted_wage"})

    # ── 13. Save all results ───────────────────────────────────────
    print("\nSaving results...")

    # RF model summary
    pd.DataFrame({
        "metric": ["Train R²", "Test R²", "Sample Size", "Num Trees",
                   "Max Depth", "Default Parallelism", "Shuffle Partitions"],
        "value": [train_r2, test_r2, df_sample.count(), 100, 10,
                  default_parallelism, int(default_partitions)]
    }).to_csv(PROCESSED / "rf_model_summary_antara.csv", index=False)
    print("✓ Saved rf_model_summary_antara.csv")

    # Feature importance (top 20)
    imp_df.head(20).to_csv(PROCESSED / "rf_feature_importance_antara.csv", index=False)
    print("✓ Saved rf_feature_importance_antara.csv")

    # Partial dependence
    pdp_pd.to_csv(PROCESSED / "rf_partial_dependence_age_gender_antara.csv", index=False)
    print("✓ Saved rf_partial_dependence_age_gender_antara.csv")

    print("\n✓ All PySpark RF results saved successfully!")
    spark.stop()

if __name__ == "__main__":
    main()
