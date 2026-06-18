# ── Classification: predict gender from labor-market profile (Julie) ──
# Logistic Regression (Spark MLlib) classifying Female vs Male from age,
# wage, occupation group, industry, state, and race. The point is not to
# "use" gender but to measure how separable the genders are by economic
# features: a high ROC AUC is itself evidence of structural sorting.
# Saves a metrics CSV and a confusion-matrix CSV for the evaluation section.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_utils as su
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)

SEED = 42
spark = su.get_spark("gender_clf_julie")

df = su.load_employed_spark(spark)
# Binary target: predict Female (1) vs Male (0).
df = df.withColumn("label", F.when(F.col("SEX_LABEL") == "Female", 1.0).otherwise(0.0))
# Grouped categoricals to keep cardinality sane.
df = (df.withColumn("OCC_GRP", (F.col("OCC") / 100).cast("int").cast("string"))
        .withColumn("STATE_S", F.col("STATEFIP").cast("string"))
        .withColumn("RACE_S", F.col("RACE").cast("string")))

cats = ["INDUSTRY", "OCC_GRP", "STATE_S", "RACE_S"]
nums = ["AGE", "INCWAGE"]

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
            for c in cats]
encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_oh") for c in cats]
assembler = VectorAssembler(inputCols=nums + [c + "_oh" for c in cats],
                            outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

train, test = df.randomSplit([0.8, 0.2], seed=SEED)
model = pipeline.fit(train)
pred = model.transform(test).cache()
print(f"Train rows: {train.count():,}  Test rows: {test.count():,}")

# ── Metrics (Spark evaluators) ─────────────────────────────────────
auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC").evaluate(pred)
mc = lambda m: MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName=m).evaluate(pred)
metrics = {
    "ROC_AUC": auc,
    "Accuracy": mc("accuracy"),
    "Precision": mc("weightedPrecision"),
    "Recall": mc("weightedRecall"),
    "F1": mc("f1"),
}
print("\n=== Gender classifier (Logistic Regression) test metrics ===")
for k, v in metrics.items():
    print(f"  {k:10s}: {v:.4f}")

# ── Confusion matrix ───────────────────────────────────────────────
conf = (pred.groupBy("label", "prediction").count()
        .orderBy("label", "prediction").toPandas())
print("\n=== Confusion matrix (label vs prediction) ===")
print(conf.to_string(index=False))

# ── Save outputs for the evaluation section ────────────────────────
import pandas as pd
mdf = pd.DataFrame(
    [{"model": "Gender classifier (Logistic Regression)", "type": "classification",
      "metric": k, "value": round(v, 4)} for k, v in metrics.items()])
out = su.PROCESSED / "gender_clf_metrics.csv"
mdf.to_csv(out, index=False)
conf.to_csv(su.PROCESSED / "gender_clf_confusion.csv", index=False)
print(f"\nSaved metrics -> {out}")
spark.stop()
