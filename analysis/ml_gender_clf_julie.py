# ── Classification: predict gender — Logistic Regression (Julie) ─────
# Logistic regression (Spark MLlib) classifying Female vs Male from a
# worker's economic profile, using DETAILED occupation and industry codes
# (not coarse groups) since occupation is the strongest gender signal.
# Measures how separable women and men are by economic features; a high
# ROC AUC reflects gender-stratified jobs. (No Random Forest here — the
# project's tree model is the regression Random Forest in Antara's notebook.)
# NOTE: this IPUMS extract has no education or hours-worked columns, so the
# features are age, wage, occupation, industry, state, and race.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import spark_utils as su
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)
from pyspark.ml.functions import vector_to_array

SEED = 42
spark = su.get_spark("gender_clf_julie")

df = su.load_employed_spark(spark)
df = df.withColumn("label", F.when(F.col("SEX_LABEL") == "Female", 1.0).otherwise(0.0))
for c in ["OCC", "IND", "STATEFIP", "RACE"]:          # detailed codes as categoricals
    df = df.withColumn(c + "_S", F.col(c).cast("string"))

cats = ["OCC_S", "IND_S", "STATEFIP_S", "RACE_S"]
nums = ["AGE", "INCWAGE"]
idx = [StringIndexer(inputCol=c, outputCol=c + "_i", handleInvalid="keep") for c in cats]
enc = [OneHotEncoder(inputCol=c + "_i", outputCol=c + "_e") for c in cats]
asm = VectorAssembler(inputCols=nums + [c + "_e" for c in cats], outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)
pipeline = Pipeline(stages=idx + enc + [asm, lr])

train, test = df.randomSplit([0.8, 0.2], seed=SEED)
model = pipeline.fit(train)
pred = model.transform(test).cache()
print(f"Train {train.count():,}  Test {test.count():,}  (detailed OCC/IND features)")

# ── Metrics ────────────────────────────────────────────────────────
auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC").evaluate(pred)
mc = lambda m: MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName=m).evaluate(pred)
metrics = {"ROC_AUC": auc, "Accuracy": mc("accuracy"),
           "Precision": mc("weightedPrecision"), "Recall": mc("weightedRecall"),
           "F1": mc("f1")}
print("=== Logistic Regression test metrics ===")
for k, v in metrics.items():
    print(f"  {k:10s}: {v:.4f}")

# ── ROC curve points ───────────────────────────────────────────────
rp = (pred.select(vector_to_array("probability")[1].alias("p"), F.col("label"))
      .toPandas().sort_values("p", ascending=False).reset_index(drop=True))
P = (rp["label"] == 1).sum(); N = (rp["label"] == 0).sum()
rp["tpr"] = (rp["label"] == 1).cumsum() / P
rp["fpr"] = (rp["label"] == 0).cumsum() / N
ii = np.linspace(0, len(rp) - 1, 300).astype(int)
pd.concat([pd.DataFrame({"fpr": [0.0], "tpr": [0.0]}),
           rp.iloc[ii][["fpr", "tpr"]],
           pd.DataFrame({"fpr": [1.0], "tpr": [1.0]})], ignore_index=True
          ).to_csv(su.PROCESSED / "gender_clf_roc.csv", index=False)

# ── Confusion matrix ───────────────────────────────────────────────
conf = (pred.groupBy("label", "prediction").count()
        .orderBy("label", "prediction").toPandas())
print("\n=== Confusion matrix (label vs prediction) ===")
print(conf.to_string(index=False))
conf.to_csv(su.PROCESSED / "gender_clf_confusion.csv", index=False)

# ── Metrics CSV (standard schema for the DRY evaluation) ───────────
pd.DataFrame([{"model": "Gender classifier (Logistic Regression)",
               "type": "classification", "metric": k, "value": round(v, 4)}
              for k, v in metrics.items()]
             ).to_csv(su.PROCESSED / "gender_clf_metrics.csv", index=False)
print(f"\nSaved metrics/roc/confusion -> {su.PROCESSED}")
spark.stop()
