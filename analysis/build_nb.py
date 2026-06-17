import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# K-Means Cluster Graph — Full Data (Julie)

This notebook is the runnable, full-data version of the cluster scatter shown on the
**Predictive Modeling** page. The website renders a 4,000-row representative sample so the
HTML stays light; here we cluster and plot **all 1.57 million IPUMS workers** so the real
structure can be inspected directly.

It reuses the same Spark helpers in `analysis/spark_utils.py` as the rest of the pipeline,
so the clustering is identical to `analysis/ml_kmeans_julie.py`. Run top to bottom. The final
scatter uses Plotly WebGL, which can render the full point cloud interactively.""")

code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd() / "analysis"))
import spark_utils as su
from utils import apply_theme
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

FEATURES = ["INCWAGE", "AGE"]
K = 4
SEED = 42

spark = su.get_spark("kmeans_fulldata_julie")
df = su.load_employed_spark(spark).cache()
print(f"Workers after cleaning: {df.count():,}")""")

md("## 1. Feature pipeline and clustering (same as the production script)")

code("""assembled = VectorAssembler(inputCols=FEATURES, outputCol="raw_features").transform(df)
scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True).fit(assembled)
data = scaler.transform(assembled)

model = KMeans(featuresCol="features", k=K, seed=SEED).fit(data)
clustered = model.transform(data).withColumnRenamed("prediction", "cluster").cache()
print(f"Clusters: {K}")""")

md("""## 2. Centroids

A K-Means **centroid** is the center point of a cluster, equal to the mean of the workers
assigned to it. Because standardization is linear, the centroid in original units is just the
average age and wage of each cluster. We rank clusters by wage to give the same descriptive
segment names used on the website.""")

code("""centroids = (clustered.groupBy("cluster")
             .agg(F.avg("AGE").alias("AGE"), F.avg("INCWAGE").alias("INCWAGE"))
             .toPandas()
             .sort_values("INCWAGE").reset_index(drop=True))
SEGMENTS = ["Low-wage, early career", "Mid-wage, late career",
            "High-wage professional", "Elite earners"]
centroids["segment"] = SEGMENTS[: len(centroids)]
cluster_to_segment = dict(zip(centroids["cluster"], centroids["segment"]))
centroids""")

md("""## 3. Full-data cluster graph

This plots every worker in the age-by-wage feature space, colored by cluster, with the
centroids overlaid as black diamonds. Pulling 1.57 million rows to pandas and rendering them
takes a little time and memory; WebGL keeps it interactive. Reduce with a `.sample(...)` on
`clustered` if your machine is constrained.""")

code("""pdf = clustered.select("AGE", "INCWAGE", "cluster").toPandas()
pdf["segment"] = pdf["cluster"].map(cluster_to_segment)
print(f"Plotting {len(pdf):,} points")

fig = px.scatter(
    pdf, x="AGE", y="INCWAGE", color="segment",
    category_orders={"segment": SEGMENTS}, opacity=0.25,
    render_mode="webgl",
    labels={"AGE": "Age", "INCWAGE": "Annual wage (USD)", "segment": "Cluster"},
    title="K-Means clusters on all 1.57M workers (age by wage)",
)
fig.add_trace(go.Scatter(
    x=centroids["AGE"], y=centroids["INCWAGE"], mode="markers",
    marker=dict(symbol="diamond", size=16, color="black"),
    name="Centroid", hovertext=centroids["segment"], hoverinfo="text",
))
apply_theme(fig)
fig.show()""")

code("""spark.stop()""")

# Generates analysis/ml_kmeans_julie.ipynb. Run from anywhere:
#   python analysis/build_nb.py
nb["cells"] = cells
out = pathlib.Path(__file__).resolve().parent / "ml_kmeans_julie.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
