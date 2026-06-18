# ── DRY model evaluation: one call, all models compared (Julie) ──────
# Reads each model's saved metric output (models are NOT re-trained here)
# and consolidates them into a single tidy comparison table, grouped by
# model type. Adding a model later = one new entry in SOURCES, no new
# metric code. The website's final section reads model_evaluation.csv.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import spark_utils as su

P = su.PROCESSED


def _rows(model, mtype, metric_values):
    """Normalize any model's metrics to the shared (model, type, metric, value) shape."""
    return [{"model": model, "type": mtype, "metric": m, "value": round(float(v), 4)}
            for m, v in metric_values.items()]


def evaluate_all():
    """Single entry point: gather every model's saved metrics into one table."""
    out = []

    # Clustering — K-Means (silhouette, k)
    km = pd.read_csv(P / "kmeans_metrics.csv")
    out += _rows("K-Means clustering", "clustering",
                 dict(zip(km["metric"], km["value"])))

    # Classification — gender logistic regression (already standardized)
    clf = pd.read_csv(P / "gender_clf_metrics.csv")
    out += _rows(clf["model"].iloc[0], "classification",
                 dict(zip(clf["metric"], clf["value"])))

    # Regression — Antara's linear models (R² from the decomposition table)
    dec = pd.read_csv(P / "wage_gap_decomposition_antara.csv")
    for _, r in dec.iterrows():
        out += _rows(f"Linear Regression — {r['model']}", "regression",
                     {"R2": r["r_squared"]})

    # Regression — Antara's Random Forest (held-out Test R²)
    rf = pd.read_csv(P / "rf_model_summary_antara.csv")
    rf_test = float(rf.loc[rf["metric"] == "Test R²", "value"].iloc[0])
    out += _rows("Random Forest Regressor", "regression", {"R2": rf_test})

    table = pd.DataFrame(out)
    # Order by type so the comparison reads clustering, classification, regression.
    order = {"clustering": 0, "classification": 1, "regression": 2}
    table = table.sort_values(["type", "model", "metric"],
                              key=lambda s: s.map(order) if s.name == "type" else s)
    table.to_csv(P / "model_evaluation.csv", index=False)
    return table


if __name__ == "__main__":
    t = evaluate_all()
    print(t.to_string(index=False))
    print(f"\nSaved -> {P / 'model_evaluation.csv'}")
