# Individual AI Disclosure — Julie Ngo Tran

**Project:** Gender Disparities in Hiring & Political Influence
**Course:** AD688 Applied Business Analytics | Summer 2026
**AI Tool:** Claude (Anthropic) — Claude Opus 4.6

This statement details my individual use of AI assistance and supplements the team disclosure ([team_disclosure.md](team_disclosure.md)). For my contributions — the K-Means clustering and Logistic Regression models — I used Claude as a coding partner, a sounding board for ideas, and a tutor for concepts I wanted to understand more deeply. Every decision, validation, and final interpretation was my own.

## Code Development (AI-assisted)

Claude helped me write and debug the PySpark code for my models:

- `analysis/ml_kmeans_julie.py` — the feature pipeline (VectorAssembler → StandardScaler), the silhouette-and-elbow sweep used to choose *k*, and the PERWT-weighted cluster profiling with the top-three industries per segment.
- `analysis/ml_gender_clf_julie.py` — the Logistic Regression pipeline (StringIndexer → OneHotEncoder → classifier), the ROC-curve point extraction, and the confusion matrix.
- `analysis/ml_state_join_julie.py` and `analysis/spark_utils.py` — the state-level supply/demand join on `STATE_CODE` and the shared Spark helpers.

## Idea Discussion & Exploration

I used Claude to think through analytical choices rather than to make them for me: selecting annual wage and age as the K-Means features, why an unsupervised model strengthens the gender-gap argument, framing the Logistic Regression as a measure of how separable — and therefore how segregated — men and women are by economic profile, and how to connect my two models to the regression and Random Forest work into one coherent story.

## Concept Explanation & Learning

I asked Claude to explain results so I could interpret my own output with confidence — the difference between ROC AUC and accuracy, the meaning of the silhouette score and cluster centroids, and why PERWT weighting reflects the population rather than the sample.

## Writing & Interpretation

Claude helped draft the clustering and classification interpretation paragraphs in `pages/ml_methods.qmd`, which I then reviewed, edited, and validated against my model outputs.

All code was executed and verified by me on our AWS EC2 instance; the model choices and final interpretations represent my own analytical judgment.
