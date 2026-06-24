# Individual AI Disclosure — Antara Sudhir

**Project:** Gender Disparities in Hiring & Political Influence  
**Course:** AD688 Applied Business Analytics | Summer 2026  
**AI Tool:** Claude (Anthropic) — Claude Sonnet 4.5 / Claude Opus 4.6

This statement details my individual use of AI assistance and supplements the team disclosure ([team_disclosure.md](team_disclosure.md)). For my contributions — the EDA analysis, linear regression decomposition, and PySpark Random Forest — I used Claude as a coding partner, a debugging assistant, and a sounding board for analytical decisions. Every model decision, result validation, and final interpretation was my own.

---

## Code Development (AI-assisted)

Claude helped me write and debug Python and PySpark code for my contributions:

- `analysis/ml_wage_gap_antara.ipynb` — the linear regression decomposition pipeline (baseline model vs. full model with occupation and industry groups), the gender coefficient extraction, and the Oaxaca-Blinder-style decomposition summary.
- `analysis/ml_rf_antara.py` — the full PySpark MLlib Random Forest pipeline including Spark session configuration, default parallelism and shuffle partition reporting, repartitioning to match available cores, the StringIndexer → OneHotEncoder → VectorAssembler → RandomForestRegressor pipeline, feature importance extraction with readable labels derived from the IPUMS OCC codebook, and partial dependence analysis by age and gender.
- `pages/eda.qmd` — Plotly visualizations for the IT vs. overall workforce gender distribution, average wage comparison, wage distribution box plots, and Lightcast occupation group analysis.
- `pages/ml_methods.qmd` — the waterfall chart for wage gap decomposition, lollipop chart for feature importance, and partial dependence line chart.
- `analysis/live_demo_antara.ipynb` — the live presentation notebook combining regression decomposition and PySpark RF results.

---

## Idea Discussion & Exploration

I used Claude to think through analytical choices rather than to make them for me:

- Deciding to use regression decomposition (Oaxaca-Blinder approach) as the primary method for quantifying the explained vs. unexplained wage gap
- Choosing Random Forest as a stress-test of the regression finding rather than a replacement for it
- Framing the two-model approach as complementary — regression measures the gap size, Random Forest confirms gender's importance non-linearly
- Deciding to combine Male and Female PySpark OHE features into a single Gender signal for the feature importance chart
- Selecting OCC group bucketing (OCC // 100) and IND group bucketing (IND // 1000) as the occupation and industry encoding strategy

---

## Concept Explanation & Learning

I asked Claude to explain results so I could interpret my own output with confidence:

- The difference between R² as a measure of fit vs. the gender coefficient as a measure of effect
- Why low R² does not invalidate the gender coefficient in a wage decomposition analysis
- What PySpark's default parallelism and shuffle partitions mean and why repartitioning improves parallel training efficiency
- Why PySpark's OneHotEncoder produces separate Male and Female binary columns (unlike scikit-learn's drop_first=True) and how to combine them correctly in feature importance rankings
- The meaning of partial dependence plots and why they isolate gender's effect by holding other features at baseline values

---

## Writing & Interpretation

Claude helped draft interpretation paragraphs for my EDA and ML sections in `pages/eda.qmd` and `pages/ml_methods.qmd`, which I then reviewed, edited, and validated against my model outputs. Claude also helped draft my presentation speech and Q&A preparation, which I reviewed and adapted into my own words.

---

All code was executed and verified by me on our shared AWS EC2 instance. Model choices, hyperparameters, and final interpretations represent my own analytical judgment.
