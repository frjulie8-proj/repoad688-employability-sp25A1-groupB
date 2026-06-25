# Gender Disparities in Hiring & Political Influence
**Course:** AD688 Cloud Data Analytics | Summer 2026  
**Group:** Antara Sudhir, Julie, Yifan, Zhiyang, Alina  
**Institution:** Boston University Metropolitan College

---

## Project Overview

This project investigates gender disparities in hiring and wages across the U.S. labor market using two datasets:

- **IPUMS USA 2024 ACS** — 1.49 million employed workers with wage, occupation, and demographic data
- **Lightcast Job Postings 2024** — 72,498 IT-focused job postings with salary, skills, and location data

**Central Research Question:** How much of the gender wage gap is explained by structural factors like occupation and industry choice — and how much persists even after controlling for those factors?

---

## Live Site

👉 [View the project site](https://frjulie8-proj.github.io/repoad688-employability-sum26-groupB/)
📊 [View the presentation slide deck](assets/AD688_Slide_Deck.pdf)

---

## Repository Structure

```
analysis/
├── utils.py                      Shared pandas/Plotly helpers
├── spark_utils.py                Shared PySpark helpers
│
├── ml_wage_gap_antara.ipynb      Linear regression decomposition
├── ml_rf_antara.py               PySpark Random Forest pipeline
├── live_demo_antara.ipynb        Live demo notebook — presentation
│
├── ml_kmeans_julie.ipynb         K-Means clustering notebook
├── ml_kmeans_julie.py            K-Means Spark MLlib pipeline
├── ml_gender_clf_julie.py        Logistic regression classifier
├── ml_state_join_julie.py        State-level demand join
├── ml_segment_sex_sql.py         Segment x sex Spark SQL
├── ml_segment_stage_sex_sql.py   Segment x stage x sex Spark SQL
├── model_eval.py                 Consolidated model evaluation
│
├── eda_exploration_antara.ipynb  EDA — IT vs overall workforce
├── eda_gender.ipynb              EDA — gender analysis
├── eda_political.ipynb           EDA — political influence
├── eda_yifan.ipynb               EDA — Yifan analysis
├── eda_zhiyang.ipynb             EDA — Zhiyang analysis
├── alina_salary_analysis.py      Salary analysis
└── industry_plot.py              Industry distribution plot
```

---

## Models Built

| Model | Author | Purpose |
|---|---|---|
| K-Means Clustering (Spark MLlib) | Julie | Segment workforce into wage tiers |
| Logistic Regression Classifier | Julie | Predict gender from economic features |
| Linear Regression Decomposition | Antara | Quantify explained vs unexplained wage gap |
| PySpark Random Forest Regressor | Antara | Confirm gender importance as wage predictor |

---

## Key Findings

- **90.6%** of the gender wage gap is unexplained by occupation and industry segregation
- Gender ranks **#4 of 172 features** in the PySpark Random Forest — behind only age and two occupation groups
- The wage gap widens sharply in the **late 20s**, suggesting it compounds during early career advancement
- The IT workforce is **72.4% male** vs 51.8% male overall

---

## Data Sources

- **IPUMS USA** — usa.ipums.org — 2024 ACS 1-year estimates
- **Lightcast** — Provided via course access — 2024 IT job postings

---

## Tech Stack

Python · PySpark · scikit-learn · Plotly · Quarto · GitHub Actions · AWS EC2
