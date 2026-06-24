# AI Disclosure

## Project: Gender Disparities in Hiring & Political Influence
**Course:** AD688 Cloud Data Analytics | Summer 2026  
**Group:** Antara, Julie, Yifan, Zhiyang, Alina

---

## AI Tools Used

### Claude (Anthropic)
- **Version:** Claude Sonnet 4.5 / Claude Opus 4.6
- **Purpose:** Used throughout the project for the following tasks:

#### Code Development
- Writing and debugging Python, PySpark, and Quarto code
- Building ETL pipelines
- Developing machine learning models
- Converting scikit-learn Random Forest to PySpark MLlib, including Spark session configuration, repartitioning, parallelism settings, and ML Pipeline construction
- Creating Plotly visualizations for EDA and ML sections
- Configuring GitHub Actions workflow for automated Quarto rendering

#### Analysis & Interpretation
- Interpreting linear regression decomposition results
- Explaining PySpark Random Forest feature importance rankings
- Drafting interpretation paragraphs for EDA and ML sections on the Quarto site

#### Writing & Documentation
- Drafting section introductions and methodology notes for `pages/eda.qmd` and `pages/ml_methods.qmd`

#### Debugging & Troubleshooting
- Resolving EC2 SSH connection issues
- Fixing GitHub Actions build failures
- Debugging git merge conflicts and `.gitignore` issues

---

## Human Contributions

All AI-generated content was reviewed, validated, and modified by team members before use. Specifically:

- **Research design** — research questions, dataset selection, and analytical approach were determined by the team
- **Data sourcing** — IPUMS USA 2024 ACS extract and Lightcast job postings dataset were independently obtained by the team
- **Model decisions** — choice of models (K-Means, Logistic Regression, Linear Regression, Random Forest) and hyperparameters were decided by the team
- **Results validation** — all model outputs, coefficients, and visualizations were verified by the team against expected values
- **Interpretation** — final interpretations and conclusions represent the team's own analytical judgment
- **Presentation** — slide design and delivery are entirely the team's own work

---

## Notes

- AI was used as a coding and writing assistant, not as a replacement for analytical thinking
- All code was executed and tested by team members on our own AWS EC2 infrastructure
- No AI tools were used to generate or fabricate data
- All datasets used are publicly available (IPUMS USA, Lightcast via course access)
