import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

salary = pd.read_csv("data/processed/lightcast_salary.csv")

print(salary.shape)
print(salary.describe())
print(salary.isna().mean() * 100)

salary_clean = salary.dropna(subset=["SALARY_FROM"])

plt.figure(figsize=(8,5))
plt.hist(salary_clean["SALARY_FROM"], bins=30)
plt.xlim(0,300000)
plt.title("Lightcast Salary Distribution")
plt.xlabel("Salary From")
plt.ylabel("Number of Job Postings")
plt.tight_layout()
plt.savefig("assets/figures/alina_salary_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(6,5))
plt.boxplot(salary_clean["SALARY_FROM"])
plt.title("Salary Outlier Detection")
plt.ylabel("Salary From")
plt.tight_layout()
plt.savefig("assets/figures/alina_salary_boxplot.png", dpi=300)
plt.show()

print("Done. Figures saved.")
