"""
generate_dataset.py
--------------------
Generates a realistic synthetic "Student Performance" dataset.
Intentionally includes missing values, duplicate rows, and a few
outliers so that the Data Cleaning step has real work to do.

Run:
    python src/generate_dataset.py
Output:
    data/student_raw.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000  # number of students

# ---- Base features ----
hours_studied = np.round(np.random.normal(5, 2.2, N), 1).clip(0, 12)
attendance_pct = np.round(np.random.normal(80, 12, N), 1).clip(30, 100)
previous_scores = np.round(np.random.normal(65, 15, N), 1).clip(0, 100)
sleep_hours = np.round(np.random.normal(6.5, 1.3, N), 1).clip(3, 10)
extracurricular = np.random.choice(["Yes", "No"], size=N, p=[0.4, 0.6])
parental_education = np.random.choice(
    ["High School", "Bachelors", "Masters", "PhD"], size=N, p=[0.35, 0.4, 0.2, 0.05]
)
internet_access = np.random.choice(["Yes", "No"], size=N, p=[0.85, 0.15])

# ---- Target: final_score (with realistic relationships + noise) ----
edu_bonus = pd.Series(parental_education).map(
    {"High School": 0, "Bachelors": 2, "Masters": 4, "PhD": 6}
).values
extra_bonus = pd.Series(extracurricular).map({"Yes": 1.5, "No": 0}).values
internet_bonus = pd.Series(internet_access).map({"Yes": 2, "No": 0}).values

noise = np.random.normal(0, 6, N)

final_score = (
    1.8 * hours_studied
    + 0.3 * attendance_pct
    + 0.35 * previous_scores
    + 0.6 * sleep_hours
    + edu_bonus
    + extra_bonus
    + internet_bonus
    + noise
)
final_score = np.round(final_score.clip(0, 100), 1)

df = pd.DataFrame({
    "hours_studied": hours_studied,
    "attendance_pct": attendance_pct,
    "previous_scores": previous_scores,
    "sleep_hours": sleep_hours,
    "extracurricular": extracurricular,
    "parental_education": parental_education,
    "internet_access": internet_access,
    "final_score": final_score,
})

# ---- Intentionally dirty the data (for cleaning practice) ----
# 1) Random missing values
for col in ["attendance_pct", "previous_scores", "sleep_hours", "parental_education"]:
    missing_idx = np.random.choice(df.index, size=int(0.04 * N), replace=False)
    df.loc[missing_idx, col] = np.nan

# 2) Duplicate a few rows
dup_rows = df.sample(15, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# 3) A few unrealistic outliers
outlier_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_idx, "hours_studied"] = 25  # impossible study hours

# Shuffle rows
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/student_raw.csv", index=False)
print(f"✅ Raw dataset created: data/student_raw.csv  (rows={len(df)})")
print(df.head())
