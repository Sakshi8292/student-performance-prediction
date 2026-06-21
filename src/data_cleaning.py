"""
data_cleaning.py
-----------------
Cleans the raw student performance dataset:
 - Removes duplicate rows
 - Handles missing values (median/mode imputation)
 - Caps unrealistic outliers
 - Encodes categorical columns
 - Saves cleaned dataset

Run:
    python src/data_cleaning.py
Input:
    data/student_raw.csv
Output:
    data/student_clean.csv
"""

import numpy as np
import pandas as pd


def load_data(path="data/student_raw.csv"):
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"🧹 Removed {before - len(df)} duplicate rows")

    # 2. Handle missing values
    numeric_cols = ["attendance_pct", "previous_scores", "sleep_hours", "hours_studied"]
    categorical_cols = ["extracurricular", "parental_education", "internet_access"]

    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"🧹 Filled {missing} missing values in '{col}' with median={median_val}")

    for col in categorical_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"🧹 Filled {missing} missing values in '{col}' with mode='{mode_val}'")

    # 3. Fix unrealistic outliers (cap study hours to a believable max)
    outliers = (df["hours_studied"] > 16).sum()
    df.loc[df["hours_studied"] > 16, "hours_studied"] = df["hours_studied"].median()
    if outliers:
        print(f"🧹 Fixed {outliers} outlier values in 'hours_studied'")

    # 4. Encode categorical variables
    df["extracurricular"] = df["extracurricular"].map({"Yes": 1, "No": 0})
    df["internet_access"] = df["internet_access"].map({"Yes": 1, "No": 0})

    edu_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    df["parental_education"] = df["parental_education"].map(edu_order)

    # 5. Reset index
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw_df = load_data()
    print(f"📥 Loaded raw data: {raw_df.shape}")
    print(f"   Missing values per column:\n{raw_df.isna().sum()}\n")

    clean_df = clean_data(raw_df)
    clean_df.to_csv("data/student_clean.csv", index=False)

    print(f"\n✅ Cleaned data saved -> data/student_clean.csv  shape={clean_df.shape}")
    print(clean_df.head())
