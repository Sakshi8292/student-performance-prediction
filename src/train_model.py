"""
train_model.py
----------------
Trains a Linear Regression model to predict a student's final_score
from study habits & demographic features.

Run:
    python src/train_model.py
Input:
    data/student_clean.csv
Output:
    models/linear_regression_model.pkl
    models/scaler.pkl
    plots/actual_vs_predicted.png
    plots/feature_importance.png
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "hours_studied",
    "attendance_pct",
    "previous_scores",
    "sleep_hours",
    "extracurricular",
    "parental_education",
    "internet_access",
]
TARGET = "final_score"


def main():
    df = pd.read_csv("data/student_clean.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (helps with interpretability of coefficients)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # ---- Evaluation ----
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("📊 Model Performance on Test Set")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R²   : {r2:.3f}")

    # ---- Save model + scaler ----
    joblib.dump(model, "models/linear_regression_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("\n✅ Saved model -> models/linear_regression_model.pkl")
    print("✅ Saved scaler -> models/scaler.pkl")

    # ---- Plot 1: Actual vs Predicted ----
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0")
    plt.plot([0, 100], [0, 100], "r--", lw=2)
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.title(f"Actual vs Predicted (R² = {r2:.3f})")
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png", dpi=120)
    plt.close()

    # ---- Plot 2: Feature importance (coefficients) ----
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_
    }).sort_values("coefficient", key=abs, ascending=True)

    plt.figure(figsize=(7, 5))
    colors = ["#C44E52" if c < 0 else "#55A868" for c in coef_df["coefficient"]]
    plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    plt.xlabel("Coefficient (impact on final score)")
    plt.title("Feature Importance - Linear Regression")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=120)
    plt.close()

    print("✅ Saved plots -> plots/actual_vs_predicted.png, plots/feature_importance.png")

    # ---- Save metrics to a small report ----
    with open("models/metrics.txt", "w") as f:
        f.write("Student Performance Prediction - Linear Regression\n")
        f.write("====================================================\n")
        f.write(f"MAE  : {mae:.2f}\n")
        f.write(f"RMSE : {rmse:.2f}\n")
        f.write(f"R2   : {r2:.3f}\n\n")
        f.write("Feature Coefficients (standardized):\n")
        for _, row in coef_df.sort_values("coefficient", ascending=False).iterrows():
            f.write(f"  {row['feature']:<20s}: {row['coefficient']:.3f}\n")


if __name__ == "__main__":
    main()
