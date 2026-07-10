"""
diabetes.py
-----------
Disease Prediction: Diabetes
Dataset: Pima Indians Diabetes Dataset (UCI ML Repository)

Features:
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
    BMI, DiabetesPedigreeFunction, Age
Target:
    Outcome (0 = No Diabetes, 1 = Diabetes)

This script:
    1. Loads and cleans the data (handles biologically impossible
       zero-values in certain columns by imputing with the median).
    2. Splits + scales the data.
    3. Trains Logistic Regression, Random Forest, SVM, and XGBoost.
    4. Evaluates each with Accuracy, Precision, Recall, F1, ROC-AUC.
    5. Saves comparison plots, confusion matrices, ROC curves, and
       feature importance charts to the `outputs/` folder.
"""

import numpy as np
import pandas as pd
from utils import (
    get_models, prepare_data, train_and_evaluate,
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_results_comparison
)

OUTPUT_DIR = "outputs/diabetes"

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Columns where a value of 0 is biologically implausible and almost
# certainly represents a missing measurement rather than a true zero.
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path="data/diabetes.csv"):
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # Replace implausible zeros with NaN, then impute with the column median.
    # Median is used instead of mean because these columns are skewed by outliers.
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].astype(float)
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading diabetes dataset...")
    df = load_data()
    print(f"Shape: {df.shape}")
    print(df.describe().round(2))

    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    feature_names = X.columns.tolist()

    print(f"\nClass balance:\n{y.value_counts(normalize=True).round(3)}")

    X_train, X_test, y_train, y_test, scaler = prepare_data(X.values, y.values)

    models = get_models()
    results_df, fitted = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        target_names=["No Diabetes", "Diabetes"]
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS — Diabetes Prediction")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    plot_results_comparison(results_df, "Diabetes", f"{OUTPUT_DIR}/comparison.png")
    plot_confusion_matrices(fitted, y_test, ["No Diabetes", "Diabetes"], "Diabetes", f"{OUTPUT_DIR}/confusion_matrices.png")
    plot_roc_curves(fitted, y_test, "Diabetes", f"{OUTPUT_DIR}/roc_curves.png")

    for name in ["Random Forest", "XGBoost"]:
        plot_feature_importance(
            fitted[name]["model"], feature_names, "Diabetes", name,
            f"{OUTPUT_DIR}/feature_importance_{name.replace(' ', '_')}.png"
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    return results_df


if __name__ == "__main__":
    main()
