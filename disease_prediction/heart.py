"""
heart.py
--------
Disease Prediction: Heart Disease
Dataset: UCI Heart Disease Dataset (Cleveland subset)

Features (13):
    age, sex, cp (chest pain type), trestbps (resting BP), chol (cholesterol),
    fbs (fasting blood sugar), restecg (resting ECG), thalach (max heart rate),
    exang (exercise-induced angina), oldpeak, slope, ca, thal
Target:
    target (0 = No Heart Disease, 1 = Heart Disease)

This script:
    1. Loads the dataset (already clean, no missing values).
    2. Splits + scales the data.
    3. Trains Logistic Regression, Random Forest, SVM, and XGBoost.
    4. Evaluates each with Accuracy, Precision, Recall, F1, ROC-AUC.
    5. Saves comparison plots, confusion matrices, ROC curves, and
       feature importance charts to the `outputs/` folder.
"""

import pandas as pd
from utils import (
    get_models, prepare_data, train_and_evaluate,
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_results_comparison
)

OUTPUT_DIR = "outputs/heart"


def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading heart disease dataset...")
    df = load_data()
    print(f"Shape: {df.shape}")
    print(df.describe().round(2))

    X = df.drop(columns="target")
    y = df["target"]
    feature_names = X.columns.tolist()

    print(f"\nClass balance:\n{y.value_counts(normalize=True).round(3)}")

    X_train, X_test, y_train, y_test, scaler = prepare_data(X.values, y.values)

    models = get_models()
    results_df, fitted = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        target_names=["No Heart Disease", "Heart Disease"]
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS — Heart Disease Prediction")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    plot_results_comparison(results_df, "Heart Disease", f"{OUTPUT_DIR}/comparison.png")
    plot_confusion_matrices(fitted, y_test, ["No Disease", "Disease"], "Heart Disease", f"{OUTPUT_DIR}/confusion_matrices.png")
    plot_roc_curves(fitted, y_test, "Heart Disease", f"{OUTPUT_DIR}/roc_curves.png")

    for name in ["Random Forest", "XGBoost"]:
        plot_feature_importance(
            fitted[name]["model"], feature_names, "Heart Disease", name,
            f"{OUTPUT_DIR}/feature_importance_{name.replace(' ', '_')}.png"
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    return results_df


if __name__ == "__main__":
    main()
