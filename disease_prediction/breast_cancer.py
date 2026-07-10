"""
breast_cancer.py
-----------------
Disease Prediction: Breast Cancer
Dataset: Wisconsin Breast Cancer Diagnostic Dataset (UCI ML Repository)
          (bundled directly in scikit-learn as `load_breast_cancer`)

Features (30):
    Mean, standard-error, and "worst" measurements of cell-nucleus
    characteristics from digitized images of breast tissue (e.g. radius,
    texture, perimeter, area, smoothness, compactness, symmetry, etc.)
Target:
    target (0 = Malignant, 1 = Benign)  [sklearn's encoding]

This script:
    1. Loads the dataset directly via sklearn.datasets.
    2. Splits + scales the data.
    3. Trains Logistic Regression, Random Forest, SVM, and XGBoost.
    4. Evaluates each with Accuracy, Precision, Recall, F1, ROC-AUC.
    5. Saves comparison plots, confusion matrices, ROC curves, and
       feature importance charts to the `outputs/` folder.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from utils import (
    get_models, prepare_data, train_and_evaluate,
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_results_comparison
)

OUTPUT_DIR = "outputs/breast_cancer"


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    # sklearn encodes 0 = malignant, 1 = benign
    class_names = list(data.target_names)
    return X, y, class_names


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading breast cancer dataset...")
    X, y, class_names = load_data()
    print(f"Shape: {X.shape}")
    print(f"Classes: {class_names}")

    feature_names = X.columns.tolist()
    print(f"\nClass balance:\n{y.value_counts(normalize=True).round(3)}")

    X_train, X_test, y_train, y_test, scaler = prepare_data(X.values, y.values)

    models = get_models()
    results_df, fitted = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        target_names=class_names
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS — Breast Cancer Prediction")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    plot_results_comparison(results_df, "Breast Cancer", f"{OUTPUT_DIR}/comparison.png")
    plot_confusion_matrices(fitted, y_test, class_names, "Breast Cancer", f"{OUTPUT_DIR}/confusion_matrices.png")
    plot_roc_curves(fitted, y_test, "Breast Cancer", f"{OUTPUT_DIR}/roc_curves.png")

    for name in ["Random Forest", "XGBoost"]:
        plot_feature_importance(
            fitted[name]["model"], feature_names, "Breast Cancer", name,
            f"{OUTPUT_DIR}/feature_importance_{name.replace(' ', '_')}.png"
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    return results_df


if __name__ == "__main__":
    main()
