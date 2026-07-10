"""
run_all.py
----------
Master script that runs all three disease prediction pipelines
(Diabetes, Heart Disease, Breast Cancer) end-to-end and prints a
combined summary table at the end.

Usage:
    python3 run_all.py
"""

import pandas as pd

import diabetes
import heart
import breast_cancer


def main():
    print("\n" + "#" * 70)
    print("#  DISEASE PREDICTION FROM MEDICAL DATA")
    print("#  Datasets: Diabetes | Heart Disease | Breast Cancer")
    print("#  Models:   Logistic Regression | Random Forest | SVM | XGBoost")
    print("#" * 70)

    print("\n\n" + "#" * 70)
    print("# 1/3  DIABETES")
    print("#" * 70)
    diabetes_results = diabetes.main()

    print("\n\n" + "#" * 70)
    print("# 2/3  HEART DISEASE")
    print("#" * 70)
    heart_results = heart.main()

    print("\n\n" + "#" * 70)
    print("# 3/3  BREAST CANCER")
    print("#" * 70)
    bc_results = breast_cancer.main()

    # Combine into one master summary table
    diabetes_results.insert(0, "Dataset", "Diabetes")
    heart_results.insert(0, "Dataset", "Heart Disease")
    bc_results.insert(0, "Dataset", "Breast Cancer")

    summary = pd.concat([diabetes_results, heart_results, bc_results], ignore_index=True)

    print("\n\n" + "=" * 70)
    print("MASTER SUMMARY — ALL DATASETS, ALL MODELS")
    print("=" * 70)
    print(summary.to_string(index=False))

    summary.to_csv("outputs/master_summary.csv", index=False)
    print("\nMaster summary saved to: outputs/master_summary.csv")

    # Best model per dataset (by ROC-AUC)
    best = summary.loc[summary.groupby("Dataset")["ROC-AUC"].idxmax()]
    print("\n" + "-" * 70)
    print("BEST MODEL PER DATASET (by ROC-AUC)")
    print("-" * 70)
    print(best[["Dataset", "Model", "Accuracy", "F1-Score", "ROC-AUC"]].to_string(index=False))


if __name__ == "__main__":
    main()
