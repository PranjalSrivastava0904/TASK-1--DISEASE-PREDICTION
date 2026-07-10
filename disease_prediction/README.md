# Disease Prediction from Medical Data

Predicting the possibility of disease from structured patient data using
classical machine learning classifiers.

## Objective

Apply classification techniques to structured medical datasets to predict
the presence (or absence) of disease, and compare how different algorithms
perform on each problem.

## Datasets

| Dataset | Source | Samples | Features | Target |
|---|---|---|---|---|
| **Diabetes** | Pima Indians Diabetes Dataset (UCI) | 768 | 8 (glucose, BMI, age, etc.) | 0 = No Diabetes, 1 = Diabetes |
| **Heart Disease** | UCI Heart Disease (Cleveland) | 303 | 13 (age, cholesterol, chest pain type, etc.) | 0 = No Disease, 1 = Disease |
| **Breast Cancer** | Wisconsin Breast Cancer Diagnostic (UCI), via `sklearn.datasets` | 569 | 30 (cell nucleus measurements) | 0 = Malignant, 1 = Benign |

## Algorithms Used

- **Logistic Regression** — simple, interpretable linear baseline
- **Random Forest** — ensemble of decision trees, handles non-linearity well
- **SVM (RBF kernel)** — effective for smaller, high-dimensional datasets
- **XGBoost** — gradient-boosted trees, typically state-of-the-art on tabular data

All four models are trained and evaluated identically on each dataset for a fair comparison.

## Project Structure

```
disease_prediction/
├── data/
│   ├── diabetes.csv          # Pima Indians Diabetes (no header; 9 columns)
│   └── heart.csv             # UCI Heart Disease (Cleveland subset)
├── utils.py                  # Shared training/evaluation/plotting functions
├── diabetes.py                # Diabetes pipeline
├── heart.py                   # Heart disease pipeline
├── breast_cancer.py           # Breast cancer pipeline (data loaded via sklearn)
├── run_all.py                  # Runs all three pipelines + combined summary
├── requirements.txt
├── outputs/                    # Generated after running — plots + CSV results
│   ├── diabetes/
│   ├── heart/
│   ├── breast_cancer/
│   └── master_summary.csv
└── README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run a single disease pipeline:
   ```bash
   python3 diabetes.py
   python3 heart.py
   python3 breast_cancer.py
   ```

3. Or run everything at once with a combined summary:
   ```bash
   python3 run_all.py
   ```

## What Each Run Produces

For every dataset, the script will:

1. Load and clean the data (the diabetes dataset has biologically
   impossible zero-values in columns like Glucose and BMI — these are
   treated as missing and imputed with the column median).
2. Split into train/test sets (80/20, stratified by class) and scale
   features with `StandardScaler`.
3. Train all four models.
4. Print a classification report (precision, recall, F1) for each model.
5. Compute and tabulate **Accuracy, Precision, Recall, F1-Score, and
   ROC-AUC** for every model.
6. Save to `outputs/<dataset>/`:
   - `results.csv` — metrics table
   - `comparison.png` — bar chart comparing all metrics across models
   - `confusion_matrices.png` — confusion matrix grid (one per model)
   - `roc_curves.png` — overlaid ROC curves with AUC scores
   - `feature_importance_*.png` — top features for Random Forest and XGBoost

## Example Results

*(Your exact numbers may vary slightly by environment/random seed, but should be close to the table below.)*

| Dataset | Best Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Breast Cancer | Logistic Regression | 0.98 | 0.99 | 0.995 |
| Heart Disease | Random Forest | 0.82 | 0.85 | 0.912 |
| Diabetes | Random Forest | 0.74 | 0.60 | 0.816 |

**Takeaways:**
- Breast Cancer is the "easiest" problem — features are highly separable, so even simple Logistic Regression nearly saturates performance.
- Heart Disease and Diabetes are harder, noisier real-world clinical datasets; ensemble methods (Random Forest, XGBoost) generally edge out linear models, though the margin is dataset-dependent.
- ROC-AUC is reported alongside accuracy because medical datasets are often imbalanced — AUC is more robust to that imbalance than raw accuracy.

## Notes on Methodology

- **Stratified splitting** ensures the train/test split preserves the original class ratio — important since these datasets aren't perfectly balanced.
- **Feature scaling** is applied uniformly. It's essential for Logistic Regression and SVM (which are distance/gradient sensitive) and harmless for tree-based models.
- **Median imputation** (diabetes dataset only) is used instead of mean to reduce sensitivity to outliers in skewed clinical measurements (e.g. Insulin).
- This is a demonstration/coursework project. Models are trained with sensible default hyperparameters, not exhaustively tuned — there is room for improvement via `GridSearchCV` / `RandomizedSearchCV` if higher performance is required.
