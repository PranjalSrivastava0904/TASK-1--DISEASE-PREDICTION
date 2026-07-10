"""
utils.py
--------
Shared helper functions for the Disease Prediction project.

This module centralizes:
  - Train/test splitting + scaling
  - Model definitions (Logistic Regression, Random Forest, SVM, XGBoost)
  - Training + evaluation loop
  - Metric reporting (accuracy, precision, recall, F1, ROC-AUC)
  - Confusion matrix and ROC curve plotting
  - Feature importance plotting (for tree-based models)

Each disease-specific script (diabetes.py, heart.py, breast_cancer.py)
imports from here so the modeling logic stays consistent and DRY.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def get_models():
    """
    Returns a dictionary of model_name -> untrained model instance.
    All models use sensible default hyperparameters suitable for
    small/medium structured medical datasets.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "SVM (RBF kernel)": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def prepare_data(X, y, test_size=0.2, scale=True):
    """
    Splits data into train/test sets and optionally scales features.
    Scaling matters most for Logistic Regression and SVM; tree-based
    models (Random Forest, XGBoost) are scale-invariant but scaling
    them does no harm, so we apply it uniformly for simplicity.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    return X_train, X_test, y_train, y_test, scaler


def train_and_evaluate(models, X_train, X_test, y_train, y_test, target_names=None):
    """
    Trains each model and computes standard classification metrics.
    Returns a results DataFrame and a dict of fitted models + predictions
    (needed later for confusion matrices / ROC curves).
    """
    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Probability of positive class, needed for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        })

        fitted[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        print(f"\n{'='*60}\n{name}\n{'='*60}")
        print(classification_report(y_test, y_pred, target_names=target_names))

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return results_df, fitted


def plot_confusion_matrices(fitted, y_test, class_labels, dataset_name, save_path=None):
    """Plots a confusion matrix grid, one per model."""
    n = len(fitted)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, info) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, info["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=class_labels, yticklabels=class_labels, ax=ax
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Confusion Matrices — {dataset_name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_roc_curves(fitted, y_test, dataset_name, save_path=None):
    """Overlays ROC curves for all models on a single plot."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, info in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, info["y_proba"])
        auc = roc_auc_score(y_test, info["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_feature_importance(model, feature_names, dataset_name, model_name, save_path=None, top_n=15):
    """
    Plots feature importance for tree-based models (Random Forest, XGBoost).
    Skipped automatically for models without `feature_importances_`.
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importances))))
    importances.plot(kind="barh", ax=ax, color="#3b6ea5")
    ax.set_title(f"Feature Importance — {model_name} ({dataset_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_results_comparison(results_df, dataset_name, save_path=None):
    """Bar chart comparing all metrics across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    melted = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Model Comparison — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
